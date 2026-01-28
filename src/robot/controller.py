"""
机器人控制器模块
整合机器人控制和检测功能的主要控制器类
"""
import time
import cv2
import Robot
from ..camera.realsense_camera import RealSenseCamera
from ..detection.color_detector import detect_object
from ..detection.pose_estimator import PoseEstimator
from ..utils.text_renderer import draw_text_with_bg
from ..config.detection_config import (
    CALIB_FILE,
    ROBOT_INITIAL_POSE,
    Z_OFFSET,
    ENABLE_NORMAL_ESTIMATION,
    TOOL_LENGTH,
    HOVER_HEIGHT,
    CENTERING_X_TRANSLATION_MM,
    CENTERING_Y_TRANSLATION_MM,
)
from ..config.robot_config import KEY_DEBOUNCE_TIME, DEFAULT_ROBOT_TOOL, DEFAULT_ROBOT_USER
from .motion_handler import RobotMotionHandler


class RobotDetectController:
    """
    机器人检测控制器
    整合了检测功能和机器人控制功能
    """
    
    def __init__(self, robot_ip, calib_file=None):
        """
        初始化机器人检测控制器
        
        Args:
            robot_ip: 机器人IP地址
            calib_file: 标定文件路径，如果为None则使用默认路径
        """
        # 初始化机器人连接
        self.robot = Robot.RPC(robot_ip)
        print(f"机器人连接成功")
        
        # 初始化相机
        if calib_file is None:
            calib_file = CALIB_FILE
        self.camera = RealSenseCamera(calib_file)
        
        # 获取标定数据
        R_cam2gripper, t_cam2gripper, R_base2cam, intrinsics = self.camera.get_calibration_data()
        
        # 初始化位姿估算器
        self.pose_estimator = PoseEstimator(R_cam2gripper, t_cam2gripper, R_base2cam, intrinsics)
        
        # 初始化运动处理器
        self.motion_handler = RobotMotionHandler(self.robot, DEFAULT_ROBOT_TOOL, DEFAULT_ROBOT_USER)
        
        # 暂停状态和记录的坐标
        # 流程状态：
        # - LIVE: 实时检测
        # - PAUSED_SNAPSHOT1: 第一次拍照/暂停（等待按W）
        # - AUTO_WAIT_ADJUST_DONE: 已按W开始调姿，等待调姿移动完成
        # - AUTO_WAIT_CENTERING_DONE: 已做居中X平移，等待完成
        # - AUTO_CAPTURE_SNAPSHOT2: 二次拍照检测（用于计算仅XYZ目标，保持rxryrz不变）
        # - AUTO_MOVING_FINAL: 正在移动到二次检测得到的XYZ目标
        self.flow_state = "LIVE"

        self.detected_pose = None  # 实时检测到的目标位姿 [x, y, z, rx, ry, rz]
        self.last_live_frame = None
        self.last_live_center = None

        self.snapshot1_pose = None
        self.snapshot1_frame = None
        self.snapshot2_pose = None
        self.snapshot2_frame = None

        # 固定姿态（来自第一次检测/调姿），用于二次复拍只改XYZ
        self.fixed_euler = None

        # 二次复拍前的居中补偿
        self.centering_x_translation_mm = float(CENTERING_X_TRANSLATION_MM)
        self.centering_y_translation_mm = float(CENTERING_Y_TRANSLATION_MM)
        
        # 按键防抖：记录上次按键时间，避免重复触发
        self.last_key_time = {}
        self.key_debounce_time = KEY_DEBOUNCE_TIME
        
        # 初始位姿
        self.initial_pose = ROBOT_INITIAL_POSE.copy()
        
        # Z偏移量
        self.z_offset = Z_OFFSET
        
        # 悬浮高度（用于两阶段运动）
        self.hover_height = HOVER_HEIGHT
    
    def get_current_pose(self):
        """获取当前机器人TCP位姿"""
        error, pose = self.robot.GetActualTCPPose()
        if error == 0:
            return pose  # [x, y, z, rx, ry, rz]
        else:
            print(f"获取当前位姿失败，错误码: {error}")
            return None
    
    def move_to_initial_pose(self):
        """
        回到初始位姿（两阶段）
        1) 先仅调整 z 到初始高度（避免碰撞）
        2) 再调整其它参数回到初始位姿
        """
        current_pose = self.get_current_pose()
        if current_pose is None:
            # 获取当前位姿失败时，退化为直接回初始位姿
            print(f"获取当前位姿失败，直接回到初始位姿: {self.initial_pose}")
            return self.motion_handler.move_to_pose(self.initial_pose)

        z_only_pose = current_pose.copy()
        z_only_pose[2] = self.initial_pose[2]

        print("\n两阶段回到初始位姿：")
        print(f"  第一阶段：仅调整Z到初始高度 z={self.initial_pose[2]:.3f}mm")
        print(f"  第二阶段：回到初始位姿 {self.initial_pose}")
        return self.motion_handler.move_to_pose_two_pose(
            z_only_pose,
            self.initial_pose,
            first_stage_desc="第一阶段：仅调整Z到初始高度",
            second_stage_desc="第二阶段：回到初始位姿",
        )
    
    def save_detected_pose(self):
        """保存第一次拍照检测到的物体坐标（用于后续按W调姿）"""
        if self.detected_pose is not None and self.last_live_frame is not None:
            self.snapshot1_pose = self.detected_pose.copy()
            self.snapshot1_frame = self.last_live_frame.copy()
            self.fixed_euler = self.snapshot1_pose[3:6].copy()
            print(
                f"\n第一次拍照已保存坐标: [{self.snapshot1_pose[0]:.3f}, {self.snapshot1_pose[1]:.3f}, "
                f"{self.snapshot1_pose[2]:.3f}, {self.snapshot1_pose[3]:.3f}, {self.snapshot1_pose[4]:.3f}, "
                f"{self.snapshot1_pose[5]:.3f}]"
            )
            return True
        else:
            print("\n未检测到物体，无法拍照保存坐标")
            return False

    def _reset_workflow(self):
        """重置 q/w 自动流程的中间状态"""
        self.flow_state = "LIVE"
        self.snapshot1_pose = None
        self.snapshot1_frame = None
        self.snapshot2_pose = None
        self.snapshot2_frame = None
        self.fixed_euler = None
        self.last_live_center = None
    
    def _start_adjust_xy_and_orientation_keep_z(self):
        """
        第一步：按W后先调整 x/y + rx/ry/rz 到第一次拍照的姿态，保持 z 不动。
        """
        if self.snapshot1_pose is None:
            print("\n没有第一次拍照坐标，请先按 'q' 拍照保存")
            return False

        if self.motion_handler.is_moving():
            print("\n机器人正在移动中，请等待当前移动完成")
            return False

        current_pose = self.get_current_pose()
        if current_pose is None:
            print("\n获取当前位姿失败，无法执行调姿")
            return False

        target_pose = current_pose.copy()
        target_pose[0] = float(self.snapshot1_pose[0])
        target_pose[1] = float(self.snapshot1_pose[1])
        # 保持Z不动
        target_pose[3] = float(self.snapshot1_pose[3])
        target_pose[4] = float(self.snapshot1_pose[4])
        target_pose[5] = float(self.snapshot1_pose[5])

        print("\n开始第一步调姿（保持Z不动）：")
        print(
            f"  目标: [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}, "
            f"{target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}]"
        )
        ok = self.motion_handler.move_to_pose(target_pose)
        if ok:
            self.flow_state = "AUTO_WAIT_ADJUST_DONE"
        return ok
    
    def run(self):
        """主运行循环"""
        try:
            mode_text = "法向量估计模式" if ENABLE_NORMAL_ESTIMATION else "固定姿态模式"
            print(f"模式: {mode_text}")
            print(f"TCP补偿长度: {TOOL_LENGTH} mm")
            print(f"Z偏移量: {self.z_offset} mm")
            print(f"悬浮高度: {self.hover_height} mm")
            print(
                f"二次复拍前居中平移: dx={self.centering_x_translation_mm:.3f} mm, "
                f"dy={self.centering_y_translation_mm:.3f} mm"
            )
            print("\n快捷键说明:")
            print("  'q' - 第一次拍照并暂停（保存坐标/姿态），再次按Q取消暂停")
            print("  'w' - 在第一次拍照暂停界面：先调XY+RxRyRz(保持Z不动)，然后自动：XY居中补偿→二次拍照→只动XYZ(保持姿态)")
            print("  'b' - 回到detect的初始位姿")
            print("  'z' - 停止机器人移动（全局快捷键）")
            print("  'ESC' - 退出程序")
            
            while True:
                current_time = time.time()

                # 1) 根据流程状态决定显示内容
                display_frame = None
                if self.flow_state == "PAUSED_SNAPSHOT1" and self.snapshot1_frame is not None:
                    display_frame = self.snapshot1_frame.copy()
                elif self.flow_state == "AUTO_MOVING_FINAL" and self.snapshot2_frame is not None:
                    display_frame = self.snapshot2_frame.copy()

                # 2) 非暂停界面时实时取流并检测（用于显示，也用于 AUTO_CAPTURE_SNAPSHOT2）
                if display_frame is None:
                    color_img, depth_frame, _ = self.camera.get_frame_data()
                    if color_img is None:
                        continue

                    center, found = detect_object(color_img, depth_frame)
                    if found:
                        u, v = center
                        current_pose = self.get_current_pose()
                        if current_pose is not None:
                            # AUTO_CAPTURE_SNAPSHOT2：二次拍照只改XYZ，保持第一次调姿的 rx/ry/rz
                            if self.flow_state == "AUTO_CAPTURE_SNAPSHOT2" and self.fixed_euler is not None:
                                result = self.pose_estimator.calculate_robot_target(
                                    u,
                                    v,
                                    depth_frame,
                                    current_robot_pose=current_pose,
                                    target_euler_override=self.fixed_euler,
                                    enable_normal_estimation_override=False,
                                )
                                if result is not None:
                                    raw_pos, _, _ = result
                                    x2, y2, z2 = raw_pos
                                    final_pose = [x2, y2, z2, *self.fixed_euler]
                                    self.snapshot2_pose = final_pose.copy()
                                    self.snapshot2_frame = color_img.copy()

                                    print("\n二次拍照完成：开始移动到二次检测XYZ（保持姿态不变）")
                                    self.motion_handler.move_to_pose_two_stage(final_pose, self.hover_height)
                                    self.flow_state = "AUTO_MOVING_FINAL"
                            else:
                                # LIVE / 其它状态：正常实时估算（始终从当前机器人位姿换算）
                                result = self.pose_estimator.calculate_robot_target(
                                    u,
                                    v,
                                    depth_frame,
                                    current_pose,
                                )
                                if result is not None:
                                    raw_pos, raw_rot, _ = result
                                    x, y, z = raw_pos
                                    rx, ry, rz = raw_rot
                                    self.detected_pose = [x, y, z, rx, ry, rz]
                                    self.last_live_center = center

                    # 保存最后一帧（用于按Q拍照）
                    self.last_live_frame = color_img.copy()
                    display_frame = color_img

                # 3) 叠加状态提示
                y_offset_left = 15
                if self.motion_handler.is_moving():
                    display_frame = draw_text_with_bg(
                        display_frame,
                        "机器人正在移动中...",
                        (15, y_offset_left),
                        font_size=16,
                        text_color=(0, 255, 255),
                        bg_color=(255, 0, 0),
                        alpha=0.5,
                    )
                    y_offset_left += 34

                state_text = f"状态: {self.flow_state}"
                display_frame = draw_text_with_bg(
                    display_frame,
                    state_text,
                    (15, y_offset_left),
                    font_size=16,
                    text_color=(255, 255, 255),
                    bg_color=(0, 0, 0),
                    alpha=0.5,
                )

                # 暂停界面提示
                if self.flow_state == "PAUSED_SNAPSHOT1":
                    img_width = display_frame.shape[1]
                    display_frame = draw_text_with_bg(
                        display_frame,
                        "第一次拍照已暂停 - 按W开始自动流程 / 按Q取消",
                        (max(15, img_width - 520), 15),
                        font_size=16,
                        text_color=(255, 255, 255),
                        bg_color=(0, 0, 255),
                        alpha=0.5,
                    )

                # 4) 显示窗口
                cv2.imshow("Detection", display_frame)

                # 5) 读取按键并处理（带防抖）
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    if 'q' not in self.last_key_time or (current_time - self.last_key_time['q']) > self.key_debounce_time:
                        if self.flow_state == "LIVE":
                            # 第一次拍照保存，并进入暂停等待W
                            if self.save_detected_pose():
                                self.flow_state = "PAUSED_SNAPSHOT1"
                                print("\n画面已暂停（第一次拍照）")
                        elif self.flow_state == "PAUSED_SNAPSHOT1":
                            print("\n取消暂停，回到实时检测")
                            self._reset_workflow()
                        else:
                            # 自动流程中按Q：紧急退出自动流程（不强制停止机器人，但允许用户接管）
                            print("\n已退出自动流程，回到实时检测（如需停止请按Z）")
                            self._reset_workflow()
                        self.last_key_time['q'] = current_time

                elif key == ord('w'):
                    if 'w' not in self.last_key_time or (current_time - self.last_key_time['w']) > self.key_debounce_time:
                        if self.flow_state == "PAUSED_SNAPSHOT1":
                            ok = self._start_adjust_xy_and_orientation_keep_z()
                            if ok:
                                print("已启动第一步调姿（后台执行）")
                        else:
                            print("\n请先按 'q' 第一次拍照暂停后再按 'w'")
                        self.last_key_time['w'] = current_time

                elif key == ord('b'):
                    if 'b' not in self.last_key_time or (current_time - self.last_key_time['b']) > self.key_debounce_time:
                        if self.motion_handler.is_moving():
                            print("\n机器人正在移动中，请等待当前移动完成")
                        else:
                            print("\n回到初始位姿...")
                            self.move_to_initial_pose()
                        self.last_key_time['b'] = current_time

                elif key == ord('z'):
                    if 'z' not in self.last_key_time or (current_time - self.last_key_time['z']) > self.key_debounce_time:
                        self.motion_handler.stop_robot_motion()
                        self.last_key_time['z'] = current_time

                elif key == 27:
                    break

                # 6) 自动流程状态机：在每帧末尾推进
                if self.flow_state == "AUTO_WAIT_ADJUST_DONE":
                    if not self.motion_handler.is_moving():
                        if (
                            abs(self.centering_x_translation_mm) > 1e-6
                            or abs(self.centering_y_translation_mm) > 1e-6
                        ):
                            pose = self.get_current_pose()
                            if pose is not None:
                                pose2 = pose.copy()
                                pose2[0] = float(pose2[0]) + self.centering_x_translation_mm
                                pose2[1] = float(pose2[1]) + self.centering_y_translation_mm
                                print("\n居中补偿：仅平移XY（保持Z/姿态不变）")
                                print(
                                    f"  dx={self.centering_x_translation_mm:.3f}mm, "
                                    f"dy={self.centering_y_translation_mm:.3f}mm"
                                )
                                print(f"  目标XY=({pose2[0]:.3f}, {pose2[1]:.3f})")
                                self.motion_handler.move_to_pose(pose2)
                                self.flow_state = "AUTO_WAIT_CENTERING_DONE"
                            else:
                                print("\n获取当前位姿失败，跳过居中补偿，进入二次拍照")
                                self.flow_state = "AUTO_CAPTURE_SNAPSHOT2"
                        else:
                            self.flow_state = "AUTO_CAPTURE_SNAPSHOT2"

                elif self.flow_state == "AUTO_WAIT_CENTERING_DONE":
                    if not self.motion_handler.is_moving():
                        print("\n居中补偿完成，进入二次拍照检测...")
                        self.flow_state = "AUTO_CAPTURE_SNAPSHOT2"

                elif self.flow_state == "AUTO_MOVING_FINAL":
                    if not self.motion_handler.is_moving():
                        print("\n自动流程完成，回到实时检测")
                        self._reset_workflow()
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            self.robot.CloseRPC()
            print("\n程序已退出")
