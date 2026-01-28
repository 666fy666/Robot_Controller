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
    CALIB_FILE, ROBOT_INITIAL_POSE, Z_OFFSET, ENABLE_NORMAL_ESTIMATION, TOOL_LENGTH, HOVER_HEIGHT
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
        self.paused = False
        self.saved_pose = None  # 保存的检测到的物体坐标 [x, y, z, rx, ry, rz]
        self.detected_pose = None  # 当前检测到的物体坐标（实时更新）
        
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
        """保存检测到的物体坐标"""
        if self.detected_pose is not None:
            self.saved_pose = self.detected_pose.copy()
            print(f"\n已保存检测到的物体坐标: [{self.saved_pose[0]:.3f}, {self.saved_pose[1]:.3f}, "
                  f"{self.saved_pose[2]:.3f}, {self.saved_pose[3]:.3f}, {self.saved_pose[4]:.3f}, "
                  f"{self.saved_pose[5]:.3f}]")
            return True
        else:
            print("\n未检测到物体，无法保存坐标")
            return False
    
    def move_to_saved_pose(self):
        """
        两阶段移动到保存的检测到的物体坐标
        第一阶段：移动到悬停位置（z坐标增加hover_height）
        第二阶段：下降到目标位置
        """
        if self.saved_pose is None:
            print("没有保存的检测坐标，请先按 'q' 暂停并保存检测到的物体坐标")
            return False
        
        print(f"\n两阶段移动到检测到的物体坐标:")
        print(f"  目标位置: [{self.saved_pose[0]:.3f}, {self.saved_pose[1]:.3f}, "
              f"{self.saved_pose[2]:.3f}, {self.saved_pose[3]:.3f}, {self.saved_pose[4]:.3f}, "
              f"{self.saved_pose[5]:.3f}]")
        print(f"  悬浮高度: {self.hover_height:.3f}mm")
        return self.motion_handler.move_to_pose_two_stage(self.saved_pose, self.hover_height)
    
    def run(self):
        """主运行循环"""
        try:
            mode_text = "法向量估计模式" if ENABLE_NORMAL_ESTIMATION else "固定姿态模式"
            print(f"模式: {mode_text}")
            print(f"TCP补偿长度: {TOOL_LENGTH} mm")
            print(f"Z偏移量: {self.z_offset} mm")
            print(f"悬浮高度: {self.hover_height} mm")
            print("\n快捷键说明:")
            print("  'q' - 暂停/恢复，暂停时会保存检测到的物体坐标")
            print("  'w' - 暂停状态下，移动到检测到的物体坐标")
            print("  'b' - 回到detect的初始位姿")
            print("  'z' - 停止机器人移动（全局快捷键）")
            print("  'ESC' - 退出程序")
            
            paused_frame = None
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                current_time = time.time()
                
                # 处理按键（带防抖）
                if key == ord('q'):
                    # 防抖：检查距离上次按q的时间
                    if 'q' not in self.last_key_time or (current_time - self.last_key_time['q']) > self.key_debounce_time:
                        self.paused = not self.paused
                        if self.paused:
                            print("\n画面已暂停")
                            # 暂停时保存检测到的物体坐标
                            self.save_detected_pose()
                        else:
                            print("\n画面已恢复")
                            paused_frame = None
                        self.last_key_time['q'] = current_time
                
                elif key == ord('w'):
                    # 防抖：检查距离上次按w的时间
                    if 'w' not in self.last_key_time or (current_time - self.last_key_time['w']) > self.key_debounce_time:
                        if self.paused:
                            # 暂停状态下，移动到保存的检测到的物体坐标
                            if self.saved_pose is not None:
                                if self.motion_handler.is_moving():
                                    print("\n机器人正在移动中，请等待当前移动完成")
                                else:
                                    print("\n执行移动命令...")
                                    success = self.move_to_saved_pose()
                                    if success:
                                        print("移动命令已启动（后台执行）")
                                    else:
                                        print("移动命令启动失败，请检查机器人状态")
                            else:
                                print("\n没有保存的检测坐标，请先按 'q' 暂停并保存检测到的物体坐标")
                        else:
                            print("\n请先按 'q' 暂停后再按 'w'")
                        self.last_key_time['w'] = current_time
                
                elif key == ord('b'):
                    # 防抖：检查距离上次按b的时间
                    if 'b' not in self.last_key_time or (current_time - self.last_key_time['b']) > self.key_debounce_time:
                        # 全局快捷键：回到初始位姿
                        if self.motion_handler.is_moving():
                            print("\n机器人正在移动中，请等待当前移动完成")
                        else:
                            print("\n回到初始位姿...")
                            self.move_to_initial_pose()
                        self.last_key_time['b'] = current_time
                
                elif key == ord('z'):
                    # 防抖：检查距离上次按z的时间
                    if 'z' not in self.last_key_time or (current_time - self.last_key_time['z']) > self.key_debounce_time:
                        # 全局快捷键：停止机器人移动
                        self.motion_handler.stop_robot_motion()
                        self.last_key_time['z'] = current_time
                
                elif key == 27:  # ESC
                    break
                
                # 如果暂停，显示暂停画面
                if self.paused:
                    if paused_frame is not None:
                        # 在暂停画面上添加状态信息
                        display_frame = paused_frame.copy()
                        
                        # 顶部状态栏（左上角）
                        y_offset = 15
                        line_height = 28
                        
                        # 显示机器人移动状态
                        if self.motion_handler.is_moving():
                            moving_text = "机器人正在移动中..."
                            display_frame = draw_text_with_bg(display_frame, moving_text, (15, y_offset), 
                                                             font_size=16, text_color=(0, 255, 255), 
                                                             bg_color=(255, 0, 0), alpha=0.5)
                            y_offset += line_height + 8
                        
                        if self.saved_pose is not None:
                            saved_text = f"已保存物体坐标: X:{self.saved_pose[0]:.1f} Y:{self.saved_pose[1]:.1f} Z:{self.saved_pose[2]:.1f}"
                            display_frame = draw_text_with_bg(display_frame, saved_text, (15, y_offset), 
                                                             font_size=16, text_color=(0, 255, 255), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                            y_offset += line_height + 5
                            
                            rot_text = f"姿态: Rx:{self.saved_pose[3]:.1f} Ry:{self.saved_pose[4]:.1f} Rz:{self.saved_pose[5]:.1f}"
                            display_frame = draw_text_with_bg(display_frame, rot_text, (15, y_offset), 
                                                             font_size=16, text_color=(0, 255, 255), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                        else:
                            no_save_text = "未保存坐标 - 按Q暂停时会保存检测到的物体坐标"
                            display_frame = draw_text_with_bg(display_frame, no_save_text, (15, y_offset), 
                                                             font_size=16, text_color=(0, 165, 255), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                        
                        # q和w动态提示（右上角，原质量位置）
                        img_width = display_frame.shape[1]
                        y_offset_right = 15
                        pause_text = "已暂停 - 按Q恢复"
                        display_frame = draw_text_with_bg(display_frame, pause_text, (img_width - 180, y_offset_right), 
                                                         font_size=16, text_color=(255, 255, 255), 
                                                         bg_color=(0, 0, 255), alpha=0.5)
                        y_offset_right += line_height + 5
                        if self.saved_pose is not None:
                            move_hint_text = "按W移动到保存坐标"
                            display_frame = draw_text_with_bg(display_frame, move_hint_text, (img_width - 180, y_offset_right), 
                                                             font_size=16, text_color=(255, 255, 0), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                        
                        # 底部快捷键提示（左下角）
                        help_text = "Q:恢复  W:移动  B:回初始  Z:停止  ESC:退出"
                        display_frame = draw_text_with_bg(display_frame, help_text, 
                                                         (15, display_frame.shape[0] - 35), 
                                                         font_size=16, text_color=(255, 255, 255), 
                                                         bg_color=(0, 0, 0), alpha=0.5)
                        
                        cv2.imshow("Detection", display_frame)
                    continue
                
                # 获取帧数据
                color_img, depth_frame, _ = self.camera.get_frame_data()
                if color_img is None:
                    continue
                
                # 检测物体
                center, found = detect_object(color_img, depth_frame)
                
                if found:
                    u, v = center
                    result = self.pose_estimator.calculate_robot_target(u, v, depth_frame)
                    
                    if result is not None:
                        raw_pos, raw_rot, normal_quality = result
                        
                        # 直接使用原始检测值
                        x, y, z = raw_pos
                        rx, ry, rz = raw_rot
                        
                        # 保存当前检测到的物体坐标（实时更新）
                        self.detected_pose = [x, y, z, rx, ry, rz]
                        
                        # 屏幕显示 - 优化布局，分散信息位置
                        # 左上角：机器人状态和位置信息
                        y_offset_left = 15
                        line_height = 26
                        
                        # 显示机器人移动状态
                        if self.motion_handler.is_moving():
                            moving_text = "机器人正在移动中..."
                            color_img = draw_text_with_bg(color_img, moving_text, (15, y_offset_left), 
                                                         font_size=16, text_color=(0, 255, 255), 
                                                         bg_color=(255, 0, 0), alpha=0.5)
                            y_offset_left += line_height + 8
                        
                        # 位置信息
                        pos_text = f"位置: X:{x:.1f}  Y:{y:.1f}  Z:{z:.1f}"
                        color_img = draw_text_with_bg(color_img, pos_text, (15, y_offset_left), 
                                                     font_size=16, text_color=(0, 255, 255), 
                                                     bg_color=(0, 0, 0), alpha=0.5)
                        y_offset_left += line_height + 5
                        
                        if ENABLE_NORMAL_ESTIMATION:
                            # 姿态信息
                            rot_text = f"姿态: Rx:{rx:.1f}  Ry:{ry:.1f}  Rz:{rz:.1f}"
                            color_img = draw_text_with_bg(color_img, rot_text, (15, y_offset_left), 
                                                         font_size=16, text_color=(255, 255, 0), 
                                                         bg_color=(0, 0, 0), alpha=0.5)
                            y_offset_left += line_height + 5
                        
                        # q和w动态提示（右上角，原质量位置）
                        img_width = color_img.shape[1]
                        y_offset_right = 15
                        if self.paused:
                            # 暂停状态提示
                            pause_text = "已暂停 - 按Q恢复"
                            color_img = draw_text_with_bg(color_img, pause_text, (img_width - 180, y_offset_right), 
                                                         font_size=16, text_color=(255, 255, 255), 
                                                         bg_color=(0, 0, 255), alpha=0.5)
                            y_offset_right += line_height + 5
                            if self.saved_pose is not None:
                                move_hint_text = "按W移动到保存坐标"
                                color_img = draw_text_with_bg(color_img, move_hint_text, (img_width - 180, y_offset_right), 
                                                             font_size=16, text_color=(255, 255, 0), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                        else:
                            # 正常运行状态提示
                            if self.saved_pose is not None:
                                move_hint_text = "按Q暂停保存坐标"
                                color_img = draw_text_with_bg(color_img, move_hint_text, (img_width - 180, y_offset_right), 
                                                             font_size=16, text_color=(0, 255, 255), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                                y_offset_right += line_height + 5
                                move_hint_text2 = "暂停后按W移动"
                                color_img = draw_text_with_bg(color_img, move_hint_text2, (img_width - 180, y_offset_right), 
                                                             font_size=16, text_color=(255, 255, 0), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                            else:
                                pause_hint_text = "按Q暂停保存坐标"
                                color_img = draw_text_with_bg(color_img, pause_hint_text, (img_width - 180, y_offset_right), 
                                                             font_size=16, text_color=(0, 255, 255), 
                                                             bg_color=(0, 0, 0), alpha=0.5)
                        
                        # 底部快捷键提示（左下角）
                        help_text = "Q:暂停  W:移动  B:回初始  Z:停止  ESC:退出"
                        color_img = draw_text_with_bg(color_img, help_text, 
                                                     (15, color_img.shape[0] - 35), 
                                                     font_size=16, text_color=(255, 255, 255), 
                                                     bg_color=(0, 0, 0), alpha=0.5)
                        
                        print(f"\r指令: [{x:.3f}, {y:.3f}, {z:.3f}, {rx:.3f}, {ry:.3f}, {rz:.3f}]", end="")
                    else:
                        # 深度无效提示
                        invalid_text = "深度无效"
                        color_img = draw_text_with_bg(color_img, invalid_text, (u - 50, v - 20), 
                                                     font_size=16, text_color=(255, 255, 255), 
                                                     bg_color=(0, 0, 255), alpha=0.5)
                else:
                    pass
                
                paused_frame = color_img.copy()
                cv2.imshow("Detection", color_img)
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            self.robot.CloseRPC()
            print("\n程序已退出")
