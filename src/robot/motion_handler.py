"""
机器人运动处理模块
处理机器人移动的线程和状态管理
"""
import threading


class RobotMotionHandler:
    """机器人运动处理器，负责管理机器人移动的线程和状态"""
    
    def __init__(self, robot, tool=0, user=0):
        """
        初始化运动处理器
        
        Args:
            robot: 机器人RPC对象
            tool: 工具号
            user: 用户坐标系号
        """
        self.robot = robot
        self.tool = tool
        self.user = user
        
        # 机器人移动状态（多线程相关）
        self.robot_moving = False  # 机器人是否正在移动
        self.robot_move_lock = threading.Lock()  # 移动操作的锁，防止并发移动
        self.move_thread = None  # 移动线程
        self.stop_requested = False  # 停止移动请求标志
    
    def _move_to_pose_thread(self, pose):
        """在后台线程中执行机器人移动"""
        with self.robot_move_lock:
            self.robot_moving = True
            self.stop_requested = False  # 重置停止标志
            try:
                # MoveL参数：desc_pos, tool, user, ...
                error = self.robot.MoveL(pose, self.tool, self.user)
                if self.stop_requested:
                    print(f"\n移动已被用户停止")
                elif error == 0:
                    print(f"\n移动到位置: [{pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}, {pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}]")
                else:
                    print(f"\n移动失败，错误码: {error}")
            except Exception as e:
                if not self.stop_requested:
                    print(f"\n移动过程中发生错误: {e}")
            finally:
                self.robot_moving = False
                self.stop_requested = False
    
    def _move_to_pose_two_stage_thread(self, target_pose, hover_height):
        """
        在后台线程中执行两阶段机器人移动
        第一阶段：移动到悬停位置（z坐标增加hover_height）
        第二阶段：移动到目标位置
        
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz]
            hover_height: 悬停高度偏移量（相对于目标z坐标），单位为毫米（mm）
        """
        with self.robot_move_lock:
            self.robot_moving = True
            self.stop_requested = False  # 重置停止标志
            try:
                # 第一阶段：移动到悬停位置（保持x, y, rx, ry, rz不变，z增加hover_height）
                hover_pose = target_pose.copy()
                hover_pose[2] = target_pose[2] + hover_height
                
                print(f"\n第一阶段：移动到悬停位置 z={hover_pose[2]:.3f}mm (目标z={target_pose[2]:.3f}mm + 悬浮高度{hover_height:.3f}mm)")
                error = self.robot.MoveL(hover_pose, self.tool, self.user)
                
                if self.stop_requested:
                    print(f"\n移动已被用户停止（第一阶段）")
                    return
                elif error != 0:
                    print(f"\n第一阶段移动失败，错误码: {error}")
                    return
                else:
                    print(f"\n第一阶段完成：已到达悬停位置 [{hover_pose[0]:.3f}, {hover_pose[1]:.3f}, {hover_pose[2]:.3f}, {hover_pose[3]:.3f}, {hover_pose[4]:.3f}, {hover_pose[5]:.3f}]")
                
                # 第二阶段：移动到目标位置
                print(f"\n第二阶段：下降到目标位置 z={target_pose[2]:.3f}mm")
                error = self.robot.MoveL(target_pose, self.tool, self.user)
                
                if self.stop_requested:
                    print(f"\n移动已被用户停止（第二阶段）")
                elif error == 0:
                    print(f"\n第二阶段完成：已到达目标位置 [{target_pose[0]:.3f}, {target_pose[1]:.3f}, {target_pose[2]:.3f}, {target_pose[3]:.3f}, {target_pose[4]:.3f}, {target_pose[5]:.3f}]")
                    print(f"\n两阶段运动完成")
                else:
                    print(f"\n第二阶段移动失败，错误码: {error}")
            except Exception as e:
                if not self.stop_requested:
                    print(f"\n移动过程中发生错误: {e}")
            finally:
                self.robot_moving = False
                self.stop_requested = False

    def _move_to_pose_two_pose_thread(
        self,
        first_pose,
        second_pose,
        first_stage_desc="第一阶段",
        second_stage_desc="第二阶段",
    ):
        """
        在后台线程中执行两段位姿序列移动
        第一段：移动到 first_pose
        第二段：移动到 second_pose

        Args:
            first_pose: 第一段目标位姿 [x, y, z, rx, ry, rz]
            second_pose: 第二段目标位姿 [x, y, z, rx, ry, rz]
            first_stage_desc: 第一阶段描述
            second_stage_desc: 第二阶段描述
        """
        with self.robot_move_lock:
            self.robot_moving = True
            self.stop_requested = False  # 重置停止标志
            try:
                pose1 = first_pose.copy()
                pose2 = second_pose.copy()

                print(
                    f"\n{first_stage_desc}：移动到中间位姿 "
                    f"[{pose1[0]:.3f}, {pose1[1]:.3f}, {pose1[2]:.3f}, {pose1[3]:.3f}, {pose1[4]:.3f}, {pose1[5]:.3f}]"
                )
                error = self.robot.MoveL(pose1, self.tool, self.user)

                if self.stop_requested:
                    print(f"\n移动已被用户停止（{first_stage_desc}）")
                    return
                elif error != 0:
                    print(f"\n{first_stage_desc}失败，错误码: {error}")
                    return
                else:
                    print(f"\n{first_stage_desc}完成")

                print(
                    f"\n{second_stage_desc}：移动到目标位姿 "
                    f"[{pose2[0]:.3f}, {pose2[1]:.3f}, {pose2[2]:.3f}, {pose2[3]:.3f}, {pose2[4]:.3f}, {pose2[5]:.3f}]"
                )
                error = self.robot.MoveL(pose2, self.tool, self.user)

                if self.stop_requested:
                    print(f"\n移动已被用户停止（{second_stage_desc}）")
                elif error == 0:
                    print(f"\n{second_stage_desc}完成")
                    print("\n两段运动完成")
                else:
                    print(f"\n{second_stage_desc}失败，错误码: {error}")
            except Exception as e:
                if not self.stop_requested:
                    print(f"\n移动过程中发生错误: {e}")
            finally:
                self.robot_moving = False
                self.stop_requested = False
    
    def move_to_pose(self, pose):
        """
        移动到指定位姿（非阻塞，在后台线程执行）
        
        Args:
            pose: 目标位姿 [x, y, z, rx, ry, rz]
        
        Returns:
            bool: 是否成功启动移动
        """
        # 检查是否已经在移动
        if self.robot_moving:
            print("\n机器人正在移动中，请等待当前移动完成")
            return False
        
        # 在后台线程中执行移动
        self.move_thread = threading.Thread(target=self._move_to_pose_thread, args=(pose,), daemon=True)
        self.move_thread.start()
        return True
    
    def stop_robot_motion(self):
        """
        停止机器人移动
        
        Returns:
            bool: 是否成功发送停止命令
        """
        # 无论 robot_moving 状态如何，都尝试停止机器人
        # 这样可以避免竞态条件，确保用户按一次 'z' 就能停止机器人
        self.stop_requested = True
        was_moving = self.robot_moving
        
        try:
            error = self.robot.StopMotion()
            if error == 0:
                if was_moving:
                    print("\n已发送停止移动命令")
                else:
                    print("\n已发送停止移动命令（机器人可能已停止）")
            else:
                print(f"\n停止移动失败，错误码: {error}")
            return True
        except Exception as e:
            # 即使出现异常（如 "Request-sent"），停止命令可能已经发送
            # 这种情况通常表示请求已发送但未收到响应，命令可能已生效
            error_msg = str(e)
            if "Request-sent" in error_msg or "Request sent" in error_msg:
                # 这种情况通常表示命令已发送，只是没有收到响应
                # 我们仍然认为命令已成功发送，避免用户需要按两次
                if was_moving:
                    print("\n已发送停止移动命令（请求已发送）")
                else:
                    print("\n已发送停止移动命令（请求已发送，机器人可能已停止）")
            else:
                # 其他异常，仍然打印成功消息，因为命令可能已发送
                # 这样可以避免用户需要按两次才能停止
                if was_moving:
                    print("\n已发送停止移动命令（可能已生效）")
                else:
                    print("\n已发送停止移动命令（可能已生效，机器人可能已停止）")
            return True  # 即使有异常，也返回 True，因为命令可能已发送
    
    def move_to_pose_two_stage(self, target_pose, hover_height):
        """
        两阶段移动到指定位姿（非阻塞，在后台线程执行）
        第一阶段：移动到悬停位置（z坐标增加hover_height）
        第二阶段：移动到目标位置
        
        Args:
            target_pose: 目标位姿 [x, y, z, rx, ry, rz]
            hover_height: 悬停高度偏移量（相对于目标z坐标），单位为毫米（mm）
        
        Returns:
            bool: 是否成功启动移动
        """
        # 检查是否已经在移动
        if self.robot_moving:
            print("\n机器人正在移动中，请等待当前移动完成")
            return False
        
        # 在后台线程中执行两阶段移动
        self.move_thread = threading.Thread(target=self._move_to_pose_two_stage_thread, args=(target_pose, hover_height), daemon=True)
        self.move_thread.start()
        return True

    def move_to_pose_two_pose(
        self,
        first_pose,
        second_pose,
        first_stage_desc="第一阶段",
        second_stage_desc="第二阶段",
    ):
        """
        两段位姿序列移动（非阻塞，在后台线程执行）

        Args:
            first_pose: 第一段目标位姿 [x, y, z, rx, ry, rz]
            second_pose: 第二段目标位姿 [x, y, z, rx, ry, rz]
            first_stage_desc: 第一阶段描述
            second_stage_desc: 第二阶段描述

        Returns:
            bool: 是否成功启动移动
        """
        if self.robot_moving:
            print("\n机器人正在移动中，请等待当前移动完成")
            return False

        self.move_thread = threading.Thread(
            target=self._move_to_pose_two_pose_thread,
            args=(first_pose, second_pose, first_stage_desc, second_stage_desc),
            daemon=True,
        )
        self.move_thread.start()
        return True
    
    def is_moving(self):
        """检查机器人是否正在移动"""
        return self.robot_moving
