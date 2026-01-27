"""
RealSense相机模块
封装RealSense相机的初始化、帧获取等功能
"""
import pyrealsense2 as rs
import numpy as np
import os
from .filters import DepthFilters
from ..config.detection_config import AXIS_SIGN, ROBOT_INITIAL_POSE, DETECT_EULER_ORDER
from scipy.spatial.transform import Rotation as R


class RealSenseCamera:
    """RealSense相机封装类"""
    
    def __init__(self, calib_file):
        """
        初始化RealSense相机
        
        Args:
            calib_file: 标定文件路径
        """
        # 确保标定文件夹存在
        calib_dir = os.path.dirname(calib_file)
        if calib_dir and not os.path.exists(calib_dir):
            os.makedirs(calib_dir, exist_ok=True)
        
        # 加载标定数据
        try:
            with np.load(calib_file) as data:
                self.R_cam2gripper = data['R_cam2gripper']
                t_raw = data['t_cam2gripper']
                if t_raw.ndim == 1:
                    t_raw = t_raw.reshape(3, 1)
                axis_sign_array = np.array(AXIS_SIGN).reshape(3, 1)
                self.t_cam2gripper = t_raw * axis_sign_array
                
            print("标定文件加载成功。")
            
            rx, ry, rz = ROBOT_INITIAL_POSE[3:6]
            self.r_init = R.from_euler(DETECT_EULER_ORDER, [rx, ry, rz], degrees=True)
            self.R_base2gripper_init = self.r_init.as_matrix()
            self.t_base2gripper_init = np.array(ROBOT_INITIAL_POSE[0:3]).reshape(3, 1)
            self.R_base2cam = self.R_base2gripper_init @ self.R_cam2gripper
            
        except Exception as e:
            print(f"错误: 无法加载标定文件 {calib_file}。原因: {e}")
            exit()

        # 初始化相机
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 初始化深度滤波器
        self.filters = DepthFilters()
        
        self.align = rs.align(rs.stream.color)
        profile = self.pipeline.start(config)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    
    def get_frame_data(self):
        """
        获取对齐后的彩色和深度帧
        
        Returns:
            tuple: (color_img, depth_frame, color_frame)
                - color_img: 彩色图像 (numpy数组)
                - depth_frame: 深度帧 (已应用滤波器)
                - color_frame: 原始彩色帧
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # 应用深度滤波器
        depth_frame = self.filters.process(depth_frame)
        
        return np.asanyarray(color_frame.get_data()), depth_frame, color_frame
    
    def stop(self):
        """停止相机流"""
        self.pipeline.stop()
    
    def get_calibration_data(self):
        """
        获取标定数据
        
        Returns:
            tuple: (R_cam2gripper, t_cam2gripper, R_base2cam, intrinsics)
        """
        return self.R_cam2gripper, self.t_cam2gripper, self.R_base2cam, self.intrinsics
