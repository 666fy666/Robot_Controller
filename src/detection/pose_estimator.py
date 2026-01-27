"""
位姿估算模块
计算机器人目标位姿，包括法向量估算和坐标转换
"""
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from ..config.detection_config import (
    ROBOT_INITIAL_POSE, TOOL_LENGTH, Z_OFFSET, X_OFFSET, Y_OFFSET,
    DETECT_EULER_ORDER, NORMAL_NEIGHBORHOOD_SIZE, MIN_POINTS_FOR_NORMAL,
    ENABLE_NORMAL_ESTIMATION, DEPTH_FILTER_THRESHOLD
)


class PoseEstimator:
    """位姿估算器，负责计算机器人目标位姿"""
    
    def __init__(self, R_cam2gripper, t_cam2gripper, R_base2cam, intrinsics):
        """
        初始化位姿估算器
        
        Args:
            R_cam2gripper: 相机到夹爪的旋转矩阵
            t_cam2gripper: 相机到夹爪的平移向量
            R_base2cam: 基座到相机的旋转矩阵
            intrinsics: 相机内参
        """
        self.R_cam2gripper = R_cam2gripper
        self.t_cam2gripper = t_cam2gripper
        self.R_base2cam = R_base2cam
        self.intrinsics = intrinsics
        
        # 初始化位姿相关矩阵
        rx, ry, rz = ROBOT_INITIAL_POSE[3:6]
        self.r_init = R.from_euler(DETECT_EULER_ORDER, [rx, ry, rz], degrees=True)
        self.R_base2gripper_init = self.r_init.as_matrix()
        self.t_base2gripper_init = np.array(ROBOT_INITIAL_POSE[0:3]).reshape(3, 1)
    
    def get_normal_in_base_frame(self, u, v, depth_frame):
        """
        在基坐标系中计算法向量
        
        Args:
            u, v: 像素坐标
            depth_frame: 深度帧
        
        Returns:
            tuple: (normal_base, quality)
                - normal_base: 基坐标系中的法向量，如果计算失败则为None
                - quality: 平面质量 (0.0-1.0)
        """
        center_depth = depth_frame.get_distance(u, v)
        if center_depth <= 0:
            return None, 0.0
        
        points_cam = []
        half_size = NORMAL_NEIGHBORHOOD_SIZE // 2
        h, w = depth_frame.get_height(), depth_frame.get_width()

        for v_idx in range(max(0, v - half_size), min(h, v + half_size + 1)):
            for u_idx in range(max(0, u - half_size), min(w, u + half_size + 1)):
                dist = depth_frame.get_distance(u_idx, v_idx)
                if dist > 0:
                    if abs(dist - center_depth) <= DEPTH_FILTER_THRESHOLD:
                        p = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u_idx, v_idx], dist)
                        points_cam.append(p)
        
        if len(points_cam) < MIN_POINTS_FOR_NORMAL:
            points_cam = []
            relaxed_threshold = DEPTH_FILTER_THRESHOLD * 2.0
            for v_idx in range(max(0, v - half_size), min(h, v + half_size + 1)):
                for u_idx in range(max(0, u - half_size), min(w, u + half_size + 1)):
                    dist = depth_frame.get_distance(u_idx, v_idx)
                    if dist > 0:
                        if abs(dist - center_depth) <= relaxed_threshold:
                            p = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u_idx, v_idx], dist)
                            points_cam.append(p)
        
        if len(points_cam) < MIN_POINTS_FOR_NORMAL:
            return None, 0.0

        points_cam = np.array(points_cam)
        centroid = np.mean(points_cam, axis=0)
        cov_matrix = np.cov((points_cam - centroid).T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)
        
        normal_cam = eigenvectors[:, sorted_indices[0]]
        if normal_cam[2] > 0:
            normal_cam = -normal_cam

        plane_quality = 1.0 - (eigenvalues[sorted_indices[0]] / (eigenvalues[sorted_indices[2]] + 1e-10))
        plane_quality = np.clip(plane_quality, 0.0, 1.0)

        normal_base = self.R_base2cam @ normal_cam
        return normal_base, plane_quality
    
    def calculate_robot_target(self, u, v, depth_frame):
        """
        计算机器人目标位姿
        
        Args:
            u, v: 像素坐标
            depth_frame: 深度帧
        
        Returns:
            tuple: (final_flange_pos, target_euler, quality)
                - final_flange_pos: 最终法兰位置 [x, y, z]
                - target_euler: 目标欧拉角 [rx, ry, rz]
                - quality: 法向量质量 (0.0-1.0)
        """
        dist = depth_frame.get_distance(u, v)
        if dist <= 0: 
            return None

        point_cam_m = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], dist)
        P_cam_mm = np.array(point_cam_m).reshape(3, 1) * 1000.0
        
        P_gripper = self.R_cam2gripper @ P_cam_mm + self.t_cam2gripper
        P_surface_base = self.R_base2gripper_init @ P_gripper + self.t_base2gripper_init
        P_surface_base = P_surface_base.flatten()
        
        target_euler = ROBOT_INITIAL_POSE[3:6]
        target_normal_base = np.array([0, 0, 1])
        quality = 0.0

        if ENABLE_NORMAL_ESTIMATION:
            normal_base, q = self.get_normal_in_base_frame(u, v, depth_frame)
            if normal_base is not None and q > 0.3:
                target_normal_base = normal_base
                quality = q
                
                z_axis = -target_normal_base
                z_axis = z_axis / np.linalg.norm(z_axis)
                ref_axis = np.array([1.0, 0.0, 0.0])
                if abs(z_axis[0]) > 0.9: 
                    ref_axis = np.array([0.0, 1.0, 0.0])
                y_axis = np.cross(z_axis, ref_axis)
                y_axis = y_axis / np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                x_axis = x_axis / np.linalg.norm(x_axis)
                
                R_target = np.column_stack([x_axis, y_axis, z_axis])
                r_obj = R.from_matrix(R_target)
                target_euler = r_obj.as_euler(DETECT_EULER_ORDER, degrees=True)
            else:
                R_target = self.R_base2gripper_init
        else:
            R_target = self.R_base2gripper_init

        # 计算偏移向量：沿法线方向的Z_OFFSET + 沿局部x和y方向的X_OFFSET和Y_OFFSET
        # 局部坐标系中的偏移向量 [X_OFFSET, Y_OFFSET, Z_OFFSET]
        offset_local = np.array([X_OFFSET, Y_OFFSET, Z_OFFSET])
        # 将局部坐标系的偏移转换到基坐标系
        offset_vector = R_target @ offset_local
        target_tip_pos = P_surface_base + offset_vector
        
        R_matrix_final = R.from_euler(DETECT_EULER_ORDER, target_euler, degrees=True).as_matrix()
        tool_offset = R_matrix_final @ np.array([0, 0, TOOL_LENGTH])
        final_flange_pos = target_tip_pos - tool_offset

        return final_flange_pos, target_euler, quality
