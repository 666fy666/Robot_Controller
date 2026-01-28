"""
位姿估算模块
计算机器人目标位姿，包括法向量估算和坐标转换
"""
import numpy as np
import pyrealsense2 as rs
from scipy.spatial.transform import Rotation as R
from ..config.detection_config import (
    TOOL_LENGTH, Z_OFFSET, X_OFFSET, Y_OFFSET,
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
        # 注意：Eye-in-hand 场景下 base->cam 并非固定，这里保留该字段仅用于兼容旧接口。
        # 正确的做法是在每次计算时基于“当前机器人TCP位姿”实时计算 R_base2cam。
        self.R_base2cam = R_base2cam
        self.intrinsics = intrinsics

    @staticmethod
    def _pose_to_base2gripper(pose):
        """
        将机器人TCP位姿 [x, y, z, rx, ry, rz] 转为 gripper->base 旋转和平移。

        说明：项目里历史命名为 R_base2gripper，但在使用上它表示“gripper坐标系到base坐标系”的旋转。
        """
        rx, ry, rz = pose[3:6]
        r = R.from_euler(DETECT_EULER_ORDER, [rx, ry, rz], degrees=True)
        R_base2gripper = r.as_matrix()
        t_base2gripper = np.array(pose[0:3], dtype=float).reshape(3, 1)
        return R_base2gripper, t_base2gripper
    
    def get_normal_in_base_frame(self, u, v, depth_frame, R_base2cam):
        """
        在基坐标系中计算法向量
        
        Args:
            u, v: 像素坐标
            depth_frame: 深度帧
            R_base2cam: 相机坐标系到基坐标系的旋转矩阵（3x3）
        
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

        normal_base = R_base2cam @ normal_cam
        return normal_base, plane_quality
    
    def calculate_robot_target(
        self,
        u,
        v,
        depth_frame,
        current_robot_pose,
        target_euler_override=None,
        enable_normal_estimation_override=None,
    ):
        """
        计算机器人目标位姿
        
        Args:
            u, v: 像素坐标
            depth_frame: 深度帧
            current_robot_pose: 当前机器人TCP位姿 [x, y, z, rx, ry, rz]
            target_euler_override: 若提供，则强制使用该欧拉角作为目标姿态（用于“二次复拍只改xyz、保持rxryrz不变”）
            enable_normal_estimation_override: 若提供，则覆盖全局 ENABLE_NORMAL_ESTIMATION
        
        Returns:
            tuple: (final_flange_pos, target_euler, quality)
                - final_flange_pos: 最终法兰位置 [x, y, z]
                - target_euler: 目标欧拉角 [rx, ry, rz]
                - quality: 法向量质量 (0.0-1.0)
        """
        if current_robot_pose is None:
            return None

        dist = depth_frame.get_distance(u, v)
        if dist <= 0: 
            return None

        point_cam_m = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], dist)
        P_cam_mm = np.array(point_cam_m).reshape(3, 1) * 1000.0
        
        P_gripper = self.R_cam2gripper @ P_cam_mm + self.t_cam2gripper

        R_base2gripper, t_base2gripper = self._pose_to_base2gripper(current_robot_pose)
        P_surface_base = R_base2gripper @ P_gripper + t_base2gripper
        P_surface_base = P_surface_base.flatten()
        
        # 默认：保持当前姿态（若二次复拍指定了 target_euler_override，则强制使用它）
        if target_euler_override is not None:
            target_euler = list(map(float, target_euler_override))
            R_target = R.from_euler(DETECT_EULER_ORDER, target_euler, degrees=True).as_matrix()
            quality = 1.0
        else:
            target_euler = list(map(float, current_robot_pose[3:6]))
            R_target = R_base2gripper
            quality = 0.0

            use_normal = ENABLE_NORMAL_ESTIMATION
            if enable_normal_estimation_override is not None:
                use_normal = bool(enable_normal_estimation_override)

            if use_normal:
                R_base2cam = R_base2gripper @ self.R_cam2gripper
                normal_base, q = self.get_normal_in_base_frame(u, v, depth_frame, R_base2cam)
                if normal_base is not None and q > 0.3:
                    quality = q

                    z_axis = -normal_base
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
