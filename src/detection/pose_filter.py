"""
位姿平滑滤波器模块
简单的低通滤波器，用于平滑坐标数值
"""
import numpy as np


class PoseFilter:
    """简单的低通滤波器，用于平滑坐标数值"""
    def __init__(self, alpha_pos, alpha_rot):
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot
        self.prev_pos = None
        self.prev_rot = None

    def update(self, pos, rot):
        # 第一次直接赋值
        if self.prev_pos is None:
            self.prev_pos = np.array(pos)
            self.prev_rot = np.array(rot)
            return pos, rot
        
        curr_pos = np.array(pos)
        curr_rot = np.array(rot)
        
        smooth_pos = self.prev_pos * (1 - self.alpha_pos) + curr_pos * self.alpha_pos
        smooth_rot = self.prev_rot * (1 - self.alpha_rot) + curr_rot * self.alpha_rot
        
        self.prev_pos = smooth_pos
        self.prev_rot = smooth_rot
        
        return smooth_pos, smooth_rot

    def reset(self):
        self.prev_pos = None
        self.prev_rot = None
