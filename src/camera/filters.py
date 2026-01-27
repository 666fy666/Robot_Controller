"""
深度滤波器模块
RealSense深度后处理滤波器封装
"""
import pyrealsense2 as rs


class DepthFilters:
    """深度后处理滤波器集合"""
    
    def __init__(self):
        """初始化所有深度滤波器"""
        # 1. 空间滤波器：平滑表面
        self.spatial = rs.spatial_filter()
        self.spatial.set_option(rs.option.filter_magnitude, 2)
        self.spatial.set_option(rs.option.holes_fill, 3)
        
        # 2. 时间滤波器：最关键的稳定器，利用历史帧平滑
        self.temporal = rs.temporal_filter()
        self.temporal.set_option(rs.option.filter_smooth_alpha, 0.4) 
        self.temporal.set_option(rs.option.filter_smooth_delta, 20)
        
        # 3. 孔洞填充
        self.hole_filling = rs.hole_filling_filter()
    
    def process(self, depth_frame):
        """
        应用所有深度滤波器
        
        Args:
            depth_frame: 原始深度帧
        
        Returns:
            处理后的深度帧
        """
        # 应用深度滤波器 (修复了报错的部分)
        # 必须显式转换回 depth_frame 才能使用 get_distance
        depth_frame = self.spatial.process(depth_frame)
        depth_frame = self.temporal.process(depth_frame)
        depth_frame = self.hole_filling.process(depth_frame)
        depth_frame = depth_frame.as_depth_frame()  # 关键修复：转回深度帧类型
        
        return depth_frame
