"""
颜色物体检测模块
检测多种颜色的物体并返回中心坐标
"""
import cv2
import numpy as np
from ..config.detection_config import COLOR_RANGES


def detect_object(color_img):
    """
    检测多种颜色的物体
    
    Args:
        color_img: 彩色图像 (BGR格式)
    
    Returns:
        tuple: (center, found) 
            - center: (cx, cy) 物体中心坐标，如果未检测到则为None
            - found: bool 是否检测到物体
    """
    # --- 高斯模糊，减少图像噪点导致的边缘跳变 ---
    blurred_img = cv2.GaussianBlur(color_img, (5, 5), 0)
    
    hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    
    for color_name, (lower, upper) in COLOR_RANGES.items():
        if color_name == 'red': 
            mask1 = cv2.inRange(hsv, COLOR_RANGES['red1'][0], COLOR_RANGES['red1'][1])
            mask2 = cv2.inRange(hsv, COLOR_RANGES['red2'][0], COLOR_RANGES['red2'][1])
            mask = cv2.bitwise_or(mask1, mask2)
        elif color_name.startswith('red'):
            mask = cv2.inRange(hsv, lower, upper)
        else:
            mask = cv2.inRange(hsv, lower, upper)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 300:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.drawContours(color_img, [c], -1, (0, 255, 0), 2)
                cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
                return (cx, cy), True
    return None, False
