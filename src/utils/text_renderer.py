"""
文本渲染工具模块
提供在OpenCV图像上绘制中文文本的功能
"""
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont


def put_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文文本
    img: OpenCV图像 (BGR格式)
    text: 要绘制的文本
    position: (x, y) 文本位置
    font_size: 字体大小
    color: 颜色 (B, G, R)
    """
    # 将OpenCV图像转换为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 尝试使用系统中文字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑 Bold
        "simhei.ttf",                       # 当前目录
    ]
    
    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    # 如果所有字体都加载失败，使用默认字体（可能不支持中文）
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 绘制文本
    if font:
        draw.text(position, text, font=font, fill=color[::-1])  # PIL使用RGB，需要反转
    else:
        draw.text(position, text, fill=color[::-1])  # 使用默认字体
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


def draw_text_with_bg(img, text, position, font_size=24, text_color=(255, 255, 255), bg_color=(0, 0, 0), alpha=0.7):
    """
    在图像上绘制带半透明背景的文本
    img: OpenCV图像 (BGR格式)
    text: 要绘制的文本
    position: (x, y) 文本位置
    font_size: 字体大小
    text_color: 文本颜色 (B, G, R)
    bg_color: 背景颜色 (B, G, R)
    alpha: 背景透明度 (0.0-1.0)
    """
    # 先绘制文本获取文本尺寸
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, 'RGBA')
    
    # 加载字体
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ]
    
    font = None
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except:
            continue
    
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # 获取文本边界框
    if font:
        bbox = draw.textbbox(position, text, font=font)
    else:
        bbox = draw.textbbox(position, text)
    
    # 计算背景矩形（添加一些padding）
    padding = 5
    bg_rect = [
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding
    ]
    
    # 绘制半透明背景
    bg_rgba = (*bg_color[::-1], int(255 * alpha))  # PIL使用RGB，需要反转
    draw.rectangle(bg_rect, fill=bg_rgba)
    
    # 绘制文本
    text_rgb = text_color[::-1]  # PIL使用RGB
    if font:
        draw.text(position, text, font=font, fill=text_rgb)
    else:
        draw.text(position, text, fill=text_rgb)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv
