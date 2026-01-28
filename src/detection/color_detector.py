"""
颜色物体检测模块
检测多种颜色的物体并返回中心坐标
"""
import cv2
import numpy as np
from ..config.detection_config import (
    COLOR_RANGES,
    ENABLE_TOP_FACE_CENTER,
    ENABLE_TOP_FACE_CENTER_2D_FALLBACK,
    TOP_FACE_DEPTH_BAND_M,
    TOP_FACE_NEAR_PERCENTILE,
    TOP_FACE_2D_CANNY_T1,
    TOP_FACE_2D_CANNY_T2,
    TOP_FACE_2D_HOUGH_MAX_LINE_GAP,
    TOP_FACE_2D_HOUGH_MIN_LINE_LEN_RATIO,
    TOP_FACE_2D_HOUGH_THRESHOLD,
    TOP_FACE_2D_MIN_REGION_AREA_PX,
    TOP_FACE_MIN_AREA_PX,
)


def _centroid_from_contour(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def _estimate_depth_scale_m(depth_frame, depth_raw, sample_uv):
    """
    估算 depth_raw 到米的比例（m/unit）。
    优先使用 depth_frame.get_units()，否则用一个采样点的 (get_distance/raw) 近似。
    """
    try:
        if hasattr(depth_frame, "get_units") and callable(depth_frame.get_units):
            units = float(depth_frame.get_units())
            if units > 0:
                return units
    except Exception:
        pass

    try:
        u, v = sample_uv
        raw = float(depth_raw[v, u])
        if raw <= 0:
            return None
        dist_m = float(depth_frame.get_distance(int(u), int(v)))
        if dist_m <= 0:
            return None
        return dist_m / raw
    except Exception:
        return None


def _select_top_face_center_from_depth(depth_frame, contour, img_shape):
    """
    在最大轮廓内部，按深度选取“最近的一层”作为顶面并计算其中心点。
    失败返回 None。
    """
    if depth_frame is None:
        return None

    h, w = img_shape[:2]

    # 轮廓mask
    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    # 深度raw
    try:
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
    except Exception:
        return None

    if depth_raw.shape[0] != h or depth_raw.shape[1] != w:
        # 理论上对齐后应同分辨率；不一致时直接回退
        return None

    # 找一个轮廓内有效深度的采样点，用于估算尺度（米）
    ys, xs = np.where((contour_mask > 0) & (depth_raw > 0))
    if len(xs) < 50:
        return None

    sample_idx = len(xs) // 2
    scale_m = _estimate_depth_scale_m(depth_frame, depth_raw, (int(xs[sample_idx]), int(ys[sample_idx])))
    if scale_m is None or scale_m <= 0:
        return None

    depth_m = depth_raw * scale_m
    depth_m[depth_raw <= 0] = np.nan

    depth_vals = depth_m[contour_mask > 0]
    depth_vals = depth_vals[np.isfinite(depth_vals)]
    if depth_vals.size < 50:
        return None

    near_depth = float(np.percentile(depth_vals, TOP_FACE_NEAR_PERCENTILE))
    threshold = near_depth + float(TOP_FACE_DEPTH_BAND_M)

    top_mask = (contour_mask > 0) & np.isfinite(depth_m) & (depth_m <= threshold)
    top_mask_u8 = (top_mask.astype(np.uint8) * 255)

    # 清理噪声，避免“最近像素点”碎片化
    kernel = np.ones((5, 5), np.uint8)
    top_mask_u8 = cv2.morphologyEx(top_mask_u8, cv2.MORPH_OPEN, kernel)
    top_mask_u8 = cv2.morphologyEx(top_mask_u8, cv2.MORPH_CLOSE, kernel)

    top_contours, _ = cv2.findContours(top_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not top_contours:
        return None

    top_c = max(top_contours, key=cv2.contourArea)
    if cv2.contourArea(top_c) < TOP_FACE_MIN_AREA_PX:
        return None

    return _centroid_from_contour(top_c), top_c


def _select_top_face_center_from_2d(color_img, contour):
    """
    纯2D方案：在轮廓ROI内做 Canny + HoughLinesP 找到“内部公共边”后将轮廓分为两块，
    默认取面积更大的一块作为顶面并返回其中心点。
    失败返回 None。
    """
    h, w = color_img.shape[:2]

    contour_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)

    x, y, bw, bh = cv2.boundingRect(contour)
    if bw < 10 or bh < 10:
        return None

    roi = color_img[y : y + bh, x : x + bw]
    roi_mask = contour_mask[y : y + bh, x : x + bw]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, TOP_FACE_2D_CANNY_T1, TOP_FACE_2D_CANNY_T2)
    edges = cv2.bitwise_and(edges, roi_mask)

    min_line_len = int(max(bw, bh) * float(TOP_FACE_2D_HOUGH_MIN_LINE_LEN_RATIO))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=int(TOP_FACE_2D_HOUGH_THRESHOLD),
        minLineLength=max(10, min_line_len),
        maxLineGap=int(TOP_FACE_2D_HOUGH_MAX_LINE_GAP),
    )
    if lines is None or len(lines) == 0:
        return None

    # 用距离变换评估“线段是否在轮廓内部而非外边界”
    dist = cv2.distanceTransform((roi_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)

    best = None  # (score, x1,y1,x2,y2)
    for line in lines.reshape(-1, 4):
        x1, y1, x2, y2 = map(int, line.tolist())
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < 10:
            continue

        # 采样判断线段大部分点是否远离边界
        n = int(max(12, min(60, length / 4)))
        xs = np.linspace(x1, x2, n).astype(np.int32)
        ys = np.linspace(y1, y2, n).astype(np.int32)

        valid = (xs >= 0) & (xs < bw) & (ys >= 0) & (ys < bh)
        if not np.any(valid):
            continue
        xs = xs[valid]
        ys = ys[valid]

        dt_vals = dist[ys, xs]
        inside_frac = float(np.mean(dt_vals > 2.0))  # >2px 认为在内部
        if inside_frac < 0.6:
            continue

        score = length * inside_frac * float(np.mean(dt_vals))
        if best is None or score > best[0]:
            best = (score, x1, y1, x2, y2)

    if best is None:
        return None

    _, lx1, ly1, lx2, ly2 = best

    # 使用该线将轮廓mask分成两部分（半平面切分）
    yy, xx = np.where(roi_mask > 0)
    if yy.size < 50:
        return None

    # 叉积符号：pos/neg 两侧
    cross = (lx2 - lx1) * (yy - ly1) - (ly2 - ly1) * (xx - lx1)
    side_a = np.zeros_like(roi_mask)
    side_b = np.zeros_like(roi_mask)
    side_a[yy[cross >= 0], xx[cross >= 0]] = 255
    side_b[yy[cross < 0], xx[cross < 0]] = 255

    kernel = np.ones((5, 5), np.uint8)
    side_a = cv2.morphologyEx(side_a, cv2.MORPH_CLOSE, kernel)
    side_b = cv2.morphologyEx(side_b, cv2.MORPH_CLOSE, kernel)

    ca, _ = cv2.findContours(side_a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cb, _ = cv2.findContours(side_b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ca or not cb:
        return None

    ca = max(ca, key=cv2.contourArea)
    cb = max(cb, key=cv2.contourArea)
    area_a = float(cv2.contourArea(ca))
    area_b = float(cv2.contourArea(cb))

    if area_a < TOP_FACE_2D_MIN_REGION_AREA_PX or area_b < TOP_FACE_2D_MIN_REGION_AREA_PX:
        return None

    # 默认取面积更大的一块作为“顶面”
    pick_a = area_a >= area_b
    picked = ca if pick_a else cb
    center = _centroid_from_contour(picked)
    if center is None:
        return None

    cx, cy = center
    cx += x
    cy += y

    # 将ROI坐标系的轮廓点平移到全图坐标系，方便可视化
    picked_global = picked.copy()
    picked_global[:, 0, 0] += x
    picked_global[:, 0, 1] += y

    # 同时返回分割线（全图坐标系），用于可视化
    line_global = (lx1 + x, ly1 + y, lx2 + x, ly2 + y)
    return (cx, cy), picked_global, line_global


def detect_object(color_img, depth_frame=None):
    """
    检测多种颜色的物体
    
    Args:
        color_img: 彩色图像 (BGR格式)
        depth_frame: 深度帧（可选）。提供时将尝试仅输出“顶面中心点”
    
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
            # 先算原始轮廓中心（作为回退）
            fallback_center = _centroid_from_contour(c)

            # 尝试用“顶面深度层”重新计算中心
            top_center = None
            top_contour = None
            split_line = None
            if ENABLE_TOP_FACE_CENTER and depth_frame is not None:
                result = _select_top_face_center_from_depth(depth_frame, c, color_img.shape)
                if result is not None:
                    top_center, top_contour = result
            elif ENABLE_TOP_FACE_CENTER and ENABLE_TOP_FACE_CENTER_2D_FALLBACK:
                result2d = _select_top_face_center_from_2d(color_img, c)
                if result2d is not None:
                    top_center, top_contour, split_line = result2d

            # 可视化
            cv2.drawContours(color_img, [c], -1, (0, 255, 0), 2)
            if top_center is not None and top_contour is not None:
                cx, cy = top_center
                cv2.drawContours(color_img, [top_contour], -1, (255, 0, 0), 2)
                if split_line is not None:
                    x1, y1, x2, y2 = split_line
                    cv2.line(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
                return (cx, cy), True

            if fallback_center is not None:
                cx, cy = fallback_center
                cv2.circle(color_img, (cx, cy), 5, (0, 0, 255), -1)
                return (cx, cy), True
    return None, False
