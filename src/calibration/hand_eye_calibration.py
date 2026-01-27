"""
手眼标定模块
从 eyeInHand.py 移动并重构
"""
import cv2
import numpy as np
import glob
import os
from scipy.spatial.transform import Rotation as R
from ..utils.text_renderer import put_chinese_text

# ================= 配置区域 =================
# 1. 标定板参数
CHECKERBOARD = (11, 8)  # 内角点 (行, 列)
SQUARE_SIZE = 15.0      # mm

# 2. 文件路径
IMAGE_DIR = './images'
POSE_FILE = './images/poses.txt'
CALIBRATION_DIR = './calibration'  # 标定文件存放文件夹
OUTPUT_FILE = os.path.join(CALIBRATION_DIR, 'hand_eye_calibration.npz')
CAMERA_INTRINSICS_FILE = os.path.join(CALIBRATION_DIR, 'camera_intrinsics.npz')  # 相机内参保存文件

# 3. 欧拉角顺序
CALIB_EULER_ORDER = 'xyz'

# 4. 优化阈值
MAX_PNP_ERROR = 0.5     # PnP重投影误差阈值(像素)，超过此值的图片会被丢弃
MIN_CALIB_IMAGES = 10    # 内参标定所需的最少图片数
RECALIBRATE_INTRINSICS = False  # 是否重新标定内参（False则尝试加载已有内参）
SAVE_INDIVIDUAL_RESULTS = True  # 是否为每个方法单独保存文件
# ===========================================

def load_robot_poses(filename):
    """读取位姿文件，返回旋转矩阵列表和平移向量列表"""
    print(f"[-] 读取机器人位姿: {filename}")
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    R_list = []
    t_list = []
    valid_indices = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line: continue
        try:
            parts = [float(x) for x in line.split(',')]
            t = np.array(parts[0:3]).reshape(3, 1)
            
            rx, ry, rz = parts[3], parts[4], parts[5]
            euler_map = {'x': rx, 'y': ry, 'z': rz}
            euler_values = [euler_map[axis] for axis in CALIB_EULER_ORDER]
            
            r = R.from_euler(CALIB_EULER_ORDER, euler_values, degrees=True)
            R_list.append(r.as_matrix())
            t_list.append(t)
            valid_indices.append(i)
        except Exception as e:
            print(f"警告: 第 {i+1} 行数据解析失败: {e}")
            continue
            
    return R_list, t_list, valid_indices

def detect_corners(image_dir):
    """提取所有图片的角点"""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    files = sorted(glob.glob(os.path.join(image_dir, '*_Color.png')), 
                   key=lambda x: int(os.path.basename(x).split('_')[0]))
    
    if not files:
        raise FileNotFoundError("未找到 *_Color.png 图片")

    img_points = []
    obj_points = []
    valid_img_indices = []
    image_shape = None

    print(f"[-] 开始处理 {len(files)} 张图片...")
    print(f"[-] 提示: 按 'q' 或 'ESC' 键切换到下一张图片\n")
    
    for idx, fname in enumerate(files):
        img = cv2.imread(fname)
        if image_shape is None:
            image_shape = img.shape[:2][::-1]  # (width, height)
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        
        # 创建可视化图像
        img_vis = img.copy()
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # 绘制角点
            cv2.drawChessboardCorners(img_vis, CHECKERBOARD, corners2, ret)
            
            img_points.append(corners2)
            obj_points.append(objp)
            valid_img_indices.append(idx)
            
            status_text = f"检测到角点 ({len(corners2)} 个)"
            print(f"    [{idx+1}/{len(files)}] {os.path.basename(fname)}: {status_text}")
        else:
            status_text = "未检测到角点"
            print(f"    [{idx+1}/{len(files)}] {os.path.basename(fname)}: {status_text}")
        
        # 在图像上添加文本信息（使用中文支持）
        img_vis = put_chinese_text(img_vis, f"图片 {idx+1}/{len(files)}: {os.path.basename(fname)}", 
                                   (10, 10), font_size=24, color=(0, 255, 0))
        img_vis = put_chinese_text(img_vis, status_text, 
                                   (10, 40), font_size=24, 
                                   color=(0, 255, 0) if ret else (0, 0, 255))
        img_vis = put_chinese_text(img_vis, "按 'q' 或 'ESC' 切换到下一张", 
                                   (10, img_vis.shape[0] - 35), font_size=20, color=(255, 255, 255))
        
        # 显示图像并等待按键
        cv2.imshow('角点检测 - 手眼标定', img_vis)
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
        
        cv2.destroyAllWindows()

    return obj_points, img_points, image_shape, valid_img_indices

def calibrate_camera_intrinsics(obj_points, img_points, image_shape, save_file=None):
    """
    完整的相机内参标定流程
    使用标定板图片进行内参标定，比直接读取SDK内参更准确
    """
    print("\n" + "="*50)
    print("【相机内参标定】")
    print("="*50)
    
    if len(obj_points) < MIN_CALIB_IMAGES:
        raise ValueError(f"内参标定需要至少 {MIN_CALIB_IMAGES} 张有效图片，当前只有 {len(obj_points)} 张")
    
    print(f"[-] 使用 {len(obj_points)} 张图片进行内参标定...")
    print(f"[-] 图像尺寸: {image_shape[0]}x{image_shape[1]}")
    
    # 执行相机标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None,
        flags=cv2.CALIB_FIX_PRINCIPAL_POINT  # 可选：固定主点
    )
    
    # 计算每张图片的重投影误差
    print("\n[-] 计算每张图片的重投影误差...")
    per_image_errors = []
    for i in range(len(obj_points)):
        imgpoints_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(img_points[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        per_image_errors.append(error)
        print(f"    图片 {i+1}: 重投影误差 = {error:.4f} 像素")
    
    mean_error = np.mean(per_image_errors)
    max_error = np.max(per_image_errors)
    min_error = np.min(per_image_errors)
    std_error = np.std(per_image_errors)
    
    print("\n[-] 内参标定质量评估:")
    print(f"    总体RMSE误差: {ret:.4f} 像素")
    print(f"    平均重投影误差: {mean_error:.4f} 像素")
    print(f"    最大重投影误差: {max_error:.4f} 像素")
    print(f"    最小重投影误差: {min_error:.4f} 像素")
    print(f"    标准差: {std_error:.4f} 像素")
    
    # 输出内参矩阵
    print("\n[-] 相机内参矩阵 (K):")
    print(f"    fx = {mtx[0,0]:.2f}")
    print(f"    fy = {mtx[1,1]:.2f}")
    print(f"    cx = {mtx[0,2]:.2f}")
    print(f"    cy = {mtx[1,2]:.2f}")
    print(f"    完整矩阵:\n{mtx}")
    
    # 输出畸变系数
    print("\n[-] 畸变系数:")
    if len(dist) >= 5:
        print(f"    k1 = {dist[0]:.6f}")
        print(f"    k2 = {dist[1]:.6f}")
        print(f"    p1 = {dist[2]:.6f}")
        print(f"    p2 = {dist[3]:.6f}")
        print(f"    k3 = {dist[4]:.6f}")
    else:
        print(f"    {dist.flatten()}")
    
    # 计算视野角度（FOV）
    fov_x = 2 * np.arctan(image_shape[0] / (2 * mtx[0,0])) * 180 / np.pi
    fov_y = 2 * np.arctan(image_shape[1] / (2 * mtx[1,1])) * 180 / np.pi
    print(f"\n[-] 视野角度 (FOV):")
    print(f"    水平FOV: {fov_x:.2f}°")
    print(f"    垂直FOV: {fov_y:.2f}°")
    
    # 质量评估建议
    print("\n[-] 标定质量评估:")
    if mean_error < 0.1:
        print("    ✓ 优秀：平均误差 < 0.1 像素")
    elif mean_error < 0.3:
        print("    ✓ 良好：平均误差 < 0.3 像素")
    elif mean_error < 0.5:
        print("    ⚠ 一般：平均误差 < 0.5 像素，建议重新标定")
    else:
        print("    ✗ 较差：平均误差 >= 0.5 像素，强烈建议重新标定")
    
    if std_error > 0.2:
        print(f"    ⚠ 警告：标准差较大 ({std_error:.4f})，可能存在异常图片")
    
    # 保存内参
    if save_file:
        np.savez(save_file,
                 camera_matrix=mtx,
                 dist_coeffs=dist,
                 image_shape=image_shape,
                 mean_error=mean_error,
                 rms_error=ret)
        print(f"\n[-] 内参已保存至: {save_file}")
    
    print("="*50 + "\n")
    
    return mtx, dist, {
        'rms_error': ret,
        'mean_error': mean_error,
        'max_error': max_error,
        'min_error': min_error,
        'std_error': std_error,
        'fov_x': fov_x,
        'fov_y': fov_y
    }

def load_camera_intrinsics(filename):
    """加载已保存的相机内参"""
    if not os.path.exists(filename):
        return None, None
    
    try:
        data = np.load(filename)
        mtx = data['camera_matrix']
        dist = data['dist_coeffs']
        print(f"[-] 成功加载已保存的内参: {filename}")
        print(f"    图像尺寸: {tuple(data.get('image_shape', (0, 0)))}")
        if 'mean_error' in data:
            print(f"    标定时的平均误差: {data['mean_error']:.4f} 像素")
        return mtx, dist
    except Exception as e:
        print(f"警告: 加载内参失败 ({e})，将重新标定")
        return None, None

def filter_pnp_errors(obj_points, img_points, mtx, dist):
    """计算每张图的 PnP 误差，剔除误差大的数据"""
    print("[-] 计算 PnP 并清洗数据...")
    R_target = []
    t_target = []
    clean_indices = []
    
    total_error = 0
    
    for i in range(len(obj_points)):
        # ---------------- 修改开始 ----------------
        # 使用 SOLVEPNP_IPPE 避免平面目标的解算翻转 (OpenCV >= 4.5.3 推荐)
        try:
            retval, rvec, tvec = cv2.solvePnP(
                obj_points[i], img_points[i], mtx, dist,
                flags=cv2.SOLVEPNP_IPPE 
            )
        except:
            # 如果IPPE失败（极少数情况），回退到 SQPNP 或 ITERATIVE
            retval, rvec, tvec = cv2.solvePnP(
                obj_points[i], img_points[i], mtx, dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
        # ---------------- 修改结束 ----------------
        
        # 使用refine优化 (可选，IPPE通常已经很准)
        rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
            obj_points[i], img_points[i], mtx, dist, rvec, tvec
        )
        
        # 重投影验证
        projected_pts, _ = cv2.projectPoints(obj_points[i], rvec_refined, tvec_refined, mtx, dist)
        error = cv2.norm(img_points[i], projected_pts, cv2.NORM_L2) / len(projected_pts)
        
        # 增加一个 Z 轴深度检查，防止相机看到板子背面的错误解
        if tvec_refined[2] <= 0:
            print(f"    剔除索引 {i}: Z轴深度异常 (z={tvec_refined[2][0]:.2f})")
            continue

        if error < MAX_PNP_ERROR:
            rmat, _ = cv2.Rodrigues(rvec_refined)
            R_target.append(rmat)
            t_target.append(tvec_refined)
            clean_indices.append(i)
            total_error += error
        else:
            print(f"    剔除索引 {i}: PnP误差 {error:.4f} > 阈值 {MAX_PNP_ERROR}")

    avg_error = total_error / len(clean_indices) if clean_indices else 0
    print(f"    保留数据: {len(clean_indices)}/{len(obj_points)}, 平均PnP误差: {avg_error:.4f} px")
    
    return R_target, t_target, clean_indices

def compute_hand_eye_error(R_base, t_base, R_target, t_target, R_cam2g, t_cam2g):
    """计算手眼标定的误差（AX=XB方程误差）"""
    errors = []
    n = len(R_base)
    
    for i in range(n - 1):
        # 计算相对变换
        R_gripper_i = R_base[i]
        t_gripper_i = t_base[i]
        R_gripper_j = R_base[i+1]
        t_gripper_j = t_base[i+1]
        
        R_target_i = R_target[i]
        t_target_i = t_target[i]
        R_target_j = R_target[i+1]
        t_target_j = t_target[i+1]
        
        # 相对变换
        R_gripper_rel = R_gripper_j @ R_gripper_i.T
        t_gripper_rel = t_gripper_j - R_gripper_rel @ t_gripper_i
        
        R_target_rel = R_target_j @ R_target_i.T
        t_target_rel = t_target_j - R_target_rel @ t_target_i
        
        # AX = XB 方程验证
        # A * X = X * B
        # 其中 A = R_gripper_rel, X = R_cam2g, B = R_target_rel
        left_side = R_gripper_rel @ R_cam2g
        right_side = R_cam2g @ R_target_rel
        R_error = left_side @ right_side.T
        
        # 旋转误差（角度）
        angle_error = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        errors.append(np.degrees(angle_error))
        
        # 平移误差
        t_left = R_gripper_rel @ t_cam2g + t_gripper_rel
        t_right = R_cam2g @ t_target_rel + t_cam2g
        t_error = np.linalg.norm(t_left - t_right)
        errors.append(t_error)
    
    return np.mean(errors) if errors else float('inf')

def run_calibration_algorithms(R_base, t_base, R_target, t_target):
    """运行多种手眼标定算法并对比，返回所有结果"""
    print("\n" + "="*50)
    print("【手眼标定 - 多种方法对比】")
    print("="*50)
    
    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud")
    ]
    
    results = {}
    errors = {}
    
    for method, name in methods:
        try:
            R_cam2g, t_cam2g = cv2.calibrateHandEye(
                R_base, t_base, R_target, t_target, method=method
            )
            
            # 计算误差
            error = compute_hand_eye_error(R_base, t_base, R_target, t_target, R_cam2g, t_cam2g)
            
            # 计算旋转矩阵质量
            orthogonality_error = np.linalg.norm(R_cam2g @ R_cam2g.T - np.eye(3))
            det_error = abs(np.linalg.det(R_cam2g) - 1.0)
            dist = np.linalg.norm(t_cam2g)
            
            results[name] = {
                'R': R_cam2g,
                't': t_cam2g,
                'error': error,
                'orthogonality_error': orthogonality_error,
                'det_error': det_error,
                'distance': dist
            }
            errors[name] = error
            
            print(f"\n[{name}]")
            print(f"    平移向量 (mm): {t_cam2g.flatten()}")
            print(f"    相机-法兰距离: {dist:.2f} mm")
            print(f"    标定误差: {error:.6f}")
            print(f"    旋转矩阵正交性误差: {orthogonality_error:.2e}")
            print(f"    旋转矩阵行列式误差: {det_error:.2e}")
            
        except cv2.error as e:
            print(f"\n[{name}] 求解失败: {e}")
            results[name] = None
    
    if not results or all(v is None for v in results.values()):
        raise RuntimeError("所有手眼标定方法都失败了")
    
    # 选择误差最小的方法作为最佳结果
    valid_results = {k: v for k, v in results.items() if v is not None}
    if not valid_results:
        raise RuntimeError("没有有效的标定结果")
    
    best_method = min(valid_results.keys(), key=lambda k: valid_results[k]['error'])
    best_result = valid_results[best_method]
    
    print("\n" + "="*50)
    print(f"【最佳方法: {best_method}】")
    print(f"    误差: {best_result['error']:.6f}")
    print("="*50 + "\n")
    
    return best_result, results, best_method

def validate_calibration(R_base, t_base, R_target, t_target, R_calib, t_calib):
    """验证标定结果"""
    dist = np.linalg.norm(t_calib)
    print(f"[-] 标定结果验证:")
    print(f"    相机-法兰距离: {dist:.2f} mm")
    print("    (请检查该距离是否与物理测量大致相符)")
    
    # 验证旋转矩阵正交性
    orthogonality_error = np.linalg.norm(R_calib @ R_calib.T - np.eye(3))
    det_error = abs(np.linalg.det(R_calib) - 1.0)
    print(f"    旋转矩阵正交性误差: {orthogonality_error:.2e}")
    print(f"    旋转矩阵行列式误差: {det_error:.2e}")

def save_all_results(output_file, best_result, all_results, best_method, mtx, dist):
    """保存所有标定结果"""
    print(f"[-] 保存所有标定结果到 {output_file}...")
    
    # 准备保存的数据
    save_dict = {
        # 最佳结果（保持向后兼容）
        'R_cam2gripper': best_result['R'],
        't_cam2gripper': best_result['t'],
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'best_method': best_method,
    }
    
    # 保存所有方法的结果
    for method_name, result in all_results.items():
        if result is not None:
            save_dict[f'R_{method_name}'] = result['R']
            save_dict[f't_{method_name}'] = result['t']
            save_dict[f'error_{method_name}'] = result['error']
            save_dict[f'orthogonality_error_{method_name}'] = result['orthogonality_error']
            save_dict[f'det_error_{method_name}'] = result['det_error']
            save_dict[f'distance_{method_name}'] = result['distance']
    
    np.savez(output_file, **save_dict)
    print(f"    ✓ 已保存所有方法的结果")
    
    # 为每个方法单独保存文件（可选）
    if SAVE_INDIVIDUAL_RESULTS:
        print(f"[-] 为每个方法单独保存文件...")
        base_name = os.path.splitext(output_file)[0]
        
        for method_name, result in all_results.items():
            if result is not None:
                individual_file = os.path.join(os.path.dirname(output_file), f"{os.path.basename(base_name)}_{method_name}.npz")
                np.savez(individual_file,
                        R_cam2gripper=result['R'],
                        t_cam2gripper=result['t'],
                        camera_matrix=mtx,
                        dist_coeffs=dist,
                        error=result['error'],
                        orthogonality_error=result['orthogonality_error'],
                        det_error=result['det_error'],
                        distance=result['distance'],
                        method=method_name)
                print(f"    ✓ {method_name}: {individual_file}")

def main():
    # 0. 创建标定文件夹（如果不存在）
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    
    # 1. 载入 Robot Pose
    all_R_robot, all_t_robot, robot_indices = load_robot_poses(POSE_FILE)
    
    # 2. 载入图片并提取角点
    obj_pts, img_pts, img_shape, img_indices = detect_corners(IMAGE_DIR)
    
    # 3. 数据对齐
    n_samples = min(len(robot_indices), len(img_indices))
    if n_samples < MIN_CALIB_IMAGES:
        print(f"错误: 有效样本数太少 (<{MIN_CALIB_IMAGES})，无法进行标定。")
        return

    obj_pts = obj_pts[:n_samples]
    img_pts = img_pts[:n_samples]
    aligned_R_robot = [all_R_robot[i] for i in range(n_samples)]
    aligned_t_robot = [all_t_robot[i] for i in range(n_samples)]
    
    print(f"[-] 数据对齐完成，共 {n_samples} 组有效数据\n")

    # 4. 相机内参标定（或加载已有内参）
    if RECALIBRATE_INTRINSICS or not os.path.exists(CAMERA_INTRINSICS_FILE):
        print("[-] 开始相机内参标定...")
        mtx, dist, calib_info = calibrate_camera_intrinsics(
            obj_pts, img_pts, img_shape, save_file=CAMERA_INTRINSICS_FILE
        )
    else:
        print("[-] 尝试加载已保存的内参...")
        mtx, dist = load_camera_intrinsics(CAMERA_INTRINSICS_FILE)
        if mtx is None:
            print("[-] 加载失败，重新标定内参...")
            mtx, dist, calib_info = calibrate_camera_intrinsics(
                obj_pts, img_pts, img_shape, save_file=CAMERA_INTRINSICS_FILE
            )
    
    # 5. PnP 解算 + 脏数据过滤
    R_target, t_target, clean_indices = filter_pnp_errors(obj_pts, img_pts, mtx, dist)
    
    if len(clean_indices) < 3:
        print("错误: 清洗后剩余数据不足。")
        return

    # 根据清洗结果，同步过滤 Robot Pose
    final_R_base = [aligned_R_robot[i] for i in clean_indices]
    final_t_base = [aligned_t_robot[i] for i in clean_indices]
    
    # 6. 手眼标定求解（所有方法）
    best_result, all_results, best_method = run_calibration_algorithms(
        final_R_base, final_t_base, R_target, t_target
    )
    
    # 7. 结果展示与保存
    validate_calibration(final_R_base, final_t_base, R_target, t_target, 
                       best_result['R'], best_result['t'])
    
    print("\n" + "="*50)
    print("【最终标定结果】")
    print("="*50)
    print(f"最佳方法: {best_method}")
    print("\n旋转矩阵 R_cam2gripper:")
    print(best_result['R'])
    print("\n平移向量 t_cam2gripper (mm):")
    print(best_result['t'].flatten())
    print("="*50)
    
    # 8. 保存所有结果
    save_all_results(OUTPUT_FILE, best_result, all_results, best_method, mtx, dist)
    print(f"\n所有结果已保存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
