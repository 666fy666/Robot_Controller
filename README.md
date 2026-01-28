# Eye-in-Hand 机器人视觉抓取系统

基于手眼标定（眼在手上）的机器人视觉抓取系统，支持颜色物体检测、位姿估算和机器人控制。

## 项目结构

```
eyeInHand/
├── src/                          # 源代码目录
│   ├── camera/                   # 相机模块
│   │   ├── __init__.py
│   │   ├── realsense_camera.py   # RealSense相机封装
│   │   └── filters.py            # 深度滤波器
│   ├── detection/                # 检测模块
│   │   ├── __init__.py
│   │   ├── color_detector.py    # 颜色物体检测
│   │   ├── pose_estimator.py    # 位姿估算
│   │   └── pose_filter.py       # 位姿平滑滤波器
│   ├── robot/                    # 机器人控制模块
│   │   ├── __init__.py
│   │   ├── controller.py        # 机器人控制器
│   │   └── motion_handler.py    # 机器人运动处理
│   ├── calibration/              # 标定模块
│   │   ├── __init__.py
│   │   └── hand_eye_calibration.py  # 手眼标定
│   ├── utils/                    # 工具模块
│   │   ├── __init__.py
│   │   └── text_renderer.py     # 中文文本渲染工具
│   └── config/                   # 配置模块
│       ├── __init__.py
│       ├── detection_config.py   # 检测相关配置
│       └── robot_config.py      # 机器人相关配置
├── calibration/                  # 标定文件目录
│   ├── hand_eye_calibration_Tsai.npz
│   ├── camera_intrinsics.npz
│   └── ...
├── images/                       # 标定图片目录
│   ├── poses.txt                 # 机器人位姿文件
│   └── *_Color.png               # 标定板图片
├── Robot.py                      # 机器人控制库（第三方）
├── robot_control.py              # 主程序入口
└── requirements.txt             # 项目依赖
```

## 功能特性

### 1. 颜色物体检测
- 支持多种颜色的物体检测（蓝色、红色、绿色、黄色、橙色、紫色等）
- 基于HSV颜色空间的阈值分割
- 形态学处理优化检测结果

### 2. 位姿估算
- 基于深度相机的3D点云处理
- 法向量估算，支持表面姿态计算
- 坐标系转换（相机坐标系 → 夹爪坐标系 → 基坐标系）
- TCP（工具中心点）补偿

### 3. 机器人控制
- 实时检测和位姿计算
- 暂停/恢复功能
- 保存检测位姿并控制机器人移动
- **两阶段运动控制**：先移动到悬停位置，再下降到目标位置，提高安全性
- 多线程非阻塞运动控制
- 紧急停止功能

### 4. 手眼标定
- 支持多种标定算法（Tsai、Park、Horaud）
- 自动选择最佳标定结果
- 相机内参标定
- PnP误差过滤和数据清洗

## 环境要求

- Python 3.7+
- Windows 10/11（推荐）
- Intel RealSense相机（D435/D435i等）
- 机器人控制器（支持RPC通信）

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置机器人IP地址

编辑 `src/config/robot_config.py`，修改默认机器人IP：

```python
DEFAULT_ROBOT_IP = '192.168.2.68'  # 修改为你的机器人IP
```

### 3. 配置检测参数

编辑 `src/config/detection_config.py`，根据需要调整：

- `ROBOT_INITIAL_POSE`: 机器人拍照时的初始位姿
- `TOOL_LENGTH`: 工具/夹爪长度（mm）
- `Z_OFFSET`: Z轴偏移量（mm）
- `HOVER_HEIGHT`: 两阶段运动时的悬浮高度（mm）
- `X_OFFSET`, `Y_OFFSET`: X、Y轴偏移补偿（mm）
- `COLOR_RANGES`: 颜色检测阈值
- `ENABLE_NORMAL_ESTIMATION`: 是否启用法向量估算

## 使用方法

### 运行主程序

```bash
python robot_control.py
```

### 快捷键说明

- **Q** - 暂停/恢复检测，暂停时会自动保存检测到的物体坐标
- **W** - 暂停状态下，控制机器人两阶段移动到保存的检测坐标（先悬停，再下降）
- **B** - 回到初始位姿（全局快捷键）
- **Z** - 停止机器人移动（全局快捷键）
- **R** - 显示提示信息
- **ESC** - 退出程序

### 手眼标定

1. 准备标定板（11x8内角点，15mm方格）
2. 将机器人移动到不同位姿，拍摄标定板图片
3. 保存图片到 `images/` 目录，命名为 `1_Color.png`, `2_Color.png`, ...
4. 将对应的机器人位姿保存到 `images/poses.txt`，格式：`x,y,z,rx,ry,rz`
5. 运行标定程序：

```bash
python -m src.calibration.hand_eye_calibration
```

标定结果将保存到 `calibration/` 目录。

## 配置说明

### 检测配置 (`src/config/detection_config.py`)

- **CALIB_FILE**: 标定文件路径
- **ROBOT_INITIAL_POSE**: 机器人初始位姿 `[x, y, z, rx, ry, rz]`
- **TOOL_LENGTH**: TCP补偿长度（mm）
- **Z_OFFSET**: 法线方向偏移量（mm），用于避免碰撞
- **HOVER_HEIGHT**: 两阶段运动时的悬浮高度（mm），控制第一阶段悬停位置
- **X_OFFSET**, **Y_OFFSET**: 局部坐标系偏移补偿（mm）
- **COLOR_RANGES**: HSV颜色阈值配置
- **ENABLE_NORMAL_ESTIMATION**: 是否启用法向量估算
- **SMOOTH_ALPHA_POS**, **SMOOTH_ALPHA_ROT**: 位姿平滑系数

### 机器人配置 (`src/config/robot_config.py`)

- **DEFAULT_ROBOT_IP**: 默认机器人IP地址
- **DEFAULT_ROBOT_TOOL**: 默认工具号
- **DEFAULT_ROBOT_USER**: 默认用户坐标系号
- **KEY_DEBOUNCE_TIME**: 按键防抖时间（秒）

## 工作流程

1. **初始化**：连接机器人，加载标定文件，初始化相机
2. **检测循环**：
   - 获取彩色和深度图像
   - 检测颜色物体，获取中心坐标
   - 计算3D位置和姿态
   - 显示检测结果
3. **控制**：
   - 按Q暂停并保存检测位姿
   - 按W控制机器人两阶段移动到保存的位姿（先移动到悬停位置，再下降到目标位置）
   - 按B回到初始位姿

## 注意事项

1. **安全**：
   - 首次使用前请确保机器人工作空间安全
   - 建议先使用较小的Z_OFFSET值进行测试
   - 随时准备使用Z键停止机器人运动

2. **标定质量**：
   - 标定质量直接影响检测精度
   - 建议使用至少15组不同位姿的标定数据
   - 标定板应清晰可见，避免反光和遮挡

3. **相机设置**：
   - 确保相机固定安装，避免振动
   - 保持相机镜头清洁
   - 根据环境光照调整颜色阈值

4. **性能优化**：
   - 法向量估算会增加计算量，如不需要可关闭
   - 平滑滤波器已禁用，如需启用可修改代码

## 故障排除

### 问题：无法检测到物体
- 检查颜色阈值配置是否正确
- 确认光照条件合适
- 检查物体是否在相机视野内

### 问题：位姿计算不准确
- 检查标定文件是否正确加载
- 验证标定质量（重投影误差）
- 检查TOOL_LENGTH和偏移量配置

### 问题：机器人移动失败
- 检查机器人IP地址是否正确
- 确认机器人处于可控制状态
- 检查目标位姿是否在机器人工作空间内

## 开发说明

### 代码架构

项目采用模块化设计，各模块职责清晰：

- **camera**: 相机硬件抽象，提供统一的图像获取接口
- **detection**: 检测算法，包括颜色检测和位姿估算
- **robot**: 机器人控制，封装运动控制和状态管理
- **calibration**: 标定工具，独立的手眼标定程序
- **utils**: 通用工具函数
- **config**: 配置管理，集中管理所有配置参数

### 扩展开发

- 添加新的颜色检测：修改 `src/config/detection_config.py` 中的 `COLOR_RANGES`
- 添加新的检测算法：在 `src/detection/` 目录下添加新模块
- 支持其他机器人：实现 `src/robot/` 中的接口

## 许可证

本项目仅供学习和研究使用。

## 更新日志

### v2.1.0 (2026-01-28)
- 新增两阶段运动控制功能
- 添加HOVER_HEIGHT参数，支持可配置的悬浮高度
- 优化机器人移动安全性，先悬停再下降

### v2.0.0 (2026-01-27)
- 重构项目结构，采用模块化设计
- 合并 detect_and_grasp.py 和 RobotTest 功能
- 优化代码架构和文件命名
- 添加完整的文档说明

### v1.0.0
- 初始版本
- 基础的颜色检测和机器人控制功能
- 手眼标定功能

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。
