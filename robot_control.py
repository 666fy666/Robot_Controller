"""
机器人控制主程序
"""
from src.robot.controller import RobotDetectController
from src.config.robot_config import DEFAULT_ROBOT_IP


if __name__ == '__main__':
    # 机器人IP地址
    robot_ip = DEFAULT_ROBOT_IP
    
    # 创建控制器并运行
    controller = RobotDetectController(robot_ip)
    controller.run()
