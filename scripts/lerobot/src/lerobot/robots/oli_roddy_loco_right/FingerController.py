import mros
import mros.controller_msgs.msg.Pose
# import mros.sensor_msgs.msg.Image
import mros.std_msgs.msg.Float32Array
import signal
import time
import numpy as np
import cv2


class FingerController:
    def __init__(self):
        # 初始化 mros 节点
        # mros.init('EEControlNode')


        # hw_T_c_pub_ = node.create_publisher(Pose, , 10)
        # hw_T_c_pub_.publish(EigenPose(hw_T_c).toMsg())
        # hw_T_c_pub_.publish(EigenPose(hw_T_c).toMsg())
        
        # 创建发布器
        self.finger_cmd_publisher = mros.advertise(
            '/brainco1/hand/cmd', 
            mros.std_msgs.msg.Float32Array
        )

    def publish_cmd(self, cmd):
        msg = mros.std_msgs.msg.Float32Array()
        msg.data = cmd
        self.finger_cmd_publisher.publish(msg)
        print(f"{msg.header.stamp} Published command: {cmd}")


    def run(self):
        """主运行循环"""
        print("Head controller started")
        try:
            while True:
                cmd = [0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 示例命令
                self.publish_cmd(cmd)
                time.sleep(1)

                # cmd = [100, 0, 100, 0, 100, 0, 100, 0, 100, 0, 100, 0]  # 示例命令
                # self.publish_cmd(cmd)
                # time.sleep(0.5)

        except KeyboardInterrupt:
            print("Shutting down...")

def signal_handler(sig, frame):
    print('Interrupted!')
    mros.shutdown()
    exit(0)

if __name__ == "__main__":
    mros.init('EEControlNode')

    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    
    # 创建控制器实例并运行
    controller = FingerController()
    controller.run()