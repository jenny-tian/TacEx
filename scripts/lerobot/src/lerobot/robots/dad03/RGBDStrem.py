# -*- coding: utf-8 -*-
import os
import threading
import subprocess
import time
from typing import Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo, CompressedImage
from cv_bridge import CvBridge
import cv2


def _decode_compressed_color(msg: CompressedImage, as_rgb: bool = False) -> np.ndarray:
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError("Failed to decode compressed color")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class DepthColorStreamer(object):
    """
    三线程维护最新 color / depth：
    - 线程A：守护运行 image_transport republish，把 compressedDepth 解成 raw
    - 线程B：持续抓 color（优先 compressed，失败回退 raw）
    - 线程C：持续抓 depth（订阅 republish 输出的 raw）
    """

    def __init__(
        self,
        ns: str = "/camera_head",
        as_rgb: bool = False,
        depth_in_m: bool = True,
        republish_out_suffix: str = "_processed_depth",  # 最终话题名会是 .../image_raw_raw
        start_immediately: bool = True,
        color_timeout: float = 1.0,
        depth_timeout: float = 1.0,
    ):
        """
        :param ns: 相机命名空间
        :param as_rgb: color 是否转 RGB（默认 False，BGR）
        :param depth_in_m: depth 是否输出米（True=米，False=毫米/原始）
        :param republish_out_suffix: 输出基名后缀；遵循 <base> + _raw
        :param start_immediately: 构造后是否立即 start()
        :param color_timeout: 单次等待 color 超时时间
        :param depth_timeout: 单次等待 depth 超时时间
        """
        self.ns = ns.rstrip("/")
        self.as_rgb = as_rgb
        self.depth_in_m = depth_in_m
        self.color_timeout = color_timeout
        self.depth_timeout = depth_timeout

        # 话题基名
        self.color_compressed = f"{self.ns}/color/image_raw/compressed"
        self.color_raw = f"{self.ns}/color/image_raw"

        # depth 输入（base，不带 /compressedDepth；republish 会自动加）
        self.depth_base = f"{self.ns}/aligned_depth_to_color/image_raw"
        # republish 输出 raw（不带 /raw 后缀，image_transport 会直接发 sensor_msgs/Image）
        # 最终我们订阅的就是这个 out_base 本身（不是 out_base/raw）
        self.depth_out_base = f"{self.depth_base}{republish_out_suffix}"

        # 线程 & 同步
        self._stop_evt = threading.Event()
        self._bridge = CvBridge()

        self._latest_color = None  # np.ndarray
        self._latest_depth = None  # np.ndarray float32(m) or uint16(mm)
        self._K = None             # 3x3 内参矩阵

        self._lock = threading.Lock()
        self._republish_thread = None
        self._color_thread = None
        self._depth_thread = None
        self._republish_proc = None

        # 初始化 ROS 节点（若尚未初始化）
        if not rospy.core.is_initialized():
            rospy.init_node("depth_color_streamer", anonymous=True, disable_signals=True)

        # 先抓一次 CameraInfo（如果存在）
        try:
            info_msg = rospy.wait_for_message(f"{self.ns}/color/camera_info", CameraInfo, timeout=3.0)
            K = np.array(info_msg.K, dtype=np.float64).reshape(3, 3)
            with self._lock:
                self._K = K
        except Exception:
            rospy.logwarn("[DepthColorStreamer] Failed to get camera_info; K will be None until available.")

        if start_immediately:
            self.start()

    # ---------------------------- public API ----------------------------
    def start(self):
        """启动三个线程"""
        if self._republish_thread is None:
            self._republish_thread = threading.Thread(target=self._republish_worker, name="republish_worker", daemon=True)
            self._republish_thread.start()

        if self._color_thread is None:
            self._color_thread = threading.Thread(target=self._color_worker, name="color_worker", daemon=True)
            self._color_thread.start()

        if self._depth_thread is None:
            self._depth_thread = threading.Thread(target=self._depth_worker, name="depth_worker", daemon=True)
            self._depth_thread.start()

    def stop(self):
        """停止三个线程并杀掉 republish 子进程"""
        self._stop_evt.set()

        # 结束 republish 子进程
        if self._republish_proc is not None:
            try:
                self._republish_proc.terminate()
                # 等一会儿，若没退出再 kill
                try:
                    self._republish_proc.wait(timeout=2.0)
                except Exception:
                    self._republish_proc.kill()
            except Exception:
                pass
            finally:
                self._republish_proc = None

        # 等线程结束
        for th in (self._color_thread, self._depth_thread, self._republish_thread):
            if th is not None and th.is_alive():
                th.join(timeout=2.0)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """返回 (color, depth, K)"""
        with self._lock:
            c = None if self._latest_color is None else self._latest_color.copy()
            d = None if self._latest_depth is None else self._latest_depth.copy()
            K = None if self._K is None else self._K.copy()
        return c, d, K

    def get_latest_color(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_color is None else self._latest_color.copy()

    def get_latest_depth(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_depth is None else self._latest_depth.copy()

    # ---------------------------- workers ----------------------------
    def _republish_worker(self):
        """
        维持 image_transport republish 常驻。
        如果进程意外退出，会自动重启（直到 stop）。
        """
        cmd = [
            "rosrun", "image_transport", "republish",
            "compressedDepth", f"in:={self.depth_base}",
            "raw",              f"out:={self.depth_out_base}",
        ]
        rospy.loginfo("[DepthColorStreamer] Starting republisher: %s", " ".join(cmd))

        while not self._stop_evt.is_set():
            try:
                # 启动子进程（继承当前环境中的 ROS_MASTER_URI 等）
                self._republish_proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=os.environ.copy(),
                )

                # 读一点输出，便于诊断
                while not self._stop_evt.is_set():
                    line = self._republish_proc.stdout.readline()
                    if not line:
                        # 进程可能退出
                        if self._republish_proc.poll() is not None:
                            break
                        time.sleep(0.1)
                        continue
                    if "compressedDepth" in line or "Publishing" in line:
                        rospy.loginfo_throttle(5.0, "[republish] " + line.strip())

                # 退出处理：若非 stop 触发，则稍后重启
                rc = self._republish_proc.poll()
                if rc is None:
                    # 正在 stop；外层会清理
                    break
                rospy.logwarn("[DepthColorStreamer] republish exited with code %s; restarting in 1s...", rc)
                time.sleep(1.0)

            except Exception as e:
                rospy.logerr("[DepthColorStreamer] republish failed to start: %s", e)
                time.sleep(1.0)
            finally:
                if self._republish_proc is not None:
                    try:
                        self._republish_proc.terminate()
                        self._republish_proc.wait(timeout=1.0)
                    except Exception:
                        try:
                            self._republish_proc.kill()
                        except Exception:
                            pass
                    self._republish_proc = None

    def _color_worker(self):
        """
        循环获取 color：优先 compressed，失败则回退 raw。
        """
        while not self._stop_evt.is_set():
            try:
                msg_c = rospy.wait_for_message(self.color_compressed, CompressedImage, timeout=self.color_timeout)
                color = _decode_compressed_color(msg_c, as_rgb=self.as_rgb)
                with self._lock:
                    self._latest_color = color
            except rospy.ROSInterruptException:
                break
            except Exception:
                # 没收到就下一轮
                time.sleep(0.02)

    def _depth_worker(self):
        """
        循环获取 depth：订阅 republish 输出的 raw 图像（sensor_msgs/Image）
        """
        while not self._stop_evt.is_set():
            try:
                msg = rospy.wait_for_message(self.depth_out_base, Image, timeout=self.depth_timeout)
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                # 统一输出单位
                if self.depth_in_m:
                    if depth.dtype == np.uint16:
                        depth = depth.astype(np.float32) * 0.001  # mm -> m
                    elif depth.dtype == np.float32:
                        # already meters (32FC1)
                        pass
                    else:
                        depth = depth.astype(np.float32)
                # else: 原样返回（通常 uint16 毫米 或 float32 米）
                with self._lock:
                    self._latest_depth = depth

            except rospy.ROSInterruptException:
                break
            except Exception:
                time.sleep(0.02)

    # ---------------------------- context manager ----------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()


# ---------------------------- 示例用法 ----------------------------
if __name__ == "__main__":
    """
    使用前请确保：
    1) 相机发布了 /<ns>/aligned_depth_to_color/image_raw/compressedDepth
    2) 已安装 image_transport & compressed_depth_image_transport
       sudo apt-get install ros-$(rosversion -d)-image-transport \
                              ros-$(rosversion -d)-compressed-depth-image-transport
    """
    streamer = DepthColorStreamer(
        ns="/camera_head",
        as_rgb=True,       # True 则输出 RGB
        depth_in_m=True,    # True 输出米，False 输出毫米/原始
        start_immediately=True
    )

    # try:
    #     rate = rospy.Rate(10)
    #     while not rospy.is_shutdown():
    #         color, depth, K = streamer.get_latest()
    #         if color is not None and depth is not None:
    #             print("color:", color.shape, color.dtype,
    #                   "depth:", depth.shape, depth.dtype,
    #                   "K is None?" , K is None)
    #         rate.sleep()
    # finally:
    #     streamer.stop()
    while True:
        color, depth, K = streamer.get_latest()
        if color is not None and depth is not None:
            print("color:", color.shape, color.dtype,
                    "depth:", depth.shape, depth.dtype,
                    "K is None?" , K is None)
        time.sleep(0.1)
