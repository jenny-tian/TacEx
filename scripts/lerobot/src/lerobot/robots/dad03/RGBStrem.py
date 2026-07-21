# -*- coding: utf-8 -*-
import threading
import time
from typing import Optional

import numpy as np
import rospy
from sensor_msgs.msg import Image, CompressedImage
import cv2


def _image_msg_to_bgr8(msg: Image) -> np.ndarray:
    """
    将 sensor_msgs/Image 转为 BGR uint8（与 CvBridge.imgmsg_to_cv2(..., 'bgr8') 常用话题一致）。
    支持常见 encoding；不依赖 cv_bridge，避免与 NumPy 2.x 的二进制冲突。
    """
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError(f"Invalid image size: {msg.width}x{msg.height}")
    h, w = msg.height, msg.width
    enc = msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    need = h * msg.step
    if buf.size < need:
        raise ValueError(
            f"Image buffer too short: got {buf.size} bytes, need >= {need} (h*step)"
        )
    row = buf[:need].reshape((h, msg.step))

    def crop_channels(n: int) -> np.ndarray:
        byte_w = w * n
        if msg.step < byte_w:
            raise ValueError(f"step {msg.step} < width * channels ({byte_w})")
        return row[:, :byte_w].reshape((h, w, n))

    if enc in ("bgr8",):
        img = crop_channels(3)
        return np.ascontiguousarray(img)
    if enc in ("rgb8", "8uc3"):
        img = crop_channels(3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8",):
        img = crop_channels(4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if enc in ("rgba8",):
        img = crop_channels(4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc in ("mono8", "8uc1"):
        img = row[:, :w].reshape((h, w))
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    raise ValueError(
        f"Unsupported Image encoding for raw decode: {msg.encoding!r} "
        f"(supported: bgr8, rgb8, 8UC3, rgba8, bgra8, mono8, 8UC1)"
    )


def _decode_compressed_color(msg: CompressedImage, as_rgb: bool = False) -> np.ndarray:
    """将 sensor_msgs/CompressedImage 解码成 numpy 图像（BGR 或 RGB）"""
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise RuntimeError("Failed to decode compressed color")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class ColorStreamer(object):
    """
    单线程只维护最新的 color：
    - 线程：循环抓取 color（优先 compressed，失败回退 raw）
    """

    def __init__(
        self,
        ns: str = "/camera_head",
        as_rgb: bool = False,
        start_immediately: bool = True,
        color_timeout: float = 1.0,
    ):
        """
        :param ns: 相机命名空间
        :param as_rgb: color 是否转为 RGB（默认 False，BGR）
        :param start_immediately: 构造后是否立即 start()
        :param color_timeout: wait_for_message 的超时时间（秒）
        """
        self.ns = ns.rstrip("/")
        self.as_rgb = as_rgb
        self.color_timeout = color_timeout

        # 话题
        self.color_compressed = f"{self.ns}/color/image_raw/compressed"
        self.color_raw = f"{self.ns}/color/image_raw"

        rospy.init_node("color_streamer", anonymous=True, disable_signals=True)

        # 同步
        self._stop_evt = threading.Event()
        self._latest_color: Optional[np.ndarray] = None

        self._lock = threading.Lock()
        self._color_thread: Optional[threading.Thread] = None

        if start_immediately:
            self.start()

    # ---------------------------- public API ----------------------------
    def start(self):
        if self._color_thread is None:
            self._color_thread = threading.Thread(
                target=self._color_worker, name="color_worker", daemon=True
            )
            self._color_thread.start()

    def stop(self):
        self._stop_evt.set()
        if self._color_thread is not None and self._color_thread.is_alive():
            self._color_thread.join(timeout=2.0)

    def get_latest_color(self) -> Optional[np.ndarray]:
        """返回最新的 color（np.ndarray），若还没有则为 None"""
        with self._lock:
            return None if self._latest_color is None else self._latest_color.copy()

    # ---------------------------- worker ----------------------------
    def _color_worker(self):
        """循环获取 color：优先 compressed，失败回退 raw。"""
        while not self._stop_evt.is_set() and not rospy.is_shutdown():
            got = False
            # 1) 先尝试 compressed
            try:
                msg_c = rospy.wait_for_message(
                    self.color_compressed, CompressedImage, timeout=self.color_timeout
                )
                color = _decode_compressed_color(msg_c, as_rgb=self.as_rgb)
                with self._lock:
                    self._latest_color = color
                got = True
            except Exception:
                pass

            # 2) 回退 raw
            if not got:
                try:
                    msg_r = rospy.wait_for_message(
                        self.color_raw, Image, timeout=self.color_timeout
                    )
                    color = _image_msg_to_bgr8(msg_r)
                    if self.as_rgb:
                        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    with self._lock:
                        self._latest_color = color
                    got = True
                except Exception:
                    pass

            if not got:
                # 没收到就稍微睡一下，避免忙等
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
    1) 相机发布了 /<ns>/color/image_raw/compressed 或 /<ns>/color/image_raw
    2) 已安装 image_transport & compressed_depth_image_transport（若使用 compressed）
       sudo apt-get install ros-$(rosversion -d)-image-transport \
                              ros-$(rosversion -d)-compressed-depth-image-transport
    """
    # 先初始化 ROS 节点（必须在创建 ColorStreamer 之前）
    
    # 如果使用仿真时间（/use_sim_time:=true），可选地等待 /clock 开始发布
    # if rospy.get_param("/use_sim_time", False):
    #     from rosgraph_msgs.msg import Clock
    #     rospy.loginfo("Waiting for /clock ...")
    #     try:
    #         rospy.wait_for_message("/clock", Clock, timeout=5.0)
    #     except Exception:
    #         rospy.logwarn("No /clock received; Rate may not advance under use_sim_time.")

    streamer = ColorStreamer(
        ns="/camera_head",
        as_rgb=True,       # True 则输出 RGB；False 为 OpenCV 常用的 BGR
        start_immediately=True
    )

    while True:
        time.sleep(0.1)
        color = streamer.get_latest_color()
        if color is not None:
            print("color:", color.shape, color.dtype)

            # 显示图像
            cv2.imshow("Color Stream", color)

            # 检查是否按下 'q' 键来退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # 退出时关闭 OpenCV 窗口
    cv2.destroyAllWindows()