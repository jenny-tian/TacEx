#! /usr/bin/env python3
import argparse
import cv2
import mros
from mros.sensor_msgs.msg import CompressedImage


DEFAULT_DEVICE = "/dev/video6"
# 覆盖角标区域的遮罩配置 (像素)，可按需调整
MASK_LEFT = (0, 0, 100, 60)   # x, y, w, h
MASK_RIGHT_WIDTH = 100        # 右上角遮罩宽度
MASK_RIGHT_HEIGHT = 60       # 右上角遮罩高度
MASK_SAMPLE_SIZE = 50         # 取色时左下角正方形边长


def configure_capture(device: str):
    """尝试多种格式/分辨率，找到可正常输出帧的配置。"""
    configs = [
        # ("MJPG", 1280, 720, 30),
        ("MJPG", 640, 480, 30),
        ("YUYV", 640, 480, 30),
        # ("YUYV", 320, 240, 30),
    ]

    cap = cv2.VideoCapture()
    for fmt, width, height, fps in configs:
        cap.open(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        ok = False
        frame = None
        for _ in range(5):  # 预读几帧，确认可用
            try:
                ok, frame = cap.read()
            except cv2.error as e:
                print(f"cap.read 出错({device}, {fmt},{width}x{height}@{fps}): {e}")
                ok = False
                break
            if ok and frame is not None:
                break

        if ok and frame is not None:
            print(f"使用设备 {device}，格式 {fmt} {width}x{height}@{fps}")
            return cap

        cap.release()

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Wrist camera compressed image publisher")
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="选择腕部相机: left 或 right，默认 left",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"视频设备路径，默认 {DEFAULT_DEVICE}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    side = args.side
    device = args.device

    # 初始化 mros
    mros.init(f"wrist_camera_node_{side}")

    # 发布压缩图像
    topic = f"/{side}_wrist_camera/color/image_raw/compressed"
    publisher = mros.advertise(topic, CompressedImage)
    print(f"发布 topic: {topic}")

    # 打开摄像头设备并自适应可用格式
    cap = configure_capture(device)
    if cap is None:
        print(f"无法打开 {device}，可尝试调整格式/分辨率或使用 v4l2-ctl 检查支持列表")
        return

    # 如需自定义分辨率/帧率可设置以下参数
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    rate = mros.Rate(30)  # 30 FPS

    while mros.ok():
        try:
            ok, frame = cap.read()
        except cv2.error as e:
            print(f"cap.read 出错: {e}")
            rate.sleep()
            continue

        if not ok or frame is None:
            print("读取帧失败")
            rate.sleep()
            continue

        # 覆盖左上、右上角的电量/容量叠加信息
        h, w, _ = frame.shape
        sample_size = min(MASK_SAMPLE_SIZE, h, w)
        roi = frame[h - sample_size:h, 0:sample_size]
        avg_color = tuple(int(c) for c in cv2.mean(roi)[:3])
        x1, y1, mw, mh = MASK_LEFT
        cv2.rectangle(frame, (x1, y1), (x1 + mw, y1 + mh), avg_color, thickness=-1)
        rx = max(0, w - MASK_RIGHT_WIDTH)
        cv2.rectangle(frame, (rx, 0), (w, MASK_RIGHT_HEIGHT), avg_color, thickness=-1)

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            print("JPEG 压缩失败")
            rate.sleep()
            continue

        msg = CompressedImage()
        msg.header.stamp = mros.Time.now()
        msg.format = "jpeg"
        msg.data = buf.tobytes()

        publisher.publish(msg)
        rate.sleep()

    cap.release()


if __name__ == "__main__":
    main()