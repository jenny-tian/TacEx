#! /usr/bin/env python3
import argparse
import importlib.util
from ctypes import POINTER, byref, cast, memset, sizeof
from pathlib import Path

import cv2
import mros
from mros.sensor_msgs.msg import CompressedImage


def default_topic(side: str) -> str:
    return f"/{side}_wrist_camera/color/image_raw/compressed"


def default_node_name(side: str) -> str:
    return f"hik_wrist_camera_publisher_{side}"


def default_frame_id(side: str) -> str:
    return f"{side}_wrist_camera"


def load_hik_sdk():
    module_path = Path(__file__).resolve().with_name("grab_display.py")
    spec = importlib.util.spec_from_file_location("hik_grab_display", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {module_path}")

    sdk_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk_module)
    return sdk_module


hik_sdk = load_hik_sdk()


def parse_args():
    parser = argparse.ArgumentParser(description="HIK camera compressed image publisher")
    parser.add_argument(
        "--side",
        choices=["left", "right"],
        default="left",
        help="安装在哪侧手腕，默认 left",
    )
    parser.add_argument("--device-index", type=int, default=0, help="海康相机索引，默认 0")
    parser.add_argument("--topic", default=None, help="发布 topic，默认按 side 自动生成")
    parser.add_argument("--node-name", default=None, help="mros 节点名，默认按 side 自动生成")
    parser.add_argument("--frame-id", default=None, help="消息 frame_id，默认按 side 自动生成")
    parser.add_argument("--fps", type=int, default=30, help="发布频率，默认 30")
    parser.add_argument(
        "--exposure-us",
        type=float,
        default=None,
        help="曝光时间(微秒)，默认读取 config.json",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=90,
        help="JPEG 压缩质量，范围 1-100，默认 90",
    )
    args = parser.parse_args()
    args.topic = args.topic or default_topic(args.side)
    args.node_name = args.node_name or default_node_name(args.side)
    args.frame_id = args.frame_id or default_frame_id(args.side)
    args.jpeg_quality = max(1, min(100, args.jpeg_quality))
    return args


def describe_device(info):
    if info.nTLayerType in (hik_sdk.MV_GIGE_DEVICE, hik_sdk.MV_GENTL_GIGE_DEVICE):
        name = hik_sdk.decoding_char(info.SpecialInfo.stGigEInfo.chModelName)
        ip_raw = info.SpecialInfo.stGigEInfo.nCurrentIp
        ip_str = "%d.%d.%d.%d" % (
            (ip_raw >> 24) & 0xFF,
            (ip_raw >> 16) & 0xFF,
            (ip_raw >> 8) & 0xFF,
            ip_raw & 0xFF,
        )
        return f"GigE | {name} | IP: {ip_str}"

    if info.nTLayerType == hik_sdk.MV_USB_DEVICE:
        name = hik_sdk.decoding_char(info.SpecialInfo.stUsb3VInfo.chModelName)
        serial = hik_sdk.decoding_char(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
        return f"USB3 | {name} | SN: {serial}"

    return "其他设备"


def get_device_info(device_index: int):
    device_list = hik_sdk.MV_CC_DEVICE_INFO_LIST()
    tl_type = (
        hik_sdk.MV_GIGE_DEVICE
        | hik_sdk.MV_USB_DEVICE
        | hik_sdk.MV_GENTL_CAMERALINK_DEVICE
        | hik_sdk.MV_GENTL_CXP_DEVICE
        | hik_sdk.MV_GENTL_XOF_DEVICE
    )

    ret = hik_sdk.MvCamera.MV_CC_EnumDevices(tl_type, device_list)
    if ret != 0:
        raise RuntimeError("枚举设备失败! ret[0x%x]" % ret)
    if device_list.nDeviceNum == 0:
        raise RuntimeError("未找到任何海康相机设备!")
    if device_index < 0 or device_index >= device_list.nDeviceNum:
        raise RuntimeError("设备索引越界: %d，当前共 %d 个设备" % (device_index, device_list.nDeviceNum))

    print("=" * 50)
    print("找到 %d 个海康设备:" % device_list.nDeviceNum)
    for idx in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[idx], POINTER(hik_sdk.MV_CC_DEVICE_INFO)).contents
        print("[%d] %s" % (idx, describe_device(info)))
    print("=" * 50)

    selected = cast(device_list.pDeviceInfo[device_index], POINTER(hik_sdk.MV_CC_DEVICE_INFO)).contents
    print("使用设备 [%d]: %s" % (device_index, describe_device(selected)))
    return selected


def configure_camera(cam, dev_info, exposure_us: float, gain_db: float, wb_r, wb_g, wb_b, width: int, height: int):
    ret = cam.MV_CC_CreateHandle(dev_info)
    if ret != 0:
        raise RuntimeError("创建句柄失败! ret[0x%x]" % ret)

    ret = cam.MV_CC_OpenDevice(hik_sdk.MV_ACCESS_Exclusive, 0)
    if ret != 0:
        raise RuntimeError("打开设备失败! ret[0x%x]" % ret)

    if dev_info.nTLayerType in (hik_sdk.MV_GIGE_DEVICE, hik_sdk.MV_GENTL_GIGE_DEVICE):
        pkt_size = cam.MV_CC_GetOptimalPacketSize()
        if int(pkt_size) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

    cam.MV_CC_SetEnumValue("TriggerMode", hik_sdk.MV_TRIGGER_MODE_OFF)
    hik_sdk.set_camera_resolution(cam, width, height)
    cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)
    cam.MV_CC_SetEnumValue("GainAuto", 0)
    cam.MV_CC_SetFloatValue("Gain", gain_db)

    if wb_r is not None and wb_g is not None and wb_b is not None:
        hik_sdk.apply_wb(cam, wb_r, wb_g, wb_b)

    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        raise RuntimeError("开始取流失败! ret[0x%x]" % ret)


def main():
    args = parse_args()

    mros.init(args.node_name)
    publisher = mros.advertise(args.topic, CompressedImage)
    print(f"发布 topic: {args.topic}")
    print(f"JPEG 质量: {args.jpeg_quality}")

    hik_sdk.MvCamera.MV_CC_Initialize()
    cam = hik_sdk.MvCamera()

    try:
        config_exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height, pub_img_width, pub_img_height = hik_sdk.load_config()
        exposure_us = args.exposure_us if args.exposure_us is not None else config_exposure_us
        gamma_lut = hik_sdk.build_gamma_lut(gamma)

        print(
            "[配置] 从 %s 加载:\n"
            "  采集分辨率: %dx%d | 发布分辨率: %dx%d\n"
            "  曝光: %.0f us | 增益: %.1f dB | Gamma: %.1f\n"
            "  白平衡: R=%s G=%s B=%s | 蓝: %.2f | 饱和: %.1f"
            % (
                hik_sdk.CONFIG_PATH,
                width, height, pub_img_width, pub_img_height,
                exposure_us, gain_db, gamma,
                hik_sdk.format_wb_value(wb_r),
                hik_sdk.format_wb_value(wb_g),
                hik_sdk.format_wb_value(wb_b),
                sw_blue, saturation,
            )
        )
        dev_info = get_device_info(args.device_index)
        configure_camera(cam, dev_info, exposure_us, gain_db, wb_r, wb_g, wb_b, width, height)
        print(
            "相机参数: 分辨率 %dx%d | 曝光 %.0f us | 增益 %.1f dB"
            % (width, height, exposure_us, gain_db)
        )

        rate = mros.Rate(max(1, args.fps))

        while mros.ok():
            frame_out = hik_sdk.MV_FRAME_OUT()
            memset(byref(frame_out), 0, sizeof(frame_out))

            ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
            if ret != 0 or frame_out.pBufAddr is None:
                rate.sleep()
                continue

            frame = hik_sdk.frame_to_numpy(cam, frame_out)
            cam.MV_CC_FreeImageBuffer(frame_out)

            if frame is None:
                print("图像转换失败")
                rate.sleep()
                continue

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            frame = hik_sdk.color_adjust(cv2.LUT(frame, gamma_lut), sw_blue, saturation)

            if frame.shape[1] != pub_img_width or frame.shape[0] != pub_img_height:
                frame = cv2.resize(frame, (pub_img_width, pub_img_height), interpolation=cv2.INTER_AREA)

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            if not ok:
                print("JPEG 压缩失败")
                rate.sleep()
                continue

            msg = CompressedImage()
            msg.header.stamp = mros.Time.now()
            msg.header.frame_id = args.frame_id
            msg.format = "jpeg"
            msg.data = encoded.tobytes()
            publisher.publish(msg)
            rate.sleep()

    except KeyboardInterrupt:
        print("收到退出信号，正在关闭...")
    except Exception as exc:
        print("运行失败: %s" % exc)
    finally:
        try:
            cam.MV_CC_StopGrabbing()
        except Exception:
            pass
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        hik_sdk.MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()
