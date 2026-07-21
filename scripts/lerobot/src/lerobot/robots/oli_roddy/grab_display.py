# -*- coding: utf-8 -*-
"""
海康工业相机实时可视化与数据采集程序

操作说明:
  ↑/↓   - 增大/减小曝光时间
  ←/→   - 减小/增大增益
  g/h   - 减小/增大 Gamma（压暗/提亮画面）
  b/n   - 减小/增大蓝色通道（消黄/加蓝）
  c/v   - 减小/增大饱和度（增艳）
  a     - 切换自动曝光
  s     - 保存当前帧到 images/ 目录
  q/ESC - 退出程序
"""

import sys
import os
import time
import json
import numpy as np
import cv2
from ctypes import *

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "MvImport"))
from MvCameraControl_class import *


HB_PIXEL_FORMATS = {
    PixelType_Gvsp_HB_Mono8, PixelType_Gvsp_HB_Mono10,
    PixelType_Gvsp_HB_Mono10_Packed, PixelType_Gvsp_HB_Mono12,
    PixelType_Gvsp_HB_Mono12_Packed, PixelType_Gvsp_HB_Mono16,
    PixelType_Gvsp_HB_RGB8_Packed, PixelType_Gvsp_HB_BGR8_Packed,
    PixelType_Gvsp_HB_RGBA8_Packed, PixelType_Gvsp_HB_BGRA8_Packed,
    PixelType_Gvsp_HB_BayerGR8, PixelType_Gvsp_HB_BayerRG8,
    PixelType_Gvsp_HB_BayerGB8, PixelType_Gvsp_HB_BayerBG8,
    PixelType_Gvsp_HB_BayerRBGG8,
    PixelType_Gvsp_HB_BayerGR10, PixelType_Gvsp_HB_BayerRG10,
    PixelType_Gvsp_HB_BayerGB10, PixelType_Gvsp_HB_BayerBG10,
    PixelType_Gvsp_HB_BayerGR12, PixelType_Gvsp_HB_BayerRG12,
    PixelType_Gvsp_HB_BayerGB12, PixelType_Gvsp_HB_BayerBG12,
    PixelType_Gvsp_HB_BayerGR10_Packed, PixelType_Gvsp_HB_BayerRG10_Packed,
    PixelType_Gvsp_HB_BayerGB10_Packed, PixelType_Gvsp_HB_BayerBG10_Packed,
    PixelType_Gvsp_HB_BayerGR12_Packed, PixelType_Gvsp_HB_BayerRG12_Packed,
    PixelType_Gvsp_HB_BayerGB12_Packed, PixelType_Gvsp_HB_BayerBG12_Packed,
    PixelType_Gvsp_HB_YUV422_Packed, PixelType_Gvsp_HB_YUV422_YUYV_Packed,
    PixelType_Gvsp_HB_RGB16_Packed, PixelType_Gvsp_HB_BGR16_Packed,
    PixelType_Gvsp_HB_RGBA16_Packed, PixelType_Gvsp_HB_BGRA16_Packed,
}

MONO_PIXEL_FORMATS = {
    PixelType_Gvsp_Mono8, PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed, PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed, PixelType_Gvsp_Mono14,
    PixelType_Gvsp_Mono16,
}


def decoding_char(c_char_array):
    raw = memoryview(c_char_array).tobytes()
    null_idx = raw.find(b'\x00')
    if null_idx != -1:
        raw = raw[:null_idx]
    for enc in ('gbk', 'utf-8', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', errors='replace')


def enum_devices():
    """枚举设备并让用户选择，返回设备信息"""
    device_list = MV_CC_DEVICE_INFO_LIST()
    tl_type = (MV_GIGE_DEVICE | MV_USB_DEVICE | MV_GENTL_CAMERALINK_DEVICE
               | MV_GENTL_CXP_DEVICE | MV_GENTL_XOF_DEVICE)

    ret = MvCamera.MV_CC_EnumDevices(tl_type, device_list)
    if ret != 0:
        raise RuntimeError("枚举设备失败! ret[0x%x]" % ret)

    if device_list.nDeviceNum == 0:
        raise RuntimeError("未找到任何设备!")

    print("=" * 50)
    print("找到 %d 个设备:" % device_list.nDeviceNum)
    print("=" * 50)

    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            name = decoding_char(info.SpecialInfo.stGigEInfo.chModelName)
            ip_raw = info.SpecialInfo.stGigEInfo.nCurrentIp
            ip_str = "%d.%d.%d.%d" % (
                (ip_raw >> 24) & 0xff, (ip_raw >> 16) & 0xff,
                (ip_raw >> 8) & 0xff, ip_raw & 0xff)
            print("[%d] GigE  | %s | IP: %s" % (i, name, ip_str))
        elif info.nTLayerType == MV_USB_DEVICE:
            name = decoding_char(info.SpecialInfo.stUsb3VInfo.chModelName)
            sn = decoding_char(info.SpecialInfo.stUsb3VInfo.chSerialNumber)
            print("[%d] USB3  | %s | SN: %s" % (i, name, sn))
        else:
            print("[%d] 其他设备" % i)

    print("=" * 50)
    idx = int(input("请输入要连接的设备编号: "))
    if idx < 0 or idx >= device_list.nDeviceNum:
        raise RuntimeError("设备编号无效!")

    return cast(device_list.pDeviceInfo[idx], POINTER(MV_CC_DEVICE_INFO)).contents


def frame_to_numpy(cam, frame_out):
    """将 MV_FRAME_OUT 转换为 numpy 数组 (BGR 或 Mono8)"""
    width = frame_out.stFrameInfo.nWidth
    height = frame_out.stFrameInfo.nHeight
    pixel_type = frame_out.stFrameInfo.enPixelType

    convert_param = MV_CC_PIXEL_CONVERT_PARAM_EX()
    memset(byref(convert_param), 0, sizeof(convert_param))

    if pixel_type in HB_PIXEL_FORMATS:
        decode_buf_len = width * height * 3
        decode_buf = (c_ubyte * decode_buf_len)()
        decode_param = MV_CC_HB_DECODE_PARAM()
        decode_param.pSrcBuf = frame_out.pBufAddr
        decode_param.nSrcLen = frame_out.stFrameInfo.nFrameLen
        decode_param.pDstBuf = decode_buf
        decode_param.nDstBufSize = decode_buf_len
        ret = cam.MV_CC_HBDecode(decode_param)
        if ret != 0:
            return None
        convert_param.pSrcData = decode_param.pDstBuf
        convert_param.nSrcDataLen = decode_param.nDstBufLen
        convert_param.enSrcPixelType = decode_param.enDstPixelType
    else:
        convert_param.pSrcData = frame_out.pBufAddr
        convert_param.nSrcDataLen = frame_out.stFrameInfo.nFrameLen
        convert_param.enSrcPixelType = pixel_type

    is_mono = convert_param.enSrcPixelType in MONO_PIXEL_FORMATS
    if is_mono:
        dst_pixel_type = PixelType_Gvsp_Mono8
        channels = 1
    else:
        dst_pixel_type = PixelType_Gvsp_BGR8_Packed
        channels = 3

    dst_buf_len = width * height * channels
    dst_buf = (c_ubyte * dst_buf_len)()

    convert_param.nWidth = width
    convert_param.nHeight = height
    convert_param.enDstPixelType = dst_pixel_type
    convert_param.pDstBuffer = dst_buf
    convert_param.nDstBufferSize = dst_buf_len

    ret = cam.MV_CC_ConvertPixelTypeEx(convert_param)
    if ret != 0:
        return None

    if is_mono:
        return np.frombuffer(dst_buf, dtype=np.uint8, count=dst_buf_len).reshape(height, width)
    else:
        return np.frombuffer(dst_buf, dtype=np.uint8, count=dst_buf_len).reshape(height, width, 3)


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
FRAME_WIDTH = 1440 #640
FRAME_HEIGHT = 1080 #480
PUB_IMG_WIDTH = 640
PUB_IMG_HEIGHT = 480
CONFIG_DEFAULTS = {"exposure_us": 30000.0, "gain_db": 10.0, "gamma": 2.2,
                   "wb_r": None, "wb_g": None, "wb_b": None,
                   "sw_blue": 1.0, "saturation": 1.0,
                   "width": FRAME_WIDTH, "height": FRAME_HEIGHT,
                   "pub_img_width": PUB_IMG_WIDTH, "pub_img_height": PUB_IMG_HEIGHT}


def format_wb_value(value):
    return "None" if value is None else "%.1f" % value


def set_camera_resolution(cam, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    cam.MV_CC_SetIntValue("OffsetX", 0)
    cam.MV_CC_SetIntValue("OffsetY", 0)

    ret = cam.MV_CC_SetIntValue("Width", width)
    if ret != 0:
        raise RuntimeError("设置宽度失败! ret[0x%x]" % ret)

    ret = cam.MV_CC_SetIntValue("Height", height)
    if ret != 0:
        raise RuntimeError("设置高度失败! ret[0x%x]" % ret)


def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
            wb_r = data.get("wb_r")
            wb_g = data.get("wb_g")
            wb_b = data.get("wb_b")
            return (float(data.get("exposure_us", CONFIG_DEFAULTS["exposure_us"])),
                    float(data.get("gain_db",     CONFIG_DEFAULTS["gain_db"])),
                    float(data.get("gamma",        CONFIG_DEFAULTS["gamma"])),
                    float(wb_r) if wb_r is not None else None,
                    float(wb_g) if wb_g is not None else None,
                    float(wb_b) if wb_b is not None else None,
                    float(data.get("sw_blue",   CONFIG_DEFAULTS["sw_blue"])),
                    float(data.get("saturation", CONFIG_DEFAULTS["saturation"])),
                    int(data.get("width", CONFIG_DEFAULTS["width"])),
                    int(data.get("height", CONFIG_DEFAULTS["height"])),
                    int(data.get("pub_img_width", CONFIG_DEFAULTS["pub_img_width"])),
                    int(data.get("pub_img_height", CONFIG_DEFAULTS["pub_img_height"])))
        except Exception:
            pass
    return (CONFIG_DEFAULTS["exposure_us"], CONFIG_DEFAULTS["gain_db"],
            CONFIG_DEFAULTS["gamma"], None, None, None,
            CONFIG_DEFAULTS["sw_blue"], CONFIG_DEFAULTS["saturation"],
            CONFIG_DEFAULTS["width"], CONFIG_DEFAULTS["height"],
            CONFIG_DEFAULTS["pub_img_width"], CONFIG_DEFAULTS["pub_img_height"])


def save_config(exposure_us, gain_db, gamma, wb_r=None, wb_g=None, wb_b=None,
                sw_blue=1.0, saturation=1.0, width=FRAME_WIDTH, height=FRAME_HEIGHT):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump({"exposure_us": exposure_us, "gain_db": gain_db, "gamma": gamma,
                       "wb_r": wb_r, "wb_g": wb_g, "wb_b": wb_b,
                       "sw_blue": sw_blue, "saturation": saturation,
                       "width": width, "height": height}, f, indent=2)
    except Exception as e:
        print("配置保存失败: %s" % e)


def color_adjust(img, sw_blue, saturation):
    """软件蓝色通道增益 + 饱和度调整。"""
    if sw_blue == 1.0 and saturation == 1.0:
        return img
    out = img.astype(np.float32)
    if sw_blue != 1.0:
        out[:, :, 0] = np.clip(out[:, :, 0] * sw_blue, 0, 255)  # BGR: 0=B
    out = out.astype(np.uint8)
    if saturation != 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def apply_wb(cam, wb_r, wb_g, wb_b):
    """将 R/G/B BalanceRatio 写入相机，自动适配浮点/整数类型。"""
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 0)
    for selector, val in [(0, wb_r), (1, wb_g), (2, wb_b)]:
        cam.MV_CC_SetEnumValue("BalanceRatioSelector", selector)
        if cam.MV_CC_SetFloatValue("BalanceRatio", val) != 0:
            cam.MV_CC_SetIntValue("BalanceRatio", int(val))


def _read_balance_ratio(cam):
    """读取当前选中通道的 BalanceRatio，先尝试浮点，再尝试整数。"""
    stF = MVCC_FLOATVALUE()
    memset(byref(stF), 0, sizeof(stF))
    if cam.MV_CC_GetFloatValue("BalanceRatio", stF) == 0:
        return stF.fCurValue

    stI = MVCC_INTVALUE_EX()
    memset(byref(stI), 0, sizeof(stI))
    if cam.MV_CC_GetIntValueEx("BalanceRatio", stI) == 0:
        return float(stI.nCurValue)

    stI2 = MVCC_INTVALUE()
    memset(byref(stI2), 0, sizeof(stI2))
    if cam.MV_CC_GetIntValue("BalanceRatio", stI2) == 0:
        return float(stI2.nCurValue)

    return None


def do_once_awb(cam):
    """触发一次自动白平衡，返回读取到的 (wb_r, wb_g, wb_b)，失败返回 None。"""
    ret = cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 1)
    if ret != 0:
        print("自动白平衡不支持 ret[0x%x]" % ret)
        return None
    time.sleep(1.0)   # 等待相机收敛
    cam.MV_CC_SetEnumValue("BalanceWhiteAuto", 0)

    ratios = []
    for selector in (0, 1, 2):
        cam.MV_CC_SetEnumValue("BalanceRatioSelector", selector)
        val = _read_balance_ratio(cam)
        ratios.append(val)

    if any(v is None for v in ratios):
        return None
    return tuple(ratios)


def build_gamma_lut(gamma):
    """生成 Gamma 校正查找表。gamma>1 提亮画面，gamma=1 不变，gamma<1 压暗。"""
    inv = 1.0 / gamma
    lut = np.array([min(255, int(((i / 255.0) ** inv) * 255 + 0.5)) for i in range(256)],
                   dtype=np.uint8)
    return lut


def main():
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(save_dir, exist_ok=True)

    MvCamera.MV_CC_Initialize()

    cam = MvCamera()
    dev_info = None

    try:
        dev_info = enum_devices()

        ret = cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise RuntimeError("创建句柄失败! ret[0x%x]" % ret)

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError("打开设备失败! ret[0x%x]" % ret)

        if dev_info.nTLayerType in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            pkt_size = cam.MV_CC_GetOptimalPacketSize()
            if int(pkt_size) > 0:
                cam.MV_CC_SetIntValue("GevSCPSPacketSize", pkt_size)

        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)

        # ---- 曝光 & 增益 & 白平衡配置 ----
        exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height, pub_img_width, pub_img_height = load_config()
        set_camera_resolution(cam, width, height)
        print("[配置] 从 %s 加载: 分辨率 %dx%d | 曝光 %.0f μs | 增益 %.1f dB | Gamma %.1f | 白平衡 R=%s G=%s B=%s | 蓝 %.2f | 饱和 %.1f"
              % (CONFIG_PATH, width, height, exposure_us, gain_db, gamma,
                 format_wb_value(wb_r), format_wb_value(wb_g), format_wb_value(wb_b),
                 sw_blue, saturation))
        cam.MV_CC_SetEnumValue("ExposureAuto", 0)       # 关闭自动曝光
        cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)

        cam.MV_CC_SetEnumValue("GainAuto", 0)            # 关闭自动增益
        cam.MV_CC_SetFloatValue("Gain", gain_db)

        if wb_r is not None:
            try:
                apply_wb(cam, wb_r, wb_g, wb_b)
                print("[配置] 白平衡已恢复: R=%.1f G=%.1f B=%.1f" % (wb_r, wb_g, wb_b))
            except Exception as e:
                print("[警告] 白平衡恢复失败: %s" % e)

        EXPOSURE_STEP = 5000.0                            # 每次调节步长 (μs)
        GAIN_STEP = 2.0                                   # 每次调节步长 (dB)
        GAMMA_STEP = 0.2                                  # 每次调节步长
        BLUE_STEP = 0.05                                  # 每次调节步长
        SAT_STEP = 0.1                                    # 每次调节步长
        gamma_lut = build_gamma_lut(gamma)

        print("\n曝光: %.0f μs | 增益: %.1f dB | Gamma: %.1f | 蓝: %.2f | 饱和: %.1f"
              % (exposure_us, gain_db, gamma, sw_blue, saturation))

        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError("开始取流失败! ret[0x%x]" % ret)

        print("\n" + "=" * 50)
        print("实时预览已启动")
        print("  ↑/↓   - 增大/减小曝光时间")
        print("  ←/→   - 减小/增大增益")
        print("  g/h   - 减小/增大 Gamma")
        print("  w     - 自动白平衡 (一次性)")
        print("  b/n   - 减小/增大蓝色通道 (消黄/加蓝)")
        print("  c/v   - 减小/增大饱和度 (增艳)")
        print("  a     - 切换自动曝光")
        print("  s     - 保存当前帧")
        print("  q/ESC - 退出")
        print("=" * 50 + "\n")

        frame_out = MV_FRAME_OUT()
        memset(byref(frame_out), 0, sizeof(frame_out))
        save_count = 0
        auto_exposure = False
        window_name = "HIK Camera Live"

        KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 82, 84, 81, 83

        while True:
            ret = cam.MV_CC_GetImageBuffer(frame_out, 1000)
            if ret != 0 or frame_out.pBufAddr is None:
                continue

            frame_num = frame_out.stFrameInfo.nFrameNum
            frame_w = frame_out.stFrameInfo.nWidth
            frame_h = frame_out.stFrameInfo.nHeight

            img = frame_to_numpy(cam, frame_out)
            cam.MV_CC_FreeImageBuffer(frame_out)

            if img is None:
                continue

            display = color_adjust(cv2.LUT(img, gamma_lut), sw_blue, saturation)

            info_text = "Frame: %d | %dx%d" % (frame_num, frame_w, frame_h)
            exp_text = "Exp:%.0fus Gain:%.1fdB G:%.1f B:%.2f S:%.1f%s" % (
                exposure_us, gain_db, gamma, sw_blue, saturation,
                " [AUTO]" if auto_exposure else "")
            wb_text = "Config:%s | WB R:%s G:%s B:%s" % (
                os.path.basename(CONFIG_PATH),
                format_wb_value(wb_r),
                format_wb_value(wb_g),
                format_wb_value(wb_b),
            )
            cv2.putText(display, info_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, exp_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)
            cv2.putText(display, wb_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') or key == ord('S'):
                save_count += 1
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = "img_%s_%04d.png" % (timestamp, save_count)
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, display)
                print("[已保存] %s" % filepath)
            elif key == ord('w') or key == ord('W'):
                print("正在执行自动白平衡，请稍候...")
                result = do_once_awb(cam)
                if result:
                    wb_r, wb_g, wb_b = result
                    print("白平衡完成: R=%.1f G=%.1f B=%.1f" % (wb_r, wb_g, wb_b))
                    save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
                else:
                    print("白平衡读取失败，效果已应用但未保存比值")
            elif key == ord('b'):
                sw_blue = max(round(sw_blue - BLUE_STEP, 2), 0.1)
                print("蓝色通道: %.2f (减蓝)" % sw_blue)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('n'):
                sw_blue = min(round(sw_blue + BLUE_STEP, 2), 3.0)
                print("蓝色通道: %.2f (加蓝)" % sw_blue)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('c'):
                saturation = max(round(saturation - SAT_STEP, 1), 0.0)
                print("饱和度: %.1f (降低)" % saturation)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('v'):
                saturation = min(round(saturation + SAT_STEP, 1), 3.0)
                print("饱和度: %.1f (增强)" % saturation)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('a') or key == ord('A'):
                auto_exposure = not auto_exposure
                cam.MV_CC_SetEnumValue("ExposureAuto", 2 if auto_exposure else 0)
                print("自动曝光: %s" % ("开启" if auto_exposure else "关闭"))
            elif key == ord('g'):
                gamma = max(round(gamma - GAMMA_STEP, 1), 0.1)
                gamma_lut = build_gamma_lut(gamma)
                print("Gamma: %.1f (压暗)" % gamma)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('h'):
                gamma = min(round(gamma + GAMMA_STEP, 1), 5.0)
                gamma_lut = build_gamma_lut(gamma)
                print("Gamma: %.1f (提亮)" % gamma)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == KEY_UP:
                exposure_us = min(exposure_us + EXPOSURE_STEP, 1000000.0)
                cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)
                print("曝光时间: %.0f μs" % exposure_us)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == KEY_DOWN:
                exposure_us = max(exposure_us - EXPOSURE_STEP, 100.0)
                cam.MV_CC_SetFloatValue("ExposureTime", exposure_us)
                print("曝光时间: %.0f μs" % exposure_us)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == KEY_RIGHT:
                gain_db = min(gain_db + GAIN_STEP, 30.0)
                cam.MV_CC_SetFloatValue("Gain", gain_db)
                print("增益: %.1f dB" % gain_db)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == KEY_LEFT:
                gain_db = max(gain_db - GAIN_STEP, 0.0)
                cam.MV_CC_SetFloatValue("Gain", gain_db)
                print("增益: %.1f dB" % gain_db)
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
            elif key == ord('q') or key == ord('Q') or key == 27:
                print("\n正在退出...")
                save_config(exposure_us, gain_db, gamma, wb_r, wb_g, wb_b, sw_blue, saturation, width, height)
                break

        cv2.destroyAllWindows()

        cam.MV_CC_StopGrabbing()
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()

    except Exception as e:
        print("错误: %s" % e)
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
    finally:
        MvCamera.MV_CC_Finalize()


if __name__ == "__main__":
    main()