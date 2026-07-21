import json
import uuid
import threading
import time
import websocket
import socket # Import socket module for pre-connection check
import sys
from datetime import datetime 
import random
import inquirer
import os


# Replace this ACCID value with your robot's actual serial number (SN)
ACCID = "None"

# Replace it with the real IP address of the robot. 
# Usually, for simulation, it is: 127.0.0.1
# for a real machine, it is: 10.192.1.2
ROBOT_IP = "10.192.1.2"
ROBOT_PORT = 5000 # WebSocket port

# Atomic flag for graceful exit (not strictly used for automated sequence completion here)
should_exit = False

# WebSocket client instance
ws_client = None

# Global flag to track WebSocket connection status
is_connected = False

# Generate dynamic GUID
def generate_guid():
    return str(uuid.uuid4())

LAST_SN_FILE = ".last_sn.json"

def load_last_sn():
    """读取上一次保存的 SN"""
    if os.path.exists(LAST_SN_FILE):
        with open(LAST_SN_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                return data.get("last_sn")
            except json.JSONDecodeError:
                return None
    return None

def save_last_sn(sn):
    """保存本次的 SN"""
    with open(LAST_SN_FILE, "w", encoding="utf-8") as f:
        json.dump({"last_sn": sn}, f, ensure_ascii=False)

def generate_sn():
    """
    交互式生成标准 SN，格式为：
    [类型代号]_[大版本+小版本]_[识别号]_[流水号]
    例如：HU_D03_03_004
    """
    # 先尝试读取上一次保存的 SN
    last_sn = load_last_sn()
    if last_sn:
        use_last = inquirer.prompt([
            inquirer.Confirm(
                'use_last',
                message=f"检测到上一次 SN: {last_sn}，是否复用？",
                default=True
            )
        ])['use_last']
        if use_last:
            return last_sn

    while True:  # 外层循环，直到用户确认
        # 类型代号
        type_choice = inquirer.prompt([
            inquirer.List(
                'type',
                message="请选择类型代号",
                choices=['HU', 'DA', 'UB', '自定义输入']
            )
        ])['type']

        if type_choice == '自定义输入':
            type_code = inquirer.prompt([
                inquirer.Text('type_custom', message="请输入类型代号")
            ])['type_custom']
        else:
            type_code = type_choice

        # 大版本
        major_choice = inquirer.prompt([
            inquirer.List(
                'major',
                message="请选择大版本",
                choices=['A', 'B', 'C', 'D', '自定义输入']
            )
        ])['major']

        if major_choice == '自定义输入':
            major = inquirer.prompt([
                inquirer.Text('major_custom', message="请输入大版本代号")
            ])['major_custom']
        else:
            major = major_choice

        # 小版本
        minor_choice = inquirer.prompt([
            inquirer.List(
                'minor',
                message="请选择小版本",
                choices=['01', '02', '03', '04', '自定义输入']
            )
        ])['minor']

        if minor_choice == '自定义输入':
            minor = inquirer.prompt([
                inquirer.Text('minor_custom', message="请输入小版本代号")
            ])['minor_custom']
        else:
            minor = minor_choice

        # 识别号
        identifier_choice = inquirer.prompt([
            inquirer.List(
                'id',
                message="请选择识别号",
                choices=['01', '02', '03', '04', '自定义输入']
            )
        ])['id']

        if identifier_choice == '自定义输入':
            identifier = inquirer.prompt([
                inquirer.Text('id_custom', message="请输入识别号")
            ])['id_custom']
        else:
            identifier = identifier_choice

        # 流水号（手动输入）
        serial = inquirer.prompt([
            inquirer.Text(
                'serial',
                message="请输入三位流水号 (例如: 001)"
            )
        ])['serial']

        # 拼接结果：大版本 + 小版本
        version = f"{major}{minor}"
        sn = f"{type_code}_{version}_{identifier}_{serial}"

        print(f"\n机器人 SN 为： {sn}")

        # 确认
        confirm = inquirer.prompt([
            inquirer.Confirm(
                'ok',
                message="是否确认以上SN？",
                default=True
            )
        ])['ok']

        if confirm:
            save_last_sn(sn)  # 保存 SN
            return sn
        else:
            print("\n输入取消，请重新输入...\n")

# Send WebSocket request with title and data
def send_request(title, data=None):
    global ACCID, is_connected
    if data is None:
        data = {}

    # Create message structure with necessary fields
    message = {
        "accid": ACCID, # Use the ACCID learned from robot or None if not yet received
        "title": title,
        "timestamp": int(time.time() * 1000),  # Current timestamp in milliseconds
        "guid": generate_guid(),
        "data": data
    }

    message_str = json.dumps(message)

    # Send the message through WebSocket ONLY if client is truly connected
    if ws_client and is_connected:
        try:
            print(f"Sending message: {message_str}")
            ws_client.send(message_str)
        except websocket.WebSocketConnectionClosedException:
            print("Error: WebSocket connection is closed, cannot send message.")
            is_connected = False # Update status if exception occurs
        except Exception as e:
            print(f"Error sending message: {e}")
            is_connected = False # Update status on other send errors
    else:
        print(f"Warning: WebSocket client not connected or connection lost, skipping message: {message_str}")


# Automated command sequence
def send_automated_commands():
    
    ACCID = generate_sn()
    
    global is_connected
    print("\n--- 即将开始强脑二代灵巧手测试，请确保机器人已进入高阶开发者模式 ---")

    # Ensure connection is still active before starting the sequence
    if not is_connected:
        print("Connection lost or not established. Aborting automated commands.")
        return
    
    # --- Helper function to create data template ---
    def create_hand_data(mode=1, pos=[0.0]*6, vel=[1.5]*6, current=[0]*6, time_ms=[2000]*6):
        return {
            "left_mode": mode,
            "left_pos": pos,
            "left_vel": vel,
            "left_current": current,
            "left_time": time_ms,
            "right_mode": mode,
            "right_pos": pos,
            "right_vel": vel,
            "right_current": current,
            "right_time": time_ms
        }

    # --- Test 1: Open All Fingers (Position-Time Mode: 1) ---
    print("Step 1: request_set_brainco2_hand_cmd (Mode 1: Position-Time)")
    print("机器人将在按下 Enter 键后张开五指 (Pos: 0.0 rad, Time: 2000ms)")
    print("请按 Enter 继续...")
    input()
    
    # 0 rad (Open) for all fingers, 2000ms to complete
    open_pos = [0.0] * 6
    open_time = [2000] * 6
    hand_data = create_hand_data(mode=1, pos=open_pos, time_ms=open_time)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(2.5) # Wait for movement
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return


    # --- Test 2: Close 4 Fingers, Thumb Open (Position-Velocity Mode: 2) ---
    print("Step 2: request_set_brainco2_hand_cmd (Mode 2: Position-Velocity)")
    print("机器人将在按下 Enter 键后收缩四指 (食指-小指: 1.4 rad, 速度: 1.5 rad/s)")
    print("请按 Enter 继续...")
    input()

    # Thumbs (0, 1) Open, Other (2-5) Closed (Max 1.4 rad)
    close_four_pos = [0.0, 0.0, 1.4, 1.4, 1.4, 1.4]
    mid_vel = [1.5] * 6 # Mid-range speed
    hand_data = create_hand_data(mode=2, pos=close_four_pos, vel=mid_vel)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(1.5) # Wait for movement
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return


    # --- Test 3: Close Thumbs, Open 4 Fingers (Position-Time Mode: 1) ---
    print("Step 3: request_set_brainco2_hand_cmd (Mode 1: Position-Time)")
    print("机器人将在按下 Enter 键后张开四指并收缩拇指 (拇指: Max Pos, Time: 2000ms)")
    print("请按 Enter 继续...")
    input()
    
    hand_data = create_hand_data(mode=1, pos=open_pos, time_ms=open_time)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(2.5) # Wait for movement
    send_request("request_get_brainco2_hand_state")
    
    # Thumb Tip (0): 1.0 rad, Thumb Base (1): 1.5 rad. Others (2-5) Open (0.0 rad)
    close_thumb_pos = [1.0, 1.5, 0.0, 0.0, 0.0, 0.0] 
    hand_data = create_hand_data(mode=1, pos=close_thumb_pos, time_ms=open_time)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(2.5) # Wait for movement
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return


    # --- Test 4: OK Gesture (Position-Time Mode: 1) ---
    print("Step 4: request_set_brainco2_hand_cmd (Mode 1: Position-Time)")
    print("机器人将在按下 Enter 键后比OK手势")
    print("请按 Enter 继续...")
    input()
    
    # Estimation for OK Gesture: Thumb Tip (0) & Index (2) partially closed/touching, others open.
    # Thumb Tip (0): 0.7 rad, Thumb Base (1): 0.6 rad, Index (2): 0.8 rad, Others (3-5): 0 rad (Open)
    ok_pos = [0.7, 0.6, 0.8, 0.0, 0.0, 0.0]
    ok_time = [1500] * 6 # Faster transition
    hand_data = create_hand_data(mode=1, pos=ok_pos, time_ms=ok_time)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(2.0)
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return


    # --- Test 5: Grasp/Fist using Force Control (Mode 3) ---
    print("Step 5: request_set_brainco2_hand_cmd (Mode 3: Force Control)")
    print("机器人将在按下 Enter 键后尝试握拳并四指保持力控 (Current: 300mA)")
    print("请按 Enter 继续...")
    input()
    
    # Apply a continuous positive current to attempt closing and holding
    force_current =  [0, 0, 300, 300, 300, 300]
    # For force control, pos, vel, time are typically ignored, but current is essential.
    hand_data = create_hand_data(mode=3, current=force_current)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(3) # Wait longer to observe the force hold
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return

    # --- Test 6: Exit Control (Mode 0) ---
    print("Step 6: request_set_brainco2_hand_cmd (Mode 0: Exit Control)")
    print("机器人将在按下 Enter 键后张开并退出控制模式")
    print("请按 Enter 继续...")
    input()
    
    # First: Send a command to explicitly open (using Position-Time)
    hand_data = create_hand_data(mode=1, pos=open_pos, time_ms=open_time)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(2.5) 
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return
    
    # Second: Send Mode 0 to stop control
    hand_data = create_hand_data(mode=0)
    send_request("request_set_brainco2_hand_cmd", hand_data)
    time.sleep(0.5) 
    send_request("request_get_brainco2_hand_state")
    if not is_connected: return


    print("\n--- 强脑二代灵巧手测试结束，Ctrl + C 退出程序 ---")


# WebSocket on_open callback 
def on_open(ws):
    global is_connected
    print("Connected!")
    is_connected = True # Set connection flag to True
    # Start automated command sequence in a separate thread
    threading.Thread(target=send_automated_commands, daemon=True).start()

# WebSocket on_message callback
def on_message(ws, message):
    global ACCID
    root = json.loads(message)
    title = root.get("title", "")
    ACCID = root.get("accid", None)

    if title != "notify_robot_info":
        print(f"Received message: {message}")  # Print the received message

# WebSocket on_close callback
def on_close(ws, close_status_code, close_msg):
    global is_connected
    print("Connection closed.")
    is_connected = False # Set connection flag to False

# Close WebSocket connection
def close_connection(ws):
    ws.close()

# Function to check if the robot's port is open (simple TCP check)
def check_robot_connection(ip, port, timeout=1):
    try:
        with socket.create_connection((ip, port), timeout) as sock:
            print(f"Successfully reached {ip}:{port} via TCP. Attempting WebSocket connection...")
            return True
    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the robot's WebSocket server running at {ip}:{port}?")
        return False
    except socket.timeout:
        print(f"Error: Connection timed out. Robot at {ip}:{port} is not responding or unreachable.")
        return False
    except socket.gaierror:
        print(f"Error: Invalid IP address or hostname: {ip}. Please check ROBOT_IP.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while checking connection to {ip}:{port}: {e}")
        return False

def main():
    global ws_client

    
    print("------------------------------------------------------------------")
    print("这个python脚本用于测试强脑二代灵巧手硬件及相关SDK")
    print("SDK接口名称：")
    print("request_set_brainco2_hand_cmd、request_get_brainco2_hand_state")
    print("张开五指->收缩四指->张开四指并收缩拇指->比OK手势->握拳->张开五指->结束")

    print("------------------------------------------------------------------")

    # Step 0: Check TCP connection to robot before proceeding
    print(f"Performing initial connection check to {ROBOT_IP}:{ROBOT_PORT}...")
    if not check_robot_connection(ROBOT_IP, ROBOT_PORT):
        print("Robot not connected or port unreachable. Please check robot status and IP address. Exiting program.")
        sys.exit(1) # Exit with an error code

    confirmation = input("Enter 'yes' or 'y' to proceed, or anything else to quit:\n").strip().lower()

    if confirmation not in ('yes', 'y'):
        print("User cancelled execution. Exiting program.")
        sys.exit(0) # Exit the program gracefully

    # Create WebSocket client instance
    ws_client = websocket.WebSocketApp(
        f"ws://{ROBOT_IP}:{ROBOT_PORT}",  # WebSocket server URI
        on_open=on_open,
        on_message=on_message,
        on_close=on_close
        # on_error is removed to strictly adhere to the original callback list
    )

    # Configure socket send and receive buffer sizes
    ws_client.sock_opt = [("socket", "SO_SNDBUF", 2 * 1024 * 1024)]
    ws_client.sock_opt.append(("socket", "SO_RCVBUF", 2 * 1024 * 1024))

    # Run WebSocket client loop
    print(f"Attempting to connect to robot WebSocket server: ws://{ROBOT_IP}:{ROBOT_PORT}")
    print("Program will automatically send commands. Press Ctrl+C to stop manually.")

    try:
        ws_client.run_forever()
    except KeyboardInterrupt:
        print("\nCtrl+C detected. Shutting down...")
    finally:
        if ws_client and ws_client.sock: # Ensure socket exists before attempting close
            ws_client.close()
        print("Program exited gracefully.")


if __name__ == "__main__":
    main()