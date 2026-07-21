import json
import uuid
import threading
import time
import websocket
from datetime import datetime

ROBOT_IP = "10.192.1.2"

class RobotController:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.accid = None
        self.ws_client = None
        self.should_exit = False
        self.connected_event = threading.Event()  # 连接状态同步事件
        self.get_new_pose_flag = False
        self.get_new_joint_state = False
        self.joint_name = []
        self.joint_q = []
        self.current_pose = None
        self.leftfinger_pos = []
        self.rightfinger_pos = []

        # 创建WebSocket客户端
        self.ws_client = websocket.WebSocketApp(
            f"ws://{ROBOT_IP}:5000",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )

        # 在后台线程中运行WebSocket客户端
        self.ws_thread = threading.Thread(target=self.ws_client.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # 等待连接建立或超时
        if not self.connected_event.wait(timeout=10.0):
            raise ConnectionError("Failed to connect to robot within 10 seconds")

    def send_request(self, title, data={}):
        if data is None:
            data = {}
        if not self.connected_event.is_set():
            raise ConnectionError("WebSocket not connected")
            
        message = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": str(uuid.uuid4()),
            "data": data
        }
        message_str = json.dumps(message)
        self.ws_client.send(message_str)

    def prepare2movep(self):
        try:
            # self.send_request("request_damping")
            # time.sleep(1)
            # self.send_request("request_prepare")
            # time.sleep(6)
            self.send_request("request_set_control_mode", {"mode": 0})
            time.sleep(0.5)
            self.send_request("request_moveP", { # request_moveP
                "left_position":[0.282278448343277,0.1601763516664505,0.0119148856028914],
                "left_pose":[-0.0324623920023441,-0.6807801723480225,-0.1766056716442108,0.7101372480392456],
                "right_position":[0.3258264362812042,-0.1893080025911331,0.0413756556808948],
                "right_pose":[0.0929329171776771,-0.693933367729187,0.000393996364437,0.7140166759490967],
                "speed": 0.1
            })
            # time.sleep(0.5)
            # self.send_request("request_moveP", {
            #     "left_position": [0.089644, 0.428712, 0.0519788],
            #     "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
            #     "right_position": [0.0835307, -0.431453, 0.13568],
            #     "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
            #     "speed": 0.1
            # })
            # self.send_request("request_get_pose")
        except ConnectionError as e:
            print(f"Error during operation: {str(e)}")

    def prepare2servop(self):
        try:
            self.send_request("request_damping")
            time.sleep(1)
            self.send_request("request_prepare")
            time.sleep(6)
            self.send_request("request_set_control_mode", {"mode": 1})
            time.sleep(0.5)
            self.send_request("request_servoP", { # request_moveP
                "left_position":[0.23500476777553558,0.2769041359424591,0.009378159418702126],
                "left_pose":[-0.03172443062067032,-0.6428214907646179,0.058355286717414856,0.7631308436393738],
                "right_position":[0.1900517195463180542,-0.27981323003768920898,0.032235391438007354736],
                "right_pose":[-0.11229092627763748169,-0.68661069869995117188,-0.031235236674547195435,0.71762168407440185547],
                "speed": 0.1
            })
            time.sleep(0.1)
            # self.send_request("request_moveP", {
            #     "left_position": [0.089644, 0.428712, 0.0519788],
            #     "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
            #     "right_position": [0.0835307, -0.431453, 0.13568],
            #     "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
            #     "speed": 0.1
            # })
            # self.send_request("request_get_pose")
        except ConnectionError as e:
            print(f"Error during operation: {str(e)}")

    def movep(self, data=None):
        if data:
            self.send_request("request_moveP", data)
        else:
            self.send_request("request_moveP", {
                "left_position": [0.089644, 0.428712, 0.0519788],
                "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
                "right_position": [0.0835307, -0.431453, 0.13568],
                "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
                "speed": 0.1
            })

    def servoP(self, data=None):
        if data:
            self.send_request("request_servoP", data)
        else:
            self.send_request("request_servoP", {
                "left_position": [0.089644, 0.428712, 0.0519788],
                "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
                "right_position": [0.0835307, -0.431453, 0.13568],
                "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
                "speed": 0.1
            })

    def movej(self, data=None):
        if data:
            self.send_request("request_moveJ", data)
        else:
            self.send_request("request_moveP", {
                "left_position": [0.089644, 0.428712, 0.0519788],
                "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
                "right_position": [0.0835307, -0.431453, 0.13568],
                "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
                "speed": 0.1
            })

    
    def servoJ(self, data=None):
        if data:
            self.send_request("request_servoJ", data)

    
    def move_finger(self, data=None):
        if data:
            self.send_request("request_set_brainco2_hand_cmd", data)


    def on_open(self, ws):
        print("WebSocket connection opened")
        self.connected_event.set()  # 标记连接已建立

    def on_message(self, ws, message):
        root = json.loads(message)
        title = root.get("title", "")
        self.accid = root.get("accid", self.accid)  # 更新accid
        
        if title != "notify_robot_info" and title != "notify_joy_data":
            # print(f"Received message: {message}")
            if title == "response_get_pose":
                self.get_new_pose_flag = True
                pose_list = root['data']['left_position'] + root['data']['left_pose'] + root['data']['right_position'] + root['data']['right_pose']
                self.current_pose = pose_list
            
            if title == "response_get_joint_state":
                self.get_new_joint_state = True
                self.joint_name = root['data']['names']
                self.joint_q = root['data']['q']

            if title == "response_get_brainco2_hand_state":
                self.leftfinger_pos = root['data']['left_pos']
                self.rightfinger_pos = root['data']['right_pos']
                


    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
        self.connected_event.clear()  # 发生错误时重置连接标志

    def on_close(self, ws, close_status_code, close_msg):
        print(f"Connection closed: {close_msg} (code: {close_status_code})")
        self.connected_event.clear()  # 连接关闭时重置标志
        self.accid = None

    def get_current_pose(self):
        self.send_request("request_get_pose")
        time.sleep(0.01)
        self.get_new_pose_flag = False
        return self.current_pose
    
    def get_current_joint_state(self):
        self.send_request("request_get_joint_state")
        time.sleep(0.01)
        # if self.get_new_joint_state:
        #     self.get_new_joint_state = False
        return self.joint_name, self.joint_q
    
    def get_current_finger_state(self):
        self.send_request("request_get_brainco2_hand_state")
        time.sleep(0.01)
        # if self.get_new_joint_state:
        #     self.get_new_joint_state = False
        return self.leftfinger_pos, self.rightfinger_pos


    
    def prepare2headmove(self):
        try:
            # self.send_request("request_damping")
            # time.sleep(1)
            # self.send_request("request_prepare")
            # time.sleep(6)
            self.send_request("request_set_control_mode", {"mode": 1})
            time.sleep(1)
            
            # self.send_request("request_moveP", {
            #     "left_position": [0.089644, 0.428712, 0.0519788],
            #     "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
            #     "right_position": [0.0835307, -0.431453, 0.13568],
            #     "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
            #     "speed": 0.1
            # })
            # self.send_request("request_get_pose")
        except ConnectionError as e:
            print(f"Error during operation: {str(e)}")
    
    def set_control_mode(self, mode):
        #0 for move, 1 for servo
        self.send_request("request_set_control_mode", {"mode": mode})
        time.sleep(1)


# 使用示例
if __name__ == "__main__":
    controller = RobotController(robot_ip=ROBOT_IP)
    time.sleep(1)
    # controller.prepare2movep()
    # controller.prepare2headmove()
    controller.set_control_mode(0)
    time.sleep(1)

    while True:
        print(controller.get_current_pose())
        #print joint pose
        print(controller.get_current_joint_state())
        time.sleep(1)
    exit()
    #init pose

    data = {
        "head_pitch": 0.9104045724868774,
        "head_yaw": 0.0021187369711697,
        "left": [0.09863018989562988, 0.08988189697265625, 0.18606318533420563, 0.0024567842483520508, 0.007707566954195499, -0.0038133200723677874, 0.337179002910852],
        "right": [0.09860610961914062, -0.09942114353179932, -0.1952073872089386, -0.0028524398803710938, -0.0026817258913069963, -0.0030990031082183123, -0.007801359985023737]
    }
    controller.movej(data=data)
    time.sleep(3)


    # data = {
    #     "head_pitch": 0.9104045724868774,
    #     "head_yaw": 0.0021187369711697,
    #     "left": [-0.0711622238159179,0.0658893585205078,-0.1171852871775627,-1.2761883735656738,0.0731456875801086,-0.221971184015274,0.0598109140992164],
    #     "right": [0.0158548355102539,-0.0897378921508789,0.1395959258079528,-1.3290026187896729,-0.0367980077862739,-0.2247629165649414,-0.0212496127933263]
    # }
    # controller.movej(data=data)


    current_ee_pose = controller.get_current_pose()
    print(current_ee_pose)
    current_joint_pose = controller.get_current_joint_state()
    print(current_joint_pose)

    # controller.movej(data={
    #     "head_pitch": 0.9104045724868774,
    #     "head_yaw": 0.0021187369711697,
    #     "left": [0.03680354356765747, 0.8941072225570679, 0.19215144217014313, -0.012020111083984375, 0.004691773094236851, 0.0120310103520751, -0.009599545039236546,],
    #     "right": [ 0.09816265106201172, -0.10265687108039856, -0.1947280466556549, -0.003283977508544922, -0.004212432540953159, 0.0015742842806503177, -0.007624289486557245],
    # })


    # time.sleep(3)

    # controller.movej(data={
    #     "head_pitch": 0.9104045724868774,
    #     "head_yaw": 0.0021187369711697,
    #     "left": [-0.04069840908050537, 0.8393632173538208, 0.08146513253450394, -1.5215052366256714, -0.03716636449098587, 0.10331329703330994, -0.00969757977873087],
    #     "right": [ 0.09816265106201172, -0.10265687108039856, -0.1947280466556549, -0.003283977508544922, -0.004212432540953159, 0.0015742842806503177, -0.007624289486557245],
    # })


    # time.sleep(5)


    # controller.movej(data={
    #     "head_pitch": 0.9104045724868774,
    #     "head_yaw": 0.0021187369711697,
    #     "left": [ -0.10808569192886353, 0.2502307891845703, 0.009679685346782207, -1.4316474199295044, -0.03558579832315445, 0.10353092849254608, -0.009694823063910007,],
    #     "right": [ 0.09816265106201172, -0.10265687108039856, -0.1947280466556549, -0.003283977508544922, -0.004212432540953159, 0.0015742842806503177, -0.007624289486557245],
    # })


    # time.sleep(3)


    # target_ee_pose = {
    #             "left_position": [0.089644, 0.428712, 0.0519788],
    #             "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
    #             "right_position": [0.0835307, -0.431453, 0.13568],
    #             "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
    #             "speed": 0.1
    #         }

    # target_ee_pose = {
    #             "left_position": [0.089644, 0.428712, 0.0519788],
    #             "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
    #             "right_position": [0.04507043957710266, -0.2602327764034271, -0.1802631914615631],
    #             "right_pose": [-0.06365077197551727, 0.032750390470027924, -0.05012422055006027, 0.9961744546890259],
    #             "speed": 0.1
    #         }
    # controller.movep(data = target_ee_pose)

    # time.sleep(3)

    # target_ee_pose = {
    #         "left_position": [0.089644, 0.478712, 0.0519788],
    #         "left_pose": [0.269296, -0.119683, -0.489868, 0.820478],
    #         "right_position": [0.0835307, -0.431453, 0.13568],
    #         "right_pose": [-0.436152, -0.285065, 0.265969, 0.81103],
    #         "speed": 0.1
    #     }


    # controller.movep(data = target_ee_pose)
    
    # time.sleep(0.01)
    # for i in range(30):
    #     print(controller.get_current_joint_state())
    #     # 等待操作完成（根据需要调整）
    #     time.sleep(1)
    
    # print(controller.get_current_joint_state())
