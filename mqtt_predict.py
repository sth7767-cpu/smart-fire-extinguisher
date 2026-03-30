# predict_mqtt.py
import cv2
import numpy as np
import os
import time
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
import platform


# ================== 카메라 / OCR / CSV 설정 ==================
CAMERA_INDEX = 0

# -----------------------------
# 모델 설정
# -----------------------------
MODEL_PATH = "smoking_zone_classifier.h5"
IMG_SIZE = (224, 224)
model = load_model(MODEL_PATH)

label_map = ["good", "Bad_low", "Bad_high"]

def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(img):
    x = preprocess_image(img)
    prob = model.predict(x)[0]
    pred_idx = np.argmax(prob)
    pred_label = label_map[pred_idx]
    pred_conf = float(prob[pred_idx])
    return pred_label, pred_conf, prob

# -----------------------------
# MQTT 설정
# -----------------------------
broker_address = "192.168.0.5"
port = 1883
TOPIC_DATA = "data/pressure"
TOPIC_CONTROL = "control/pressure"

resend_flag = False
last_message = ""
QOS_LEVEL = 1

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("MQTT 연결 성공")
        client.subscribe(TOPIC_CONTROL, qos=QOS_LEVEL)
    else:
        print("MQTT 연결 실패:", rc)

def on_message_control(client, userdata, msg):
    global resend_flag
    payload = msg.payload.decode()
    if payload == "resend_request":
        print("📌 재전송 요청 수신")
        resend_flag = True

# MQTT 클라이언트 생성
client = mqtt.Client(
    client_id="predict_publisher",
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2
)

client.on_connect = on_connect
client.on_message = on_message_control

try:
    client.connect(broker_address, port)
except Exception as e:
    print("브로커 연결 실패:", e)
    exit(1)

client.loop_start()

# ================== 촬영 / 세션 설정 ==================
CAPTURE_INTERVAL_SEC = 2      # 사진 간격: 2초
IMAGES_PER_SESSION = 15       # 세션당 사진 15장
SESSION_INTERVAL_SEC = 180    # 세션 간격: 3분(180초)


# ================== 카메라 관련 함수 ==================
def initialize_camera(index):
    system = platform.system()
    print(f"[DEBUG] OS: {system}, CAMERA_INDEX: {index}")

    if system == "Windows":
        backends = [0, cv2.CAP_DSHOW, cv2.CAP_MSMF]
        for be in backends:
            if be == 0:
                print(f"[DEBUG] VideoCapture({index}) 기본 백엔드 시도")
                cap = cv2.VideoCapture(index)
            else:
                print(f"[DEBUG] VideoCapture({index}, backend={be}) 시도")
                cap = cv2.VideoCapture(index, be)

            if cap.isOpened():
                print(f"[INFO] Windows에서 카메라 장치 {index} 연결 성공 (backend={be})")
                return cap
            cap.release()

        print(f"[ERROR] Windows에서 카메라 장치 {index}를 열 수 없습니다.")
        return None
    else:
        print(f"[DEBUG] Linux/라즈베리파이에서 V4L2 백엔드 사용 시도")
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print(f"[ERROR] Linux에서 카메라 장치 {index}를 열 수 없습니다. (/dev/video{index} 확인 필요)")
            return None
        print(f"[INFO] Linux에서 카메라 장치 {index} 연결 성공 (V4L2)")
        return cap



# -----------------------------
# 카메라 예측 반복
# -----------------------------
cap = cv2.VideoCapture(0)   # 0번 기본 카메라

if not cap.isOpened():
    print("카메라 열기 실패!")
    exit()

print("📸 압력계 카메라 예측 시작... (CTRL+C로 종료)")

while True:
    ret, frame = cap.read()
    if not ret:
        print("카메라 캡처 오류")
        continue

    pred_label, pred_conf, full_probs = predict_image(frame)
    message = f"{pred_label}/{pred_conf:.4f}"

    if resend_flag:
        publish_msg = last_message
        resend_flag = False
        print(f"[재전송] {publish_msg}")
        client.publish(TOPIC_DATA, publish_msg, qos=QOS_LEVEL)
    else:
        publish_msg = message
        last_message = publish_msg
        print(f"[전송] 상태={pred_label}, 확률={pred_conf:.4f}")
        client.publish(TOPIC_DATA, publish_msg, qos=QOS_LEVEL)

    time.sleep(5)