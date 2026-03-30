import cv2
import numpy as np
import os
import json
import math

IMAGE_FOLDER = "dataset"
OUTPUT_JSON = "labels_auto.json"


# -------------------------------------------------------------
# 1) 원형 중심 검출
# -------------------------------------------------------------
def detect_circle_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=80,
        param1=120, param2=30,
        minRadius=40, maxRadius=200
    )

    if circles is not None:
        cx, cy, r = np.round(circles[0][0]).astype(int)
        return cx, cy, r

    # fallback
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        return int(x), int(y), int(r)

    return None


# -------------------------------------------------------------
# 2) 초록 mask (왼쪽·아래쪽 제한 포함) 핵심
# -------------------------------------------------------------
def detect_green_mask(img, cx):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 게이지 초록 최적 범위
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)

    # 노이즈 제거 및 연결
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    H, W = mask.shape

    # 안정화 핵심 ① — 초록은 "오직 중심 왼쪽"에만 존재
    mask[:, cx:] = 0

    # 안정화 핵심 ② — 초록은 아래쪽 40% 범위에서만 존재
    mask[:int(H * 0.35), :] = 0

    # 확장으로 부드럽게
    mask = cv2.dilate(mask, kernel, 1)
    return mask


# -------------------------------------------------------------
# 3) 초록 x 좌우 경계
# -------------------------------------------------------------
def get_green_bounds(mask):
    cols = np.where(mask.max(axis=0) > 0)[0]
    if len(cols) == 0:
        return None
    return int(cols[0]), int(cols[-1])


# -------------------------------------------------------------
# 4) 바늘 끝: 오렌지 중 가장 위(y 최소) 픽셀 핵심
# -------------------------------------------------------------
def detect_needle_tip(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_orange = np.array([5, 80, 50])
    upper_orange = np.array([25, 255, 255])

    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    idx = np.argmin(ys)  # 가장 위쪽 픽셀 = 진짜 TIP
    return (int(xs[idx]), int(ys[idx]))


# -------------------------------------------------------------
# 5) GOOD / LOW / HIGH 분류
# -------------------------------------------------------------
def classify(tip_x, tip_y, left_x, right_x, green_mask):
    H, W = green_mask.shape

    # 1) 초록 mask 내부 → GOOD
    if 0 <= tip_x < W and 0 <= tip_y < H:
        if green_mask[tip_y, tip_x] > 0:
            return "good"

    # 2) x 범위 기반 GOOD
    if left_x <= tip_x <= right_x:
        return "good"

    # 3) 왼쪽이면 LOW
    if tip_x < left_x:
        return "Bad_low"

    # 4) 오른쪽이면 HIGH
    return "Bad_high"


# -------------------------------------------------------------
# 6) 전체 이미지 라벨링
# -------------------------------------------------------------
def auto_label_dataset(folder):
    result = []

    for root, dirs, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(root, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue

            center = detect_circle_center(img)
            if center is None:
                print("❌ 중심 검출 실패:", fname)
                continue
            cx, cy, r = center

            green_mask = detect_green_mask(img, cx)
            bounds = get_green_bounds(green_mask)

            if bounds is None:
                print("❌ 초록 없음:", fname)
                continue

            left_x, right_x = bounds

            tip = detect_needle_tip(img)
            if tip is None:
                print("❌ 바늘 검출 실패:", fname)
                continue

            tip_x, tip_y = tip

            label = classify(tip_x, tip_y, left_x, right_x, green_mask)

            result.append({
                "image_path": img_path.replace("\\", "/"),
                "tip_x": tip_x,
                "tip_y": tip_y,
                "green_left_x": left_x,
                "green_right_x": right_x,
                "label": label
            })

            print(f"✔ {fname} → {label}")

    return result


# -------------------------------------------------------------
# 7) 저장
# -------------------------------------------------------------
data = auto_label_dataset(IMAGE_FOLDER)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("\n 라벨링 완료!")
print("📁 저장:", OUTPUT_JSON)