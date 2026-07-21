"""
trafficlight_lib.py — 신호등 인식 (빨강=STOP / 초록=GO), YOLO 없이 단독 동작

[조명 대응 — 실내 커튼 환경, 일조량이 날마다 다름]
 절대 밝기 임계값(v_min=200 고정)은 조명이 0.85배만 어두워져도 빨간불을 놓쳤다.
 그래서 '자기 색상 대역 안에서의 상대적 밝기'로 바꿨다:
     임계값 = max(해당 hue 픽셀들의 99퍼센타일 * 0.75, 절대하한 60)
 조명이 어두워지면 기준선도 함께 내려가므로 판정이 흔들리지 않는다.

 검증 (밝기 0.3배~2.0배 × 사진 7장 = 77건):
   절대 임계값  : 0.85배에서 이미 실패
   상대 임계값  : 74/77. 0.3~1.2배 완전, 1.5배 이상에서만 초록 LED가
                  배경 나뭇잎과 병합되어 실패

[제일 중요한 것 — 소프트웨어보다 카메라 설정]
 신호등 LED는 스스로 빛을 낸다. 실내 조명이 변해도 LED가 내뿜는 빛은 그대로다.
 카메라 자동노출/자동화이트밸런스를 켜두면 카메라가 배경 밝기에 맞춰 노출을
 바꾸면서 LED 픽셀값까지 흔들어버린다. setup_camera()로 반드시 고정할 것.
 이걸 하면 위 시뮬레이션 같은 전역 밝기 변동 자체가 거의 사라진다.
"""

import cv2
import numpy as np
from collections import deque, Counter


# ==================================================================
# [섹터 1] 튜닝 파라미터
# ==================================================================
PARAMS = {
    "s_min": 100,           # 유채색 판정 채도 하한
    "bright_ratio": 0.75,   # 상대 밝기 임계값 비율 (자기 hue 대역 99퍼센타일 대비)
    "v_floor": 60,          # 절대 하한. 너무 높으면 어두운 프레임에서 전멸한다
                            # (80으로 뒀더니 0.3배 이미지의 최대 V가 76이라 검출 0개)
    "roi_top_ratio": 0.6,
    "min_area_ratio": 0.0005,  # 화면 면적의 0.05% 이상. 픽셀 개수가 아니라 '비율'이라
                               # 해상도가 바뀌어도 그대로 동작한다.
                               # (실측: 배경 자판기 0.010% / 진짜 신호등 0.91%)
    "close_ksize": 15,      # LED 알갱이 병합. 해상도에 맞춰 줄이면 오히려 나빠진다
                            # (480~1092px 전부 15가 최적. 320px 이하는 성능 저하)
    "open_ksize": 3,        # 3보다 키우면 LED 도트까지 갉아먹어 오히려 나빠진다
    "circularity_th": 0.75,
    "fill_ratio_th": 0.65,
    "solidity_th": 0.50,
    "vote_window": 5,
    "vote_min_count": 3,

    # --- [1차 판정] 타버린 핵 + 색 고리 방식 ---
    # 켜진 LED는 중심이 하얗게 타고(V≈255, S 낮음) 주위에 색 번짐 고리가 생긴다.
    # 커튼·벽 같은 "색만 비슷한 배경"은 하얀 핵이 없어 원천 배제된다.
    # (실측: 체육관 청록 커튼 V=108로 핵 조건 미달 → 오탐 차단)
    "core_v_min": 245,     # 핵 밝기 하한
    "core_s_max": 60,      # 핵 채도 상한 (하얀색이어야 함)
    "halo_s_min": 100,     # 고리 픽셀 채도 하한
    "halo_v_min": 120,
    "halo_r_in": 1.1,      # 고리 안쪽 반경 (핵 반경 배수)
    "halo_r_out": 1.7,     # 고리 바깥 반경
    "halo_min_votes": 30,  # 고리 최소 픽셀 수
    "halo_dominance": 0.6, # 우세 색 비율 하한
    "halo_density_min": 0.30,  # 승자표/고리넓이 하한. 진짜 LED는 고리가 빽빽함
                               # (실측: 진짜 0.59~0.83, 오탐 0.00~0.18)
    "core_max_r_ratio": 0.20,  # 핵 반지름 상한 (화면 짧은 변 대비)
}

# 과노출 시 붉은 번짐이 주황(h≈27)까지 밀리므로 고리용 대역은 넓게 잡는다.
HALO_BANDS = {
    "RED":   [(0, 35), (160, 180)],
    "GREEN": [(50, 100)],
}

HUE_BANDS = {
    "RED":   [(0, 8), (170, 180)],   # OpenCV hue는 0~179라 빨강이 양 끝에 걸침
    "GREEN": [(75, 95)],
}


# ==================================================================
# [섹터 2] 카메라 설정 — 이게 소프트웨어 튜닝보다 효과가 크다
# ==================================================================
def setup_camera(cap, exposure=-6, wb_temp=4500):
    """자동노출·자동화이트밸런스를 끄고 값을 고정한다.

    exposure 값은 카메라마다 다르다. 신호등이 살짝 과노출되되 배경은
    어둡게 나오는 값을 실험으로 찾을 것. (UVC 웹캠은 보통 -4 ~ -8)
    """
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # 0.25 = manual (백엔드마다 다름)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb_temp)
    return cap


# ==================================================================
# [섹터 3] 색 마스크 — 상대 밝기 임계값
# ==================================================================
def make_color_mask(hsv, color):
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    hue_ok = np.zeros(h.shape, dtype=bool)
    for lo, hi in HUE_BANDS[color]:
        hue_ok |= (h >= lo) & (h <= hi)
    candidate = hue_ok & (s > PARAMS["s_min"])

    if candidate.sum() < 20:
        return np.zeros(h.shape, dtype=np.uint8)

    # 핵심: 절대값이 아니라 '이 색 대역 안에서 얼마나 밝은가'.
    # 켜진 LED는 자기 색 대역에서 압도적으로 밝다.
    v_ref = np.percentile(v[candidate], 99)
    thr = max(v_ref * PARAMS["bright_ratio"], PARAMS["v_floor"])

    return (candidate & (v >= thr)).astype(np.uint8) * 255


# ==================================================================
# [섹터 4] 블롭 검출 및 원형 판정
# ==================================================================
def find_light_blobs(mask, frame_area):
    ok_, ck = PARAMS["open_ksize"], PARAMS["close_ksize"]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok_, ok_)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ck, ck)))

    min_area = frame_area * PARAMS["min_area_ratio"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        # 볼록 껍질이 핵심. LED 도트 매트릭스의 삐죽삐죽한 둘레를 고무줄처럼 감싼다.
        # 없으면 원형도 0.06, 씌우면 0.94.
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < min_area:
            continue

        peri = cv2.arcLength(hull, True)
        circularity = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        (cx, cy), radius = cv2.minEnclosingCircle(hull)
        fill_ratio = area / (np.pi * radius * radius) if radius > 0 else 0.0

        blobs.append({
            "center": (int(cx), int(cy)),
            "radius": int(radius),
            "area": area,
            "circularity": circularity,
            "fill_ratio": fill_ratio,
            "solidity": cv2.contourArea(c) / area,
        })
    return blobs


def is_light_shape(blob):
    return (blob["circularity"] > PARAMS["circularity_th"]
            and blob["fill_ratio"] > PARAMS["fill_ratio_th"]
            and blob["solidity"] > PARAMS["solidity_th"])




# ==================================================================
# [섹터 4.5] 1차 판정 — 타버린 핵 + 색 고리
# ==================================================================
def detect_by_glow(hsv, frame_area):
    """켜진 LED의 '하얀 핵'을 찾고 둘레 고리의 색으로 판정.

    반환: (color, blob) 또는 (None, None)
    핵이 없으면 아무것도 반환하지 않는다 → 커튼/벽 오탐 원천 차단.
    """
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    core = ((v >= PARAMS["core_v_min"]) & (s <= PARAMS["core_s_max"])).astype(np.uint8) * 255
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    contours, _ = cv2.findContours(core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = frame_area * PARAMS["min_area_ratio"]

    best = (None, None)
    for c in contours:
        hull = cv2.convexHull(c)
        area = cv2.contourArea(hull)
        if area < min_area:
            continue
        peri = cv2.arcLength(hull, True)
        circ = 4.0 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        (cx, cy), r = cv2.minEnclosingCircle(hull)
        fill = area / (np.pi * r * r) if r > 0 else 0.0
        if circ < 0.6 or fill < 0.6:
            continue
        # 과노출로 화면이 통째로 타면 거대한 흰 덩어리가 생긴다 → 크기 상한
        if r > min(h.shape) * PARAMS["core_max_r_ratio"]:
            continue

        # 핵 둘레 고리에서 색 투표
        ring_mask = np.zeros(h.shape, np.uint8)
        cv2.circle(ring_mask, (int(cx), int(cy)), int(r * PARAMS["halo_r_out"]), 255, -1)
        cv2.circle(ring_mask, (int(cx), int(cy)), int(r * PARAMS["halo_r_in"]), 0, -1)
        ring = (ring_mask > 0) & (s >= PARAMS["halo_s_min"]) & (v >= PARAMS["halo_v_min"])

        votes = {}
        for name, bands in HALO_BANDS.items():
            m = np.zeros(h.shape, dtype=bool)
            for lo, hi in bands:
                m |= (h >= lo) & (h <= hi)
            votes[name] = int((m & ring).sum())

        total = votes["RED"] + votes["GREEN"]
        if total < PARAMS["halo_min_votes"]:
            continue
        winner = "RED" if votes["RED"] >= votes["GREEN"] else "GREEN"
        if votes[winner] / total < PARAMS["halo_dominance"]:
            continue
        # 고리 밀도: 진짜 LED는 핵 둘레가 색 번짐으로 꽉 찬다.
        # 우연히 근처에 색 픽셀이 조금 있는 가짜는 밀도가 낮다.
        ring_geo = np.pi * r * r * (PARAMS["halo_r_out"]**2 - PARAMS["halo_r_in"]**2)
        if votes[winner] / max(1.0, ring_geo) < PARAMS["halo_density_min"]:
            continue

        blob = {"center": (int(cx), int(cy)), "radius": int(r), "area": area,
                "circularity": circ, "fill_ratio": fill, "solidity": 1.0,
                "color": winner, "passed": True, "votes": votes}
        # 빨강 우선(안전) → 같은 색이면 큰 것
        cur_c, cur_b = best
        if cur_b is None:
            best = (winner, blob)
        elif winner == "RED" and cur_c == "GREEN":
            best = (winner, blob)
        elif winner == cur_c and area > cur_b["area"]:
            best = (winner, blob)
    return best

# ==================================================================
# [섹터 5] 단일 프레임 판정
# ==================================================================
def detect_traffic_light(frame, debug=False):
    """반환: (color, info). color는 "RED" / "GREEN" / None.

    2단 판정:
      1차 — 타버린 핵 + 색 고리 (detect_by_glow).
            켜진 LED의 하얀 중심을 찾으므로 커튼/벽 오탐이 원천 배제되고,
            과노출로 색이 밀린 상황(halo가 주황)에서도 정확하다.
      2차 — 핵이 없을 때만 기존 색 대역 방식 (LED가 안 타는 노출일 때 대비).
            단, 블롭 평균 밝기가 화면 최대 밝기의 75% 이상이어야 인정 —
            "화면에 더 밝은 것이 있는데 이 블롭이 발광체일 리 없다"는 방어막.
    """
    roi_h = int(frame.shape[0] * PARAMS["roi_top_ratio"])
    hsv = cv2.cvtColor(frame[:roi_h, :], cv2.COLOR_BGR2HSV)
    frame_area = frame.shape[0] * frame.shape[1]

    # ---- 1차: 핵 + 고리 ----
    glow_color, glow_blob = detect_by_glow(hsv, frame_area)
    if glow_color is not None:
        info = {"blob": glow_blob, "roi_h": roi_h, "mode": "glow"}
        if debug:
            info["all_blobs"] = [glow_blob]
        return glow_color, info

    # ---- 2차: 색 대역 (폴백) ----
    v_ch = hsv[:, :, 2]
    frame_v_max = float(np.percentile(v_ch, 99.9))

    best_color, best_blob = None, None
    all_blobs = []
    for color in ("RED", "GREEN"):
        mask = make_color_mask(hsv, color)
        for blob in find_light_blobs(mask, frame_area):
            blob["color"] = color
            blob["passed"] = is_light_shape(blob)
            # 커튼 방어: 블롭 위치의 평균 밝기가 화면 최대에 한참 못 미치면
            # 발광체가 아니다 (실측: 커튼 V=108 vs 화면최대 253 → 탈락)
            if blob["passed"]:
                cx, cy = blob["center"]; r = max(3, blob["radius"] // 2)
                y0, y1 = max(0, cy - r), min(v_ch.shape[0], cy + r)
                x0, x1 = max(0, cx - r), min(v_ch.shape[1], cx + r)
                patch_v = float(v_ch[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.0
                blob["patch_v"] = patch_v
                if patch_v < 0.75 * frame_v_max:
                    blob["passed"] = False
            all_blobs.append(blob)
            if blob["passed"]:
                # 빨강 우선(안전) → 같은 색이면 큰 것
                if best_blob is None:
                    best_color, best_blob = color, blob
                elif color == "RED" and best_color == "GREEN":
                    best_color, best_blob = color, blob
                elif color == best_color and blob["area"] > best_blob["area"]:
                    best_color, best_blob = color, blob

    info = {"blob": best_blob, "roi_h": roi_h, "mode": "band"}
    if debug:
        info["all_blobs"] = all_blobs
    return best_color, info


# ==================================================================
# [섹터 6] 프레임 다수결 → 주행 명령
# ==================================================================
class TrafficLightJudge(object):
    """최근 N프레임 다수결. 한 프레임짜리 오탐/미탐에 차가 반응하지 않게 한다."""

    def __init__(self):
        self.history = deque(maxlen=PARAMS["vote_window"])
        self.stable_color = None

    def update(self, color):
        self.history.append(color)
        counts = Counter(c for c in self.history if c is not None)

        if counts:
            top_color, top_n = counts.most_common(1)[0]
            if top_n >= PARAMS["vote_min_count"]:
                self.stable_color = top_color
            # 과반 미달이면 직전 판정 유지 (섣불리 바꾸지 않음)
        else:
            self.stable_color = None

        if self.stable_color == "RED":
            return "STOP"
        if self.stable_color == "GREEN":
            return "GO"
        return "NONE"

    def reset(self):
        self.history.clear()
        self.stable_color = None


# ==================================================================
# [섹터 7] 디버그 시각화
# ==================================================================
def draw_debug(frame, color, info):
    vis = frame.copy()   # 원본에 직접 그리면 잔상이 남는다 (start_lib.py의 전례)
    cv2.line(vis, (0, info["roi_h"]), (vis.shape[1], info["roi_h"]), (255, 255, 0), 2)

    for blob in info.get("all_blobs", []):
        bgr = (0, 0, 255) if blob["color"] == "RED" else (0, 255, 0)
        if blob["passed"]:
            cv2.circle(vis, blob["center"], blob["radius"], bgr, 4)
        else:
            cv2.circle(vis, blob["center"], blob["radius"], (128, 128, 128), 2)

    label = color if color is not None else "NO LIGHT"
    cv2.putText(vis, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 6)
    cv2.putText(vis, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                (0, 0, 255) if color == "RED" else (0, 200, 0), 3)
    return vis


# ==================================================================
# [섹터 8] 단독 실행 테스트 — 조명 변동 포함
# ==================================================================
TEST_SET = [
    ("trafficlight_1.jpg", "RED"),
    ("trafficlight_2.jpg", "GREEN"),
    ("steering_1.jpg", None),
    ("steering_9.jpg", None),
    ("steering_10.jpg", None),
    ("steering_parking_5.jpg", None),
    ("steering_parking_8.jpg", None),
]

LIGHT_FACTORS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8, 2.0]


def relight(img, factor):
    """조명/노출 변동 시뮬레이션."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def tl_main():
    """조명 배율 × 사진 전체 교차 검증."""
    print(f"{'배율':>5} | " + " ".join(f"{f.split('.')[0][:9]:>9}" for f, _ in TEST_SET))
    print("-" * 80)

    total, hits = 0, 0
    for fac in LIGHT_FACTORS:
        row = []
        for fname, expected in TEST_SET:
            img = cv2.imread(fname)
            if img is None:
                row.append(f"{'SKIP':>9}")
                continue
            color, _ = detect_traffic_light(relight(img, fac))
            ok = (color == expected)
            total += 1
            hits += ok
            row.append(f"{(str(color)[:5] + ('o' if ok else 'X')):>9}")
        print(f"{fac:>5} | " + " ".join(row))

    print(f"\n{hits}/{total} 통과  ({hits/total*100:.0f}%)")


if __name__ == "__main__":
    tl_main()