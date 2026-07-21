"""
stopline_lib.py — 정지선 검출 (BEV 기반)

신호등이 "빨간불이다"를 알려줘도, "정지선 앞이다"를 모르면 차는
신호등이 보이자마자 그 자리에 서버린다. 이 파일이 그 거리 판단을 맡는다.

[원리]
 BEV(하늘에서 내려다본 시점)로 펴면 정지선은 '가로로 곧게 뻗은 흰 띠'가 된다.
 각 행(row)에서 흰 픽셀이 폭의 몇 %를 차지하는지 세고(수평 투영),
 그 비율이 충분히 높은 행을 정지선으로 본다.

 정지선 vs 횡단보도 구분 (실측 검증됨):
   정지선   = 하나의 긴 가로선  → 한 행이 폭의 85%까지 참
   횡단보도 = 여러 세로 줄무늬  → 어느 행도 폭의 47%를 못 넘음
 "가로로 꽉 찬 행"이라는 기준 하나로 둘이 갈린다. 차선(세로선)도 같은 이유로 걸러진다.

[거리]
 BEV에서 아래쪽 = 차에 가까움. 정지선 행의 y좌표를 실제 거리(cm)로 바꾸려면
 BEV 캘리브레이션이 필요하다 (bev_bottom_cm, bev_top_cm). 트랙에서 실측해 넣을 것.

[입력]
 이 파일은 '이미 BEV로 변환된 이미지'를 받는다. 원본→BEV 변환은
 start_lib.py의 get_bird_eye_view()를 재사용한다 (lane 파이프라인과 동일한 BEV).
"""

import cv2
import numpy as np


# ==================================================================
# [섹터 1] 파라미터
# ==================================================================
PARAMS = {
    "white_v": 140,      # 흰색 판정: 밝기 하한
    "white_s": 90,       # 흰색 판정: 채도 상한 (흰색은 채도가 낮다)
    "fill_th": 0.55,     # 한 행이 정지선이려면 폭의 이 비율 이상이 흰색이어야
    "min_band": 3,       # 정지선 밴드의 최소 두께(행). 이보다 얇으면 잡음
    "gap_merge": 8,      # 이 간격 안의 흰 행들은 한 밴드로 묶음

    # --- 거리 캘리브레이션 (트랙에서 실측) ---
    "bev_bottom_cm": 30,   # BEV 맨 아래 행이 카메라 앞 몇 cm인지
    "bev_top_cm": 150,     # BEV 맨 위 행이 카메라 앞 몇 cm인지
    "stop_trigger_cm": 40, # 이 거리 안에 정지선이 들어오면 STOP 신호
}


# ==================================================================
# [섹터 2] 흰색 수평 투영
# ==================================================================
def white_row_fraction(bev):
    """각 행에서 흰 픽셀이 차지하는 비율(0~1) 배열을 반환."""
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)
    white = (v > PARAMS["white_v"]) & (s < PARAMS["white_s"])
    return white.sum(axis=1) / white.shape[1]


# ==================================================================
# [섹터 3] 정지선 검출
# ==================================================================
def detect_stopline(bev, debug=False):
    """반환: (y, info)
      y   : 정지선의 BEV 행 좌표. 없으면 None
      info: {"fraction": 행별 비율, "distance_cm": 추정 거리}
    """
    frac = white_row_fraction(bev)
    info = {"fraction": frac, "distance_cm": None, "band": None}

    hot = np.where(frac > PARAMS["fill_th"])[0]
    if len(hot) == 0:
        return None, info

    # 연속된 흰 행들을 밴드로 그룹핑 (gap_merge 이하 간격은 이어붙임)
    groups = np.split(hot, np.where(np.diff(hot) > PARAMS["gap_merge"])[0] + 1)

    # 가장 아래(=차에 가까운) 밴드를 정지선으로 채택
    bottom = max(groups, key=lambda g: g.max())
    if len(bottom) < PARAMS["min_band"]:
        return None, info

    y = int(bottom.mean())
    info["band"] = (int(bottom.min()), int(bottom.max()))
    info["distance_cm"] = row_to_distance(y, bev.shape[0])
    return y, info


def row_to_distance(y, bev_height):
    """BEV 행 좌표 → 카메라 앞 거리(cm). 아래(y 큼)일수록 가깝다."""
    bottom, top = PARAMS["bev_bottom_cm"], PARAMS["bev_top_cm"]
    ratio = y / max(1, bev_height - 1)      # 0=맨 위(멀다), 1=맨 아래(가깝다)
    return top + (bottom - top) * ratio


# ==================================================================
# [섹터 4] 정지 판단
# ==================================================================
def should_stop(bev):
    """정지선이 정지 트리거 거리 안에 있으면 True.

    신호등 판정(빨간불)과 AND로 묶어서 쓴다:
        if tl_cmd == "STOP" and should_stop(bev): 실제로 멈춤
    빨간불이어도 정지선이 아직 멀면 계속 접근한다.
    """
    y, info = detect_stopline(bev)
    if y is None:
        return False, info
    return info["distance_cm"] <= PARAMS["stop_trigger_cm"], info


# ==================================================================
# [섹터 5] 디버그 시각화
# ==================================================================
def draw_debug(bev, y, info):
    vis = bev.copy()
    h, w = vis.shape[:2]

    # 우측에 행별 흰색 비율 그래프
    frac = info["fraction"]
    for r in range(h):
        length = int(frac[r] * w * 0.3)
        cv2.line(vis, (w - length, r), (w, r), (80, 80, 200), 1)
    # fill_th 기준선
    tx = int(w - PARAMS["fill_th"] * w * 0.3)
    cv2.line(vis, (tx, 0), (tx, h), (0, 165, 255), 1)

    if y is not None:
        b0, b1 = info["band"]
        cv2.rectangle(vis, (0, b0), (w, b1), (0, 255, 0), 2)
        d = info["distance_cm"]
        cv2.putText(vis, f"STOPLINE {d:.0f}cm", (10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(vis, "no stopline", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return vis


# ==================================================================
# [섹터 6] 단독 테스트 — 합성 BEV로 알고리즘 검증
# ==================================================================
def _synth(kind, W=400, H=500):
    img = np.full((H, W, 3), 40, np.uint8)
    if kind in ("stopline", "both"):
        cv2.rectangle(img, (30, 300), (W - 30, 320), (240, 240, 240), -1)
    if kind == "crosswalk":
        for x in range(40, W - 40, 55):
            cv2.rectangle(img, (x, 200), (x + 30, 400), (240, 240, 240), -1)
    if kind == "both":
        for x in range(40, W - 40, 55):
            cv2.rectangle(img, (x, 150), (x + 30, 280), (240, 240, 240), -1)
    if kind == "lane":
        cv2.rectangle(img, (60, 0), (85, H), (240, 240, 240), -1)
        cv2.rectangle(img, (W - 85, 0), (W - 60, H), (240, 240, 240), -1)
    return img


def sl_main():
    expect = {"stopline": True, "crosswalk": False, "both": True, "lane": False}
    for kind, exp in expect.items():
        bev = _synth(kind)
        y, info = detect_stopline(bev)
        got = y is not None
        ok = "PASS" if got == exp else "FAIL"
        d = f"{info['distance_cm']:.0f}cm" if y is not None else "-"
        print(f"[{ok}] {kind:10s}: 검출={got!s:5s} 거리={d:8s} "
              f"(기대 {exp})")
        cv2.imwrite(f"debug_stopline_{kind}.jpg", draw_debug(bev, y, info))


if __name__ == "__main__":
    sl_main()
