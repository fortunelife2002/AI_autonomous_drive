"""
-------------------------------------------------------------------
  FILE NAME: obstacle_avoidance_v2.py  (캘리브레이션 주석판)
  PURPOSE  : 차선 위 정차 차량 회피 주행 (Follow-the-Gap 기반)
  AUTHOR   : Dohyung
-------------------------------------------------------------------
  [v1 대비 변경점]
   1) 타이머 기반 4-상태 FSM  ->  Follow-the-Gap 연속 제어 (타이머 제거)
   2) 섹터 min()             ->  직교변환 + 코리도 필터 + 클러스터링 + N-of-M 지속성
   3) bang-bang 조향         ->  PD 제어 (헤딩 오차 기반)
   4) Hough 차선             ->  하단 ROI 컬럼 히스토그램 (연속 오차값)
   5) 상태 전이 히스테리시스 추가

  [참고 기법]
   - Follow-the-Gap Method : Sezer & Gokasan (2012)
   - VFH                   : Borenstein & Koren (1991)
   - Pure Pursuit 계열 횡제어 : DARPA Grand Challenge

  [주석 표기]
   [실측] 자·각도기·상자로 물리적으로 재야 하는 값
   [튜닝] 트랙에서 반복 주행하며 조정하는 값
   [계산] 다른 실측값으로부터 유도되는 값

  [캘리브레이션 순서]  앞 단계가 틀리면 뒷 단계가 전부 무의미합니다.
   1. 포트 / BAUDRATE
   2. LIDAR_ZERO_DEG, LIDAR_CW          <- 최우선
   3. STEER_CENTER, STEER_LIMIT          (차를 들어올린 채)
   4. VEHICLE_HALF_WIDTH, MIN_RANGE
   5. CORRIDOR_HALF_WIDTH                (정지 상태에서 기둥 배제 확인)
   6. V_THRESHOLD, S_THRESHOLD           (저장한 프레임으로)
   7. SPEED_MIN/MAX, EMERGENCY_DIST      (직선 주행)
   8. KP_HEADING, KD_HEADING             (직선 -> 곡선 순서로)
   9. AVOID_ENTER_DIST                   (마지막, 실제 차 놓고)
-------------------------------------------------------------------
"""

import time
import platform
from collections import deque

import cv2
import numpy as np

import Function_Library as fl


# ==================== 하드웨어 ====================
IS_WINDOWS = (platform.system() == "Windows")

# [실측] 포트 이름
#   측정: Windows -> 장치관리자 > 포트(COM & LPT) 에서 확인
#         Ubuntu  -> 라이다 뽑고 `ls /dev/tty*` , 다시 꽂고 `ls /dev/tty*` 해서
#                    새로 생긴 이름이 그것. 보통 라이다=ttyUSB0, 아두이노=ttyACM0
#   수정: 아래 문자열을 확인한 이름으로 교체
#   주의: Ubuntu 는 `sudo usermod -aG dialout $USER` 후 재로그인 해야 권한이 생김
LIDAR_PORT = "COM4" if IS_WINDOWS else "/dev/ttyUSB0"
ARDUINO_PORT = "COM3" if IS_WINDOWS else "/dev/ttyACM0"

# [실측] 아두이노 Serial.begin() 값과 반드시 동일해야 함
#   측정: Vehicle_Control.ino 의 setup() 안 Serial.begin(9600) 확인
#   수정: 아두이노 쪽을 115200 으로 올렸다면 여기도 115200 으로
BAUDRATE = 9600

# ==================== 라이다 장착 보정 ====================
# ※ 이 두 개가 틀리면 아래 모든 값이 무의미해집니다. 가장 먼저 맞추세요.

# [실측] 라이다 raw 각도 중 '차량 정면'에 해당하는 값
#   측정: 차를 세우고 정면 1m 지점에만 상자를 하나 놓는다.
#         LiDAR_Exercise_2_A1.py 처럼 scan 을 그대로 print 해서
#         거리 약 1000mm 가 찍히는 raw 각도를 읽는다.
#   수정: 그 각도를 그대로 적는다. (예: 정면 상자가 182도에 찍혔으면 182.0)
LIDAR_ZERO_DEG = 0.0

# [실측] raw 각도의 증가 방향
#   측정: 위 상자를 차량 '왼쪽' 1m 로 옮긴다.
#         LIDAR_CW = True 로 두고 아래 진단 코드를 돌린다.
#             pts = ScanProcessor.to_cartesian(scan); print(pts[:, 3])
#         y 값(4번째 열)이 양수(+)로 나오면 True 가 맞음.
#   수정: y 가 음수로 나오면 False 로 바꾼다.
LIDAR_CW = True

# ==================== 차량 제원 (mm) ====================
# [실측] 차폭의 절반
#   측정: 자로 차량의 '가장 넓은 부분'(보통 앞바퀴 바깥~바깥)을 재고 2로 나눈다.
#         범퍼, 센서 브래킷이 더 튀어나왔다면 그걸 기준으로.
#   수정: 예) 실측 차폭 270mm -> 135.0 으로 수정
VEHICLE_HALF_WIDTH = 130.0

# [튜닝] 차폭에 더할 안전 여유
#   측정: 트랙 차선폭에서 차폭을 뺀 값의 절반이 '이론상 최대 여유'.
#         예) 차선폭 800mm, 차폭 260mm -> 여유 270mm 까지 가능
#   수정: 처음엔 100mm 로 크게 잡고, 회피가 너무 소심하면 60mm 까지 줄인다.
#         너무 작으면 스치고, 너무 크면 통과 가능한 갭도 막혀버림.
SAFETY_MARGIN = 90.0

BUBBLE_RADIUS = VEHICLE_HALF_WIDTH + SAFETY_MARGIN   # 자동 계산 (수정 불필요)

# ==================== 코리도 (mm) ====================
# 라이다는 평면 트랙(천)을 못 봄 -> 기둥/벤치 배제를 위해 반드시 필요

# [실측] 관심 영역의 좌우 반폭
#   측정: 줄자로 트랙 '양쪽 차선 바깥 흰선 사이' 폭을 잰다. (편도 2차로면 전체 폭)
#   수정: 측정한 전체 폭의 절반 + 100mm 여유.
#         예) 전체 폭 1000mm -> 500 + 100 = 600.0
#   검증: 값을 넣고 정지 상태에서 실행. 트랙 옆 기둥/벤치가 클러스터로
#         잡히면 너무 넓은 것이니 50씩 줄인다. 반대로 비스듬히 선 차를
#         놓치면 50씩 늘린다.
CORRIDOR_HALF_WIDTH = 620.0

# [튜닝] 전방 관심 거리
#   측정: SPEED_MAX 로 달리다 정지 명령 후 실제로 멈출 때까지의 거리(제동거리)를
#         3회 재서 평균낸다. 그 값의 3배 정도가 적당.
#   수정: 제동거리 500mm 였다면 1500~2000 사이. 너무 크면 저 멀리 벽까지
#         반응해 느려지고, 너무 작으면 늦게 반응해 못 피함.
LOOKAHEAD = 2200.0

# [실측] 이보다 가까운 점은 자기 차체/노이즈로 간주
#   측정: 라이다 회전 중심에서 차체 앞·옆 부품 중 '가장 먼 지점'까지의 거리를 잰다.
#   수정: 그 값 + 30mm. 실행 시 아무것도 없는데 계속 물체가 잡히면
#         차체를 보고 있는 것이므로 이 값을 올린다.
MIN_RANGE = 120.0

# ==================== 클러스터링 ====================
# [계산] 인접한 두 점을 같은 물체로 묶는 거리 임계
#   측정: 라이다 각분해능을 확인한다. (RPLidar A1 ≈ 1도)
#         LOOKAHEAD 거리에서 이웃 점 사이 간격 = LOOKAHEAD × tan(1도) ≈ 38mm
#   수정: 위 간격의 3배 정도. 2200mm 기준이면 115~130.
#         너무 작으면 차 한 대가 여러 조각으로 쪼개지고,
#         너무 크면 차 두 대가 하나로 붙어버린다.
#   검증: 차 2대를 나란히 놓고 클러스터가 2개로 잡히는지 print 로 확인.
CLUSTER_TOL = 130.0

# [튜닝] 클러스터로 인정할 최소 점 개수
#   측정: LOOKAHEAD 거리에 실제 차를 놓고 몇 점이 찍히는지 센다.
#         (차폭 200mm / 38mm ≈ 5점)
#   수정: 그 값의 절반 정도. 올리면 노이즈에 강해지지만 먼 물체를 놓친다.
CLUSTER_MIN_PTS = 3

# [실측] 물체로 인정할 최소 폭
#   측정: 대회에 놓일 정차 차량의 '가장 좁게 보이는 면'(보통 정면 폭)을 잰다.
#   수정: 그 값의 1/3 정도. 예) 차 폭 180mm -> 60.0
#         라바콘 같은 얇은 물체도 피해야 한다면 더 낮춘다.
CLUSTER_MIN_WIDTH = 60.0

# ==================== 지속성 필터 (N-of-M) ====================
# [계산] 최근 M 프레임 중 N 번 이상 보여야 진짜 장애물로 인정
#   측정: 라이다 스캔 주기를 잰다. 루프 안에서
#             t=time.time() ... print(1.0/(time.time()-t))
#         RPLidar A1 은 보통 5~10Hz.
#   수정: 10Hz 이면 5프레임 = 0.5초 지연. 이게 너무 느리면 WINDOW=3, HITS=2 로.
#         반대로 오검출이 잦으면 WINDOW=7, HITS=4 로 올린다.
#   주의: HITS 를 올릴수록 반응이 느려짐 -> LOOKAHEAD 도 같이 늘려야 함
PERSIST_WINDOW = 5
PERSIST_HITS = 3

# ==================== 히스테리시스 (mm) ====================
# [튜닝] 회피를 시작할 거리
#   측정: SPEED_MAX 로 달리며 STEER_LIMIT 까지 꺾었을 때, 차가 차선 하나
#         (약 400mm) 옆으로 이동하는 데 걸린 주행 거리를 잰다. = 최소 회피거리
#   수정: 그 값의 2배. 예) 최소 회피거리 450mm -> 900~1000
#   검증: 너무 작으면 아슬아슬하게 스치고, 너무 크면 멀리서부터 계속 피해 다님
AVOID_ENTER_DIST = 950.0

# [계산] 회피를 해제할 거리 (채터링 방지)
#   수정: AVOID_ENTER_DIST × 1.4 로 둔다. 반드시 ENTER 보다 커야 함.
#         두 값이 같으면 경계에서 CRUISE/AVOID 가 초당 수십 번 뒤집힌다.
AVOID_EXIT_DIST = 1350.0

# [실측] 비상 정지 거리
#   측정: SPEED_MIN 으로 달리다 T0 명령을 보내고 멈출 때까지 거리를 3회 잰다.
#   수정: 평균 제동거리 + 100mm 여유. 예) 제동거리 220mm -> 320.0
EMERGENCY_DIST = 320.0

# ==================== 제어 ====================
# [실측] 직진에 해당하는 조향 명령값
#   측정: 차를 들어올린 채 S85, S90, S95 ... 를 하나씩 보내면서
#         앞바퀴가 정확히 정면을 보는 값을 찾는다.
#   수정: 그 값으로 교체. (기구부 조립 오차 때문에 90 이 아닐 수 있음)
STEER_CENTER = 90

# [실측] 중앙 기준 최대 조향각
#   측정: 앞바퀴를 좌/우 끝까지 물리적으로 돌려보고, 기구부가 걸리지 않는
#         안전한 한계 각도를 각도기로 잰다.
#   수정: 실측값보다 5도 작게. 끝까지 밀면 조향 모터가 스톨(과열)된다.
#   ★ 중요: Vehicle_Control.ino 의 STEER_MIN/STEER_MAX 와 반드시 일치시킬 것
#           여기가 35 이면 .ino 는 STEER_MIN=55, STEER_MAX=125 (=90±35)
STEER_LIMIT = 35

# [계산] FGM 이 제안할 수 있는 최대 목표 헤딩
#   수정: STEER_LIMIT × 1.4 정도. 조향 한계보다 크게 두어야 PD 가
#         포화 구간에서도 방향을 잃지 않는다.
MAX_HEADING_DEG = 50.0

# [튜닝] PD 게인 — 아래 순서를 지켜서 잡을 것
#   1) KD_HEADING = 0 으로 두고 KP 를 0.3 부터 0.1 씩 올린다.
#   2) 차가 좌우로 규칙적으로 흔들리기 시작하는 KP 를 찾는다. (= 임계 게인)
#   3) KP = 임계 게인 × 0.6 으로 낮춘다.
#   4) 남은 흔들림이 있으면 KD 를 KP 의 1/4 부터 올린다.
#   증상별: 반응이 굼뜸 -> KP↑ / 좌우 진동 -> KP↓ 또는 KD↑ / 덜컥거림 -> KD↓
KP_HEADING = 0.75
KD_HEADING = 0.18

# [실측] 차선 중심 오차(픽셀) -> 조향각(도) 변환 계수
#   측정: 차를 차선 중앙에서 '오른쪽으로 100mm' 옮겨 세우고 실행한다.
#         출력 로그의 lane 값(px 기반 각도)이 아니라, LaneEstimator 가 준
#         raw offset px 를 print 해서 읽는다. (예: -120px)
#   수정: KP_LANE = (그 상황에서 원하는 조향각) / (읽은 px 절대값)
#         예) 120px 일 때 8도 꺾고 싶다 -> 8 / 120 ≈ 0.067
KP_LANE = 0.055

# [실측] 속도 명령 (아두이노 analogWrite PWM, 0~255)
#   측정: 평지에서 T60, T80, T100 을 각각 보내고 2m 주파 시간을 재서
#         실제 속도(m/s)를 구한다. 대회 규정 속도 제한도 확인할 것.
#   수정: SPEED_MAX 는 직선에서 안정적으로 제어되는 최대값.
#         SPEED_MIN 은 '바퀴가 실제로 굴러가기 시작하는 최소 PWM + 10'.
#         (모터는 PWM 이 너무 낮으면 아예 안 돈다 -> 그 값을 먼저 찾을 것)
SPEED_MAX = 90
SPEED_MIN = 45
SPEED_STOP = 0

# [계산] 아두이노로 명령을 보내는 주기
#   측정: 아두이노 loop() 가 도는 주기와 시리얼 버퍼 여유를 본다.
#   수정: 0.05 = 20Hz. 아두이노 쪽 CMD_TIMEOUT(500ms) 보다 훨씬 짧아야 하고,
#         라이다 스캔 주기보다도 짧거나 같아야 한다.
CMD_PERIOD = 0.05


# ===================================================================
#  LiDAR 전처리 : 극좌표 -> 차체 직교좌표
# ===================================================================
class ScanProcessor(object):
    """
    출력 좌표계 (차체 기준)
        x : 전방 (+)
        y : 좌측 (+)
        theta : 정면 0, 좌측 +, 우측 -  [rad]
    """

    @staticmethod
    def to_cartesian(scan):
        if scan is None or len(scan) == 0:
            return np.empty((0, 4))

        data = np.asarray(scan, dtype=float)
        raw_ang, dist = data[:, 0], data[:, 1]

        # 유효 거리만
        valid = (dist > MIN_RANGE) & (dist < 12000.0)
        raw_ang, dist = raw_ang[valid], dist[valid]
        if len(dist) == 0:
            return np.empty((0, 4))

        # raw 각도 -> 차체 각도(정면 0, 좌 +)
        rel = raw_ang - LIDAR_ZERO_DEG
        if LIDAR_CW:
            rel = -rel
        rel = (rel + 180.0) % 360.0 - 180.0      # -180 ~ +180
        theta = np.deg2rad(rel)

        x = dist * np.cos(theta)
        y = dist * np.sin(theta)

        # [theta, dist, x, y] 를 theta 오름차순으로
        out = np.column_stack([theta, dist, x, y])
        return out[np.argsort(out[:, 0])]

    @staticmethod
    def corridor_filter(pts):
        """주행 코리도 안의 점만 남긴다 (기둥/벤치/화분 배제)"""
        if len(pts) == 0:
            return pts
        cond = (pts[:, 2] > 0.0) & (pts[:, 2] < LOOKAHEAD) & \
               (np.abs(pts[:, 3]) < CORRIDOR_HALF_WIDTH)
        return pts[cond]


# ===================================================================
#  클러스터링 : 인접 점 연결 (라이다 스캔은 각도 정렬이므로 1D 인접으로 충분)
# ===================================================================
class Cluster(object):
    def __init__(self, pts):
        self.pts = pts
        self.n = len(pts)
        self.min_dist = float(np.min(pts[:, 1]))
        self.cx = float(np.mean(pts[:, 2]))
        self.cy = float(np.mean(pts[:, 3]))
        self.width = float(np.hypot(pts[0, 2] - pts[-1, 2],
                                    pts[0, 3] - pts[-1, 3]))
        self.theta_min = float(np.min(pts[:, 0]))
        self.theta_max = float(np.max(pts[:, 0]))

    def __repr__(self):
        return "Cluster(n=%d, d=%.0f, x=%.0f, y=%.0f, w=%.0f)" % (
            self.n, self.min_dist, self.cx, self.cy, self.width)


def cluster_points(pts):
    """연속한 점들 사이 유클리드 거리가 CLUSTER_TOL 이내면 같은 물체로 묶음"""
    if len(pts) < CLUSTER_MIN_PTS:
        return []

    clusters, start = [], 0
    for i in range(1, len(pts)):
        gap = np.hypot(pts[i, 2] - pts[i - 1, 2], pts[i, 3] - pts[i - 1, 3])
        if gap > CLUSTER_TOL:
            if i - start >= CLUSTER_MIN_PTS:
                clusters.append(Cluster(pts[start:i]))
            start = i

    if len(pts) - start >= CLUSTER_MIN_PTS:
        clusters.append(Cluster(pts[start:]))

    return [c for c in clusters if c.width >= CLUSTER_MIN_WIDTH]


# ===================================================================
#  Follow-the-Gap
# ===================================================================
class FollowTheGap(object):
    """
    Sezer & Gokasan (2012) 방식:
      1) 최근접 점 주변을 안전 버블로 0 처리
      2) 남은 구간 중 가장 넓은 연속 갭 탐색
      3) 갭의 대표 방향을 목표 헤딩으로 반환
    """

    # [계산] 탐색 각도 범위와 해상도
    #   측정: 라이다 각분해능(A1 ≈ 1도)을 확인.
    #   수정: N_BINS = (ANG_MAX - ANG_MIN) / 각분해능 + 1
    #         범위는 MAX_HEADING_DEG 보다 넉넉히 넓게 (여기선 ±60 > ±50).
    #         너무 넓히면 옆 차선 밖까지 갭으로 잡아 코스를 이탈한다.
    N_BINS = 121
    ANG_MIN, ANG_MAX = -60.0, 60.0

    def __init__(self):
        self.bin_deg = np.linspace(self.ANG_MIN, self.ANG_MAX, self.N_BINS)

    def _build_histogram(self, pts):
        """각 각도 bin 의 자유 거리 (장애물 없으면 LOOKAHEAD)"""
        hist = np.full(self.N_BINS, LOOKAHEAD, dtype=float)
        if len(pts) == 0:
            return hist

        deg = np.rad2deg(pts[:, 0])
        inside = (deg >= self.ANG_MIN) & (deg <= self.ANG_MAX)
        deg, dist = deg[inside], pts[inside, 1]

        idx = np.clip(
            np.round((deg - self.ANG_MIN) /
                     (self.ANG_MAX - self.ANG_MIN) * (self.N_BINS - 1)),
            0, self.N_BINS - 1).astype(int)

        # 같은 bin 에 여러 점이면 가장 가까운 것 채택
        for i, d in zip(idx, dist):
            if d < hist[i]:
                hist[i] = d
        return hist

    def _apply_bubble(self, hist):
        """최근접 점 주변을 물리적 차폭만큼 막음"""
        near_idx = int(np.argmin(hist))
        near_dist = hist[near_idx]
        if near_dist >= LOOKAHEAD:
            return hist

        # 거리 near_dist 에서 BUBBLE_RADIUS 를 각도로 환산
        half_deg = np.rad2deg(np.arctan2(BUBBLE_RADIUS, max(near_dist, 1.0)))
        half_bins = int(np.ceil(half_deg))

        lo = max(0, near_idx - half_bins)
        hi = min(self.N_BINS, near_idx + half_bins + 1)
        hist[lo:hi] = 0.0
        return hist

    def _largest_gap(self, hist, threshold):
        """자유거리가 threshold 이상인 최장 연속 구간 [lo, hi)"""
        free = hist >= threshold
        best_lo = best_hi = -1
        best_len = 0
        lo = None

        for i in range(self.N_BINS):
            if free[i] and lo is None:
                lo = i
            elif not free[i] and lo is not None:
                if i - lo > best_len:
                    best_lo, best_hi, best_len = lo, i, i - lo
                lo = None

        if lo is not None and self.N_BINS - lo > best_len:
            best_lo, best_hi, best_len = lo, self.N_BINS, self.N_BINS - lo

        return best_lo, best_hi, best_len

    def compute(self, pts, bias_deg=0.0):
        """
        bias_deg : 차선 추종이 원하는 방향. 갭이 여러 개일 때 이쪽에 가까운 걸 선호.
        return   : (목표 헤딩 deg, 그 방향의 자유거리)
        """
        hist = self._build_histogram(pts)
        hist = self._apply_bubble(hist.copy())

        lo, hi, length = self._largest_gap(hist, threshold=AVOID_ENTER_DIST)

        if length <= 0:
            # 통과 가능한 갭이 없음 -> 가장 여유 있는 방향이라도 반환
            best = int(np.argmax(hist))
            return float(self.bin_deg[best]), float(hist[best])

        # 갭 안에서 bias 에 가장 가까운 방향 선택 (VFH 의 goal-directed 선택)
        seg = self.bin_deg[lo:hi]
        target = seg[int(np.argmin(np.abs(seg - bias_deg)))]
        clearance = float(np.max(hist[lo:hi]))

        return float(np.clip(target, -MAX_HEADING_DEG, MAX_HEADING_DEG)), clearance


# ===================================================================
#  차선 추정 : 하단 ROI 컬럼 히스토그램
# ===================================================================
class LaneEstimator(object):
    """
    흰 테이프 / 어두운 천 트랙에 맞춘 간단 BEV-free 추정기.
    Hough 대비 장점: 차선 중심 오차를 '연속값(px)'으로 준다.
    """

    # [실측] 화면에서 사용할 하단 영역 비율
    #   측정: 카메라를 차에 장착한 상태로 한 프레임을 저장해서 열어본다.
    #         지평선(트랙 끝나고 기둥/창문이 시작되는 높이)의 y 픽셀을 읽는다.
    #   수정: ROI_TOP_RATIO = (지평선 y) / (전체 높이). 지평선보다 아래만 봐야
    #         창틀·기둥의 흰색 세로선을 차선으로 오인하지 않는다.
    ROI_TOP_RATIO = 0.62

    # [실측] 흰색 차선 판정 임계 (HSV)
    #   측정: 실제 트랙에서 프레임을 저장하고 아래를 돌려 눈으로 확인한다.
    #             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #             _, s, v = cv2.split(hsv)
    #             mask = ((v > V_THRESHOLD) & (s < S_THRESHOLD)).astype('uint8')*255
    #             cv2.imshow('mask', mask); cv2.waitKey(0)
    #   수정: 차선만 하얗게 남을 때까지 조정.
    #         차선이 끊겨 보이면 V_THRESHOLD 를 낮춘다(예 165 -> 140).
    #         바닥 반사광까지 잡히면 V_THRESHOLD 를 올린다.
    #         초록 잔디가 섞이면 S_THRESHOLD 를 낮춘다(예 90 -> 60).
    #   ★ 이 트랙은 실내등 + 창문 자연광이 섞여 시간대별로 달라짐.
    #     대회 당일 아침/오후에 각각 다시 확인할 것.
    V_THRESHOLD = 165
    S_THRESHOLD = 90

    def __init__(self):
        self.last_offset = 0.0

    def estimate(self, frame):
        """return (center_offset_px, confidence 0~1)"""
        if frame is None:
            return self.last_offset, 0.0

        h, w = frame.shape[:2]
        roi = frame[int(h * self.ROI_TOP_RATIO):, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)
        mask = ((v > self.V_THRESHOLD) & (s < self.S_THRESHOLD)).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        hist = mask.sum(axis=0).astype(float)
        if hist.sum() < 50:
            return self.last_offset, 0.0

        mid = w // 2
        left_hist, right_hist = hist[:mid], hist[mid:]

        left_ok = left_hist.max() > 8
        right_ok = right_hist.max() > 8

        if left_ok and right_ok:
            lx = int(np.argmax(left_hist))
            rx = int(np.argmax(right_hist)) + mid
            center = (lx + rx) / 2.0
            conf = 1.0
        elif left_ok:
            lx = int(np.argmax(left_hist))
            center = lx + w * 0.25          # 차선폭 가정으로 보간
            conf = 0.5
        elif right_ok:
            rx = int(np.argmax(right_hist)) + mid
            center = rx - w * 0.25
            conf = 0.5
        else:
            return self.last_offset, 0.0

        offset = center - mid              # + 면 차선이 오른쪽 = 좌로 치우침
        self.last_offset = offset
        return offset, conf


# ===================================================================
#  아두이노 인터페이스
# ===================================================================
class Driver(object):
    def __init__(self, port, baudrate):
        self.ser = fl.libARDUINO().init(port, baudrate)
        self.last_send = 0.0

    def send(self, steer_deg, throttle):
        now = time.time()
        if now - self.last_send < CMD_PERIOD:
            return None
        self.last_send = now

        steer = int(np.clip(STEER_CENTER + steer_deg,
                            STEER_CENTER - STEER_LIMIT,
                            STEER_CENTER + STEER_LIMIT))
        thr = int(np.clip(throttle, 0, 255))
        self.ser.write(("S%d,T%d\n" % (steer, thr)).encode())
        return steer, thr

    def stop(self):
        try:
            self.ser.write(("S%d,T0\n" % STEER_CENTER).encode())
            time.sleep(0.1)
        finally:
            self.ser.close()


# ===================================================================
#  메인
# ===================================================================
CRUISE, AVOID = 0, 1
STATE_NAME = ("CRUISE", "AVOID")


class ObstacleAvoiderV2(object):
    def __init__(self):
        self.cam = fl.libCAMERA()
        self.ch0, _ = self.cam.initial_setting(capnum=1)

        self.lidar = fl.libLIDAR(LIDAR_PORT)
        self.lidar.init()

        self.driver = Driver(ARDUINO_PORT, BAUDRATE)
        self.fgm = FollowTheGap()
        self.lane = LaneEstimator()

        self.hits = deque(maxlen=PERSIST_WINDOW)
        self.state = CRUISE
        self.prev_err = 0.0
        self.prev_t = time.time()
        self.passed = 0
        self.was_avoiding = False

    # ---------- 지속성 필터 ----------
    def persistent_obstacle(self, clusters):
        """N-of-M 검증을 통과한 '진짜' 장애물의 최근접 거리"""
        threshold = AVOID_EXIT_DIST if self.state == AVOID else AVOID_ENTER_DIST
        near = [c for c in clusters if c.min_dist < threshold]
        self.hits.append(len(near) > 0)

        if sum(self.hits) < PERSIST_HITS:
            return None
        return min(near, key=lambda c: c.min_dist) if near else None

    # ---------- PD 조향 ----------
    def steer_pd(self, heading_err_deg):
        now = time.time()
        dt = max(now - self.prev_t, 1e-3)
        d_err = (heading_err_deg - self.prev_err) / dt

        out = KP_HEADING * heading_err_deg + KD_HEADING * d_err

        self.prev_err = heading_err_deg
        self.prev_t = now
        return float(np.clip(out, -STEER_LIMIT, STEER_LIMIT))

    # ---------- 메인 루프 ----------
    def run(self):
        try:
            for scan in self.lidar.scanning():
                pts_all = ScanProcessor.to_cartesian(scan)
                pts = ScanProcessor.corridor_filter(pts_all)
                clusters = cluster_points(pts)

                frame = None
                res = self.cam.camera_read(self.ch0)
                if res and res[0]:
                    frame = res[1]

                lane_offset, lane_conf = self.lane.estimate(frame)
                lane_bias_deg = float(np.clip(-lane_offset * KP_LANE,
                                              -25.0, 25.0))

                obstacle = self.persistent_obstacle(clusters)

                # ---------- 상태 갱신 (히스테리시스) ----------
                if obstacle is not None:
                    self.state = AVOID
                    self.was_avoiding = True
                else:
                    if self.was_avoiding and self.state == AVOID:
                        self.passed += 1
                        print("[INFO] 장애물 %d대 통과" % self.passed)
                        self.was_avoiding = False
                    self.state = CRUISE

                # ---------- 목표 헤딩 결정 ----------
                if self.state == AVOID:
                    target_deg, clearance = self.fgm.compute(pts, bias_deg=lane_bias_deg)
                else:
                    target_deg = lane_bias_deg if lane_conf > 0.0 else 0.0
                    clearance = LOOKAHEAD

                steer = self.steer_pd(target_deg)

                # ---------- 속도 : 여유거리·조향량에 비례 ----------
                clear_ratio = np.clip(clearance / LOOKAHEAD, 0.0, 1.0)
                steer_ratio = 1.0 - abs(steer) / STEER_LIMIT
                speed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * \
                    min(clear_ratio, steer_ratio)

                nearest = min([c.min_dist for c in clusters]) if clusters else float('inf')
                if nearest < EMERGENCY_DIST:
                    speed = SPEED_STOP
                    print("[SAFETY] 전방 %.0fmm 비상정지" % nearest)

                self.driver.send(steer, speed)

                print("%-7s | obj:%d near:%6.0f | lane:%+6.1f conf:%.1f "
                      "| tgt:%+5.1f steer:%+5.1f spd:%3d"
                      % (STATE_NAME[self.state], len(clusters), nearest,
                         lane_bias_deg, lane_conf, target_deg, steer, speed))

                if self.cam.loop_break():
                    break

        except KeyboardInterrupt:
            print("\n[EXIT] 사용자 중단")
        finally:
            self.shutdown()

    def shutdown(self):
        self.driver.stop()
        self.lidar.stop()
        if self.ch0 is not None:
            self.ch0.release()
        cv2.destroyAllWindows()
        print("[EXIT] 정상 종료 (통과 %d대)" % self.passed)


if __name__ == "__main__":
    ObstacleAvoiderV2().run()
