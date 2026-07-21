"""
mission_trafficlight.py — 신호등 미션 통합 골격

신호등 색 판정(trafficlight_lib) + 정지선 거리(stopline_lib)를 결합해
실제 주행 명령을 만든다. 카메라 루프와 Arduino 전송부는 팀 통합 시 채운다.

[핵심 로직]
 "빨간불" 하나만으로는 멈추지 않는다. 두 조건을 AND로 묶는다:
     빨간불이다  AND  정지선이 코앞이다   → 정지
     빨간불이다  AND  정지선이 아직 멀다   → 계속 접근
     초록불이다                          → 출발/주행
 이래야 신호등이 멀리 보일 때 그 자리에 서버리는 사고를 막는다.
"""

import cv2
import sys
sys.path.insert(0, "/mnt/project")   # start_lib.py 위치 (실차에선 경로 조정)

from start_lib import get_bird_eye_view
import trafficlight_lib as tl
import stopline_lib as sl


# 트랙에서 start_lib.py로 캘리브레이션한 BEV 4점 (좌하,우하,우상,좌상)
# 여기 값은 예시. 실측으로 교체할 것.
BEV_SRC_POINTS = [(80, 780), (760, 720), (620, 600), (120, 640)]
BEV_SIZE = (400, 500)


def process_frame(frame, judge):
    """한 프레임 → (drive_command, 디버깅용 상태)

    drive_command: "STOP" / "GO" / "CRUISE"
      STOP   : 빨간불 + 정지선 코앞 → 멈춰라
      GO     : 초록불 → 출발/주행
      CRUISE : 그 외 → 차선 주행 로직에 맡겨라 (신호등 개입 없음)
    """
    # 1) 신호등 색 (원본 프레임에서)
    color, _ = tl.detect_traffic_light(frame)
    tl_cmd = judge.update(color)          # 다수결 스무딩된 "STOP"/"GO"/"NONE"

    # 2) 정지선 거리 (BEV에서)
    bev = get_bird_eye_view(frame, BEV_SIZE, BEV_SRC_POINTS)
    near_stopline, sl_info = sl.should_stop(bev)

    # 3) 결합
    if tl_cmd == "STOP" and near_stopline:
        drive = "STOP"
    elif tl_cmd == "GO":
        drive = "GO"
    else:
        drive = "CRUISE"                  # 빨간불이어도 정지선 멀면 계속 접근

    state = {
        "color": color,
        "tl_cmd": tl_cmd,
        "stopline_cm": sl_info["distance_cm"],
        "near": near_stopline,
    }
    return drive, state


def main():
    """실차 메인 루프 골격. 카메라/Arduino 부분은 통합 시 연결."""
    cap = cv2.VideoCapture(1)
    tl.setup_camera_hint = None          # (섹터 2의 setup_camera를 여기서 호출)

    judge = tl.TrafficLightJudge()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        drive, state = process_frame(frame, judge)

        # TODO(통합): drive를 시리얼로 Arduino에 전송
        #   STOP  -> motor_hold
        #   GO    -> 정상 속도
        #   CRUISE-> 차선 주행 조향각 그대로
        print(f"{drive:7s} | 색={state['color']} "
              f"정지선={state['stopline_cm']}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


if __name__ == "__main__":
    main()
