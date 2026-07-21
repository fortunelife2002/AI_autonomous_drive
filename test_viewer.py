"""
test_viewer.py — 카메라 화면에 신호등 색 + 정지선을 실시간 표시

mission_trafficlight.py는 판단만 하고 화면을 안 띄운다(통합 골격이라서).
이 파일은 눈으로 확인하는 용도 — 카메라 영상 위에 결과를 그려서 보여준다.

실행:
    python test_viewer.py           # 기본으로 카메라 1번 사용
    python test_viewer.py 0         # 인자를 주면 해당 번호로 강제 지정

  q 키: 종료

카메라가 안 열리면 번호를 0,1,2로 바꿔가며 시도할 것.
이 스크립트는 시작할 때 사용 가능한 카메라를 자동으로 찾아준다.
"""

import cv2
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/project")

import trafficlight_lib as tl
import stopline_lib as sl

try:
    from start_lib import get_bird_eye_view

    HAS_BEV = True
except Exception:
    HAS_BEV = False


def find_camera(preferred=None):
    """열리는 카메라 번호를 찾는다. preferred부터 시도."""
    # 1번 카메라를 우선순위 맨 앞으로 배치 (preferred가 지정되면 그것부터)
    candidates = ([preferred] if preferred is not None else []) + [1, 0, 2, 3]
    for idx in candidates:
        if idx is None:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"카메라 {idx}번 사용")
                return cap, idx
            cap.release()
        print(f"카메라 {idx}번: 열기 실패")
    return None, None


def main():
    # 기본 카메라 번호를 1번으로 설정
    preferred = 1
    args = sys.argv[1:]
    skip_next = False

    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a == "--exposure":
            skip_next = True
            continue
        # 실행할 때 뒤에 숫자를 직접 적어줬다면 (예: python test_viewer.py 0) 그 값을 따름
        if a.lstrip("-").isdigit():
            preferred = int(a)
            break

    cap, idx = find_camera(preferred)
    if cap is None:
        print("\n[에러] 열리는 카메라가 없습니다.")
        print("  · USB 웹캠이 꽂혀 있는지 확인")
        print("  · 다른 프로그램(줌 등)이 카메라를 쓰고 있지 않은지 확인")
        print("  · Windows면 카메라 개인정보 설정에서 앱 접근 허용 확인")
        return

    # 노출 고정 여부: 기본은 자동노출(화면이 잘 보임).
    #   --exposure 인자를 주면 그 값으로 고정 (신호등 인식 최종 튜닝 시 사용)
    #   예: python test_viewer.py 1 --exposure -6
    fix_exposure = None
    if "--exposure" in sys.argv:
        i = sys.argv.index("--exposure")
        if i + 1 < len(sys.argv):
            fix_exposure = int(sys.argv[i + 1])

    if fix_exposure is not None:
        try:
            tl.setup_camera(cap, exposure=fix_exposure)
            print(f"노출 고정: {fix_exposure}")
        except Exception as e:
            print(f"카메라 설정 고정 실패(무시하고 진행): {e}")
    else:
        print("자동노출 사용 (화면 확인용). 신호등 최종 튜닝 시 --exposure 값 지정")

    judge = tl.TrafficLightJudge()
    print("실행 중... 'q' 키로 종료\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("프레임 읽기 실패")
            break

        # --- 신호등 ---
        color, tl_info = tl.detect_traffic_light(frame, debug=True)
        cmd = judge.update(color)

        # --- 정지선 (BEV 가능할 때만) ---
        stopline_cm = None
        if HAS_BEV:
            h, w = frame.shape[:2]
            pts = [(int(w * 0.02), int(h * 0.98)), (int(w * 0.98), int(h * 0.98)),
                   (int(w * 0.80), int(h * 0.55)), (int(w * 0.20), int(h * 0.55))]
            try:
                bev = get_bird_eye_view(frame, (400, 600), pts)
                _, sl_info = sl.should_stop(bev)
                stopline_cm = sl_info["distance_cm"]
                # 정지선 시각화 창도 함께
                sl_y, sl_dbg_info = sl.detect_stopline(bev)
                bev_vis = sl.draw_debug(bev, sl_y, sl_dbg_info)
                cv2.imshow("BEV + Stopline", bev_vis)
            except Exception as e:
                pass

        # --- 신호등 결과를 원본 위에 표시 ---
        vis = tl.draw_debug(frame, color, tl_info)
        label = f"{color}  cmd={cmd}"
        if stopline_cm is not None:
            label += f"  stopline={stopline_cm:.0f}cm"
        cv2.putText(vis, label, (10, vis.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Camera - Traffic Light", vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()