import argparse
import cv2
import copy
import numpy as np

from src import util
from src.body import Body
from src.hand import Hand


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exclude-hand", action="store_true")
    parser.add_argument("--exclude-body", action="store_true")

    parser.add_argument("--input-video", type=str, default="0")
    parser.add_argument("--input-video-height", type=int, default=480)
    parser.add_argument("--input-video-width", type=int, default=640)

    parser.add_argument("--body-model-ckpt-path", type=str,
                        default="model/body_pose_model.pth")
    parser.add_argument("--hand-model-ckpt-path", type=str,
                        default="model/hand_pose_model.pth")

    return parser.parse_args()


def main(args):
    assert not (args.exclude_body and args.exclude_hand)
    body_estimation = None
    hand_estimation = None
    if not args.exclude_hand:
        hand_estimation = Hand(args.hand_model_ckpt_path)
    if not args.exclude_body:
        body_estimation = Body(args.body_model_ckpt_path)

    try:
        cap = cv2.VideoCapture(int(args.input_video))
    except:
        cap = cv2.VideoCapture(args.input_video)

    cap.set(3, args.input_video_width)
    cap.set(4, args.input_video_height)
    while True:
        ret, oriImg = cap.read()

        if args.exclude_body:
            peaks = hand_estimation(oriImg)
            canvas = util.draw_handpose(oriImg, [peaks], True)
        elif args.exclude_hand:
            candidate, subset = body_estimation(oriImg)
            canvas = util.draw_bodypose(oriImg, candidate, subset)
        else:
            candidate, subset = body_estimation(oriImg)
            canvas = copy.deepcopy(oriImg)
            canvas = util.draw_bodypose(canvas, candidate, subset)

            # detect hand
            hands_list = util.handDetect(candidate, subset, oriImg)

            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
                peaks[:, 0] = np.where(
                    peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0]+x)
                peaks[:, 1] = np.where(
                    peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1]+y)
                all_hand_peaks.append(peaks)

            canvas = util.draw_handpose(canvas, all_hand_peaks)

        cv2.imshow('demo', canvas)  # 一个窗口用以显示原视频
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(_parse_args())
