"""
Extracts a single frame using cv2.VideoCapture — the SAME read path
run_pipeline.py uses — so the pixel coordinates you click in label_zones.py
are guaranteed to match what the pipeline actually sees at inference time.

Usage:
    python scripts/extract_frame.py --video CCTV_Restaurant.mp4 --frame_number 300 \
        --out outputs/sample_frame_01.jpg
"""
import argparse

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame_number", type=int, default=300,
                         help="Pick a frame with people/furniture clearly visible, not a blank one.")
    parser.add_argument("--out", default="outputs/sample_frame.jpg")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open '{args.video}'")

    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_number)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame_number} — video may be shorter than that.")

    cv2.imwrite(args.out, frame)
    print(f"Frame {args.frame_number} ({frame.shape[1]}x{frame.shape[0]}) saved -> {args.out}")
    print("Label zones on THIS file, against THIS video, to keep coordinates aligned.")
    cap.release()


if __name__ == "__main__":
    main()
