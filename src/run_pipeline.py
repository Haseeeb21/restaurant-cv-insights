"""
Single combined pipeline for all restaurant-operations CV solutions in this
project. One detection model + one tracker feeding several zone-analytics
modules, rather than multiple stacked models.

Solutions implemented here:
    - Entrance in/out counting (explicit side-of-line crossing, debuggable -
      see ZoneTracker in zone_utils.py)
    - Waiting-area occupancy + dwell time (anchor-point-in-polygon)
    - Table occupied/empty status (anchor-point-in-polygon)
    - Host-stand exclusion (overlap-ratio test, NOT anchor-point — a person
      occluded behind a counter has a bbox whose bottom isn't their real feet,
      so a single anchor point unreliably misses them; overlap-ratio checks
      how much of their whole box falls inside the zone instead)
    - Live on-frame summary overlay
    - Per-run MLflow logging (params, metrics, sample annotated frames, CSV)

Usage:
    python src/run_pipeline.py --video path/to/video.mp4 --zones configs/zones.json \
        --weights yolo26s.pt --out outputs/annotated.mp4 --debug

Model classes used (COCO ids, no custom training required):
    0  = person
    56 = chair         # Not used
    60 = dining table. # Not used
"""
import argparse
import time

import cv2
import mlflow
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from zone_utils import ZoneTracker, load_zones, overlap_ratio

PERSON_CLASS = 0
CHAIR_CLASS = 56
TABLE_CLASS = 60
EXCLUSION_OVERLAP_THRESHOLD = 0.30  # fraction of a person's bbox inside the exclusion polygon


def bbox_anchor(xyxy):
    """Bottom-center of the bbox (approx. feet position). Used for waiting-area
    and table membership. NOT used for exclusion — see overlap_ratio instead."""
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2), float(y2))


def draw_zones(frame, zones):
    for z in zones:
        color = {
            "entrance_line": (0, 0, 255),
            "waiting_area": (0, 255, 255),
            "table": (255, 0, 0),
            "exclusion": (128, 128, 128),
        }.get(z["type"], (0, 255, 0))
        pts = np.array(z["points"], dtype=np.int32)
        if z["type"] == "entrance_line":
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, 2)
        else:
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def draw_summary_panel(frame, summary_lines):
    x, y = 15, 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (430, 30 + 26 * len(summary_lines)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for line in summary_lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", default="configs/zones.json")
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument("--csv_out", default="outputs/metrics.csv")
    parser.add_argument("--experiment", default="restaurant-cv-insights")
    parser.add_argument("--min_track_age", type=int, default=8,
                         help="Frames a track must persist before it's allowed to trigger an "
                              "entrance in/out crossing. Filters out phantom crossings caused by "
                              "ByteTrack ID switches at the door (a brand-new ID appearing already "
                              "past the line looks like a crossing but isn't a real one).")
    parser.add_argument("--invert_inout", action="store_true",
                         help="Swap the in/out labels without re-labeling zones.json. Use this if "
                              "a calibration walk-through shows the counts are backwards — that "
                              "just means start/end of your entrance_line are flipped relative to "
                              "the physical doorway.")
    parser.add_argument("--exclusion_overlap_threshold", type=float, default=EXCLUSION_OVERLAP_THRESHOLD)
    parser.add_argument("--debug", action="store_true",
                         help="Print a line to console every time an in/out count changes, so you "
                              "can watch it live against the video and confirm crossings register.")
    args = parser.parse_args()

    zones = load_zones(args.zones)  # raises clearly if an entrance_line has != 2 points
    area_zones = [z for z in zones if z["type"] in ("waiting_area", "table")]
    exclusion_zones = [z for z in zones if z["type"] == "exclusion"]
    zone_tracker = ZoneTracker(zones)  # handles waiting/table dwell AND line-crossing counts

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = YOLO(args.weights)  # auto-downloads pretrained weights on first use
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video at '{args.video}'. Check the path and extension "
            f"(e.g. .mp4 vs .mov) match an actual file in this directory."
        )
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rows = []
    frame_idx = 0

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run():
        mlflow.log_params({
            "model_weights": args.weights,
            "confidence_threshold": args.conf,
            "classes": [PERSON_CLASS],   # removed CHAIR_CLASS and TABLE_CLASS because table area is marked
            "tracker": "ByteTrack",
            "video": args.video,
            "min_track_age": args.min_track_age,
            "invert_inout": args.invert_inout,
            "waiting_table_method": "anchor_point_bottom_center",
            "exclusion_method": "overlap_ratio",
            "exclusion_overlap_threshold": args.exclusion_overlap_threshold,
        })

        id_switch_estimate = 0
        prev_active_ids = set()
        track_first_seen = {}  # track_id -> frame_idx it was first observed
        start_time = time.time()
        last_printed_in, last_printed_out = 0, 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame, classes=[PERSON_CLASS], # only detect people, not chairs/tables
                conf=args.conf, verbose=False,
            )[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            person_mask = detections.class_id == PERSON_CLASS
            person_detections = detections[person_mask]

            active_ids = set()
            people_now = 0
            waiting_counts = {}
            table_names = {z["name"] for z in area_zones if z["type"] == "table"}
            table_occupied = {name: False for name in table_names}

            for i in range(len(person_detections)):
                xyxy = person_detections.xyxy[i]
                track_id = (int(person_detections.tracker_id[i])
                            if person_detections.tracker_id is not None else -1)

                if track_id not in track_first_seen:
                    track_first_seen[track_id] = frame_idx
                is_stable = (frame_idx - track_first_seen[track_id]) >= args.min_track_age

                active_ids.add(track_id)

                # Exclusion via overlap-ratio (whole-bbox test), not a single anchor
                # point — robust to a person being occluded behind the counter.
                excluded = any(
                    overlap_ratio(xyxy, z["points"]) >= args.exclusion_overlap_threshold
                    for z in exclusion_zones
                )
                if excluded:
                    continue  # detected (drawn on screen) but excluded from customer metrics

                people_now += 1
                anchor = bbox_anchor(xyxy)
                # count_line=is_stable: area-zone membership always processed, but
                # a freshly-spawned track (likely ID switch) can't trigger a count.
                current_zones = zone_tracker.update(track_id, anchor, count_line=is_stable)
                for zname in current_zones:
                    if zname in table_names:
                        table_occupied[zname] = True
                    else:
                        waiting_counts[zname] = waiting_counts.get(zname, 0) + 1

            id_switch_estimate += len(prev_active_ids.symmetric_difference(active_ids)) if frame_idx > 0 else 0
            prev_active_ids = active_ids

            total_in, total_out = zone_tracker.in_count, zone_tracker.out_count
            if args.invert_inout:
                total_in, total_out = total_out, total_in

            if args.debug and (total_in != last_printed_in or total_out != last_printed_out):
                print(f"[frame {frame_idx}] in={total_in} out={total_out}")
                last_printed_in, last_printed_out = total_in, total_out

            frame = draw_zones(frame, zones)
            frame = box_annotator.annotate(frame, detections)
            id_labels = [
                f"#{int(tid)} {model.names[int(cid)]}"
                for tid, cid in zip(
                    detections.tracker_id if detections.tracker_id is not None else [-1] * len(detections),
                    detections.class_id,
                )
            ]
            frame = label_annotator.annotate(frame, detections, labels=id_labels)

            summary = [
                f"People in frame: {people_now}",
                f"In: {total_in}  Out: {total_out}",
                f"Waiting: {sum(waiting_counts.values())}",
                f"Tables occupied: {sum(table_occupied.values())}/{len(table_occupied) or '-'}",
            ]
            frame = draw_summary_panel(frame, summary)
            writer.write(frame)

            rows.append({
                "frame": frame_idx,
                "people_in_frame": people_now,
                "cumulative_in": total_in,
                "cumulative_out": total_out,
                "waiting_total": sum(waiting_counts.values()),
                "tables_occupied": sum(table_occupied.values()),
                "tables_total": len(table_occupied),
            })
            frame_idx += 1

        writer.release()
        cap.release()

        df = pd.DataFrame(rows)
        df.to_csv(args.csv_out, index=False)

        elapsed = time.time() - start_time
        mlflow.log_metrics({
            "frames_processed": frame_idx,
            "avg_fps": frame_idx / elapsed if elapsed > 0 else 0,
            "max_people_in_frame": int(df["people_in_frame"].max()) if not df.empty else 0,
            "final_in_count": total_in,
            "final_out_count": total_out,
            "id_switch_estimate": id_switch_estimate,
        })
        mlflow.log_artifact(args.csv_out)
        mlflow.log_artifact(args.zones)

    print(f"Done. Annotated video -> {args.out}, metrics CSV -> {args.csv_out}")


if __name__ == "__main__":
    main()
