"""
Same detector (YOLO26, person) and same tracker (ByteTrack)
as run_pipeline.py — the ONLY thing this file changes is how zone membership
is decided: bbox-overlap-ratio instead of anchor-point-in-polygon. Run both on
the same video and compare outputs/*.csv for the write-up.

Overlap metric used: intersection_area / person_bbox_area (NOT classic
symmetric IoU = intersection / union). Classic IoU would be almost always
near-zero here regardless of true containment, because a zone polygon
(e.g. the whole waiting area) is typically far larger than a single person's
bounding box — union would be dominated by the zone's own area. What actually
answers "is this person inside the zone" is what fraction of THEIR box falls
inside it, which is intersection / person_bbox_area. A person's bbox is
counted as "in" a zone once that fraction crosses --overlap_threshold.

Entrance in/out here is done differently too: the entrance_line from
zones.json is expanded into a thin rectangular "gate" (+/- --gate_width_px on
either side), and a person is counted as crossing when their bbox overlap
with the gate crosses the threshold AND their anchor's position two frames
apart shows net movement to one side of the line (this replaces sv.LineZone's
built-in crossing state machine with an explicit overlap+direction check).

Usage:
    python src/run_pipeline_iou.py --video path/to/video.mp4 --zones configs/zones.json \
        --weights yolo26s.pt --out outputs/annotated_iou.mp4 --csv_out outputs/metrics_iou.csv
"""
import argparse
import time

import cv2
import mlflow
import numpy as np
import pandas as pd
import supervision as sv
from zone_utils import load_zones, overlap_ratio, side_of_line
from ultralytics import YOLO

PERSON_CLASS = 0
CHAIR_CLASS = 56
TABLE_CLASS = 60


def make_gate_polygon(line_points, gate_width_px):
    (x1, y1), (x2, y2) = line_points
    dx, dy = x2 - x1, y2 - y1
    length = (dx ** 2 + dy ** 2) ** 0.5 or 1.0
    nx, ny = -dy / length, dx / length  # unit normal
    w = gate_width_px
    return [
        (x1 + nx * w, y1 + ny * w), (x2 + nx * w, y2 + ny * w),
        (x2 - nx * w, y2 - ny * w), (x1 - nx * w, y1 - ny * w),
    ]


def bbox_anchor(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2), float(y2))


def draw_zones(frame, zones, gate_polys):
    for z in zones:
        color = {"waiting_area": (0, 255, 255), "table": (255, 0, 0),
                 "exclusion": (128, 128, 128)}.get(z["type"], (0, 255, 0))
        if z["type"] == "entrance_line":
            continue
        pts = np.array(z["points"], dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    for name, poly_pts in gate_polys.items():
        pts = np.array(poly_pts, dtype=np.int32)
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, f"gate:{name}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame


def draw_summary_panel(frame, lines):
    x, y = 15, 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 30 + 26 * len(lines)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        y += 26
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--zones", default="configs/zones.json")
    parser.add_argument("--weights", default="yolo26s.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--out", default="outputs/annotated_iou.mp4")
    parser.add_argument("--csv_out", default="outputs/metrics_iou.csv")
    parser.add_argument("--experiment", default="restaurant-cv-insights")
    parser.add_argument("--overlap_threshold", type=float, default=0.35,
                         help="Fraction of a person's bbox that must fall inside a zone/gate "
                              "polygon to count as 'in' it.")
    parser.add_argument("--gate_width_px", type=float, default=140.0,
                         help="Half-width of the rectangular gate built around each entrance_line. "
                              "Widened from an earlier default of 60px — too narrow a gate means a "
                              "fast-moving person may only ever be observed on one side of the "
                              "physical line, so no crossing is ever detected. Widen further if "
                              "in/out still isn't registering.")
    parser.add_argument("--min_track_age", type=int, default=8)
    parser.add_argument("--invert_inout", action="store_true")
    parser.add_argument("--debug", action="store_true",
                         help="Print a line to console every time an in/out count changes.")
    args = parser.parse_args()

    zones = load_zones(args.zones)
    area_zones = [z for z in zones if z["type"] in ("waiting_area", "table", "exclusion")]
    line_zones = [z for z in zones if z["type"] == "entrance_line"]
    gate_polys = {z["name"]: make_gate_polygon(z["points"], args.gate_width_px) for z in line_zones}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    model = YOLO(args.weights)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open '{args.video}'")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    rows = []
    frame_idx = 0
    track_first_seen = {}
    track_last_side = {}   # track_id -> last side-of-line, per line name
    in_count, out_count = 0, 0

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="iou_overlap_variant"):
        mlflow.log_params({
            "model_weights": args.weights, "confidence_threshold": args.conf,
            "tracker": "ByteTrack", "video": args.video,
            "zone_membership_method": "bbox_overlap_ratio",
            "overlap_threshold": args.overlap_threshold,
            "gate_width_px": args.gate_width_px,
            "min_track_age": args.min_track_age,
        })

        start_time = time.time()
        last_printed_in, last_printed_out = 0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(frame, classes=[PERSON_CLASS], # removed CHAIR_CLASS and TABLE_CLASS because table area is marked
                                     conf=args.conf, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)
            person_mask = detections.class_id == PERSON_CLASS
            person_detections = detections[person_mask]

            people_now = 0
            waiting_counts, table_occupied = {}, {z["name"]: False for z in area_zones if z["type"] == "table"}

            for i in range(len(person_detections)):
                tid = int(person_detections.tracker_id[i]) if person_detections.tracker_id is not None else -1
                xyxy = person_detections.xyxy[i]
                if tid not in track_first_seen:
                    track_first_seen[tid] = frame_idx
                is_stable = (frame_idx - track_first_seen[tid]) >= args.min_track_age

                # Exclusion + waiting + table via overlap ratio
                excluded = False
                for z in area_zones:
                    ratio = overlap_ratio(xyxy, z["points"])
                    if ratio >= args.overlap_threshold:
                        if z["type"] == "exclusion":
                            excluded = True
                            break
                        elif z["type"] == "table":
                            table_occupied[z["name"]] = True
                        elif z["type"] == "waiting_area":
                            waiting_counts[z["name"]] = waiting_counts.get(z["name"], 0) + 1
                if excluded:
                    continue
                people_now += 1

                # Entrance crossing via gate overlap + direction of movement
                if is_stable:
                    anchor = bbox_anchor(xyxy)
                    for z in line_zones:
                        ratio = overlap_ratio(xyxy, gate_polys[z["name"]])
                        if ratio >= args.overlap_threshold:
                            side = side_of_line(anchor, z["points"])
                            key = (tid, z["name"])
                            prev = track_last_side.get(key)
                            if prev is not None and prev != side:
                                if side > 0:
                                    in_count += 1
                                else:
                                    out_count += 1
                            track_last_side[key] = side

            if args.invert_inout:
                disp_in, disp_out = out_count, in_count
            else:
                disp_in, disp_out = in_count, out_count

            if args.debug and (disp_in != last_printed_in or disp_out != last_printed_out):
                print(f"[frame {frame_idx}] in={disp_in} out={disp_out}")
                last_printed_in, last_printed_out = disp_in, disp_out

            frame = draw_zones(frame, zones, gate_polys)
            frame = box_annotator.annotate(frame, detections)
            id_labels = [f"#{int(t)} {model.names[int(c)]}" for t, c in
                         zip(detections.tracker_id if detections.tracker_id is not None
                             else [-1] * len(detections), detections.class_id)]
            frame = label_annotator.annotate(frame, detections, labels=id_labels)
            frame = draw_summary_panel(frame, [
                f"People in frame: {people_now}",
                f"In: {disp_in}  Out: {disp_out}",
                f"Waiting: {sum(waiting_counts.values())}",
                f"Tables occupied: {sum(table_occupied.values())}/{len(table_occupied) or '-'}",
                f"[IoU-overlap variant, threshold={args.overlap_threshold}]",
            ])
            writer.write(frame)

            rows.append({
                "frame": frame_idx, "people_in_frame": people_now,
                "cumulative_in": disp_in, "cumulative_out": disp_out,
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
            "final_in_count": disp_in, "final_out_count": disp_out,
        })
        mlflow.log_artifact(args.csv_out)
        mlflow.log_artifact(args.zones)

    print(f"Done. Annotated video -> {args.out}, metrics CSV -> {args.csv_out}")


if __name__ == "__main__":
    main()
