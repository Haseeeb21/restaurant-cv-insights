"""
Manual zone-labeling tool. You are NOT training a model to find zones -
these are static polygons on a fixed camera, so you draw them once by hand.

Usage:
    python scripts/label_zones.py --frame path/to/reference_frame.jpg --out configs/zones.json

Controls:
    - Left click to add a point to the current polygon.
    - Press 'n' to finish the current polygon and start a new one (you'll be
      prompted in the terminal for a zone name and a zone type).
    - Press 's' to save all zones to the output JSON and quit.
    - Press 'r' to reset the current (unfinished) polygon.
    - Press 'q' to quit without saving.

Zone types (used by the analytics logic downstream):
    - "entrance_line"   -> a 2-point line for in/out crossing counts
    - "waiting_area"    -> polygon, used for dwell-time / queue counts
    - "table"           -> polygon, used for occupancy status
    - "exclusion"       -> polygon, e.g. host stand — detections here are
                            excluded from customer-facing metrics
"""
import argparse
import json

import cv2

ZONE_TYPES = ["entrance_line", "waiting_area", "table", "exclusion"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", required=True, help="Path to a reference frame (jpg/png)")
    parser.add_argument("--out", default="configs/zones.json", help="Output JSON path")
    args = parser.parse_args()

    img = cv2.imread(args.frame)
    if img is None:
        raise FileNotFoundError(f"Could not read frame at {args.frame}")

    clone = img.copy()
    current_points = []
    zones = []

    def redraw():
        display = clone.copy()
        for z in zones:
            pts = z["points"]
            color = {
                "entrance_line": (0, 0, 255),
                "waiting_area": (0, 255, 255),
                "table": (255, 0, 0),
                "exclusion": (128, 128, 128),
            }.get(z["type"], (0, 255, 0))
            for i in range(len(pts) - 1):
                cv2.line(display, tuple(pts[i]), tuple(pts[i + 1]), color, 2)
            if z["type"] != "entrance_line" and len(pts) > 2:
                cv2.line(display, tuple(pts[-1]), tuple(pts[0]), color, 2)
            cv2.putText(display, z["name"], tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        for i in range(len(current_points) - 1):
            cv2.line(display, current_points[i], current_points[i + 1], (0, 255, 0), 2)
        for p in current_points:
            cv2.circle(display, p, 4, (0, 255, 0), -1)
        cv2.imshow("label_zones", display)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))
            redraw()

    cv2.namedWindow("label_zones")
    cv2.setMouseCallback("label_zones", on_click)
    redraw()

    print("Click points, then press 'n' to name+save this polygon, 's' to finish, 'q' to quit.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and len(current_points) >= 2:
            name = input("Zone name (e.g. 'entrance_door', 'waiting_bench_left'): ").strip()
            print(f"Zone types: {ZONE_TYPES}")
            ztype = input("Zone type: ").strip()
            if ztype not in ZONE_TYPES:
                print("Unknown type, defaulting to 'waiting_area'")
                ztype = "waiting_area"
            zones.append({"name": name, "type": ztype, "points": [list(p) for p in current_points]})
            current_points = []
            redraw()
        elif key == ord("r"):
            current_points = []
            redraw()
        elif key == ord("s"):
            with open(args.out, "w") as f:
                json.dump({"zones": zones}, f, indent=2)
            print(f"Saved {len(zones)} zones to {args.out}")
            break
        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
