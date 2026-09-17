# Restaurant CV Insights

Computer vision pipeline that analyzes restaurant CCTV footage from a single fixed entrance camera and produces operational metrics: entrance in/out counts, waiting area occupancy and dwell time, table occupancy, and a staff exclusion zone at the host stand.

Demo video used: https://youtu.be/FsKal7K2zJY?si=QTnriBd3hFXkOhuD

## Sample frame and prediction output

Reference frame (used to label zones):

<img width="1920" height="1080" alt="sample_frame" src="https://github.com/user-attachments/assets/2ca89d95-c650-485e-86ad-a96ea7ad4a00" />

Annotated prediction output:

<img width="1512" height="866" alt="labelled_reference" src="https://github.com/user-attachments/assets/b34faaf3-97a8-4167-b626-1c53867e467b" />

## What this does

One camera angle is available, pointed at the entrance. Given that constraint, the pipeline implements everything that a single fixed camera can reliably support:

- Entrance in/out counting (crossing detection, still being tuned, see `docs/limitations.md`)
- Waiting area occupancy and dwell time, for two separate waiting zones
- Table occupancy status for the tables visible in frame
- Host stand exclusion, so staff at reception aren't counted as customers
- Live on-screen summary overlay (people count, in/out, waiting, tables occupied)
- Per-run experiment tracking with MLflow

Solutions that would need a different camera angle, more data, or more time (table turnover across the full floor, food delivery timing, staff efficiency, pose-based sitting/standing, POS integration, a natural language Q&A layer, and more) are documented instead of implemented, see `docs/proposed_solutions.md`.

## How it works

One detection model feeds several zone-analytics modules, rather than chaining multiple stacked models:

1. YOLO26s (Ultralytics, COCO-pretrained) detects people in each frame. Only the person class is used, chair and dining table detection were tried and dropped, see `docs/limitations.md` for why.
2. ByteTrack (via the `supervision` library) assigns a persistent ID to each detected person across frames.
3. Hand-labeled zone polygons (`configs/zones.json`) turn tracked detections into metrics: an entrance line for in/out counts, waiting-area and table polygons tested against each person's approximate foot position, and an exclusion polygon around the host stand tested against how much of a person's box overlaps it.
4. Results are drawn on the output video, written to a CSV, and logged to MLflow.

Full write-up of assumptions and known failure modes: `docs/limitations.md`. Ideas that didn't make it in, and why: `docs/proposed_solutions.md`.

## Repo structure

```
restaurant-cv-insights/
├── README.md
├── LICENSE
├── docs/
│   ├── limitations.md          # assumptions, failure modes, mitigations for what's implemented
│   ├── proposed_solutions.md   # solutions not implemented, required data, recommended approach
├── outputs/
│   └── metrics.csv             # results of detected objects per frame
│   └── sample_frame.jpg             # sample frame from video used
├── configs/
│   └── zones.json              # hand-labeled zone polygons for this camera
├── scripts/
│   ├── extract_frame.py        # pulls a reference frame for zone labeling
│   └── label_zones.py          # interactive tool to draw zone polygons
├── src/
│   ├── zone_utils.py           # zone geometry: point-in-polygon, overlap ratio, dwell tracking
│   ├── run_pipeline.py         # main pipeline: anchor-point zone membership
│   └── run_pipeline_iou.py     # comparison pipeline: overlap-ratio zone membership
├── requirements.txt
└── .gitignore
```

Video files, model weights, and the MLflow store are intentionally not committed, see `.gitignore`.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Extract a reference frame and label zones on it (only needs to be done once per camera):

```bash
python scripts/extract_frame.py --video your_video.mp4 --frame_number 300 --out outputs/sample_frame.jpg
python scripts/label_zones.py --frame outputs/sample_frame.jpg --out configs/zones.json
```

Run the main pipeline:

```bash
python src/run_pipeline.py --video your_video.mp4 --weights yolo26s.pt --debug
```

Run the comparison (overlap-ratio) pipeline:

```bash
python src/run_pipeline_iou.py --video your_video.mp4 --weights yolo26s.pt --debug
```

View experiment tracking:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Model and settings

- Model: YOLO26s, Ultralytics pretrained COCO checkpoint, person class only
- Confidence threshold: 0.35
- Tracker: ByteTrack (supervision library, default settings)
- Zone membership: anchor-point-in-polygon for waiting areas and tables, overlap-ratio for the exclusion zone
- Minimum track age before a crossing counts: 8 frames

Full reasoning behind these choices is in `docs/limitations.md`.

## License

MIT, see `LICENSE`.
