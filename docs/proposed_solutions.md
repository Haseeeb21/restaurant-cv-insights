# Proposed Solutions Not Implemented

Everything below is a solution that would add real value to a restaurant operator, but couldn't be built with this specific footage. The common thread is that this is a single fixed camera pointed at the entrance, so anything needing a different view of the restaurant, closer detail on faces or tables, or data this camera simply doesn't capture, falls into this category. For each one: what it is, what data or setup it would need, and how it would likely be built.

## Timestamp extraction from the on-screen clock overlay

What it would do: read the burned-in CCTV timestamp visible in the corner of the frame, so every count and zone event in the analytics output could be tagged with a real time of day instead of just a frame number, letting a report say something like "the queue peaked around 6:12pm" instead of "at frame 9,400."

What was tried: an OCR pass (EasyOCR) on the timestamp region, sampled once per second rather than every frame. It didn't come out reliable enough to trust for a client-facing report, the readings were inconsistent enough that it was pulled from the final pipeline rather than shipped in a half-working state.

Required data or setup: nothing further is strictly needed data-wise, this is more a matter of engineering time than missing data. A cleaner crop of just the timestamp region, some contrast preprocessing (thresholding the text to pure black and white before OCR), and possibly a different OCR engine or a small template-matching approach built specifically for this fixed digital font, would likely get this working reliably.

Recommended approach: rather than OCR at all, the more robust fix is usually simpler than it looks. If the video's frame rate and the wall-clock time of its very first frame are known (even just read once by eye), every later frame's timestamp can be computed directly as start_time plus frame_number divided by fps, with no OCR involved. This sidesteps the reliability problem entirely and is the approach worth trying first if this is picked back up.

## Full restaurant occupancy and table turnover

What it would do: track how full the restaurant is overall and how quickly tables turn over across the whole dining room, not just the couple of tables visible at the edge of this entrance camera.

Required data: cameras covering every table in the restaurant, ideally mounted so each table is visible without being blocked by pillars, the lamp, or other guests.

Recommended approach: run the same YOLO person detection and zone logic used here, but per-camera, then stitch the zones together into one occupancy map using a homography (a per-camera coordinate transform) so a single dashboard reflects the whole floor rather than isolated camera views.

## Food and order delivery time tracking

What it would do: measure how long it takes from an order being placed to food actually reaching the table, a genuinely useful operational metric for a restaurant.

Required data: a camera with a clear view of the kitchen pass (where plates leave the kitchen) and a closer, less angled view of each table than what this entrance camera provides.

Recommended approach: detect plates leaving the kitchen pass with a lightweight object detector, then detect the same plates arriving at a table, and compute the time between the two. This would also need a POS integration to know when the order was actually placed, since vision alone can't see that.

## Staff efficiency and server-to-table assignment

What it would do: see how many tables each server is handling and how quickly they respond to a table's needs.

Required data: some way to actually tell staff apart from customers, either a uniform color that's clearly visible from the camera, a badge, or labeled training footage of staff versus guests. None of that exists in this footage; visually, servers and customers look the same from this camera's angle and distance.

Recommended approach: once such labeling exists, fine-tune a classifier on top of the person detector to tag each detection as staff or customer, then track which staff member spends time near which table.

## Sitting versus standing classification using pose estimation

What it would do: more precisely tell whether someone is sitting, standing, or waiting, beyond the current zone-based occupancy proxy.

Required data: this actually doesn't need new data exactly, it needs a better camera angle. Pose estimation (detecting keypoints like shoulders, hips, and knees) works best on a roughly front-facing or side-facing view of a person. This camera looks almost straight down from a fisheye lens, which distorts limb geometry badly and would make pose keypoints unreliable.

Recommended approach: with a more typical eye-level or moderately angled camera, a pose model such as YOLO26-pose could classify sitting versus standing from joint angles far more robustly than the current bounding-box-and-zone heuristic. It wasn't attempted here because the expected accuracy gain, given this specific camera's angle, didn't justify the added compute and complexity.

## Spill, hazard, or safety incident detection

What it would do: flag things like a spill on the floor, a fall, or another safety incident automatically.

Required data: labeled footage of actual incidents, or at minimum near-miss events, none of which appear in this clip. Training or even reliably prompting a model for this needs examples of what the failure case actually looks like.

Recommended approach: once such footage exists, this would likely be either a fine-tuned action-recognition model or an anomaly-detection approach that flags unusual motion patterns compared to normal foot traffic.

## Repeat customer recognition via face matching

What it would do: recognize returning customers for loyalty or personalization purposes.

Required data: a front-facing camera with clear, well-lit faces. This entrance camera's overhead fisheye angle mostly shows the tops of people's heads and shoulders, faces are barely visible.

Recommended approach: even setting aside the camera limitation, this is flagged as something to think carefully about before building at all. Face recognition of customers raises real privacy and consent questions that go beyond a technical feasibility problem, and would need a clear legal and policy review before implementation regardless of camera quality.

## POS-integrated wait time analysis

What it would do: correlate how long someone waited to be seated with how long they then waited to order, get their food, and pay, giving a full picture of the customer's time in the restaurant rather than just the entrance-to-seating piece this project covers.

Required data: an export or live feed from the restaurant's point-of-sale system with order and payment timestamps.

Recommended approach: this is mostly a data engineering problem rather than a computer vision one, joining the vision pipeline's timestamps (already logged per frame) with POS timestamps on a shared customer or table identifier.

## Natural language scene summary and question answering

What it would do: let a manager ask something like "how many people are here right now" or "what's happening at the entrance" and get a plain-language answer.

Required data: none beyond what this pipeline already produces, the structured counts and zone stats in the metrics CSV.

Recommended approach: this is deliberately not built as a vision-language model answering questions directly from video frames. Vision-language models are not reliable at precise counting, and running one continuously on video is expensive. The better architecture is to keep the counting logic exactly as it is now, and add a lightweight text-only language model on top that reads the structured output (the CSV or a small JSON summary) and answers questions in natural language grounded in those real numbers. A true vision-language model would only be worth adding on top of that for open-ended "describe what's happening" requests that go beyond the metrics this pipeline already tracks, and that's a genuine stretch goal rather than a near-term addition.

## Age or demographic analytics

What it would do: break down foot traffic by rough age group or other demographic signals.

Required data: better face visibility than this camera provides.

Recommended approach: not recommended even if the camera limitation were solved. This raises privacy and fairness concerns serious enough that it's flagged here as something to think twice about rather than something to build, regardless of data availability.
