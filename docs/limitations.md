# Limitations and Failure Modes: Implemented Solutions

This document covers everything that was actually built and run on the CCTV footage: the model and settings used, and for each solution, what it assumes, where it breaks, and how that could be fixed with more time, data, or a better camera setup.

## Model and pipeline settings

Detection uses YOLO26s, Ultralytics' pretrained COCO checkpoint, filtered to a single class: person (class id 0). Chair and dining table classes (ids 56 and 60) were tried early on since COCO already includes them, but they were dropped from the final pipeline. At this camera's angle and distance, chairs and tables were either too small, too occluded by the lamp and other furniture, or simply not detected reliably by the pretrained weights, so keeping them only added noise without giving any real occupancy signal. This is a direct example of a solution idea that looked promising on paper but didn't hold up once tested against this specific footage.

Person detection runs at a confidence threshold of 0.35. This was chosen empirically: lower thresholds (around 0.2) started pulling in false positives from reflections and the overexposed lamp area, while higher thresholds (0.5+) started missing people at the edges of the fisheye frame or partially behind furniture. 0.35 was the best balance found by watching the annotated output at a few different settings.

Tracking uses ByteTrack through the supervision library, with its default association settings (no custom tuning of its internal matching thresholds). ByteTrack's own IoU-based matching (linking a detection in the current frame to an existing track) is a separate thing from the overlap-ratio checks described below; the two shouldn't be confused, they solve different problems, one links detections across frames, the other checks whether a person's box falls inside a hand-drawn zone.

Two zone-membership methods are used depending on the solution:
- Anchor-point-in-polygon: a single point, the bottom-center of a person's bounding box (an approximation of where their feet are), is tested against a zone polygon. Used for waiting areas and tables.
- Overlap-ratio: what fraction of a person's whole bounding box falls inside a zone polygon. Used for the exclusion zone, set at a threshold of 0.30 (30% of their box must overlap), because people at the host stand are frequently occluded from the waist down by the counter, so a single foot-position point often misses them entirely while their visible upper body clearly overlaps the zone.

Entrance crossings require a track to have persisted for at least 8 frames before it's allowed to register an in/out count, to filter out phantom crossings caused by the tracker losing and re-acquiring a person right at the doorway.

## Total person count in frame

The count excludes anyone whose box overlaps the exclusion zone, so it reflects customers, not staff at the host stand.

Assumption: every visible person in the camera's field of view is detected.

Limitation: this only reflects what one camera can see. It says nothing about the rest of the dining room, kitchen, or restrooms, which are entirely outside this frame. Occlusion between guests standing close together, and the overexposed lamp directly in the middle of the shot, both cause brief missed detections, so this number is best understood as a lower bound, not an exact headcount.

Mitigation: additional cameras covering the rest of the floor would be needed for a true restaurant-wide count. Within this single camera, some gamma or exposure correction on the video before running detection could recover a bit of the lamp's blown-out region, though this wasn't tested given time constraints.

## Waiting area occupancy and dwell time

This works well and reliably reflects people sitting on the bench and sofa near the door.

Assumption: the bottom-center anchor point accurately represents where a person is standing or sitting.

Limitation: if the tracker briefly loses someone, for example when another guest walks in front of them, a new track ID gets created. Dwell time is accumulated per track ID, so a lost-and-reacquired person's wait time resets and understates how long they actually waited.

Mitigation: increasing ByteTrack's track buffer (how long it keeps a lost track alive before giving up on it) would let brief occlusions bridge over without spawning a new ID. A more involved fix would add appearance-based re-identification so a person who briefly leaves and re-enters the frame is matched back to their original ID.

## Table occupancy

Since chair and table detection wasn't usable at this angle, table occupancy is purely zone-based: a polygon hand-drawn around each visible table, tested the same way as the waiting areas.

Assumption: anyone whose anchor point falls inside a table's zone is seated at or standing right next to that table.

Limitation: this can't tell the difference between a guest actually seated and a server standing there for a moment to take an order or clear plates. It's an occupancy proxy, not a real seating detector.

Known failure: a table can briefly flicker between occupied and empty if someone stands up for a second, or gets marked occupied by a passing server.

Mitigation: require a person to stay in a table's zone for a minimum number of seconds before flipping its status, which would smooth out these short, spurious transitions. This wasn't implemented due to time, but would be a quick addition. Longer term, a camera mounted directly above each table, close enough to reliably detect chairs, would allow a proper chair-plus-person occupancy check instead of a pure zone proxy.

## Host stand / exclusion zone

This works reliably now that it uses the overlap-ratio method instead of a single anchor point.

Assumption: anyone whose box substantially overlaps the host stand polygon is either staff or a customer briefly interacting with staff, and either way should not count toward customer-facing metrics like the waiting queue.

Limitation: there is no real staff-versus-customer classification here, it's a deliberate simplification. A customer checking in at the host stand is also excluded from the count during that moment, which is expected and fine for this use case (it avoids double-counting them in the waiting-area metric at the same time), but it should be understood as a simplification rather than a true staff detector.

Mitigation: if a genuine staff/customer distinction becomes a real requirement, that would need labeled footage of staff (uniforms, badges, or a fixed appearance cue) to train a proper classifier. For now the exclusion-zone approach is the right tradeoff given no such labeled data exists.

## Entrance in/out counting

This is the one solution that is not working reliably yet, and it's worth being upfront about why.

Assumption: each unique tracked person crosses the entrance line exactly once in each direction.

Limitation: the doorway has strong backlight from outside daylight, and people cluster right at the threshold, both of which hurt detection and tracking quality at exactly the spot where accuracy matters most.

Known failure: ID switches at the door cause both missed crossings (the tracker loses someone mid-crossing and the "before" and "after" positions belong to two different IDs, so no single ID is ever seen on both sides) and occasional false crossings (a brand new ID appears already on the far side of the line, which looks like a crossing that never actually happened). The 8-frame track-age gate reduces this but doesn't eliminate it.

Mitigation: a camera specifically positioned for the doorway, ideally without the backlight, would help significantly. Increasing the tracker's track buffer so brief losses don't spawn new IDs is a cheap software fix worth trying next. For a genuinely reliable count, a secondary sensor such as an infrared beam counter, or a dedicated overhead camera pointed straight down at the threshold, would give a ground truth to calibrate against. As it stands, this metric should be reported to a client as an estimate with a known margin of error, not treated as an exact count.
