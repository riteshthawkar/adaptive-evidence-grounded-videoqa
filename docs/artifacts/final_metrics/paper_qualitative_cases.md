# Paper Qualitative Cases

## Diversity Removes Redundant Evidence

**Example:** `nextgqa:5600915537:9`; type `TN`; target `8.2-11.8s, 33.1-37.2s`

**Question:** what does the man do after stopping half way

**Options:** A. exit stage; B. adjust the tricycle; C. changes direction; D. lower the camera on his shirt; E. jumps to cross the water

**Gold:** C. changes direction

- Linear min-2: pred C, correct 1, IoP 0.000, IoU 0.000, evidence S19@39.2-41.7s, F30@40.2s
- MLP top-2: pred C, correct 1, IoP 0.000, IoU 0.000, evidence S19@39.2-41.7s, S18@37.2-41.2s
- MLP top-2 + NMS: pred C, correct 1, IoP 1.000, IoU 0.976, evidence S19@39.2-41.7s, S16@33.2-37.2s
- Oracle IoP top-2: pred C, correct 1, IoP 1.000, IoU 0.976, evidence F26@35.2s, S16@33.2-37.2s
- Fixed 3+3: pred C, correct 1, IoP 1.000, IoU 0.976, evidence F30@40.2s, S19@39.2-41.7s, F26@35.2s, F3@5.2s, S18@37.2-41.2s, S16@33.2-37.2s

**Takeaway:** NMS keeps the answer correct while replacing a redundant nearby segment with temporally grounded evidence.

## Oracle Exposes the Router Gap

**Example:** `nextgqa:5139599690:4`; type `TC`; target `0.0-5.0s, 17.9-21.0s`

**Question:** what does the man in black do as the boy was cutting the wood

**Options:** A. turn away; B. stretch his hands; C. move his hand over the ground; D. smile into the camera; E. bend over and look

**Gold:** E. bend over and look

- Linear min-2: pred C, correct 0, IoP 0.075, IoU 0.034, evidence S2@4.7-8.7s, S3@6.7-10.7s
- MLP top-2: pred A, correct 0, IoP 0.000, IoU 0.000, evidence S5@10.7-14.7s, S3@6.7-10.7s
- MLP top-2 + NMS: pred A, correct 0, IoP 0.000, IoU 0.000, evidence S5@10.7-14.7s, S3@6.7-10.7s
- Oracle IoP top-2: pred E, correct 1, IoP 1.000, IoU 0.000, evidence F6@4.7s, F0@0.7s
- Fixed 3+3: pred E, correct 1, IoP 1.000, IoU 0.034, evidence S2@4.7-8.7s, S3@6.7-10.7s, S5@10.7-14.7s, F10@7.2s, F6@4.7s, F0@0.7s

**Takeaway:** The retrieved pool already contains compact supporting evidence, but the learned router does not always select it.

## High-Coverage Evidence Still Helps

**Example:** `nextgqa:2400084970:2`; type `CW`; target `7.3-11.0s`

**Question:** why did the lady in green cover her mouth and bend down in the middle

**Options:** A. laughing; B. tie shoelace; C. adjust shoes; D. focused on the ground; E. attract dog's attention

**Gold:** A. laughing

- Linear min-2: pred D, correct 0, IoP 0.850, IoU 0.791, evidence S3@6.7-10.7s, F16@11.2s
- MLP top-2: pred D, correct 0, IoP 0.850, IoU 0.791, evidence S4@8.7-12.7s, S3@6.7-10.7s
- MLP top-2 + NMS: pred D, correct 0, IoP 0.575, IoU 0.426, evidence S4@8.7-12.7s, S2@4.7-8.7s
- Oracle IoP top-2: pred D, correct 0, IoP 1.000, IoU 0.791, evidence F11@7.7s, S3@6.7-10.7s
- Fixed 3+3: pred A, correct 1, IoP 1.000, IoU 0.791, evidence F16@11.2s, S3@6.7-10.7s, F10@7.2s, S2@4.7-8.7s, S4@8.7-12.7s, F11@7.7s

**Takeaway:** Some questions still need broader visual context or redundant views even when compact oracle evidence overlaps the target span.

## Compact Oracle Nearly Matches Fixed Budget

**Example:** `nextgqa:7064920441:8`; type `TN`; target `62.7-70.6s`

**Question:** how does the baby react after the man in green shows his phone

**Options:** A. lean forward to look; B. throws it at the centre of the table; C. turns to face the green ball; D. vomits; E. walk around the table

**Gold:** A. lean forward to look

- Linear min-2: pred A, correct 1, IoP 0.000, IoU 0.000, evidence S24@50.2-54.2s, F6@15.2s
- MLP top-2: pred A, correct 1, IoP 0.000, IoU 0.000, evidence S24@50.2-54.2s, S23@48.2-52.2s
- MLP top-2 + NMS: pred A, correct 1, IoP 0.000, IoU 0.000, evidence S24@50.2-54.2s, S7@16.2-20.2s
- Oracle IoP top-2: pred A, correct 1, IoP 1.000, IoU 0.000, evidence F30@68.7s, S24@50.2-54.2s
- Fixed 3+3: pred A, correct 1, IoP 1.000, IoU 0.000, evidence S24@50.2-54.2s, S23@48.2-52.2s, S7@16.2-20.2s, F6@15.2s, F22@50.7s, F30@68.7s

**Takeaway:** Two well-chosen items can be enough; the large fixed budget mainly compensates for imperfect learned selection.
