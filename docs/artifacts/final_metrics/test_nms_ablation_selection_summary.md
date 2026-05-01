| Method | N | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Pair IoU | Top Combo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mlp_plain_selection | 5553 | 3.000 | 2.000 | 0.407 | 0.423 | 0.238 | 0.213 | 0.183 | segment+segment (5553) |
| mlp_nms0_selection | 5553 | 2.935 | 2.000 | 0.440 | 0.453 | 0.246 | 0.210 | 0.000 | segment+segment (4826) |
| mlp_nms03_selection | 5553 | 2.935 | 2.000 | 0.440 | 0.453 | 0.246 | 0.210 | 0.000 | segment+segment (4826) |
| mlp_nms05_selection | 5553 | 3.000 | 2.000 | 0.407 | 0.423 | 0.238 | 0.213 | 0.183 | segment+segment (5553) |
