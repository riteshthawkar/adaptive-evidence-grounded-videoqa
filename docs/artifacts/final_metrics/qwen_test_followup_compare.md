| Method | Acc@QA | Cost | Count | mIoP | IoP@0.5 | mIoU | IoU@0.5 | Acc@GQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qwen_fixed_segment1 | 0.668 | 1.500 | 1.000 | 0.301 | 0.311 | 0.165 | 0.133 | 0.221 |
| qwen_linear_min2 | 0.696 | 2.795 | 2.000 | 0.403 | 0.410 | 0.206 | 0.181 | 0.297 |
| qwen_router_mlp_top2 | 0.687 | 3.000 | 2.000 | 0.407 | 0.423 | 0.238 | 0.213 | 0.301 |
| qwen_router_mlp_top2_nms0 | 0.689 | 2.935 | 2.000 | 0.440 | 0.453 | 0.246 | 0.210 | 0.321 |
| qwen_oracle_iop_top2 | 0.701 | 2.493 | 2.000 | 0.618 | 0.615 | 0.224 | 0.207 | 0.442 |
| qwen_oracle_iop_top3 | 0.710 | 3.793 | 3.000 | 0.618 | 0.615 | 0.278 | 0.268 | 0.448 |
| qwen_fixed_f3_s3 | 0.724 | 7.500 | 6.000 | 0.618 | 0.615 | 0.296 | 0.287 | 0.454 |

### Bootstrap confidence intervals

| Method | Acc@QA 95% CI | mIoP 95% CI | IoP@0.5 95% CI | Acc@GQA 95% CI |
| --- | --- | --- | --- | --- |
| qwen_fixed_segment1 | [0.655, 0.681] | [0.291, 0.311] | [0.298, 0.322] | [0.210, 0.231] |
| qwen_linear_min2 | [0.683, 0.708] | [0.392, 0.413] | [0.398, 0.423] | [0.285, 0.308] |
| qwen_router_mlp_top2 | [0.674, 0.699] | [0.397, 0.418] | [0.410, 0.435] | [0.290, 0.313] |
| qwen_router_mlp_top2_nms0 | [0.676, 0.701] | [0.430, 0.451] | [0.442, 0.467] | [0.309, 0.333] |
| qwen_oracle_iop_top2 | [0.688, 0.713] | [0.607, 0.629] | [0.602, 0.627] | [0.429, 0.455] |
| qwen_oracle_iop_top3 | [0.698, 0.723] | [0.607, 0.629] | [0.602, 0.627] | [0.435, 0.461] |
| qwen_fixed_f3_s3 | [0.711, 0.735] | [0.607, 0.629] | [0.602, 0.627] | [0.441, 0.467] |

### Question-type breakdown

| Method | Question type | N | Acc@QA | mIoP | IoP@0.5 | Acc@GQA |
| --- | --- | --- | --- | --- | --- | --- |
| qwen_fixed_segment1 | CH | 796 | 0.662 | 0.332 | 0.343 | 0.246 |
| qwen_fixed_segment1 | CW | 2456 | 0.688 | 0.334 | 0.346 | 0.247 |
| qwen_fixed_segment1 | TC | 852 | 0.671 | 0.300 | 0.312 | 0.218 |
| qwen_fixed_segment1 | TN | 1356 | 0.634 | 0.227 | 0.231 | 0.163 |
| qwen_fixed_segment1 | TP | 93 | 0.624 | 0.256 | 0.258 | 0.172 |
| qwen_linear_min2 | CH | 796 | 0.693 | 0.436 | 0.438 | 0.323 |
| qwen_linear_min2 | CW | 2456 | 0.715 | 0.441 | 0.453 | 0.331 |
| qwen_linear_min2 | TC | 852 | 0.688 | 0.400 | 0.408 | 0.296 |
| qwen_linear_min2 | TN | 1356 | 0.667 | 0.322 | 0.324 | 0.227 |
| qwen_linear_min2 | TP | 93 | 0.710 | 0.317 | 0.312 | 0.226 |
| qwen_router_mlp_top2 | CH | 796 | 0.688 | 0.442 | 0.460 | 0.324 |
| qwen_router_mlp_top2 | CW | 2456 | 0.704 | 0.448 | 0.469 | 0.339 |
| qwen_router_mlp_top2 | TC | 852 | 0.674 | 0.392 | 0.405 | 0.279 |
| qwen_router_mlp_top2 | TN | 1356 | 0.664 | 0.330 | 0.338 | 0.243 |
| qwen_router_mlp_top2 | TP | 93 | 0.677 | 0.303 | 0.280 | 0.183 |
| qwen_router_mlp_top2_nms0 | CH | 796 | 0.692 | 0.469 | 0.487 | 0.347 |
| qwen_router_mlp_top2_nms0 | CW | 2456 | 0.710 | 0.481 | 0.498 | 0.362 |
| qwen_router_mlp_top2_nms0 | TC | 852 | 0.664 | 0.426 | 0.437 | 0.293 |
| qwen_router_mlp_top2_nms0 | TN | 1356 | 0.666 | 0.363 | 0.370 | 0.255 |
| qwen_router_mlp_top2_nms0 | TP | 93 | 0.656 | 0.382 | 0.366 | 0.258 |
| qwen_oracle_iop_top2 | CH | 796 | 0.710 | 0.648 | 0.651 | 0.476 |
| qwen_oracle_iop_top2 | CW | 2456 | 0.724 | 0.660 | 0.659 | 0.481 |
| qwen_oracle_iop_top2 | TC | 852 | 0.681 | 0.585 | 0.573 | 0.398 |
| qwen_oracle_iop_top2 | TN | 1356 | 0.668 | 0.549 | 0.542 | 0.383 |
| qwen_oracle_iop_top2 | TP | 93 | 0.688 | 0.585 | 0.581 | 0.398 |
| qwen_oracle_iop_top3 | CH | 796 | 0.715 | 0.648 | 0.651 | 0.484 |
| qwen_oracle_iop_top3 | CW | 2456 | 0.729 | 0.660 | 0.659 | 0.486 |
| qwen_oracle_iop_top3 | TC | 852 | 0.696 | 0.585 | 0.573 | 0.406 |
| qwen_oracle_iop_top3 | TN | 1356 | 0.684 | 0.549 | 0.542 | 0.388 |
| qwen_oracle_iop_top3 | TP | 93 | 0.688 | 0.585 | 0.581 | 0.419 |
| qwen_fixed_f3_s3 | CH | 796 | 0.721 | 0.648 | 0.651 | 0.482 |
| qwen_fixed_f3_s3 | CW | 2456 | 0.735 | 0.660 | 0.659 | 0.487 |
| qwen_fixed_f3_s3 | TC | 852 | 0.712 | 0.585 | 0.573 | 0.413 |
| qwen_fixed_f3_s3 | TN | 1356 | 0.711 | 0.549 | 0.542 | 0.407 |
| qwen_fixed_f3_s3 | TP | 93 | 0.720 | 0.585 | 0.581 | 0.409 |
