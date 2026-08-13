# QAT-fix closed-loop success (pc_success %, up to 4 suites x 20 ep)

Winner to beat: **cnn 95.0 / tinyvit 93.8** (W4A6 capped-QAT).

## CNN (FPGA)

| arm | spatial | object | goal | long | mean | Δ vs winner |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 95 | 85 | 80 | 85 | 86.2 | -8.8 |
| W8A8 PTQ | 95 | 85 | 85 | 95 | 90.0 | -5.0 |
| W4A6 PTQ | 75 | 35 | 75 | 40 | 56.2 | -38.8 |
| W4A6 QAT (WINNER) | 90 | 95 | 95 | 100 | 95.0 |  |
| qatfix_ctrl | 95 | 95 | 90 | 95 | 93.8 | -1.2 |
| qatfix_ds | 95 | 100 | 80 | 100 | 93.8 | -1.2 |
| qatfix_dsmild | 100 | 100 | 85 | 90 | 93.8 | -1.2 |
| qatfix_mild | 95 | 95 | 90 | 100 | 95.0 | +0.0 |

## TinyViT (nano)

| arm | spatial | object | goal | long | mean | Δ vs winner |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 90 | 95 | 90 | 95 | 92.5 | -1.2 |
| W8A8 PTQ | 95 | 100 | 100 | 95 | 97.5 | +3.8 |
| W4A6 PTQ | 90 | 30 | 70 | 15 | 51.2 | -42.5 |
| W4A6 QAT (WINNER) | 100 | 100 | 95 | 80 | 93.8 |  |
| qatfix_ctrl | 80 | 100 | 95 | 85 | 90.0 | -3.8 |
| qatfix_ds | 100 | 100 | 100 | 95 | 98.8 | +5.0 |
| qatfix_dsmild | 100 | 95 | 100 | 85 | 95.0 | +1.2 |
| qatfix_mild | 95 | 100 | 95 | 80 | 92.5 | -1.2 |

