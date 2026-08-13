# Quantization closed-loop success (pc_success %, 4 suites x 20 episodes)

## CNN (FPGA)

| stage | spatial | object | goal | long | mean | Δ vs fp32 |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 95 | 85 | 80 | 85 | 86.2 |  |
| W8A8 PTQ | 95 | 85 | 85 | 95 | 90.0 | +3.8 |
| W4A6 PTQ (post-quant) | 75 | 35 | 75 | 40 | 56.2 | -30.0 |
| W4A6 QAT (LIBERO) | 90 | 95 | 95 | 100 | 95.0 | +8.8 |
| W4A6 re-distill (DROID mix, 3ep) | 50 | 75 | 70 | 45 | 60.0 | -26.2 |
| W4A6 re-distill + QAT-ft (LIBERO, 3ep) | 65 | 25 | 80 | 20 | 47.5 | -38.8 |

## TinyViT (nano)

| stage | spatial | object | goal | long | mean | Δ vs fp32 |
|---|---|---|---|---|---|---|
| fp32 (baseline) | 90 | 95 | 90 | 95 | 92.5 |  |
| W8A8 PTQ | 95 | 100 | 100 | 95 | 97.5 | +5.0 |
| W4A6 PTQ (post-quant) | 90 | 30 | 70 | 15 | 51.2 | -41.2 |
| W4A6 QAT (LIBERO) | 100 | 100 | 95 | 80 | 93.8 | +1.2 |
| W4A6 re-distill (DROID mix, 3ep) | 75 | 80 | 95 | 75 | 81.2 | -11.2 |
| W4A6 re-distill + QAT-ft (LIBERO, 3ep) | 70 | 35 | 75 | 45 | 56.2 | -36.2 |

