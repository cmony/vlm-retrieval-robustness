# Released result sentinels

The committed outputs correspond to the corrected metric-first reanalysis.

## Natural distortions

- Aggregate rows: 152
- Repetition-level metric rows: 280
- Bootstrap rows: 720
- Per-image rows: 720,000 (`per_image_rank_shift.csv.gz`)
- Stored natural rank checkpoints: 140
- Transition identities checked: 456

Corrected stochastic sentinels:

- SigLIP crop I-to-T mean change in R@1: `-0.052950`
- BLIP crop I-to-T mean change in R@1: `-0.034850`
- CLIP ViT-B/32 crop T-to-I mean change in R@1: `-0.040573`
- CLIP ViT-B/16 noise T-to-I mean change in R@1: `-0.021577`

## Checkpoint contrasts

- Six checkpoint pairs x five families: 30 contrasts
- Bootstrap repetitions: 5,000
- Bootstrap seed: 2026
- Holm family size: 30
- Holm reporting floor: `2/5001 x 30 = 0.011997600479904018`
- Significant contrasts at 0.05: 11

## Adapted RoCOCO stress test

- Result rows: 20
- Caption-injection conditions: four per checkpoint
- Image mixing: `0.9 I + 0.1 I'`, seeds 1, 10, and 100
- Results are descriptive point estimates without bootstrap intervals

