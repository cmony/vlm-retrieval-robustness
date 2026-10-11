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
- Relevance-policy decomposition: 24 rows (16 caption, 4 image mixing, 4 image
  mixing excluding self-paired images)
- Exact-item R@1 counts only original ground truth; source-aware R@1 also
  counts injected candidates derived from the query's own ground truth
- Image-mixing source-aware change: +0.0067 (CLIP ViT-B/32), +0.0078
  (CLIP ViT-B/16), -0.0014 (BLIP), -0.0016 (SigLIP)
- Self-paired images per seed: 75, 71, 82 (seeds 1, 10, 100); only BLIP
  produces identical original and mixed embeddings, exactly on these images

## Exposure audit (BLIP Karpathy-train overlap)

- Exposed / unexposed images: 3,799 / 1,201
- Queries restricted to each subset; gallery is the full evaluation set
- BLIP clean R@1 gap (exposed - unexposed): +0.0648 image-to-text,
  +0.0626 text-to-image; all control-checkpoint intervals include zero
- Bootstrap repetitions: 5,000; seed 2026; resampling within each subset

