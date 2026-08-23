# VLM Retrieval Robustness

Code, configurations, rank checkpoints, and result tables for the IEEE Access
manuscript **“Bidirectional Retrieval Robustness of Four Vision-Language
Checkpoints Under Natural Distortions and Gallery-Contamination Stress Tests.”**

Repository: <https://github.com/cmony/vlm-retrieval-robustness>

The study evaluates four released checkpoints on bidirectional COCO retrieval:

- `openai/clip-vit-base-patch32`
- `openai/clip-vit-base-patch16`
- `Salesforce/blip-itm-base-coco`
- `google/siglip-base-patch16-224`

The natural-distortion protocol uses the full COCO 2017 validation set (5,000
images and 25,000 captions), 18 distorted settings, metric-first averaging for
the repeated Gaussian-noise and crop conditions, image-level cluster
bootstrap, and one 30-test Holm family for checkpoint contrasts. The adapted
RoCOCO analysis is an exploratory gallery-contamination stress test and is
reported with descriptive point estimates.

## Repository contents

```text
configs/                    Experiment configuration
vlmrobust/                  Retrieval, distortion, analysis, and statistics code
tests/                      CPU-only unit tests
artifacts/ranks/            140 compressed natural-distortion rank checkpoints
artifacts/dataset_index.csv Frozen COCO evaluation order and fingerprint
results/natural/            Corrected aggregate, bootstrap, contrast, and per-image results
results/rococo/             Adapted RoCOCO summary results
manifests/                  Recorded configuration and execution environment
paper/figures/              Figures generated from the released result tables
scripts/                    RoCOCO, figure, and release-validation utilities
```

Model weights and COCO images are not redistributed. They are downloaded from
their original providers when inference is rerun. The 1.2 GB RoCOCO embedding
cache is also omitted because it can be regenerated from the public model
checkpoints and COCO `val2014` images.

## Quick verification without a GPU

Create a Python 3.11 environment, install the lightweight dependencies, and run
the release gate:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python scripts/verify_release.py
python -m unittest discover -s tests -v
```

Expected release-gate sentinels:

```text
aggregate=152
metrics=280
bootstrap=720
contrasts=30
significant=11
rococo=20
rank_checkpoints=140
per_image=720000
figures=5
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for rank-first reanalysis, full
GPU inference, RoCOCO execution, and figure regeneration commands.

## Key statistical definitions

For a cutoff `k`, the clean-success-to-failure rate is conditional on clean
success:

```text
S->F@k = count(clean rank <= k and distorted rank > k)
         / count(clean rank <= k)
```

For stochastic conditions, success indicators and retrieval metrics are
computed separately for each repetition and then averaged. The corrected
aggregate obeys the identity

```text
distorted R@k - clean R@k = (F->S - S->F) / N
```

for all 456 stored condition-direction-cutoff combinations.

## Data and third-party assets

- COCO images and captions remain subject to the original COCO terms.
- RoCOCO annotations are obtained from the original RoCOCO repository by the
  evaluation script and remain subject to the original project terms.
- Hugging Face model checkpoints remain subject to their respective licenses.
- This repository does not redistribute model weights or COCO image files.

## Citation

The final bibliographic record and DOI will be added after acceptance. Until
then, use the metadata in `CITATION.cff`, which identifies this public code
repository.

## License status

No repository license has been selected yet. Before making the repository
public, select a software license consistent with the author's institution and
the included third-party dependencies. MIT is the recommended default for the
original code in this repository.
