# Recovered training release

This release contains the reproducible two-stage Jittor training pipeline recovered
and completed on 2026-08-03.

## Metrics

| Artifact | DOTA-v1.0 test mAP50 | Local validation mAP50 |
|---|---:|---:|
| Stage 1 Point2RBox-v2 | 0.4895466803 | 0.5452640912 |
| Stage 2 pseudo-label Rotated FCOS | 0.5939490341 | 0.6750881105 |

Weights and correctly packaged DOTA Task1 submissions are hosted in the
[Hugging Face model repository](https://huggingface.co/Mingqian-233/Point2RBox-v2-jittor).

## Runtime

- Python 3.10
- Jittor 1.3.8.5
- NumPy 1.26.4
- Jittor bundled CUDA 11.2
- g++-10 (`cc_path=/usr/bin/g++-10`)

The recovered DOTA patches use names like `name__1024__x___y`; the result merger
therefore treats the middle field as tile size rather than scale. Submission files
use the official `Task1_<class>.txt` names.
