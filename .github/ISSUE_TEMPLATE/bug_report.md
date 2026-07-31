---
name: Chorus bug report
about: Report a reproducible problem with a Chorus backend or benchmark
title: "[BUG] "
labels: bug
assignees: ''
---

## Problem

Describe what happened and what you expected instead.

## Reproduction

Provide the complete `run_server.sh` or Slurm command, including whether the `deepspeed` or `simplefsdp` backend was selected. Remove credentials and private model paths.

## Environment

- Chorus commit:
- Backend:
- Python version:
- PyTorch version:
- CUDA version:
- GPU model and count:
- Interconnect/topology:
- Transformers, Accelerate, and SciPy versions:

For DeepSpeed backend problems, also attach the output of `ds_report`.

## Logs

Attach the relevant `examples/chorus/logs/` output and the earliest complete traceback. Do not attach model weights or private datasets.
