# Chorus benchmark and launch tools

This directory is the public entry point for Chorus. It contains a DeepSpeed backend powered by DeepCompile's `global_layer_scheduler` pass and a SimpleFSDP backend powered by PyTorch compiled autograd.

## Requirements

Use Python 3.10 and a CUDA-enabled PyTorch 2.7.x build compatible with your NVIDIA driver. Historical DeepCompile experiments in this tree recorded Python 3.10.12, CUDA 12.4, and PyTorch 2.6.0; the later SimpleFSDP implementation targets PyTorch 2.7 APIs. The unified Chorus extra therefore constrains PyTorch to 2.7.x. Both backends rely on compiler or distributed APIs that changed in later PyTorch releases.

Install the appropriate CUDA-enabled PyTorch build first, following the PyTorch installation instructions for your system. Then, from the repository root:

```bash
python -m pip install -e ".[chorus]"
```

Model access is handled by Transformers. Pass either a Hugging Face model identifier or an explicit local model directory. Gated models require the corresponding Hugging Face authorization.

## Single-server usage (no Slurm)

Run from any working directory:

```bash
# DeepSpeed Chorus
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --gpus 4 \
  --batch-size 2 \
  --seq-length 1024

# SimpleFSDP Chorus
bash examples/chorus/run_server.sh \
  --backend simplefsdp \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --gpus 4 \
  --batch-size 2 \
  --seq-length 1024
```

For a model stored locally:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model-path /models/Mistral-7B-Instruct-v0.3
```

The launcher uses all GPUs visible to PyTorch by default. `CUDA_VISIBLE_DEVICES` is respected. Inspect the exact command without starting a GPU process:

```bash
bash examples/chorus/run_server.sh --backend deepspeed --dry-run
bash examples/chorus/run_server.sh --backend simplefsdp --dry-run
```

Run `bash examples/chorus/run_server.sh --help` for all options. Arguments after `--` are forwarded to the benchmark program.

### Backend expansion

The public launcher deliberately presents Chorus terminology while retaining compatible internal DeepSpeed names:

| Backend | Internal arguments |
| --- | --- |
| `deepspeed` | `--compile --deepcompile --passes global_layer_scheduler` |
| `simplefsdp` | `--compile --simplefsdp-enable-compiled-autograd --simplefsdp-enable-chorus` |

Do not combine `global_layer_scheduler` with the legacy `prefetch` or `selective_gather` passes in the same DeepSpeed schedule.

## Slurm usage

`run_slurm.sbatch` is a portable starting point and contains no cluster-specific module, environment, account, partition, or QoS commands.

Run these commands from the repository root so the Slurm job can resolve the repository after Slurm spools the submission script:

```bash
BACKEND=deepspeed MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  sbatch --nodes=1 --gpus-per-node=4 examples/chorus/run_slurm.sbatch
```

For SimpleFSDP:

```bash
BACKEND=simplefsdp MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  sbatch --nodes=1 --gpus-per-node=4 examples/chorus/run_slurm.sbatch
```

Add site-specific environment initialization in a wrapper or export `CHORUS_ENV_SETUP` with the path of a shell file to source before the run. If submitting from another directory, export `CHORUS_REPO_ROOT=/absolute/path/to/Chorus`.

## Benchmark sweep

`run_benchmark.sh` runs the batch-size and sequence-length values supplied through environment variables:

```bash
BACKEND=deepspeed \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
BATCH_SIZES="1 2 4" \
SEQ_LENGTHS="512 1024 2048" \
bash examples/chorus/run_benchmark.sh
```

Useful environment variables include:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `deepspeed` | `deepspeed` or `simplefsdp` |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face identifier |
| `MODEL_PATH` | empty | Explicit local model directory |
| `BATCH_SIZES` | `2` | Space-separated sweep values |
| `SEQ_LENGTHS` | `1024` | Space-separated sweep values |
| `NUM_NODES` | `1` | Number of nodes for Slurm runs |
| `NGPUS_PER_NODE` | visible GPU count | Processes per node |
| `MASTER_PORT` | `29500` | Distributed rendezvous port |
| `PROFILE_DIR` | `profiles` | Profile output directory |

The bundled German OpenAssistant/Guanaco subset is used by default. Override it with `--dataset-path /path/to/data.jsonl`; JSON and JSONL files are supported. Dataset provenance and licensing are recorded in the repository's third-party notices.

## Outputs

Runtime logs are written to `logs/`. Profiler output is written under `profiles/` when profiling is enabled. The benchmark reports iteration time, estimated compile overhead, and peak allocated CUDA memory, together with backend-specific Chorus diagnostics.

## Code map

| File | Purpose |
| --- | --- |
| `run_server.sh` | Public single-server, multi-GPU entry point |
| `run_slurm.sbatch` | Slurm entry point |
| `run_benchmark.sh` | Benchmark sweep driver |
| `launch.sh` | Single- or multi-node dispatch |
| `launch_worker.sh` | Per-node Accelerate configuration and launch |
| `benchmark.py` | Model, backend, training, and measurement harness |
| `native_simplefsdp.py` | SimpleFSDP and SimpleFSDP Chorus implementation |
| `configs/` | Backend-specific Accelerate and DeepSpeed templates |

DeepSpeed Chorus scheduling is implemented in `deepspeed/compile/passes/global_layer_scheduler.py`. DeepCompile's internal API and configuration names remain unchanged because they are the underlying execution infrastructure, not the public project name.
