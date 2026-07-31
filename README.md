# Chorus

Chorus is a compiler-guided global scheduling system for memory-efficient distributed LLM training. This repository provides a DeepSpeed ZeRO backend built on DeepCompile and a PyTorch SimpleFSDP backend built around compiled autograd.

> This repository is a research fork of [DeepSpeed](https://github.com/deepspeedai/DeepSpeed). It is not an official DeepSpeed release.

## Chorus backends

| Public backend | Chorus implementation enabled by the launcher |
| --- | --- |
| `--backend deepspeed` | DeepSpeed ZeRO-3 compilation with `global_layer_scheduler` |
| `--backend simplefsdp` | SimpleFSDP compiled autograd with Chorus scheduling |

DeepCompile remains the internal execution infrastructure for the DeepSpeed backend. The `deepspeed` Python package, `deepspeed/compile`, and the internal `--deepcompile` option retain their names for compatibility; Chorus is the public project and scheduling system.

The public commands run the paper's short reproducibility benchmark. The benchmark stops after approximately 11 optimizer-update windows and reports timing and memory metrics; it does not save a training checkpoint or act as a general-purpose fine-tuning CLI.

## Requirements

- Linux with NVIDIA GPUs and NCCL-capable networking
- Python 3.10
- CUDA-enabled PyTorch 2.7.x for the unified DeepSpeed and SimpleFSDP setup
- GPUs with bfloat16 support
- For the DeepSpeed backend, a matching local CUDA toolkit (`CUDA_HOME`) and C++ build toolchain for JIT compilation
- The same Chorus commit, Python environment, model, and dataset on every node

Install a CUDA-enabled PyTorch build suitable for the NVIDIA driver on every server. Then clone and install Chorus:

```bash
git clone https://github.com/zuoyanzhang/Chorus.git
cd Chorus

python -m pip install -e ".[chorus]"
```

Verify the environment before launching a distributed job:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("gpus:", torch.cuda.device_count()); print("bf16:", torch.cuda.is_bf16_supported())'
```

## Single-server quick start

`run_server.sh` does not require Slurm. It uses every GPU visible to PyTorch by default. Use `CUDA_VISIBLE_DEVICES` to select physical GPUs and `--gpus` to choose how many visible GPUs become training processes.

DeepSpeed Chorus on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

SimpleFSDP Chorus on four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend simplefsdp \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

Use `--model-path /absolute/path/to/model` for a local model and `--dataset-path /absolute/path/to/data.jsonl` for a local dataset. The public SimpleFSDP launcher currently requires `--gradient-accumulation-steps 1`.

Inspect the resolved distributed settings and per-node worker command without loading a model or starting GPU processes:

```bash
bash examples/chorus/run_server.sh --backend deepspeed --gpus 4 --dry-run
```

## Multi-node, multi-GPU without Slurm

Run `run_server.sh` once on every server. All arguments must be identical except `--node-rank`, which must be unique in the range `0` to `nodes - 1`. `--master-addr` is a hostname or IP address of rank 0 that every other node can reach.

For two servers with four GPUs each, assume rank 0 is reachable at `10.0.0.10`.

On node rank 0:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --nodes 2 \
  --node-rank 0 \
  --master-addr 10.0.0.10 \
  --master-port 29500 \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

On node rank 1:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --nodes 2 \
  --node-rank 1 \
  --master-addr 10.0.0.10 \
  --master-port 29500 \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

To run the SimpleFSDP implementation, use the same pair of commands with `--backend simplefsdp`.

In this example, the distributed world size is `2 nodes × 4 GPUs = 8` processes. The effective global batch size is:

```text
batch size per process × gradient accumulation steps × nodes × GPUs per node
```

The rank 0 rendezvous port and the cluster's required NCCL peer traffic must be permitted between servers. On multi-interface hosts, set `NCCL_SOCKET_IFNAME` consistently before both commands. Local model and dataset paths must exist on every node, either through a shared filesystem or replicated files.

## Slurm

Submit from the repository root. For example, DeepSpeed Chorus on two nodes with eight GPUs per node:

```bash
sbatch \
  --nodes=2 \
  --gpus-per-node=8 \
  --export=ALL,BACKEND=deepspeed,MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  examples/chorus/run_slurm.sbatch
```

Use `BACKEND=simplefsdp` for the SimpleFSDP implementation. The Slurm template deliberately contains no cluster-specific account, partition, QoS, module, or Conda settings.

## Full usage guide

See [`examples/chorus/README.md`](examples/chorus/README.md) for:

- model and dataset preparation
- all public launcher options
- detailed non-Slurm and Slurm commands
- benchmark sweeps and advanced pass-through options
- output locations and troubleshooting

Run `bash examples/chorus/run_server.sh --help` for the command-line reference.

## Repository layout

| Path | Purpose |
| --- | --- |
| `examples/chorus/` | Public launchers, benchmark harness, configurations, and documentation |
| `deepspeed/compile/passes/global_layer_scheduler.py` | DeepSpeed Chorus global planner and graph-rewrite pass |
| `examples/chorus/native_simplefsdp.py` | SimpleFSDP runtime and Chorus scheduling integration |
| `deepspeed/compile/` | DeepCompile infrastructure used by the DeepSpeed backend |
| `csrc/compile/` | Native DeepCompile runtime support |

## Versions and branches

| Ref | Meaning |
| --- | --- |
| `main` | Maintained Chorus implementation with the DeepSpeed and SimpleFSDP backends |
| `chorus-v0.1` | Preserved historical implementation corresponding to the original paper code |
| `memory` | Preserved follow-on development branch that introduced SimpleFSDP |

Neither historical branch is rewritten by `main`. For an immutable paper artifact, cite a release tag created from the exact paper revision rather than a moving branch.

## Citation

The accepted paper's final title, author order, venue, and persistent URL are not stored in the source history, so they are intentionally not guessed here. Add the camera-ready metadata in `CITATION.cff` before creating the archival paper release.

## License and attribution

Chorus retains the upstream Apache-2.0 license and original copyright headers. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
