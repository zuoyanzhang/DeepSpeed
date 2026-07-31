# Using Chorus

This directory is the public entry point for Chorus. It contains launchers, backend configurations, a training benchmark, and the two Chorus implementations:

| Public backend | Internal implementation selected automatically |
| --- | --- |
| `deepspeed` | `--compile --deepcompile --passes global_layer_scheduler` |
| `simplefsdp` | `--compile --simplefsdp-enable-compiled-autograd --simplefsdp-enable-chorus` |

The commands below support single-node, non-Slurm multi-node, and Slurm multi-node runs. Users should select a backend through `--backend`; the internal flags in the table are documented for reproducibility and do not need to be supplied manually.

## 1. Requirements

Use the following environment on every participating server:

- Linux and NVIDIA GPUs
- Python 3.10
- CUDA-enabled PyTorch 2.7.x
- GPUs with bfloat16 support
- NCCL-capable connectivity between nodes
- For DeepSpeed Chorus, a matching local CUDA toolkit (`CUDA_HOME`) and C++ compiler for its JIT-built extension
- the same Chorus commit and Python dependencies on every node

Historical DeepCompile experiments in this repository used PyTorch 2.6, but the later SimpleFSDP implementation targets PyTorch 2.7 APIs. PyTorch 2.7.x is therefore the supported unified environment for both public backends.

Install a CUDA-enabled PyTorch build compatible with the local NVIDIA driver first. Then run this from the Chorus repository root on every server:

```bash
python -m pip install -e ".[chorus]"
```

Verify each server before starting a distributed run:

```bash
python -c 'import torch; print("torch:", torch.__version__); print("cuda:", torch.cuda.is_available()); print("gpus:", torch.cuda.device_count()); print("bf16:", torch.cuda.is_bf16_supported())'
```

The expected result is PyTorch `2.7.x`, `cuda: True`, a positive GPU count, and `bf16: True`.

## 2. Model and dataset preparation

### Hugging Face model

Pass a model identifier with `--model`:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --gpus 4
```

For gated models, make the corresponding Hugging Face token available on every node and ensure every node can access its model cache.

### Local model

Use `--model-path` for an exact local model directory:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model-path /models/Mistral-7B-Instruct-v0.3 \
  --gpus 4
```

The directory must contain `config.json`, model weights, and tokenizer files. In a multi-node run, that path must exist on every server or point to a shared filesystem.

Relative paths are converted to absolute paths before the worker changes directory. If `--model` is omitted, the local directory basename is used as the model identity in logs and model-specific configuration. Supply both options when a more precise identity is needed:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model my-org/My-Mixtral-Checkpoint \
  --model-path ./checkpoints/checkpoint-1000 \
  --gpus 4
```

### Dataset

The bundled German OpenAssistant/Guanaco subset is used when `--dataset-path` is omitted. To use a local file or directory:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --dataset-path /datasets/train.jsonl \
  --gpus 4
```

JSON and JSONL records must contain a `text` field:

```json
{"text": "Example training text."}
```

When a directory is supplied, Chorus reads top-level `*.json` and `*.jsonl` files from that directory. The path must be available on every node.

## 3. Single-node, multi-GPU usage

`run_server.sh` is the recommended non-Slurm entry point. With no `--gpus` argument, a single-node run uses all GPUs visible to PyTorch.

### DeepSpeed Chorus

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

### SimpleFSDP Chorus

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash examples/chorus/run_server.sh \
  --backend simplefsdp \
  --gpus 4 \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --batch-size 2 \
  --seq-length 1024
```

`CUDA_VISIBLE_DEVICES` selects physical GPUs. `--gpus` selects how many of those visible GPUs become local training processes. For example:

```bash
CUDA_VISIBLE_DEVICES=2,3,6,7 \
bash examples/chorus/run_server.sh --backend deepspeed --gpus 4
```

This launches four processes on physical GPUs 2, 3, 6, and 7. The launcher rejects a requested GPU count larger than the number visible to PyTorch.

## 4. Multi-node, multi-GPU usage without Slurm

The public launcher uses a one-command-per-node model. It does not SSH into other machines. Start the command once on every server, using the same values for:

- `--backend`
- `--nodes`
- `--master-addr`
- `--master-port`
- `--gpus`
- model, dataset, batch size, sequence length, and other training options

Only `--node-rank` changes. Ranks must be unique and cover `0` through `nodes - 1`.

The example below runs two nodes with four GPUs per node. The rank 0 server is reachable at `10.0.0.10`; TCP port `29500` and the cluster's required NCCL peer traffic are permitted between the servers.

### Node rank 0

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

### Node rank 1

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

Use the same commands with `--backend simplefsdp` to run SimpleFSDP Chorus. The public SimpleFSDP launcher currently requires gradient accumulation to remain at its default value of 1.

For this example:

```text
world size = nodes × GPUs per node = 2 × 4 = 8 processes

effective global batch size
  = batch size per process × gradient accumulation steps × world size
  = 2 × 1 × 8
  = 16 samples
```

Before starting the real job, inspect both node commands:

```bash
# Run on node rank 0
bash examples/chorus/run_server.sh \
  --backend deepspeed --nodes 2 --node-rank 0 \
  --master-addr 10.0.0.10 --master-port 29500 --gpus 4 --dry-run

# Run on node rank 1
bash examples/chorus/run_server.sh \
  --backend deepspeed --nodes 2 --node-rank 1 \
  --master-addr 10.0.0.10 --master-port 29500 --gpus 4 --dry-run
```

The output must show identical `NUM_NODES`, `NGPUS_PER_NODE`, `WORLD_SIZE`, `MASTER_ADDR`, and `MASTER_PORT`, with a different `MACHINE_RANK` on each node.

### Multi-node network checklist

- Do not use `localhost`, `127.0.0.1`, `::1`, or `0.0.0.0` as `--master-addr`.
- Confirm every node can resolve or reach the rank 0 address.
- Allow the selected `--master-port` and the required NCCL peer traffic through host and network firewalls.
- Use a different port for concurrent Chorus jobs sharing the same rank 0 host.
- If NCCL selects the wrong interface, export the same interface policy on every node, for example `NCCL_SOCKET_IFNAME=eth0`.
- If any node exits before rendezvous completes, terminate the remaining node processes before retrying.

All nodes must use the same number of GPUs per node. Heterogeneous per-node GPU counts are not supported by this launcher.

## 5. Slurm usage

`run_slurm.sbatch` is a portable template with no cluster-specific account, partition, QoS, module, or environment commands.

Submit from the repository root. DeepSpeed Chorus on two nodes with eight GPUs per node:

```bash
sbatch \
  --nodes=2 \
  --gpus-per-node=8 \
  --export=ALL,BACKEND=deepspeed,MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  examples/chorus/run_slurm.sbatch
```

SimpleFSDP Chorus:

```bash
sbatch \
  --nodes=2 \
  --gpus-per-node=8 \
  --export=ALL,BACKEND=simplefsdp,MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  examples/chorus/run_slurm.sbatch
```

Use `CHORUS_ENV_SETUP=/absolute/path/to/setup.sh` to source site-specific modules or a Python environment inside the allocation. When submitting from outside the repository root, export `CHORUS_REPO_ROOT=/absolute/path/to/Chorus`.

For example, from any directory:

```bash
export CHORUS_REPO_ROOT=/absolute/path/to/Chorus
sbatch "$CHORUS_REPO_ROOT/examples/chorus/run_slurm.sbatch"
```

The Slurm allocation supplies the node count and GPUs per node. `launch.sh` chooses the first allocated host as the rendezvous address and starts one launcher task per node with `srun`.

## 6. Public launcher options

Run the built-in help at any time:

```bash
bash examples/chorus/run_server.sh --help
```

| Option | Meaning |
| --- | --- |
| `--backend deepspeed\|simplefsdp` | Select the Chorus implementation |
| `--model ID` | Hugging Face model identifier |
| `--model-path DIR` | Exact local model directory |
| `--dataset-path PATH` | Local JSON/JSONL file or directory |
| `--gpus N` | GPU processes per node; required for multi-node runs |
| `--nodes N` | Number of participating servers |
| `--node-rank R` | Unique zero-based rank of this server |
| `--master-addr HOST` | Reachable rank 0 hostname or IP |
| `--master-port PORT` | Distributed rendezvous port |
| `--batch-size N` | Batch size per GPU process |
| `--seq-length N` | Token sequence length |
| `--gradient-accumulation-steps N` | Gradient accumulation count; SimpleFSDP currently requires 1 |
| `--no-activation-checkpointing` | Disable the default activation checkpointing |
| `--random-init` | Initialize from model configuration instead of loading model weights |
| `--profile` | Enable the PyTorch profiler |
| `--profile-dir DIR` | Set profiler output and enable profiling |
| `--dry-run` | Print the resolved distributed environment and command without launching |

The equivalent environment variables `NUM_NODES`, `NGPUS_PER_NODE`, `MACHINE_RANK`, `MASTER_ADDR`, and `MASTER_PORT` are accepted, but the explicit command-line form is recommended for non-Slurm runs. `CHORUS_RUN_ID` may be set to the same short identifier on every node when a custom log/run label is desired.

## 7. Backend behavior and advanced options

### DeepSpeed Chorus

The public `deepspeed` backend enables compilation, DeepCompile, ZeRO stage 3, and `global_layer_scheduler`. Do not combine `global_layer_scheduler` with the legacy `prefetch` or `selective_gather` passes in the same DeepSpeed schedule.

### SimpleFSDP Chorus

The public `simplefsdp` backend enables PyTorch compilation, compiled autograd, and the SimpleFSDP Chorus scheduling implementation. It requires PyTorch 2.7.x and currently requires `--gradient-accumulation-steps 1`.

### Advanced benchmark arguments

Arguments after `--` pass through the launcher. Managed Chorus options such as the backend, compilation pass, model, dataset, batch size, and sequence length cannot be overridden there.

For a smaller randomly initialized model useful for a smoke test:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --gpus 4 \
  --random-init \
  -- --num-layers 8 --attn-impl sdpa
```

DeepSpeed parameter and optimizer CPU offload can be enabled as an advanced launcher argument:

```bash
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --gpus 4 \
  -- --ds-offload
```

## 8. Benchmark sweeps

`run_benchmark.sh` iterates over batch-size and sequence-length lists. A single-node example is:

```bash
BACKEND=deepspeed \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
NGPUS_PER_NODE=4 \
BATCH_SIZES="1 2 4" \
SEQ_LENGTHS="512 1024 2048" \
bash examples/chorus/run_benchmark.sh
```

Useful variables are:

| Variable | Default | Meaning |
| --- | --- | --- |
| `BACKEND` | `deepspeed` | Public Chorus backend |
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face identifier |
| `MODEL_PATH` | empty | Exact local model directory |
| `DATASET_PATH` | bundled dataset | Local JSON/JSONL input |
| `BATCH_SIZES` | `2` | Space-separated per-process batch sizes |
| `SEQ_LENGTHS` | `1024` | Space-separated sequence lengths |
| `GRADIENT_ACCUMULATION_STEPS` | `1` | Gradient accumulation count |
| `NUM_NODES` | `1` | Number of nodes |
| `NGPUS_PER_NODE` | visible GPU count | Processes per node |
| `MACHINE_RANK` | empty | Required unique node rank for a non-Slurm multi-node sweep |
| `MASTER_ADDR` | `127.0.0.1` for one node | Rendezvous address |
| `MASTER_PORT` | `29500` outside Slurm | Base rendezvous port; each sweep case increments it by one |
| `ACTIVATION_CHECKPOINTING` | `1` | Enable activation checkpointing |
| `LOAD_WEIGHTS` | `1` | Load pretrained model weights |
| `PROFILE` | `0` | Set to `1` to enable profiling |
| `PROFILE_DIR` | `profiles` | Profiler output directory |
| `LOG_DIR` | `examples/chorus/logs` | Per-node worker log directory |
| `SRUN_LOG_FILE` | generated path | Aggregated `srun` log path for Slurm |

For a non-Slurm two-node sweep with four GPUs per node, run the first command on rank 0:

```bash
NUM_NODES=2 \
NGPUS_PER_NODE=4 \
MACHINE_RANK=0 \
MASTER_ADDR=10.0.0.10 \
MASTER_PORT=29500 \
BACKEND=deepspeed \
BATCH_SIZES="1 2" \
SEQ_LENGTHS="512 1024" \
bash examples/chorus/run_benchmark.sh
```

Run the matching command on rank 1:

```bash
NUM_NODES=2 \
NGPUS_PER_NODE=4 \
MACHINE_RANK=1 \
MASTER_ADDR=10.0.0.10 \
MASTER_PORT=29500 \
BACKEND=deepspeed \
BATCH_SIZES="1 2" \
SEQ_LENGTHS="512 1024" \
bash examples/chorus/run_benchmark.sh
```

The sweep lists and their order must match on every node. Each combination uses a different port starting at `MASTER_PORT`, preventing consecutive distributed runs from reusing a rendezvous endpoint. Permit the full range from `MASTER_PORT` through `MASTER_PORT + number of configurations - 1` in the firewall. For routine multi-node runs, `run_server.sh` is simpler and safer.

## 9. Outputs

- Per-node worker logs are written under `examples/chorus/logs/` unless `LOG_DIR` is set. Their names include the run or Slurm job ID and machine rank.
- Slurm writes combined batch stdout/stderr to `chorus-<job-id>.out.log` in the submission directory.
- `launch.sh` also records aggregated `srun` output in `examples/chorus/logs/<job-name>-<job-id>.out.log` unless `SRUN_LOG_FILE` is set.
- Profiler traces are written under `examples/chorus/profiles/` by default when profiling is enabled. `result.txt` and `compile_time.txt` are appended in the selected profile directory.
- Each launcher prints its machine rank, global process count, model, runtime configuration directory, and log path.
- The benchmark reports steady-state iteration time, estimated compile overhead, and peak allocated CUDA memory, along with backend-specific Chorus diagnostics.

Temporary configuration files created by `run_server.sh` are private to that node and removed when the launcher exits.

This harness normally stops after approximately 11 optimizer-update windows. Profiling may run a few additional microsteps so the profiler can flush its trace. It is a short Chorus performance and memory benchmark and does not save a model checkpoint.

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| PyTorch version error | Use CUDA-enabled PyTorch 2.7.x on every node |
| DeepSpeed extension build fails | Check `CUDA_HOME`, the local CUDA toolkit version, and the C++ compiler |
| `CUDA is not available` | Check the NVIDIA driver, CUDA PyTorch build, and `CUDA_VISIBLE_DEVICES` |
| bfloat16 error | Use a GPU architecture with bfloat16 support |
| Requested GPUs exceed visible GPUs | Reduce `--gpus` or correct `CUDA_VISIBLE_DEVICES` |
| Model download returns 401/403 | Provide model access credentials on every node |
| Local model error | Check `config.json`, weights, tokenizer files, and the path on every node |
| Dataset error | Ensure the path exists everywhere and each record has a `text` field |
| Multi-node job waits forever | Check unique ranks, identical node/GPU counts, rank 0 address, port, and firewall |
| NCCL connects on the wrong network | Set `NCCL_DEBUG=INFO` and an appropriate `NCCL_SOCKET_IFNAME` on every node |
| Address already in use | Choose a different `--master-port` and update every node command |
| First iteration is slow | Compilation and kernel generation occur during warm-up |
| CUDA out of memory | Reduce `--batch-size` or `--seq-length`; keep activation checkpointing enabled |

## 11. Code map

| File | Purpose |
| --- | --- |
| `run_server.sh` | Public non-Slurm single-node and multi-node entry point |
| `run_slurm.sbatch` | Portable Slurm entry point |
| `run_benchmark.sh` | Benchmark sweep driver |
| `launch.sh` | Single-node, non-Slurm multi-node, or Slurm dispatch |
| `launch_worker.sh` | Per-node Accelerate configuration and process launch |
| `generate_config.py` | Runtime Accelerate and DeepSpeed configuration rendering |
| `benchmark.py` | Model, data, training, and measurement harness |
| `native_simplefsdp.py` | SimpleFSDP runtime and Chorus scheduling implementation |
| `configs/` | Backend-specific configuration templates |

DeepSpeed Chorus scheduling is implemented in `deepspeed/compile/passes/global_layer_scheduler.py`. Internal DeepCompile API and configuration names remain unchanged because they are the underlying execution infrastructure rather than the public project name.
