# Chorus

Chorus is a compiler-guided global scheduling system for memory-efficient distributed LLM training. This repository provides a DeepSpeed ZeRO backend built on DeepCompile and a PyTorch SimpleFSDP backend built around compiled autograd.

> This repository is a research fork of [DeepSpeed](https://github.com/deepspeedai/DeepSpeed). It is not an official DeepSpeed release.

## Implementations

| Chorus backend | Public launcher | Internal implementation |
| --- | --- | --- |
| DeepSpeed | `--backend deepspeed` | `--compile --deepcompile --passes global_layer_scheduler` |
| SimpleFSDP | `--backend simplefsdp` | `--compile --simplefsdp-enable-compiled-autograd --simplefsdp-enable-chorus` |

DeepCompile remains part of the internal DeepSpeed implementation. Chorus is the project and scheduling system exposed by this repository; `deepspeed/compile`, the `--deepcompile` switch, and related configuration keys retain their upstream names for compatibility.

## Quick start on a GPU server

The public server launcher does not require Slurm. It automatically uses all GPUs visible to PyTorch unless `--gpus` is specified. Use Python 3.10 and install a CUDA-enabled PyTorch 2.7.x build appropriate for your system before installing Chorus; the unified extra keeps PyTorch on that supported release line.

```bash
git clone https://github.com/zuoyanzhang/Chorus.git
cd Chorus

python -m pip install -e ".[chorus]"

# DeepSpeed Chorus
bash examples/chorus/run_server.sh \
  --backend deepspeed \
  --model mistralai/Mistral-7B-Instruct-v0.3

# SimpleFSDP Chorus
bash examples/chorus/run_server.sh \
  --backend simplefsdp \
  --model mistralai/Mistral-7B-Instruct-v0.3
```

Use `--model-path /path/to/model` for a local model directory, `--gpus N` to select the process count, and `--dry-run` to inspect the generated command without launching training. See [`examples/chorus/README.md`](examples/chorus/README.md) for the full interface and environment notes.

## Slurm

The Slurm entry point is:

```bash
sbatch examples/chorus/run_slurm.sbatch
```

Run that command from the repository root. The Slurm file intentionally contains no site-specific module, Conda, account, partition, or QoS settings. Supply those settings at submission time or in your own wrapper.

## Repository layout

| Path | Purpose |
| --- | --- |
| `examples/chorus/` | Public launchers, benchmark harness, configurations, and plotting tools |
| `deepspeed/compile/passes/global_layer_scheduler.py` | DeepSpeed Chorus global planning and graph-rewrite pass |
| `examples/chorus/native_simplefsdp.py` | SimpleFSDP runtime and Chorus scheduling integration |
| `deepspeed/compile/` | DeepCompile infrastructure used by the DeepSpeed backend |
| `csrc/compile/` | Native DeepCompile runtime support |

## Versions and branches

| Ref | Meaning |
| --- | --- |
| `main` | Maintained Chorus implementation containing the DeepSpeed and SimpleFSDP backends |
| `chorus-v0.1` | Preserved historical implementation corresponding to the original paper code |
| `memory` | Preserved follow-on development branch that introduced the SimpleFSDP implementation |

Neither historical branch is rewritten by the `main` branch. For an immutable paper artifact, cite a release tag created from the exact paper revision rather than a moving branch.

## Relationship to DeepSpeed

Chorus is distributed as a modified research fork because its DeepSpeed backend changes compiler passes, runtime hooks, configuration, and native operators across the DeepSpeed source tree. The Python package therefore remains `deepspeed`, and installation continues to use the DeepSpeed build system.

The repository retains the upstream Apache-2.0 license and original copyright headers. Additional attribution for adapted third-party code is recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Citation

The accepted paper's final title, author order, venue, and persistent URL are not stored in the source history, so they are intentionally not guessed here. Add the camera-ready metadata in `CITATION.cff` before creating the archival paper release.

## License

See [`LICENSE`](LICENSE). Upstream and third-party attribution is described above and in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
