# Contributing to Chorus

Chorus is a research fork of DeepSpeed. Issues and pull requests about the Chorus global scheduler, its DeepSpeed or SimpleFSDP integrations, and the launch or benchmark tools are welcome in this repository.

Before opening a change:

1. Search existing Chorus issues and describe the backend, model, GPU type, Python, CUDA, and PyTorch versions involved.
2. For a substantial feature, open an issue first so its scope and evaluation plan can be discussed.
3. Keep compatibility-oriented internal names such as `deepspeed`, `deepcompile`, and `deepspeed/compile` unless the change deliberately updates the underlying API.
4. Add focused tests or a reproducible benchmark command, and report correctness as well as performance or memory results.

Install development dependencies and run the checks relevant to the files you change. At minimum, run formatting and targeted unit tests; GPU compiler/runtime changes should also be exercised on a supported CUDA-enabled PyTorch 2.6 or 2.7 environment. Changes to the public Chorus launchers should test both `--help` and `--dry-run`.

General DeepSpeed bugs that reproduce on the unmodified upstream project should be reported to [DeepSpeed](https://github.com/deepspeedai/DeepSpeed). Contributions made here are reviewed and maintained by the Chorus repository maintainers, not by Microsoft or the upstream DeepSpeed team.

By contributing, you confirm that you have the right to submit the work under the repository's applicable licenses. Follow the [Code of Conduct](CODE_OF_CONDUCT.md).
