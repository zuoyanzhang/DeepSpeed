# Security Policy

## Reporting a vulnerability

Do not disclose an exploitable vulnerability in a public issue. Report Chorus-specific vulnerabilities privately through this repository's [GitHub security advisory form](https://github.com/zuoyanzhang/Chorus/security/advisories/new).

If the issue also affects unmodified DeepSpeed, report it privately to the [upstream DeepSpeed security channel](https://github.com/deepspeedai/DeepSpeed/security/advisories/new). Non-sensitive correctness and reliability bugs may be filed as normal Chorus issues.

## Scope

Security fixes are provided on a best-effort basis for the maintained `main` branch. Historical research branches are preserved for reproducibility and may not receive fixes.

Chorus executes distributed training code and may download models or datasets through third-party libraries. Review remote code, model, dataset, and checkpoint sources before use; do not run untrusted artifacts with elevated privileges or cluster credentials.
