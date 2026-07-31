# Disabled upstream workflows

These workflow definitions come from the upstream DeepSpeed repository and are retained here for source-history reference only. They have been moved out of `.github/workflows/` so GitHub does not execute DeepSpeed's scheduled hardware tests, issue automation, or PyPI release pipeline in the Chorus research fork.

Do not move `release.yml` back into `.github/workflows/`: it builds and publishes the `deepspeed` package when version-like tags are created. Chorus releases and archival paper tags must not trigger that upstream publication workflow.

Future Chorus-specific continuous integration should be introduced as a new, narrowly scoped workflow after its runner requirements and permissions have been reviewed.
