# Zephyr Runtime (Local Copy)

This directory is a local copy of Zephyr container tooling for running validation jobs from this repository.

Default image in this repo copy is set to `sygaldry/zephyr:sglang` so SGLang tooling is available by default.

## Quick checks

```bash
infra/zephyr/container/launch_container.sh --entrypoint=verify-spack.sh
```

## Run a background job

```bash
infra/zephyr/tools/zephyr_job run \
  --project-id mardia-qwen3 \
  --job qwen3-vl-edu-smoke \
  --project-root infra/zephyr \
  -- "cd /workspace && python -m qwen3_vl.validation.run_educational_smoke"
```

## Inspect status/logs

```bash
infra/zephyr/tools/zephyr_job status --project-id mardia-qwen3 --job qwen3-vl-edu-smoke --project-root infra/zephyr
infra/zephyr/tools/zephyr_job tail --project-id mardia-qwen3 --job qwen3-vl-edu-smoke --project-root infra/zephyr --lines 80
```
