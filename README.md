# eScriptorium with Intel ARC

Custom container recipes for [eScriptorium](https://gitlab.com/scripta/escriptorium), tracking its official `develop` branch. Includes Kraken 7.1.1, a single-device Intel XPU compatibility adapter, crop geometry repairs, and optional Videm region-training extensions.

This is an independent compatibility build, not an official eScriptorium or Intel release.

## Current status

The recipe was extracted from a working local deployment based on upstream commit `5f17889fe571d8fa25feb5deebec4221d9485d32`. That deployment passed ARC inference and a short training/checkpoint test. Those results do not automatically validate future upstream commits or the newly parameterized public recipe.

The GitHub workflow builds the portable ARC runtime from the pinned public vendor image and checksum-verified Intel packages, then builds a candidate from the latest official `develop` commit. Runtime images are reused by a hash of their recipe and dependencies; candidates reference the exact registry digest. All jobs run on GitHub-hosted runners using the short-lived workflow token.

The first complete registry build is being validated. A published candidate is not a production release: real Intel ARC inference/training and isolated application checks remain a promotion gate.

## Pipeline

1. Fetch the official `develop` HEAD and resolve it to an immutable SHA.
2. Apply compatibility patches in a temporary copy. Stop if upstream code no longer matches.
3. Build or reuse the portable runtime, then skip already published upstream/recipe/runtime combinations.
4. Build with GitHub-hosted runners, run CPU import checks, and push a versioned candidate to GHCR using the workflow's short-lived GitHub token.
5. Validate candidates separately on isolated staging storage and real Intel hardware while the GPU is idle.
6. Promote a tested image only after a fresh production backup and an active-job check.

Schedules are best effort; GitHub can delay scheduled runs or disable schedules after prolonged repository inactivity. Enable Actions notifications for workflow failures. Candidate results appear in the Actions run summary.

No production deployment, ARC test runner, or production credentials are configured by this repository. Public pull requests are not wired to a local runner. Do not attach a persistent production-connected runner to this public repository; use a separately controlled local staging process or isolated private automation.

## Local build

Use a clean Linux checkout, Docker, `dpkg-deb`, and Python 3.12 or newer:

```sh
python3 scripts/prepare-intel-runtime.py
docker build -f Dockerfile.runtime -t escriptorium-arc-runtime:local .
python3 scripts/fetch-upstream.py --commit 5f17889fe571d8fa25feb5deebec4221d9485d32
python3 scripts/check-source.py
docker build --build-arg ARC_RUNTIME_IMAGE=escriptorium-arc-runtime:local --build-arg UPSTREAM_COMMIT=5f17889fe571d8fa25feb5deebec4221d9485d32 -t escriptorium-arc:candidate .
```

The included `Dockerfile.runtime` supplies the eScriptorium Python/runtime system dependencies, Kraken 7.1.1, Lightning 2.6.1, PyTorch 2.14.0+xpu, torchvision 0.29.0+xpu, dfine-kraken 0.4.3, and compatible Intel user-space libraries. It must contain unmodified Kraken segmentation source for the exact patch to apply. Application dependencies are constrained against the runtime; new upstream dependency requirements require review and an updated runtime.

Runtime registration must work without an attached GPU; GPU execution requires the host's compatible driver/device mapping. The validated WSL worker uses `/dev/dxg` and `/usr/lib/wsl`, single-worker execution, float32 and eager mode. Native Linux device setup is separate and is not validated by the WSL checks.

## Data and configuration

Credentials, databases, document images, model weights, backups and training configuration are excluded. This repository contains code only.

Videm inference applies only to models with the expected metadata. Videm training is disabled by default. To use it, supply a private writable mount at `/usr/src/app/videm-seg-v4` containing your reviewed `training-config.json` and starting weights, then explicitly enable `VIDEM_V4_ENABLED=1` only on the intended worker. This specialized training code has fixed architecture and dataset checks; it is not a generic replacement for normal Kraken training.

Keep PostgreSQL, media and Redis in persistent volumes. Enable Redis AOF and snapshot persistence in your deployment configuration. Database migrations may require restoring the matching database backup to roll back an application image; changing the image alone is not a complete rollback.

## Attribution

Upstream eScriptorium and Kraken retain their respective licenses; see `licenses/`. The Videm training extension adapts eScriptorium training code. Third-party runtime components and model files retain their own licenses. Public source availability does not grant rights to redistribute private training material or third-party binary assets.

## Runtime provenance

`runtime/intel-packages.json` records exact package URLs, versions and SHA-256 checksums from the [Intel-documented Ubuntu PPA](https://dgpu-docs.intel.com/installation-guides/installing-packages-from-the-intel-ppa.html). Downloads are verified before extraction. Original package copyright/license notices and changelogs are included in `/opt/intel-runtime/licenses/`; package provenance is in `/opt/intel-runtime/package-provenance.json`.

The public eScriptorium base is pinned by digest. Python constraints are recorded in `runtime/python-constraints.txt`, and the actual installed inventory is retained in `/opt/arc-runtime/installed.lock.txt`. OS package installation and wheel availability still depend on upstream repositories; this is not a promise of byte-for-byte reproducibility. No local container filesystem, models, credentials, or private configuration are copied into the runtime. GitHub Packages visibility is separate from repository visibility; package settings must allow public pulls for unauthenticated users.
