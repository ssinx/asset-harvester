---
name: Bug report
about: Create a bug report to help us improve NuRec
title: "[BUG]"
labels: "? - Needs Triage, bug"
assignees: 'daehyoungko'

---

**Describe the bug**
A clear and concise description of what the bug is.

**Steps/Code to reproduce bug**
Follow this guide http://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports to craft a minimal bug report. This helps us reproduce the issue and resolve it more quickly.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Environment overview (please complete the following information)**
 - Asset Harvester workflow/component: [NCore parser, multiview diffusion, camera estimator, TokenGS lifting, NuRec asset preparation, benchmark, or post-training]
 - Installation method: [`bash setup.sh` or manual editable install; include the command and selected extras]
 - Asset Harvester version: [release version or Git commit]
 - Model checkpoints used: [checkpoint filenames or Hugging Face revision]
 - Input type: [NCore V4 clip/component store, posed multiview images, or single-view images and masks]

**Environment details**
 - GPU model(s), GPU count, and VRAM per GPU
 - NVIDIA driver and CUDA toolkit versions
 - Operating system, Python version, and Conda environment name
 - GCC version
 - PyTorch and relevant dependency versions (for example, `gsplat`, `diffusers`, and `transformers`)
 - NCore version, if the issue involves NCore parsing

**Additional context**
Add any other context about the problem here.
