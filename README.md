# Mask-Guided InstructPix2Pix (Training-Free Spatial Guidance)

This repository hosts the submission-ready materials for CS 5404's final project: a mask-blended inference pipeline that adds stronger spatial control to the InstructPix2Pix image editor without additional training.

## Repository Layout
- `submission/` – final deliverables: proposal (`submission/proposal/proposal.pdf`), report (`submission/report/final_report.{md,pdf}`), and runnable code (`submission/code/project_run.py`).
- `data/images/` – curated test images (shirted woman, dog on grass, futuristic car) used by the scripted examples.
- `models/` – SAM checkpoint `sam_vit_h_4b8939.pth` (downloaded; not tracked in git). Fetch via `bash download_models.sh`.
- `results/` – auto-created by the pipeline; stores the original image, SAM mask, baseline InstructPix2Pix edit, and mask-blended edit for each case.
- `docs/` – assignment instructions and supporting text extracted from the provided PDFs.

## Quick Start (GPU recommended)
Ensure PyTorch + CUDA are available before running the script. Required packages:
```
python3 -m pip install torch torchvision torchaudio diffusers transformers accelerate scipy safetensors opencv-python segment-anything matplotlib
```
Alternatively, install via the bundled requirements file:
```
python3 -m pip install -r requirements.txt
```

1. Download the SAM checkpoint (about 2.5 GB) into `models/`:
```
bash download_models.sh
```
2. Run the mask-blended inference pipeline (this also runs metrics):
```
bash run_pipeline.sh
```
3. Review the outputs in `results/`. Each example produces `*_baseline.png` and `*_ours.png` for comparison. The metrics script writes `results/metrics_summary.csv` and prints mIoU/CLIP scores for each case.

_Note: On CPU this script is very slow; use a CUDA-enabled machine or Colab for reasonable runtimes (5–10 minutes per image otherwise)._  

### Running in Colab
1. Clone this repository into a Colab workspace.
2. Run `bash download_models.sh` to fetch the SAM checkpoint.
3. Execute `bash run_pipeline.sh` from the repository root to install dependencies, generate edits, and compute metrics.

## Deliverables
- Proposal: see `submission/proposal/proposal.pdf`.  
- Final Report: `submission/report/final_report.pdf` (generated from Markdown).  
- Code: `submission/code/project_run.py` plus the SAM checkpoint in `models/`.  
- Sample images: under `data/images/` for reproducibility.

## Next Steps (before final submission)
1. Complete the GPU run(s) to generate the requested example outputs.  
2. Annotate `results/` with qualitative observations or evaluation notes.  
3. Package `submission/` (proposal, report, code) and point Canvas/gradescope to it.
