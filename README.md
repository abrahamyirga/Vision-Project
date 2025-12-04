# Mask-Guided InstructPix2Pix (Training-Free Spatial Guidance)

This repo hosts the submission-ready materials for CS 5404's final project: a mask-blended inference pipeline that adds stronger spatial control to the InstructPix2Pix image editor without any additional training.

## Repository Layout
- `submission/` – what you would submit. Contains the official proposal (`submission/proposal/proposal.pdf`), the final report (`submission/report/final_report.{md,pdf}`), and the runnable code (`submission/code/project_run.py`).
- `data/images/` – curated test images (shirted man, dog on grass, futuristic car). These are the inputs for the scripted examples.
- `models/` – place the SAM checkpoint `sam_vit_h_4b8939.pth` here (downloaded; not tracked in git). Run `bash download_models.sh` to fetch it.
- `results/` – automatically created by the script; stores the original image, SAM mask, baseline InstructPix2Pix edit, and mask-blended edit for each case.
- `docs/` – assignment instructions and supporting text extracted from the provided PDFs.

## Quick Start (GPU recommended)
Make sure you have PyTorch + CUDA installed before running the script. The required packages were installed earlier with:
```
python3 -m pip install torch torchvision torchaudio diffusers transformers accelerate scipy safetensors opencv-python segment-anything matplotlib
```
Alternatively, install via the bundled requirements file:
```
python3 -m pip install -r requirements.txt
```

1. Download the SAM checkpoint (about 2.5 GB) into `models/` by running:
```
bash download_models.sh
```
2. Run the mask-blended inference pipeline:
```
python3 submission/code/project_run.py
```
3. Review the outputs in `results/`. Each example produces `*_baseline.png` and `*_ours.png` for comparison.
4. After generating the PNGs, run the metric evaluation script:
```
python3 submission/code/evaluate_metrics.py
```
The script writes `results/metrics_summary.csv` and prints mIoU/CLIP scores for each case.

_Note: On CPU this script is very slow; use a CUDA-enabled machine or Colab for reasonable runtimes (5–10 minutes per image otherwise)._  

## Deliverables
- Proposal: see `submission/proposal/proposal.pdf`.  
- Final Report: `submission/report/final_report.pdf` (generated from Markdown).  
- Code: `submission/code/project_run.py` plus the SAM checkpoint in `models/`.  
- Sample images: under `data/images/` for reproducibility.

## Next Steps (before final submission)
1. Complete the GPU run(s) to generate the requested example outputs.  
2. Annotate `results/` with qualitative observations or evaluation notes.  
3. Package `submission/` (proposal, report, code) and point Canvas/gradescope to it.
