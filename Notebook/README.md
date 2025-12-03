# Mask-Guided InstructPix2Pix – Colab Workspace

This folder mirrors the minimal assets you need to run the mask-guided InstructPix2Pix pipeline inside Google Colab (or any other GPU-backed notebook).

## Contents
- `requirements.txt`: same dependency list as the main repo (`torch`, `diffusers`, `segment-anything`, etc.).
- `models/sam_vit_h_4b8939.pth`: the SAM ViT-H checkpoint required for mask generation.
- `data/images/`: three canonical test inputs (`man_shirt.jpg`, `dog_grass.jpg`, `car_street.jpg`).
- `project_run.py` / `evaluate_metrics.py`: scripts that reproduce the inference and metric evaluation pipeline.
- `results/`: the directory where the scripts will write their PNGs and CSV results.
- `run_pipeline.sh`: helper shell script you can execute in Colab to install deps and run both scripts automatically.

## Running inside Colab
1. Upload this folder to your Colab VM (e.g., via `!unzip` or `git clone` into `/content/Notebook`).
2. Open a terminal cell and run `bash run_pipeline.sh` (it installs the dependencies, runs `project_run.py`, then executes `evaluate_metrics.py`).
3. Once it finishes, the generated assets and `results/metrics_summary.csv` will be available in `results/`.
4. Download `results/` (and, optionally, `models/` if you want to keep the checkpoint) back to your local machine for final documentation.

If you prefer to step through the workflow manually, install the packages with `pip install -r requirements.txt`, run `python project_run.py`, then run `python evaluate_metrics.py`.
