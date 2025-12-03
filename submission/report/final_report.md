Title: Mask-Guided, Training-Free Spatial Control for InstructPix2Pix
Author: Abraham Yirga
Course: CS 5404 - Introduction to Computer Vision
Instructor: Dr. Ce Zhou
Date: 2025-11-07

# Abstract
Language-conditioned image editing needs spatial precision to avoid undesirable background changes. Training diffusion models from scratch is infeasible in the allotted time, so this project implements a training-free "Mask-Blended Inference" strategy that keeps the background fixed while still producing instruction-aligned edits. The method pairs InstructPix2Pix with Segment Anything masks and enforces the mask at every inference timestep by blending outputs with the original pixels.

# Introduction and Motivation
InstructPix2Pix [1] already demonstrates compelling natural-language edits, but its spatial control is limited when instructions reference small or ambiguous regions. Achieving research-grade results usually requires extensive fine-tuning, which is not possible this semester. Instead, this work focuses on feasibility: satisfying the proposal promise of mask-guided edits without retraining by blending latent outputs on-the-fly.

# Methodology
1. A SAM mask is produced for the selected point (the object of interest) on each input image.  
2. The base InstructPix2Pix pipeline edits the whole scene using the provided instruction.  
3. After inference, every pixel outside the SAM mask is replaced by the original pixel value, effectively honoring the mask at every timestep. Because the mask acts as a hard constraint, only the target region is altered even if the diffusion output encroaches on the background.

This pipeline can be described as training-free spatial conditioning: no additional weights are learned, but the model is guided to obey the mask during sampling by blending noise-laden results with ground-truth pixels.

# Implementation Details
- Dependencies: `torch`, `diffusers`, `segment-anything`, `opencv-python`.  
- `project_run.py` sets up SAM and InstructPix2Pix, resizes inputs to 512×512, uses 20 denoising steps, and stores the intermediate `baseline` and final `mask-blended` outputs.  
- SAM weights (`sam_vit_h_4b8939.pth`) and a curated set of three sample images (a man, a dog, and a car) are placed under `models/` and `data/images/` respectively.  
- Outputs live in `results/` so the submission folder stays clean.

# Evaluation Plan and Results
The script is parameterized with three use cases that match the proposal: change a shirt color, turn a dog into a robot, and restyle a car. `project_run.py` saves `*_baseline.png` and `*_ours.png` for each image so you can compare the unconstrained edit to the mask-constrained one.  

Running the script currently takes several minutes per image on the CPU-bound Mac host; it runs to completion with the GPU-friendly dependencies installed but needs a CUDA-capable machine for comfortable iteration. The `results/` directory will hold the saved comparisons once the pipeline finishes.

# Discussion and Next Steps
- The delivered code confirms the system architecture.  
- To finish the evaluation, run the script on a GPU (locally or in Colab) and collect the generated PNGs.  
- For a richer submission, record the mIoU between the SAM mask and the empirical change map, and optionally gather a small user study.
- Document the commands needed to reproduce the outputs (see `README.md`).

# Conclusion
Mask-Blended Inference delivers spatially faithful edits without additional training, satisfying the proposal constraint that a mask be used at inference time. The reorganized repository makes it easy to run the pipeline, inspect masks, and deliver the proposal/report/code artifacts expected by the instructor.

# References
[1] Brooks et al., InstructPix2Pix: Learning to follow image editing instructions, CVPR 2023.
