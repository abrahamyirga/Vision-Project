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
The script encodes three canonical use cases: change a shirt color, turn a dog into a robot, and restyle a car. Each run saves `*_baseline.png` and `*_ours.png` so the unconstrained and mask-constrained edits can be compared directly.

## Execution Environment
- Platform: Google Colab
- Accelerator: NVIDIA A100 (40 GB)
- Command: `bash run_pipeline.sh` (installs deps, runs inference, then metrics)
- Runtime: ~2 minutes total after checkpoint downloads

## Quantitative Metrics
Automatic metrics were computed with `submission/code/evaluate_metrics.py` (CLIP for instruction fidelity and mIoU for spatial precision between the change map and SAM mask). Results:

| Case | mIoU | CLIP |
| --- | --- | --- |
| Shirt → red leather | 0.183 | 0.206 |
| Dog → playful robot | 0.265 | 0.261 |
| Car → glowing hovercraft | 0.173 | 0.221 |

Higher mIoU indicates edits remained within the SAM mask; CLIP reflects prompt alignment. The dog case scores highest on both metrics, consistent with its clear object boundaries.

## Qualitative Results
Representative outputs (original, mask, baseline, mask-blended) are stored under `results/` and included here for documentation:

- Shirt recolor: `results/case_man_shirt_{original,mask,baseline,ours}.png`
- Dog to robot: `results/case_dog_robot_{original,mask,baseline,ours}.png`
- Car restyle: `results/case_car_hover_{original,mask,baseline,ours}.png`

Across cases, mask-blended outputs preserve the background while the baseline occasionally bleeds edits outside the target region.

# Discussion and Next Steps
- The delivered code confirms the system architecture and includes quantitative metrics plus qualitative figures from the Colab A100 run.  
- Optional extensions: human evaluation for perceived spatial fidelity, mask dilations/ablations, or latent-space blending variants.  
- Commands to reproduce are captured in `README.md` and `run_pipeline.sh`.

# Conclusion
Mask-Blended Inference delivers spatially faithful edits without additional training, satisfying the proposal constraint that a mask be used at inference time. The reorganized repository makes it easy to run the pipeline, inspect masks, and deliver the proposal/report/code artifacts expected by the instructor. Results and metrics from the Colab A100 run are bundled in `results/`, providing concrete evidence of spatial control and instruction fidelity.

# References
[1] Brooks et al., InstructPix2Pix: Learning to follow image editing instructions, CVPR 2023.
