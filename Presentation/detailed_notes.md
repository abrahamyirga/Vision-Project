# Mask-Guided, Training-Free Spatial Control for InstructPix2Pix — Deep Dive Notes
Purpose: Detailed, plain-language walkthrough with analogies so anyone (classmate, TA, non-vision friend) can grasp what was done and why.

---

## 1) Big Picture: What Problem Are We Solving?
Modern text-to-image editors like InstructPix2Pix are good at “what” to change (follow a prompt) but bad at “where” to change it. If you ask “make the jacket red,” the model might also tint the background. This is like spray-painting without masking tape: you hit the target but also splatter the table.

**Goal:** Add spatial control (only edit the intended region) without retraining the diffusion model.

---

## 2) The Ingredients (Base Papers and Tools)
### InstructPix2Pix (Brooks et al., CVPR 2023)
- A diffusion model trained on synthetic triplets: (source image, edited image, instruction).
- Strength: follows natural-language instructions well.
- Weakness: no explicit notion of “where” to edit; edits can bleed.

### Segment Anything (SAM, Kirillov et al., ICCV 2023)
- Promptable segmentation: click or box → binary mask of the object.
- Analogy: universal scissors that neatly cut out whatever you point at.

We combine them: SAM provides the “painter’s tape” (mask); InstructPix2Pix provides the “airbrush” (edit). No retraining, just smarter postprocessing.

---

## 3) Core Idea: Mask-Blended Inference (Training-Free)
We leave all weights untouched. The workflow:
1. Use SAM to get a binary mask for the target region.
2. Run InstructPix2Pix to generate an edited image.
3. Blend the edited image with the original using the mask.

**Formula:**  
\~y = M ⊙ y_baseline + (1 – M) ⊙ x₀  
Where M is 1 inside the mask (edit allowed) and 0 outside (keep original).  
**Analogy:** Painter’s tape + airbrush. The tape blocks paint; only exposed areas get colored.

Why this works for us:
- Zero training cost; just a final compositing step.
- Deterministic when seeds and masks are fixed.
- Reuses public checkpoints; minimal engineering.

---

## 4) Implementation Map (Where Things Live)
- **Inference script:** `submission/code/project_run.py`
  - Loads SAM checkpoint (`models/sam_vit_h_4b8939.pth`) and InstructPix2Pix (`timbrooks/instruct-pix2pix`).
  - Resizes to 512×512; 20 denoising steps; fixed seeds per case.
  - Saves four PNGs per case: original, mask, baseline edit, mask-blended edit.
  - Mask resize uses nearest-neighbor to keep crisp edges; mask normalized to [0,1] to avoid overflow.
- **Metrics script:** `submission/code/evaluate_metrics.py`
  - mIoU: overlap between change map (baseline vs. original) and mask → spatial precision (“did edits stay under the tape?”).
  - CLIP: similarity between edited image and instruction → semantic fidelity (“does it match the caption?”).
- **Helpers:**  
  - `download_models.sh` — fetch SAM checkpoint.  
  - `run_pipeline.sh` — install deps, run inference, then metrics.  
  - `clean_run.sh` — optional reset (wipe results, rerun).
- **Data:** `data/images/` — shirted woman, dog on grass, car on street.

---

## 5) Experiment Setup and Results (Colab A100)
- **Environment:** Google Colab, NVIDIA A100 (40 GB), ~2 minutes after model downloads.
- **Cases (20 steps, guidance 1.5):**
  1) Shirt → bright red leather (woman)  
  2) Dog → playful robot  
  3) Car → glowing hovercraft
- **Quantitative (mIoU, CLIP):**
  - Shirt: 0.183, 0.206  
  - Dog: 0.265, 0.261  
  - Car: 0.173, 0.221  
  Analogy: mIoU tells how well the paint stayed inside the tape; CLIP tells how well the new look matches the caption.
- **Qualitative:** For each case, show a row: original | mask | baseline | blended. Baseline sometimes bleeds; blended preserves background.

---

## 6) Strengths, Limits, and What’s Next
### Strengths
- Training-free; uses public checkpoints.
- Deterministic with fixed seeds and masks.
- Simple to explain and reproduce; clear visual before/after.

### Limitations
- Post-hoc: does not steer the diffusion trajectory—only the final frame.  
- Relies on SAM accuracy; bad click → bad mask.  
- Models are large (SAM + diffusion); GPU recommended for speed.

### Next Steps
- Inject mask earlier (attention/latent masking) to steer denoising.  
- Experiment with mask dilations/soft edges; more diverse test images.  
- Optional human study on “did it stay inside?” and “did it follow the prompt?”

---

## 7) Reproducibility Checklist
- Commands: `bash download_models.sh`; `bash run_pipeline.sh`.  
- Assets: SAM checkpoint in `models/`; sample images included; results in `results/` and `submission/report/figures/`.  
- Report: CVPR format (6–9 pages); code separate; repo: https://github.com/abrahamyirga/Vision-Project

---

## 8) Q&A Cheat Sheet
- **Why not train a mask-aware model?** Time/compute; this gives fast 80/20 spatial control.  
- **Does blending lose semantics?** Inside the mask we keep edited pixels; CLIP scores show alignment; background stays original.  
- **How robust is SAM?** Good on clear objects; if cluttered or misclicked, consider multi-click or dilations.  
- **CPU vs GPU?** CPU works but is slow; recommend GPU/Colab.  
- **Future work?** Attention/latent masking, softer masks, more cases, human evaluation.

---

## 9) Mental Model (Everyday Analogy)
Think of editing as painting a mural with an airbrush (diffusion). Without masking tape (SAM), paint drifts. Our method applies tape first, sprays the whole mural, then peels the tape to reveal that only the intended shape was recolored. No need to buy new paint (no retraining), just smarter taping and cleanup.

---

These notes are meant to be verbose for self-study and can be condensed into slides. Focus on the “painter’s tape for diffusion” story, show the before/after grids, and cite the mIoU/CLIP numbers as evidence. 
