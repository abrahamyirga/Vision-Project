import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"

CLIP_MODEL = "openai/clip-vit-base-patch32"

CASE_INSTRUCTIONS = {
    "case_man_shirt": "Change the shirt to bright red leather",
    "case_dog_robot": "Turn the dog into a playful robot",
    "case_car_hover": "Make the car glow like a futuristic hovercraft",
}


def load_image(path):
    return Image.open(path).convert("RGB")


def compute_miou(mask, change_map, threshold=None):
    if threshold is None:
        threshold = max(10, change_map.mean())
    change_binary = change_map > threshold
    intersection = (change_binary & mask).sum()
    union = (change_binary | mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def main():
    cases = sorted({p.stem.rsplit("_", 1)[0] for p in RESULTS_DIR.glob("*_baseline.png")})
    if not cases:
        raise SystemExit("No results found; run project_run.py first.")

    clip_model = CLIPModel.from_pretrained(CLIP_MODEL)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model.eval()

    rows = []
    for case in cases:
        original = load_image(RESULTS_DIR / f"{case}_original.png")
        baseline = load_image(RESULTS_DIR / f"{case}_baseline.png")
        mask = np.array(Image.open(RESULTS_DIR / f"{case}_mask.png").convert("L")) > 127
        mask_bool = mask.astype(bool)

        baseline_np = np.array(baseline).astype(np.float32)
        original_np = np.array(original).astype(np.float32)
        change_map = np.abs(baseline_np - original_np).mean(axis=2)

        miou = compute_miou(mask_bool, change_map)

        blended = load_image(RESULTS_DIR / f"{case}_ours.png")
        instruction = CASE_INSTRUCTIONS.get(case, f"An edited image of {case.replace('_', ' ')}")
        inputs = clip_processor(text=[instruction], images=blended, return_tensors="pt")
        with torch.no_grad():
            clip_outputs = clip_model(**inputs)
        clip_score = float(torch.nn.functional.cosine_similarity(clip_outputs.image_embeds, clip_outputs.text_embeds).item())

        rows.append({"case": case, "mIoU": miou, "clip_score": clip_score})

    output_csv = BASE_DIR / "results" / "metrics_summary.csv"
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["case", "mIoU", "clip_score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Metrics written to {output_csv}")
    for row in rows:
        print(f"{row['case']}: mIoU={row['mIoU']:.3f}, CLIP={row['clip_score']:.3f}")


if __name__ == "__main__":
    main()
