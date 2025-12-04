import torch
import cv2
import numpy as np
import PIL.Image
from pathlib import Path
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from segment_anything import SamPredictor, sam_model_registry

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "images"
MODEL_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ================= CONFIGURATION =================
# SAM checkpoint must be downloaded manually:
# https://github.com/facebookresearch/segment-anything
SAM_CHECKPOINT = MODEL_DIR / "sam_vit_h_4b8939.pth"
MODEL_ID = "timbrooks/instruct-pix2pix"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 1337

print(f"Running on {DEVICE}...")

# ================= LOAD MODELS =================
print("Loading InstructPix2Pix...")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32, safety_checker=None
)
pipe.to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

print("Loading SAM (Segment Anything)...")
try:
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
except Exception as e:
    print(f"Error loading SAM: {e}")
    print("SAM checkpoint missing; expected 'sam_vit_h_4b8939.pth' in the models directory.")
    exit()

# ================= HELPER FUNCTIONS =================
def get_mask_from_sam(image_path, point_coords):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=np.array([point_coords]),
        point_labels=np.array([1]),
        multimask_output=False,
    )
    return masks[0]

# ================= MAIN PIPELINE =================
def run_masked_edit(image_path, prompt, click_x, click_y, output_name, seed=None):
    original_pil = PIL.Image.open(image_path).convert("RGB")
    original_pil = original_pil.resize((512, 512))

    print(f"Generating mask for {image_path} at point {click_x},{click_y}...")
    mask_np = get_mask_from_sam(image_path, [click_x, click_y])

    mask_pil = PIL.Image.fromarray(mask_np)
    mask_pil = mask_pil.resize((512, 512), resample=PIL.Image.NEAREST)
    mask_arr = np.array(mask_pil).astype(np.float32) / 255.0  # normalize to [0,1] for blending

    if seed is None:
        seed = DEFAULT_SEED
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    print(f"Running Baseline (seed={seed})...")
    res_baseline = pipe(
        prompt,
        image=original_pil,
        num_inference_steps=20,
        image_guidance_scale=1.5,
        generator=generator,
    ).images[0]

    res_np = np.array(res_baseline)
    orig_np = np.array(original_pil)
    mask_3ch = np.stack([mask_arr] * 3, axis=-1)

    final_np = res_np * mask_3ch + orig_np * (1 - mask_3ch)
    final_image = PIL.Image.fromarray(final_np.astype(np.uint8))

    original_pil.save(RESULTS_DIR / f"{output_name}_original.png")
    mask_pil.save(RESULTS_DIR / f"{output_name}_mask.png")
    res_baseline.save(RESULTS_DIR / f"{output_name}_baseline.png")
    final_image.save(RESULTS_DIR / f"{output_name}_ours.png")
    print(f"Saved results to {output_name}_*.png")

# ================= EXECUTION =================
if __name__ == "__main__":
    examples = [
        ("woman_shirt.jpg", "Change the shirt to bright red leather", 270, 260, "case_woman_shirt", DEFAULT_SEED),
        ("dog_grass.jpg", "Turn the dog into a playful robot", 260, 250, "case_dog_robot", DEFAULT_SEED + 1),
        ("car_street.jpg", "Make the car glow like a futuristic hovercraft", 280, 330, "case_car_hover", DEFAULT_SEED + 2),
    ]

    for image_name, prompt, x, y, name, seed in examples:
        image_path = DATA_DIR / image_name
        run_masked_edit(str(image_path), prompt, x, y, name, seed=seed)
