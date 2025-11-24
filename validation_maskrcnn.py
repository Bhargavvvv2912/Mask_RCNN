import os
import sys
import random
import numpy as np
import skimage.io

# We import the CORE library only. 
# We do NOT import 'coco' (which is just a sample script).
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import config

# --- Define a Local Configuration ---
# Instead of importing CocoConfig from samples/coco/coco.py,
# we define a compatible config here. This avoids needing 
# pycocotools or messing with sys.path.
class ValidationConfig(config.Config):
    """Configuration for validation smoke test."""
    NAME = "coco_validation"
    
    # Set to match the COCO dataset (80 classes + 1 background)
    # This is required because we are loading COCO weights.
    NUM_CLASSES = 1 + 80 
    
    # run on 1 GPU with 1 image
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def run_maskrcnn_smoke_test():
    """
    Performs a full, end-to-end inference workflow to validate
    that the 'mrcnn' library is installed and working.
    """
    print("--- Starting Mask R-CNN Smoke Test ---")
    
    try:
        # --- Stage 1: Configuration and Model Loading ---
        print("\n--> Stage 1: Configuring environment and loading model...")
        
        ROOT_DIR = os.path.abspath(".")
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        
        # Download weights if needed
        if not os.path.exists(COCO_MODEL_PATH):
            print("COCO model weights not found. Downloading...")
            utils.download_trained_weights(COCO_MODEL_PATH)
            print("Download complete.")

        # Initialize our local config
        config = ValidationConfig()
        
        # Create model in inference mode
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        
        # Load weights
        # We use by_name=True to be safe, though strict matching is usually fine with correct classes
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        print("Model and weights loaded successfully.")

        # --- Stage 2: Run Inference on a Sample Image ---
        print("\n--> Stage 2: Generating dummy image and running detection...")
        
        # Instead of relying on files on disk, we generate a random noise image.
        # This ensures the validation relies ONLY on the code, not on 'images/' existing.
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Run detection
        # verbose=0 suppresses Keras logs
        results = model.detect([image], verbose=0)
        r = results[0]
        
        # --- Stage 3: Verify the Results ---
        print("\n--> Stage 3: Verifying output structure...")
        
        required_keys = ['rois', 'class_ids', 'scores', 'masks']
        for key in required_keys:
            if key not in r:
                raise ValueError(f"Output missing key '{key}'")
        
        # Check shapes
        if r['rois'].shape[0] != r['class_ids'].shape[0]:
            raise ValueError("Mismatch between ROIs and Class IDs count")

        print(f"Verification PASSED. Model successfully processed image.")
        print(f"Detected {len(r['class_ids'])} objects (likely 0 for noise, but pipeline worked).")
        
        print("\n--- Mask R-CNN Smoke Test: ALL STAGES PASSED ---")
        sys.exit(0)

    except Exception as e:
        print(f"\n--- Mask R-CNN Smoke Test: FAILED ---", file=sys.stderr)
        print(f"An error occurred during the smoke test: {type(e).__name__} - {e}", file=sys.stderr)
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_maskrcnn_smoke_test()