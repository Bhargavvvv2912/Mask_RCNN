# validation_maskrcnn.py (The Final, Correctly Implemented Version)

import os
import sys
import random
import numpy as np
import skimage.io

# This script assumes it is being run with the 'Mask_RCNN' directory as the CWD.
# This allows the 'mrcnn' package to be found on the sys.path.

# --- START OF DEFINITIVE FIX: Use correct relative imports from the 'mrcnn' package ---
try:
    from mrcnn import coco
    from mrcnn import utils
    from mrcnn import model as modellib
    from mrcnn import visualize
except ImportError as e:
    print("--- Smoke Test FAILED: Critical import error ---", file=sys.stderr)
    print(f"Could not import the 'mrcnn' library. This indicates a fundamental installation problem.", file=sys.stderr)
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
# --- END OF DEFINITIVE FIX ---


def run_maskrcnn_smoke_test():
    """
    Performs a full, end-to-end inference workflow with Mask R-CNN to
    validate its core functionality.
    """
    print("--- Starting Mask R-CNN Smoke Test ---")
    
    try:
        # --- Stage 1: Configuration and Model Loading ---
        print("\n--> Stage 1: Configuring environment and loading model...")
        
        ROOT_DIR = os.path.abspath(".")
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        
        if not os.path.exists(COCO_MODEL_PATH):
            print("COCO model weights not found. Downloading...")
            utils.download_trained_weights(COCO_MODEL_PATH)
            print("Download complete.")

        class InferenceConfig(coco.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
        
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        print("Model and weights loaded successfully.")

        # --- Stage 2: Run Inference on a Sample Image ---
        print("\n--> Stage 2: Running object detection and segmentation...")
        
        image_dir = os.path.join(ROOT_DIR, "images")
        file_names = next(os.walk(image_dir))[2]
        image = skimage.io.imread(os.path.join(image_dir, random.choice(file_names)))

        results = model.detect([image], verbose=0)
        r = results[0]
        
        # --- Stage 3: Verify the Results ---
        print("\n--> Stage 3: Verifying inference results...")
        
        required_keys = ['rois', 'class_ids', 'scores', 'masks']
        for key in required_keys:
            assert key in r, f"Verification failed: Output missing key '{key}'."
            
        num_detections = r['rois'].shape[0]
        assert num_detections > 0, f"Verification failed: Model detected 0 objects."

        print(f"Verification PASSED. Model successfully detected {num_detections} objects.")
        
        print("\n--- Mask R-CNN Smoke Test: ALL STAGES PASSED ---")
        sys.exit(0)

    except Exception as e:
        print(f"\n--- Mask R-CNN Smoke Test: FAILED ---", file=sys.stderr)
        print(f"An error occurred during the smoke test: {type(e).__name__} - {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_maskrcnn_smoke_test()