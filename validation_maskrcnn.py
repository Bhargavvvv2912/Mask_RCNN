# validation_maskrcnn.py (Lightweight Version)
import sys
import numpy as np

def run_lightweight_smoke_test():
    print("--- Starting Lightweight Mask R-CNN Smoke Test ---")
    
    try:
        # We purposefully import ONLY config and utils.
        # We DO NOT import 'mrcnn.model' because that triggers the 
        # TensorFlow/Keras version incompatibility in this legacy repo.
        from mrcnn import config
        from mrcnn import utils
        print("Imports successful: mrcnn.config, mrcnn.utils")

        # --- Test 1: Configuration Class ---
        print("\n--> Stage 1: Testing Config instantiation...")
        class TestConfig(config.Config):
            NAME = "test_config"
            IMAGES_PER_GPU = 1
            GPU_COUNT = 1
            NUM_CLASSES = 80
        
        c = TestConfig()
        print(f"    Config Name: {c.NAME}")
        
        # --- Test 2: Utils & Numpy Operations ---
        print("\n--> Stage 2: Testing Utils (Numpy/Image processing)...")
        # Create a dummy image (100x100 RGB)
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Test Resizing (Relies on Scikit-Image)
        # This is a great test for your Agent because updating scikit-image
        # often breaks this specific function in legacy code.
        molded_image, window, scale, padding, crop = utils.resize_image(
            image, min_dim=64, max_dim=128, min_scale=0, mode="square"
        )
        print(f"    Resized shape: {molded_image.shape}")
        
        if molded_image.shape != (128, 128, 3):
             raise ValueError(f"Resize failed. Got {molded_image.shape}")

        print("\n--- Lightweight Smoke Test: PASSED ---")
        sys.exit(0)

    except Exception as e:
        print(f"\n--- Lightweight Smoke Test: FAILED ---", file=sys.stderr)
        print(f"Error: {type(e).__name__} - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_lightweight_smoke_test()