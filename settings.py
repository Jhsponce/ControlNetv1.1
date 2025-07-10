import os

import numpy as np

DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")

MAX_NUM_IMAGES = int(os.getenv("MAX_NUM_IMAGES", "4"))
DEFAULT_NUM_IMAGES = min(MAX_NUM_IMAGES, int(os.getenv("DEFAULT_NUM_IMAGES", "4")))
MAX_IMAGE_RESOLUTION = int(os.getenv("MAX_IMAGE_RESOLUTION", "1024"))
DEFAULT_IMAGE_RESOLUTION = min(MAX_IMAGE_RESOLUTION, int(os.getenv("DEFAULT_IMAGE_RESOLUTION", "1024")))

ALLOW_CHANGING_BASE_MODEL = os.getenv("SPACE_ID") != "Scythd/Sketch_Rendering_Test"
SHOW_DUPLICATE_BUTTON = os.getenv("SHOW_DUPLICATE_BUTTON") == "1"

MAX_SEED = np.iinfo(np.int32).max
