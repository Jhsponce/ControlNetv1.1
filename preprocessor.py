import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import PIL.Image
import torch
from controlnet_aux import (
    CannyDetector,
    LineartAnimeDetector,
    LineartDetector,
    MLSDdetector,
)
from controlnet_aux.util import HWC3

from cv_utils import resize_image
from depth_estimator import DepthEstimator
from image_segmentor import ImageSegmentor


class Preprocessor:
    MODEL_ID = "lllyasviel/Annotators"

    def __init__(self) -> None:
        self.model: Callable = None  # type: ignore
        self.name = ""

    def load(self, name: str) -> None:  # noqa: C901, PLR0912
        if name == self.name:
            return
        elif name == "MLSD":
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Lineart":
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == "LineartAnime":
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Canny":
            self.model = CannyDetector.from_pretrained(self.MODEL_ID)
        else:
            raise ValueError
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:  # noqa: ANN003
        if self.name == "Canny":
            if "detect_resolution" in kwargs:
                detect_resolution = kwargs.pop("detect_resolution")
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)
        if self.name == "Midas":
            detect_resolution = kwargs.pop("detect_resolution", 512)
            image_resolution = kwargs.pop("image_resolution", 512)
            image = np.array(image)
            image = HWC3(image)
            image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            return PIL.Image.fromarray(image)
        return self.model(image, **kwargs)
