import gc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np
import PIL.Image
import torch
from controlnet_aux import (
    CannyDetector,
    LineartDetector,
    LineartAnimeDetector,
    MLSDdetector,
)
from controlnet_aux.util import HWC3

from cv_utils import resize_image


class Preprocessor:
    MODEL_ID = "lllyasviel/Annotators"

    def __init__(self) -> None:
        self.model: Callable = None  # type: ignore
        self.name = ""

    def load(self, name: str) -> None:
        if name == self.name:
            return
        if name == "MLSD":
            self.model = MLSDdetector.from_pretrained(self.MODEL_ID)
        elif name == "Lineart":
            self.model = LineartDetector.from_pretrained(self.MODEL_ID)
        elif name == "LineartAnime":
            self.model = LineartAnimeDetector.from_pretrained(self.MODEL_ID)
        elif name == "Canny":
            self.model = CannyDetector()
        else:
            raise ValueError(f"Unknown preprocessor: {name}")
        torch.cuda.empty_cache()
        gc.collect()
        self.name = name

    def __call__(self, image: PIL.Image.Image, **kwargs) -> PIL.Image.Image:
        if self.name == "Canny":
            if "detect_resolution" in kwargs:
                detect_resolution = kwargs.pop("detect_resolution")
                image = np.array(image)
                image = HWC3(image)
                image = resize_image(image, resolution=detect_resolution)
            image = self.model(image, **kwargs)
            return PIL.Image.fromarray(image)

        image = self.model(image, **kwargs)
        return PIL.Image.fromarray(image) if isinstance(image, np.ndarray) else image
