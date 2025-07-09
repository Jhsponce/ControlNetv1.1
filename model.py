import gc
from typing import Optional
import numpy as np
import PIL.Image
import torch
from controlnet_aux.util import HWC3
from diffusers import (
    ControlNetModel,
    DiffusionPipeline,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)

from cv_utils import resize_image
from preprocessor import Preprocessor
from settings import MAX_IMAGE_RESOLUTION, MAX_NUM_IMAGES
from ipadapter_patch import apply_ipadapter
from ipadapter_patch import apply_ipadapter

CONTROLNET_MODEL_IDS = {
    "Canny": "lllyasviel/control_v11p_sd15_canny",
    "MLSD": "lllyasviel/control_v11p_sd15_mlsd",
    "softedge": "lllyasviel/control_v11p_sd15_softedge",
    "lineart": "lllyasviel/control_v11p_sd15_lineart",
    "lineart_anime": "lllyasviel/control_v11p_sd15s2_lineart_anime",
    "ip2p": "lllyasviel/control_v11e_sd15_ip2p",
    "inpaint": "lllyasviel/control_v11e_sd15_inpaint",
}


def download_all_controlnet_weights() -> None:
    for model_id in CONTROLNET_MODEL_IDS.values():
        ControlNetModel.from_pretrained(model_id)


class Model:
    def __init__(
        self,
        base_model_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5",
        task_name: str = "Canny"
    ) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.base_model_id = base_model_id
        self.task_name = task_name  # last used task

        self.pipes: dict[str, dict[str, DiffusionPipeline]] = {}
        self.preprocessor = Preprocessor()

    def load_pipe(self, base_model_id: str, task_name: str, use_ip_adapter: bool = True) -> DiffusionPipeline:
        model_id = CONTROLNET_MODEL_IDS[task_name]

        controlnet = ControlNetModel.from_pretrained(
            model_id,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_id,
            safety_checker=None,
            controlnet=controlnet,
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16
        )

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        if self.device.type == "cuda":
            pipe.enable_xformers_memory_efficient_attention()

        pipe.to(self.device)

        if use_ip_adapter:
            pipe = apply_ipadapter(pipe, device=self.device)

        torch.cuda.empty_cache()
        gc.collect()

        return pipe

    def unload_pipe(self, task_name: str) -> None:
        if task_name in self.pipes:
            if "styled" in self.pipes[task_name]:
                del self.pipes[task_name]["styled"]
            if "plain" in self.pipes[task_name]:
                del self.pipes[task_name]["plain"]
            if not self.pipes[task_name]:
                del self.pipes[task_name]

        torch.cuda.empty_cache()
        gc.collect()

    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        reference_image: Optional[PIL.Image.Image] = None,
        task_name: str = "Canny",
    ) -> list[PIL.Image.Image]:
        generator = torch.Generator().manual_seed(seed)

        if self.task_name != task_name:
            self.unload_pipe(self.task_name)
            self.task_name = task_name

        if task_name not in self.pipes:
            self.pipes[task_name] = {
                "styled": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=True),
                "plain": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=False),
            }

        pipe = self.pipes[task_name]["styled" if reference_image is not None else "plain"]

        pipe_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "num_inference_steps": num_steps,
            "generator": generator,
            "image": control_image,
        }

        if reference_image is not None:
            pipe_args["ip_adapter_image"] = reference_image

        return pipe(**pipe_args).images



    def set_base_model(self, base_model_id: str) -> str:
        if not base_model_id or base_model_id == self.base_model_id:
            return self.base_model_id
        del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        try:
            self.pipe = self.load_pipe(base_model_id, self.task_name)
        except Exception:  # noqa: BLE001
            self.pipe = self.load_pipe(self.base_model_id, self.task_name)
        return self.base_model_id

    def load_controlnet_weight(self, task_name: str) -> None:
        if task_name == self.task_name:
            return

        # Remove the previous task's pipes before switching
        self.unload_pipe(self.task_name)

        # Load new task's pipes if not present
        if task_name not in self.pipes:
            self.pipes[task_name] = {
                "styled": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=True),
                "plain": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=False),
            }

        pipe = self.pipes[task_name]["styled"]

        torch.cuda.empty_cache()
        gc.collect()

        model_id = CONTROLNET_MODEL_IDS[task_name]
        controlnet = ControlNetModel.from_pretrained(model_id, torch_dtype=torch.float16)
        controlnet.to(self.device)

        torch.cuda.empty_cache()
        gc.collect()

        pipe.controlnet = controlnet
        self.task_name = task_name



    def get_prompt(self, prompt: str, additional_prompt: str) -> str:
        return additional_prompt if not prompt else f"{prompt}, {additional_prompt}"

    @torch.autocast("cuda")
    def run_pipe(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: PIL.Image.Image,
        num_images: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        reference_image: Optional[PIL.Image.Image] = None,
        task_name: str = "Canny",
    ) -> list[PIL.Image.Image]:
        generator = torch.Generator().manual_seed(seed)

        # Lazy-load pipeline if not already cached
        if task_name not in self.pipes:
            self.pipes[task_name] = {
                "styled": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=True),
                "plain": self.load_pipe(self.base_model_id, task_name, use_ip_adapter=False),
            }

        pipe = self.pipes[task_name]["styled" if reference_image else "plain"]

        pipe_args = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_images_per_prompt": num_images,
            "num_inference_steps": num_steps,
            "generator": generator,
            "image": control_image,
        }

        if reference_image is not None:
            pipe_args["ip_adapter_image"] = reference_image

        return pipe(**pipe_args).images



    @torch.inference_mode()
    def process_canny(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        low_threshold: int,
        high_threshold: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load("Canny")
        control_image = self.preprocessor(
            image=image, low_threshold=low_threshold, high_threshold=high_threshold, detect_resolution=image_resolution
        )

        self.load_controlnet_weight("Canny")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=reference_image,
            task_name="Canny",
        )
        if hasattr(self.preprocessor, "clear"):
            self.preprocessor.clear()

        torch.cuda.empty_cache()
        gc.collect()
        return [control_image, *results]

    @torch.inference_mode()
    def process_mlsd(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        value_threshold: float,
        distance_threshold: float,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        self.preprocessor.load("MLSD")
        control_image = self.preprocessor(
            image=image,
            image_resolution=image_resolution,
            detect_resolution=preprocess_resolution,
            thr_v=value_threshold,
            thr_d=distance_threshold,
        )
        self.load_controlnet_weight("MLSD")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=reference_image,
            task_name="MLSD",
        )
        if hasattr(self.preprocessor, "clear"):
            self.preprocessor.clear()

        torch.cuda.empty_cache()
        gc.collect()
        return [control_image, *results]


    @torch.inference_mode()
    def process_softedge(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        if preprocessor_name == "None":
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["HED", "HED safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("HED")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                scribble=safe,
            )
        elif preprocessor_name in ["PidiNet", "PidiNet safe"]:
            safe = "safe" in preprocessor_name
            self.preprocessor.load("PidiNet")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                safe=safe,
            )
        else:
            raise ValueError
        self.load_controlnet_weight("softedge")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=reference_image,
            task_name="softedge",
        )
        if hasattr(self.preprocessor, "clear"):
            self.preprocessor.clear()

        torch.cuda.empty_cache()
        gc.collect()
        return [control_image, *results]



    @torch.inference_mode()
    def process_lineart(
        self,
        image: np.ndarray,
        reference_image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        preprocess_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
        preprocessor_name: str,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        task_name = "lineart_anime" if "anime" in preprocessor_name else "lineart"

        if preprocessor_name in ["None", "None (anime)"]:
            image = HWC3(image)
            image = resize_image(image, resolution=image_resolution)
            control_image = PIL.Image.fromarray(image)
        elif preprocessor_name in ["Lineart", "Lineart coarse"]:
            coarse = "coarse" in preprocessor_name
            self.preprocessor.load("Lineart")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
                coarse=coarse,
            )
        elif preprocessor_name == "Lineart (anime)":
            self.preprocessor.load("LineartAnime")
            control_image = self.preprocessor(
                image=image,
                image_resolution=image_resolution,
                detect_resolution=preprocess_resolution,
            )

        self.load_controlnet_weight(task_name)

        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=reference_image,
            task_name=task_name,
        )

        if hasattr(self.preprocessor, "clear"):
            self.preprocessor.clear()

        torch.cuda.empty_cache()
        gc.collect()
        return [control_image, *results]


    

    @torch.inference_mode()
    def process_ip2p(
        self,
        image: np.ndarray,
        prompt: str,
        additional_prompt: str,
        negative_prompt: str,
        num_images: int,
        image_resolution: int,
        num_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> list[PIL.Image.Image]:
        if image is None:
            raise ValueError
        if image_resolution > MAX_IMAGE_RESOLUTION:
            raise ValueError
        if num_images > MAX_NUM_IMAGES:
            raise ValueError

        image = HWC3(image)
        image = resize_image(image, resolution=image_resolution)
        control_image = PIL.Image.fromarray(image)
        self.load_controlnet_weight("ip2p")
        results = self.run_pipe(
            prompt=self.get_prompt(prompt, additional_prompt),
            negative_prompt=negative_prompt,
            control_image=control_image,
            num_images=num_images,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            reference_image=reference_image,
            task_name="ip2p",
        )
        if hasattr(self.preprocessor, "clear"):
            self.preprocessor.clear()

        torch.cuda.empty_cache()
        gc.collect()
        return [control_image, *results]
