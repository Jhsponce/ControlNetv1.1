#!/usr/bin/env python

import gradio as gr
import numpy as np

from settings import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_NUM_IMAGES,
    MAX_IMAGE_RESOLUTION,
    MAX_NUM_IMAGES,
    MAX_SEED,
)
from utils import randomize_seed_fn


def create_canvas(w: int, h: int) -> dict[str, np.ndarray | list[np.ndarray]]:
    return {
        "background": np.full((h, w), 255, dtype=np.uint8),
        "composite": np.full((h, w), 255, dtype=np.uint8),
        "layers": [np.full((h, w), 255, dtype=np.uint8)],
    }


def create_demo(process):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                canvas_width = gr.Slider(
                    label="Canvas width",
                    minimum=256,
                    maximum=MAX_IMAGE_RESOLUTION,
                    value=DEFAULT_IMAGE_RESOLUTION,
                    step=1,
                )
                canvas_height = gr.Slider(
                    label="Canvas height",
                    minimum=256,
                    maximum=MAX_IMAGE_RESOLUTION,
                    value=DEFAULT_IMAGE_RESOLUTION,
                    step=1,
                )
                create_button = gr.Button("Open drawing canvas!")
                image = gr.ImageEditor(
                    value=create_canvas(DEFAULT_IMAGE_RESOLUTION, DEFAULT_IMAGE_RESOLUTION),
                    image_mode="L",
                    width=MAX_IMAGE_RESOLUTION + 50,
                    height=MAX_IMAGE_RESOLUTION + 50,
                    sources=None,
                    transforms=(),
                    layers=False,
                    brush=gr.Brush(default_size=2, default_color="black", color_mode="fixed"),
                )
                prompt = gr.Textbox(label="Prompt", submit_btn=True)
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(
                        label="Number of images", minimum=1, maximum=MAX_NUM_IMAGES, value=DEFAULT_NUM_IMAGES, step=1
                    )
                    image_resolution = gr.Slider(
                        label="Image resolution",
                        minimum=256,
                        maximum=MAX_IMAGE_RESOLUTION,
                        value=DEFAULT_IMAGE_RESOLUTION,
                        step=256,
                    )
                    num_steps = gr.Slider(label="Number of steps", minimum=1, maximum=100, value=20, step=1)
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    a_prompt = gr.Textbox(label="Additional prompt", value="best quality, extremely detailed")
                    n_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                    )
            with gr.Column():
                result = gr.Gallery(label="Output", show_label=False, columns=2, object_fit="scale-down")

        create_button.click(
            fn=create_canvas,
            inputs=[canvas_width, canvas_height],
            outputs=image,
            queue=False,
            api_name=False,
        )

        inputs = [
            image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            num_steps,
            guidance_scale,
            seed,
        ]
        prompt.submit(
            fn=randomize_seed_fn,
            inputs=[seed, randomize_seed],
            outputs=seed,
            queue=False,
            api_name=False,
        ).then(
            fn=process,
            inputs=inputs,
            outputs=result,
            api_name=False,
            concurrency_id="main",
        )
    return demo


if __name__ == "__main__":
    from model import Model

    model = Model(task_name="scribble")
    demo = create_demo(model.process_scribble_interactive)
    demo.queue().launch()
