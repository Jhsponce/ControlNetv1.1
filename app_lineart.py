#!/usr/bin/env python

import gradio as gr

from settings import (
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_NUM_IMAGES,
    MAX_IMAGE_RESOLUTION,
    MAX_NUM_IMAGES,
    MAX_SEED,
)
from utils import randomize_seed_fn


def create_demo(process):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image = gr.Image()
                reference_image = gr.Image(type="pil", label="Reference Image (optional, for style)")
                prompt = gr.Textbox(label="Prompt", submit_btn=True)
                with gr.Accordion("Advanced options", open=False):
                    preprocessor_name = gr.Radio(
                        label="Preprocessor",
                        choices=[
                            "Lineart",
                            "Lineart coarse",
                            "None",
                            "Lineart (anime)",
                            "None (anime)",
                        ],
                        type="value",
                        value="Lineart",
                        info='Note that "Lineart (anime)" and "None (anime)" are for anime base models like Anything-v3.',
                    )
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
                    preprocess_resolution = gr.Slider(
                        label="Preprocess resolution", minimum=128, maximum=512, value=512, step=1
                    )
                    num_steps = gr.Slider(label="Number of steps", minimum=1, maximum=100, value=20, step=1)
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    a_prompt = gr.Textbox(label="Additional prompt", value="Best quality, extremely detailed, real life photography, real life textures, real life materials, wood, plaster, marble, stone, concrete, metal, glass, realistic lighting, photo style")
                    n_prompt = gr.Textbox(
                        label="Negative prompt",
                        value="Extra windows, extra doors, distorted glass, fantasy rendering, watercolor, loose brushwork, illustration, incorrect perspective, people, vehicles, foliage overlap, lens flare, reflections in pavement, AI artifacts, duplicate elements, curved walls, irregular structures, sketch-like effects",
                    )
            with gr.Column():
                result = gr.Gallery(label="Output", show_label=False, columns=2, object_fit="scale-down")
        inputs = [
            image,
            reference_image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            preprocess_resolution,
            num_steps,
            guidance_scale,
            seed,
            preprocessor_name,
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
            api_name="lineart",
            concurrency_id="main",
        )
    return demo


if __name__ == "__main__":
    from model import Model

    model = Model(task_name="lineart")
    demo = create_demo(model.process_lineart)
    demo.queue().launch()
