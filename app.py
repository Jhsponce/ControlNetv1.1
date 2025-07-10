#!/usr/bin/env python

import gradio as gr
import torch

from app_canny import create_demo as create_demo_canny
from app_lineart import create_demo as create_demo_lineart
from app_mlsd import create_demo as create_demo_mlsd
from model import Model
from settings import ALLOW_CHANGING_BASE_MODEL, DEFAULT_MODEL_ID, SHOW_DUPLICATE_BUTTON

DESCRIPTION = "# Sketch Rendering with ControlNet v1.1"

if not torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CPU This demo does not work on CPU.</p>"

model = Model(base_model_id=DEFAULT_MODEL_ID, task_name="Canny")

with gr.Blocks(css_paths="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    gr.DuplicateButton(
        value="Duplicate Space for private use",
        elem_id="duplicate-button",
        visible=SHOW_DUPLICATE_BUTTON,
    )

    with gr.Tabs():
        with gr.Tab("Canny"):
            create_demo_canny(model.process_canny)
        with gr.Tab("MLSD"):
            create_demo_mlsd(model.process_mlsd)
        with gr.Tab("Lineart"):
            create_demo_lineart(model.process_lineart)

    with gr.Accordion(label="Base model", open=False):
        with gr.Row():
            with gr.Column(scale=5):
                current_base_model = gr.Text(label="Current base model")
            with gr.Column(scale=1):
                check_base_model_button = gr.Button("Check current base model")
        with gr.Row():
            with gr.Column(scale=5):
                new_base_model_id = gr.Text(
                    label="New base model",
                    max_lines=1,
                    placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    info="The base model must be compatible with Stable Diffusion v1.5.",
                    interactive=ALLOW_CHANGING_BASE_MODEL,
                )
            with gr.Column(scale=1):
                change_base_model_button = gr.Button("Change base model", interactive=ALLOW_CHANGING_BASE_MODEL)
        if not ALLOW_CHANGING_BASE_MODEL:
            gr.Markdown(
                """The base model is not allowed to be changed in this Space so as not to slow down the demo, but it can be changed if you duplicate the Space."""
            )

    check_base_model_button.click(
        fn=lambda: model.base_model_id,
        outputs=current_base_model,
        queue=False,
        api_name="check_base_model",
    )
    gr.on(
        triggers=[new_base_model_id.submit, change_base_model_button.click],
        fn=model.set_base_model,
        inputs=new_base_model_id,
        outputs=current_base_model,
        api_name=False,
        concurrency_id="main",
    )

if __name__ == "__main__":
    demo.queue(max_size=20).launch()
