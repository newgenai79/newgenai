"""
Copyright NewGenAI
Code can't be included in commercial app used for monetary gain. No derivative code allowed.
"""
import torch
import gradio as gr
import os

from pathlib import Path
from diffusers import CogView3PlusPipeline
from datetime import datetime
from huggingface_hub import hf_hub_download

# Queue to hold batch processing tasks
queue = []
RESOLUTIONS = [
    "512x512",
    "720x480",
    "1024x1024",
    "1280x720",
    "2048x2048"
]
def get_dimensions(resolution):
    width, height = map(int, resolution.split('x'))
    return width, height

def add_to_queue(prompt, resolution, guidance_scale, num_inference_steps):
    width, height = get_dimensions(resolution)
    task = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps
    }
    queue.append(task)
    return f"Added to queue. Current queue size: {len(queue)}"

def clear_queue():
    queue.clear()
    return "Queue cleared."

def process_queue():
    results = []
    for task in queue:
        result, _ = generate_image(
            task["prompt"],
            task["height"],
            task["width"],
            task["guidance_scale"],
            task["num_inference_steps"]
        )
        results.append(result)
    queue.clear()
    return f"Processed {len(results)} items in the queue."



repo_id = "THUDM/CogView3-Plus-3B"
base_path = repo_id
files_to_download = [
        "model_index.json",
        "configuration.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/added_tokens.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
    ]
os.makedirs(base_path, exist_ok=True)
for file_path in files_to_download:
    try:
        # Create the full directory path for this file
        full_dir = os.path.join(base_path, os.path.dirname(file_path))
        os.makedirs(full_dir, exist_ok=True)
        
        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=base_path,
        )
        
        print(f"Successfully downloaded: {file_path}")
        
    except Exception as e:
        print(f"Error downloading {file_path}: {str(e)}")
        raise



# Load the model once
pipe = CogView3PlusPipeline.from_pretrained(
    "THUDM/CogView3-Plus-3B", 
    torch_dtype=torch.bfloat16
    )

pipe.text_encoder = pipe.text_encoder.to("cpu")
pipe.vae = pipe.vae.to("cuda")
pipe.transformer = pipe.transformer.to("cuda")

pipe.enable_sequential_cpu_offload()  # This will move unused components to CPU
# pipe.enable_attention_slicing(1)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

# Define inference function
def generate_image(prompt, height, width, guidance_scale, num_inference_steps):
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        num_images_per_prompt=1,
    )[0]

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output", exist_ok=True)
    filename = f"output/{prompt[:10].replace(' ', '_')}_{timestamp}.png"

    # Save and return image
    image[0].save(filename)
    return image[0], filename

with gr.Blocks() as demo:
    gr.Markdown("# CogView3 Plus 3B: Text 2 Image")

    with gr.Tabs():
        with gr.Tab("Generate image"):
            gr.Markdown("## Generate image")
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt", 
                    placeholder="Enter your text prompt here", 
                    lines=7, 
                    value="A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
                )
                with gr.Column():
                    resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTIONS,
                        value="512x512",
                        label="Resolution"
                    )
                    guidance_scale_slider = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.1
                    )
                    num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps",
                        value=50
                    )

            generate_button = gr.Button("Generate image")
            with gr.Row():
                output_image = gr.Image(label="Generated Image", type="pil")
                output_filename = gr.Textbox(label="Saved File Path", interactive=False)

            def generate_wrapper(prompt, resolution, guidance_scale, num_inference_steps):
                width, height = get_dimensions(resolution)
                return generate_image(prompt, height, width, guidance_scale, num_inference_steps)

            generate_button.click(
                fn=generate_wrapper,
                inputs=[
                    prompt_input,
                    resolution_dropdown,
                    guidance_scale_slider,
                    num_inference_steps_input
                ],
                outputs=[output_image, output_filename]
            )

        # Batch Processing Tab
        with gr.Tab("Batch Processing"):
            gr.Markdown("## Batch Processing")
            with gr.Row():
                batch_prompt_input = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your text prompt here",
                    lines=7,
                    value="A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
                )
                with gr.Column():
                    batch_resolution_dropdown = gr.Dropdown(
                        choices=RESOLUTIONS,
                        value="512x512",
                        label="Resolution"
                    )
                    batch_guidance_scale_slider = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=20.0,
                        value=7.0,
                        step=0.1
                    )
                    batch_num_inference_steps_input = gr.Number(
                        label="Number of Inference Steps",
                        value=50
                    )

            with gr.Row():
                add_to_queue_button = gr.Button("Add to Queue")
                clear_queue_button = gr.Button("Clear Queue")

            process_queue_button = gr.Button("Process Queue")
            queue_status = gr.Textbox(label="Queue Status", interactive=False)

            add_to_queue_button.click(
                fn=add_to_queue,
                inputs=[
                    batch_prompt_input,
                    batch_resolution_dropdown,
                    batch_guidance_scale_slider,
                    batch_num_inference_steps_input
                ],
                outputs=[queue_status]
            )
            clear_queue_button.click(fn=clear_queue, outputs=[queue_status])
            process_queue_button.click(fn=process_queue, outputs=[queue_status])

demo.launch()