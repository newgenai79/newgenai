"""
Copyright NewGenAI
Code can't be included in commercial app used for monetary gain. No derivative code allowed.
"""
import torch
import gradio as gr
import os

from pathlib import Path
from diffusers import SanaPAGPipeline
from datetime import datetime
from huggingface_hub import hf_hub_download

# Queue to hold batch processing tasks
queue = []

def add_to_queue(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed):
    task = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "seed": seed
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
            task["negative_prompt"],
            task["width"],
            task["height"],
            task["guidance_scale"],
            task["num_inference_steps"],
            task["seed"]
        )
        results.append(result)
    queue.clear()
    return f"Processed {len(results)} items in the queue."



repo_id = "Efficient-Large-Model/Sana_1600M_2Kpx_BF16_diffusers"
base_path = repo_id
files_to_download = [
        "vae/diffusion_pytorch_model.bf16.safetensors",
        "vae/config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.bf16.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model.bf16-00001-of-00002.safetensors",
        "text_encoder/model.bf16-00002-of-00002.safetensors",
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer.model",
        "tokenizer/tokenizer_config.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.bf16.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json"
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
pipe = SanaPAGPipeline.from_pretrained(
    repo_id,
    variant="bf16",
    torch_dtype=torch.bfloat16,
    pag_applied_layers="transformer_blocks.8",
    use_safetensors=True,

)
pipe.to("cuda")
pipe.vae.to(torch.bfloat16)
pipe.text_encoder.to(torch.bfloat16)
pipe.enable_model_cpu_offload()

# Define inference function
def generate_image(prompt, negative_prompt, width, height, guidance_scale, num_inference_steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        pag_scale=2.0,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )[0]

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("output_sana_2K", exist_ok=True)
    filename = f"output_sana_2K/{prompt[:10].replace(' ', '_')}_{timestamp}.png"

    # Save and return image
    image[0].save(filename)
    return image[0], filename

# Define Gradio Interface
def random_seed():
    return torch.randint(0, 2**32 - 1, (1,)).item()

with gr.Blocks() as demo:
    gr.Markdown("# Sana Image Generation 1.6B 2K")

    with gr.Tabs():
        # Generate image Tab
        with gr.Tab("Generate image"):
            gr.Markdown("## Generate image")
            with gr.Row():
                with gr.Column():
                    prompt_input = gr.Textbox(
                        label="Prompt", 
                        placeholder="Enter your text prompt here", 
                        lines=3, 
                        value="A grand library within a hollowed-out ancient tree, its shelves carved directly into the wood and filled with glowing, magical books."
                    )
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt here",
                        lines=3,
                        value=""
                    )
                with gr.Column():
                    with gr.Row():
                        width_input = gr.Number(label="Width", value=2048, minimum=512, maximum=2048, step=64)
                        height_input = gr.Number(label="Height", value=2048, minimum=512, maximum=2048, step=64)
                    guidance_scale_slider = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=5.0, step=0.1)
                    num_inference_steps_input = gr.Number(label="Number of Inference Steps", value=20)
                    seed_input = gr.Number(label="Seed", value=0)
                    random_button = gr.Button("Randomize Seed")

            generate_button = gr.Button("Generate image")
            with gr.Row():
                output_image = gr.Image(label="Generated Image", type="pil")
                output_filename = gr.Textbox(label="Saved File Path", interactive=False)

            random_button.click(fn=random_seed, outputs=[seed_input])
            generate_button.click(
                fn=generate_image,
                inputs=[
                    prompt_input,
                    negative_prompt_input,
                    width_input,
                    height_input,
                    guidance_scale_slider,
                    num_inference_steps_input,
                    seed_input
                ],
                outputs=[output_image, output_filename]
            )

        # Batch Processing Tab
        with gr.Tab("Batch Processing"):
            gr.Markdown("## Batch Processing")
            with gr.Row():
                with gr.Column():
                    batch_prompt_input = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your text prompt here",
                        lines=3,
                        value="Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
                    )
                    batch_negative_prompt_input = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="Enter negative prompt here",
                        lines=3,
                        value=""
                    )
                with gr.Column():
                    with gr.Row():
                        batch_width_input = gr.Number(label="Width", value=2048, minimum=512, maximum=2048, step=64)
                        batch_height_input = gr.Number(label="Height", value=2048, minimum=512, maximum=2048, step=64)
                    batch_guidance_scale_slider = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=5.0, step=0.1)
                    batch_num_inference_steps_input = gr.Number(label="Number of Inference Steps", value=20)
                    batch_seed_input = gr.Number(label="Seed", value=0)
                    batch_random_button = gr.Button("Randomize Seed")

            with gr.Row():
                add_to_queue_button = gr.Button("Add to Queue")
                clear_queue_button = gr.Button("Clear Queue")

            process_queue_button = gr.Button("Process Queue")
            queue_status = gr.Textbox(label="Queue Status", interactive=False)

            batch_random_button.click(fn=random_seed, outputs=[batch_seed_input])
            add_to_queue_button.click(
                fn=add_to_queue,
                inputs=[
                    batch_prompt_input,
                    batch_negative_prompt_input,
                    batch_width_input,
                    batch_height_input,
                    batch_guidance_scale_slider,
                    batch_num_inference_steps_input,
                    batch_seed_input
                ],
                outputs=[queue_status]
            )
            clear_queue_button.click(fn=clear_queue, outputs=[queue_status])
            process_queue_button.click(fn=process_queue, outputs=[queue_status])

demo.launch()
