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

# Ensure output directory exists
os.makedirs("output", exist_ok=True)

# Define inference function
def generate_image(prompt, guidance_scale, num_inference_steps, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=2048,  # Fixed value
        width=2048,   # Fixed value
        guidance_scale=guidance_scale,
        pag_scale=2.0,
        num_inference_steps=num_inference_steps,
        generator=generator,
    )[0]

    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"output/{prompt[:10].replace(' ', '_')}_{timestamp}.png"

    # Save and return image
    image[0].save(filename)
    return image[0], filename

# Define Gradio Interface
def random_seed():
    return torch.randint(0, 2**32 - 1, (1,)).item()

with gr.Blocks() as demo:
    gr.Markdown("# Sana Image Generation 1.6M 2K")
    
    with gr.Row():
        prompt_input = gr.Textbox(
            label="Prompt", 
            placeholder="Enter your text prompt here", 
            lines=3
        )
    
    with gr.Row():
        guidance_scale_slider = gr.Slider(
            label="Guidance Scale", 
            minimum=1.0, 
            maximum=20.0, 
            value=5.0, 
            step=0.1
        )
        num_inference_steps_input = gr.Number(
            label="Number of Inference Steps", 
            value=20
        )
    
    with gr.Row():
        seed_input = gr.Number(
            label="Seed", 
            value=0
        )
        random_button = gr.Button("Randomize Seed")

    with gr.Row():
        height_display = gr.Textbox(
            label="Height", 
            value="2048", 
            interactive=False
        )
        width_display = gr.Textbox(
            label="Width", 
            value="2048", 
            interactive=False
        )

    with gr.Row():
        generate_button = gr.Button("Generate Image")
    
    with gr.Row():
        output_image = gr.Image(label="Generated Image", type="pil")
        output_filename = gr.Textbox(label="Saved File Path", interactive=False)
    
    # Button functionality
    random_button.click(fn=random_seed, outputs=[seed_input])
    generate_button.click(
        fn=generate_image, 
        inputs=[prompt_input, guidance_scale_slider, num_inference_steps_input, seed_input],
        outputs=[output_image, output_filename]
    )

demo.launch()
