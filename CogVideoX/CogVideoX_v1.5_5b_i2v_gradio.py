"""
Copyright NewGenAI
Code can't be included in commercial app used for monetary gain. No derivative code allowed.
"""
import json
import gradio as gr
import random
import time
from datetime import datetime
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

import torch
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video


STATE_FILE = "CogVideoX_v1.5_5b_i2v_state.json"
queue = []

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as file:
            return json.load(file)
    return {}

def save_state(state):
    with open(STATE_FILE, "w") as file:
        json.dump(state, file)

initial_state = load_state()

def add_to_queue(image, guidance_scale, prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed, use_dynamic_cfg):
    task = {
        "image": image,
        "guidance_scale": guidance_scale,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
        "use_dynamic_cfg": use_dynamic_cfg
    }
    queue.append(task)
    return f"Task added to queue. Current queue length: {len(queue)}"

def clear_queue():
    queue.clear()
    return "Queue cleared."

def process_queue():
    if not queue:
        return "Queue is empty."

    for i, task in enumerate(queue):
        generate_video(**task)
        time.sleep(1)

    queue.clear()
    return "All tasks in the queue have been processed."

def save_ui_state(guidance_scale, prompt, negative_prompt, height, width, num_frames, 
                  num_inference_steps, fps, seed, use_dynamic_cfg, memory_optimization):
    state = {
        "guidance_scale": guidance_scale,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
        "use_dynamic_cfg": use_dynamic_cfg,
        "memory_optimization": memory_optimization
    }
    save_state(state)
    return "State saved!"


repo_id = "THUDM/CogVideoX1.5-5B-I2V"
base_path = repo_id
files_to_download = [
        "model_index.json",
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
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    ]
os.makedirs(base_path, exist_ok=True)
for file_path in files_to_download:
    try:
        full_dir = os.path.join(base_path, os.path.dirname(file_path))
        os.makedirs(full_dir, exist_ok=True)
        
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=file_path,
            local_dir=base_path,
        )
        
        print(f"Successfully downloaded: {file_path}")
        
    except Exception as e:
        print(f"Error downloading {file_path}: {str(e)}")
        raise


pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16
)

pipe.vae.enable_tiling()

def setup_memory_optimization(optimization_mode):
    print("setup_memory_optimization: ", optimization_mode)
    if optimization_mode == "Low VRAM":
        pipe.enable_model_cpu_offload()
    else:  # "Extremely Low VRAM"
        pipe.enable_sequential_cpu_offload()

setup_memory_optimization(initial_state.get("memory_optimization", "Extremely Low VRAM"))
# pipe.enable_model_cpu_offload()
# pipe.enable_sequential_cpu_offload()

def generate_video(image, guidance_scale, prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed, use_dynamic_cfg):
    if seed == 0:
        seed = random.randint(0, 999999)

    generator = torch.Generator(device='cuda').manual_seed(seed)
    
    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        generator=generator,
    ).frames[0]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt[:15]}_{timestamp}.mp4"
    
    os.makedirs("output_cogvideox_v1.5_5b_i2v", exist_ok=True)
    output_path = f"./output_cogvideox_v1.5_5b_i2v/{filename}"
    export_to_video(video, output_path, fps=fps)
    
    return output_path

def randomize_seed():
    return random.randint(0, 999999)

with gr.Blocks() as demo:
    gr.Markdown("# CogVideoX v1.5-5B (image 2 video)")
    with gr.Tabs():
        with gr.Tab("Generate Video"):
            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil")
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=7, 
                                  value=initial_state.get("prompt", ""))
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=5,
                                          value=initial_state.get("negative_prompt", "blurry, low quality, distorted"))
                memory_optimization = gr.Radio(
                    choices=["Low VRAM", "Extremely Low VRAM"],
                    label="Memory Optimization",
                    value=initial_state.get("memory_optimization", "Extremely Low VRAM")
                )
            with gr.Row():
                height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1,
                                 value=initial_state.get("height", 320))
                width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1,
                                value=initial_state.get("width", 512))
                guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, step=0.1,
                                         value=initial_state.get("guidance_scale", 6))
            with gr.Row():
                num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1,
                                     value=initial_state.get("num_frames", 81))
                num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1,
                                              value=initial_state.get("num_inference_steps", 50))
            with gr.Row():
                fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1,
                              value=initial_state.get("fps", 15))
                seed = gr.Number(label="Seed", value=initial_state.get("seed", 0))
                random_seed_button = gr.Button("Randomize Seed")
                use_dynamic_cfg = gr.Checkbox(label="Use Dynamic CFG", 
                                            value=initial_state.get("use_dynamic_cfg", True))

            output_video = gr.Video(label="Generated Video", show_label=True)
            generate_button = gr.Button("Generate Video")
            save_state_button = gr.Button("Save State")
            memory_optimization.change(
                setup_memory_optimization,
                inputs=[memory_optimization]
            )
            random_seed_button.click(randomize_seed, outputs=seed)
            generate_button.click(
                generate_video,
                inputs=[input_image, guidance_scale, prompt, negative_prompt, height, width, num_frames,
                       num_inference_steps, fps, seed, use_dynamic_cfg],
                outputs=output_video
            )
            save_state_button.click(
                save_ui_state,
                inputs=[guidance_scale, prompt, negative_prompt, height, width, num_frames,
                       num_inference_steps, fps, seed, use_dynamic_cfg, memory_optimization],
                outputs=gr.Text(label="State Status")
            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                batch_input_image = gr.Image(label="Input Image", type="pil")
            with gr.Row():
                batch_prompt = gr.Textbox(label="Prompt", lines=3,
                                        value="")
                batch_negative_prompt = gr.Textbox(label="Negative Prompt", lines=3,
                                                 value="blurry, low quality, distorted")
            with gr.Row():
                batch_height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1, value=320)
                batch_width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1, value=512)
                batch_guidance_scale = gr.Slider(label="Guidance Scale", minimum=1, maximum=20, step=0.1, value=6)
            with gr.Row():
                batch_num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1, value=81)
                batch_num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100,
                                                    step=1, value=50)
            with gr.Row():
                batch_fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=15)
                batch_seed = gr.Number(label="Seed", value=0)
                random_seed_batch_button = gr.Button("Randomize Seed")
                batch_use_dynamic_cfg = gr.Checkbox(label="Use Dynamic CFG", value=True)

            add_to_queue_button = gr.Button("Add to Queue")
            clear_queue_button = gr.Button("Clear Queue")
            process_queue_button = gr.Button("Process Queue")

            queue_status = gr.Text(label="Queue Status")

            random_seed_batch_button.click(randomize_seed, outputs=batch_seed)
            add_to_queue_button.click(
                add_to_queue,
                inputs=[batch_input_image, batch_guidance_scale, batch_prompt, batch_negative_prompt, batch_height,
                       batch_width, batch_num_frames, batch_num_inference_steps, batch_fps,
                       batch_seed, batch_use_dynamic_cfg],
                outputs=queue_status
            )
            clear_queue_button.click(clear_queue, outputs=queue_status)
            process_queue_button.click(process_queue, outputs=queue_status)

demo.launch()