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
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers import GGUFQuantizationConfig
from diffusers.utils import export_to_video


STATE_FILE = "HunyuanVideo_state.json"
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

def add_to_queue(prompt, height, width, num_frames, num_inference_steps, fps, seed):
    task = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
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

def save_ui_state(prompt, height, width, num_frames, num_inference_steps, fps, seed):
    state = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
    }
    save_state(state)
    return "State saved!"


repo_id = "hunyuanvideo-community/HunyuanVideo"
base_path = repo_id
files_to_download = [
        "model_index.json",
        "config.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "text_encoder/model.safetensors.index.json",
        "text_encoder_2/config.json",
        "text_encoder_2/model.safetensors",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json",
        "tokenizer_2/vocab.json",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00006.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00006.safetensors",
        "transformer/diffusion_pytorch_model-00003-of-00006.safetensors",
        "transformer/diffusion_pytorch_model-00004-of-00006.safetensors",
        "transformer/diffusion_pytorch_model-00005-of-00006.safetensors",
        "transformer/diffusion_pytorch_model-00006-of-00006.safetensors",
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

# GGUF
base_path_GGUF = "city96/HunyuanVideo-gguf"
files_to_download_GGUF = "hunyuan-video-t2v-720p-Q3_K_S.gguf"

os.makedirs(base_path_GGUF, exist_ok=True)
try:
    os.makedirs(base_path_GGUF, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id=base_path_GGUF,
        filename=files_to_download_GGUF,
        local_dir=base_path_GGUF,
    )
    
    print(f"Successfully downloaded: {files_to_download_GGUF}")
    
except Exception as e:
    print(f"Error downloading {files_to_download_GGUF}: {str(e)}")
    raise



# GGUF https://huggingface.co/city96/HunyuanVideo-gguf/tree/main
# LORA TBI https://github.com/huggingface/diffusers/pull/10376

transformer_path = (
    "https://huggingface.co/city96/HunyuanVideo-gguf/blob/main/hunyuan-video-t2v-720p-Q3_K_M.gguf"
)

model_id = repo_id

transformer = HunyuanVideoTransformer3DModel.from_single_file(
    transformer_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)

pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, 
    transformer=transformer,
    torch_dtype=torch.float16
)

pipe.vae.enable_tiling()
pipe.enable_model_cpu_offload()

def generate_video(prompt, height, width, num_frames, num_inference_steps, fps, seed):
    if seed == 0:
        seed = random.randint(0, 999999)
    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator(device='cuda').manual_seed(seed),
    ).frames[0]
    
    # Create output filename based on prompt and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt[:15]}_{timestamp}.mp4"
    
    # Save the video to the output folder
    os.makedirs("output_hunyuan", exist_ok=True)
    output_path = f"./output_hunyuan/{filename}"
    export_to_video(video, output_path, fps=fps)
    
    return output_path

def randomize_seed():
    return random.randint(0, 999999)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Generate Video"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=3, value=initial_state.get("prompt", "A dramatic view of the pyramids at Giza during sunset."))
            with gr.Row():
                height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1, value=initial_state.get("height", 320))
                width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1, value=initial_state.get("width", 512))
            with gr.Row():
                num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1, value=initial_state.get("num_frames", 81))
                num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=initial_state.get("num_inference_steps", 30))
            with gr.Row():
                fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=initial_state.get("fps", 15))
                seed = gr.Number(label="Seed", value=initial_state.get("seed", 0))
                random_seed_button = gr.Button("Randomize Seed")

            output_video = gr.Video(label="Generated Video", show_label=True)
            generate_button = gr.Button("Generate Video")
            save_state_button = gr.Button("Save State")

            random_seed_button.click(lambda: random.randint(0, 999999), outputs=seed)
            generate_button.click(
                generate_video,
                inputs=[prompt, height, width, num_frames, num_inference_steps, fps, seed],
                outputs=output_video
            )
            save_state_button.click(
                save_ui_state,
                inputs=[prompt, height, width, num_frames, num_inference_steps, fps, seed],
                outputs=gr.Text(label="State Status")
            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                batch_prompt = gr.Textbox(label="Prompt", lines=3, value="A batch of videos depicting different landscapes.")
            with gr.Row():
                batch_height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1, value=320)
                batch_width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1, value=512)
            with gr.Row():
                batch_num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1, value=81)
                batch_num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=30)
            with gr.Row():
                batch_fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=15)
                batch_seed = gr.Number(label="Seed", value=0)
                random_seed_batch_button = gr.Button("Randomize Seed")

            add_to_queue_button = gr.Button("Add to Queue")
            clear_queue_button = gr.Button("Clear Queue")
            process_queue_button = gr.Button("Process Queue")

            queue_status = gr.Text(label="Queue Status")

            random_seed_batch_button.click(lambda: random.randint(0, 999999), outputs=batch_seed)
            add_to_queue_button.click(
                add_to_queue,
                inputs=[batch_prompt, batch_height, batch_width, batch_num_frames, batch_num_inference_steps, batch_fps, batch_seed],
                outputs=queue_status
            )
            clear_queue_button.click(clear_queue, outputs=queue_status)
            process_queue_button.click(process_queue, outputs=queue_status)

demo.launch()
