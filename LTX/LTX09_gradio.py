
"""
Copyright NewGenAI
Code can't be included in commercial app used for monetary gain. No derivative code allowed.
"""
import json
import torch
import gradio as gr
import random
import time
from datetime import datetime
import os

from diffusers.utils import export_to_video
from diffusers import LTXPipeline
from transformers import T5EncoderModel, T5Tokenizer
from pathlib import Path
from datetime import datetime
from huggingface_hub import hf_hub_download

STATE_FILE = "LTX_state.json"
QUEUE_FILE = "LTX_queue.json"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as file:
            return json.load(file)
    return {}

# Function to save the current state
def save_state(state):
    with open(STATE_FILE, "w") as file:
        json.dump(state, file)

# Load initial state
initial_state = load_state()
def load_queue():
    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, "r") as file:
            return json.load(file)
    return []

def save_queue(queue):
    with open(QUEUE_FILE, "w") as file:
        json.dump(queue, file)

queue = load_queue()

# Function to add to queue
def add_to_queue(prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed):
    task = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
    }
    queue.append(task)
    save_queue(queue)
    return f"Task added to queue. Current queue length: {len(queue)}"

# Function to clear the queue
def clear_queue():
    queue.clear()
    save_queue(queue)
    return "Queue cleared."

# Function to process the queue
def process_queue():
    if not queue:
        return "Queue is empty."

    for i, task in enumerate(queue):
        generate_video(**task)
        time.sleep(1)  # Simulate processing time

    queue.clear()
    save_queue(queue)
    return "All tasks in the queue have been processed."

def save_ui_state(prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed):
    state = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_inference_steps": num_inference_steps,
        "fps": fps,
        "seed": seed,
    }
    save_state(state)
    return "State saved!"

repo_id = "Lightricks/LTX-Video"
base_path = repo_id
files_to_download = [
        "model_index.json",
        "ltx-video-2b-v0.9.1.safetensors",
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
        "scheduler/scheduler_config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
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



single_file_url = repo_id+"/ltx-video-2b-v0.9.safetensors"
text_encoder = T5EncoderModel.from_pretrained(
  repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
)
tokenizer = T5Tokenizer.from_pretrained(
  repo_id, subfolder="tokenizer", torch_dtype=torch.bfloat16
)
pipe = LTXPipeline.from_single_file(
    single_file_url, 
    text_encoder=text_encoder, 
    tokenizer=tokenizer, 
    torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

def generate_video(prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed):
    # Randomize seed if seed is 0
    if seed == 0:
        seed = random.randint(0, 999999)
    
    # Generating the video <Does not support seed :( >
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
    ).frames[0]

    # Create output filename based on prompt and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prompt[:10]}_{timestamp}.mp4"
    
    # Save the video to the output folder
    os.makedirs("output_LTX", exist_ok=True)
    output_path = f"./output_LTX/{filename}"
    export_to_video(video, output_path, fps=fps)
    
    return output_path

# Gradio UI setup
def randomize_seed():
    return random.randint(0, 999999)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("Generate Video"):
            with gr.Row():
                prompt = gr.Textbox(label="Prompt", lines=3, value=initial_state.get("prompt", "A dramatic view of the pyramids at Giza during sunset."))
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=3, value=initial_state.get("negative_prompt", "worst quality, blurry, distorted"))
            with gr.Row():
                height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1, value=initial_state.get("height", 480))
                width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1, value=initial_state.get("width", 704))
            with gr.Row():
                num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1, value=initial_state.get("num_frames", 161))
                num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=initial_state.get("num_inference_steps", 50))
            with gr.Row():
                fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=initial_state.get("fps", 24))
                seed = gr.Number(label="Seed", value=initial_state.get("seed", 0))
                random_seed_button = gr.Button("Randomize Seed")

            output_video = gr.Video(label="Generated Video", show_label=True)
            generate_button = gr.Button("Generate Video")
            save_state_button = gr.Button("Save State")

            random_seed_button.click(lambda: random.randint(0, 999999), outputs=seed)
            generate_button.click(
                generate_video,
                inputs=[prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed],
                outputs=output_video
            )
            save_state_button.click(
                save_ui_state,
                inputs=[prompt, negative_prompt, height, width, num_frames, num_inference_steps, fps, seed],
                outputs=gr.Text(label="State Status")
            )

        with gr.Tab("Batch Processing"):
            with gr.Row():
                batch_prompt = gr.Textbox(label="Prompt", lines=3, value="A batch of videos depicting different landscapes.")
                batch_negative_prompt = gr.Textbox(label="Negative Prompt", lines=3, value="low quality, inconsistent, jittery")
            with gr.Row():
                batch_height = gr.Slider(label="Height", minimum=240, maximum=1080, step=1, value=480)
                batch_width = gr.Slider(label="Width", minimum=320, maximum=1920, step=1, value=704)
            with gr.Row():
                batch_num_frames = gr.Slider(label="Number of Frames", minimum=1, maximum=500, step=1, value=161)
                batch_num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=1, maximum=100, step=1, value=50)
            with gr.Row():
                batch_fps = gr.Slider(label="FPS", minimum=1, maximum=60, step=1, value=24)
                batch_seed = gr.Number(label="Seed", value=0)
                random_seed_batch_button = gr.Button("Randomize Seed")

            add_to_queue_button = gr.Button("Add to Queue")
            clear_queue_button = gr.Button("Clear Queue")
            process_queue_button = gr.Button("Process Queue")

            queue_status = gr.Text(label="Queue Status")

            random_seed_batch_button.click(lambda: random.randint(0, 999999), outputs=batch_seed)
            add_to_queue_button.click(
                add_to_queue,
                inputs=[batch_prompt, batch_negative_prompt, batch_height, batch_width, batch_num_frames, batch_num_inference_steps, batch_fps, batch_seed],
                outputs=queue_status
            )
            clear_queue_button.click(clear_queue, outputs=queue_status)
            process_queue_button.click(process_queue, outputs=queue_status)

demo.launch()
