#!/usr/bin/env python3
"""
Image-in of the people
"""

import argparse
import asyncio
import hashlib
import json
import yaml
import os
import subprocess
import sys
import re
import time
import toml


# Add project root to python path to import modules
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType


def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def merge_model_config(default_config: dict, model_config: dict) -> dict:
    """Merge default config with model-specific overrides."""
    merged = {}

    if isinstance(default_config, dict):
        merged.update(default_config)

    if isinstance(model_config, dict):
        merged.update(model_config)

    return merged if merged else None
def get_config_for_model(lrs_config: dict, model_name: str, specific_only: bool = False) -> dict:
    """Get configuration overrides based on model name."""
    if not isinstance(lrs_config, dict):
        return None

    data = lrs_config.get("data")
    default_config = lrs_config.get("default", {})

    if isinstance(data, dict) and model_name in data:
        return merge_model_config(default_config, data.get(model_name))

    if specific_only:
        return None

    if default_config:
        return default_config

    return None

def load_lrs_config(model_type: str, is_style: bool) -> dict:
    """Load the appropriate LRS configuration based on model type and training type"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, "lrs")

    if model_type == "flux":
        config_file = os.path.join(config_dir, "flux.json")
    elif is_style:
        config_file = os.path.join(config_dir, "style_config.json")
    else:
        config_file = os.path.join(config_dir, "person_config.json")
    
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load LRS config from {config_file}: {e}", flush=True)
        return None


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word=None):
    """Get the training data directory"""
    train_data_dir = train_paths.get_image_training_images_dir(task_id)

    """Create the diffusion config file"""
    config_template_path, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)

    is_ai_toolkit = model_type in [ImageModelType.Z_IMAGE.value, ImageModelType.QWEN_IMAGE.value]
    
    if is_ai_toolkit:
        with open(config_template_path, "r") as file:
            config = yaml.safe_load(file)
        if 'config' in config and 'process' in config['config']:
            for process in config['config']['process']:
                if 'model' in process:
                    process['model']['name_or_path'] = model_path
                    if 'training_folder' in process:
                        output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name or "output")
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir, exist_ok=True)
                        process['training_folder'] = output_dir
                
                if 'datasets' in process:
                    for dataset in process['datasets']:
                        dataset['folder_path'] = train_data_dir

                if trigger_word:
                    process['trigger_word'] = trigger_word
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        print(f"Created ai-toolkit config at {config_path}", flush=True)
        return config_path
    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)

        # Define network configurations
        network_config_person = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 467,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 467,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 228,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 699,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 467,
        "dataautogpt3/TempestV0.1": 456,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 467,
        "fluently/Fluently-XL-Final": 228,
        "mann-e/Mann-E_Dreams": 456,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 228,
        "recoilme/colorfulxl": 228,
        "zenless-lab/sdxl-aam-xl-anime-mix": 456,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 228,
        "zenless-lab/sdxl-anything-xl": 228,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 467,
        "Corcelio/mobius": 228,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 228
    }

    network_config_style = {
        "stabilityai/stable-diffusion-xl-base-1.0": 235,
        "Lykon/dreamshaper-xl-1-0": 235,
        "Lykon/art-diffusion-xl-0.9": 235,
        "SG161222/RealVisXL_V4.0": 235,
        "stablediffusionapi/protovision-xl-v6.6": 235,
        "stablediffusionapi/omnium-sdxl": 235,
        "GraydientPlatformAPI/realism-engine2-xl": 235,
        "GraydientPlatformAPI/albedobase2-xl": 235,
        "KBlueLeaf/Kohaku-XL-Zeta": 235,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 235,
        "John6666/nova-anime-xl-pony-v5-sdxl": 235,
        "cagliostrolab/animagine-xl-4.0": 235,
        "dataautogpt3/CALAMITY": 235,
        "dataautogpt3/ProteusSigma": 235,
        "dataautogpt3/ProteusV0.5": 235,
        "dataautogpt3/TempestV0.1": 228,
        "ehristoforu/Visionix-alpha": 235,
        "femboysLover/RealisticStockPhoto-fp16": 235,
        "fluently/Fluently-XL-Final": 235,
        "mann-e/Mann-E_Dreams": 235,
        "misri/leosamsHelloworldXL_helloworldXL70": 235,
        "misri/zavychromaxl_v90": 235,
        "openart-custom/DynaVisionXL": 235,
        "recoilme/colorfulxl": 235,
        "zenless-lab/sdxl-aam-xl-anime-mix": 235,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 235,
        "zenless-lab/sdxl-anything-xl": 235,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 235,
        "Corcelio/mobius": 235,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 235,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 235
    }

    config_mapping = {
        228: {
            "network_dim": 32,
            "network_alpha": 32,
            "network_args": []
        },
        235: {
            "network_dim": 32,
            "network_alpha": 32,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
        },
        456: {
            "network_dim": 64,
            "network_alpha": 64,
            "network_args": []
        },
        467: {
            "network_dim": 64,
            "network_alpha": 64,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
        },
        699: {
            "network_dim": 96,
            "network_alpha": 96,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]
        },
    }
    
    # --- CONFIGURATION PIPELINE ---
    
    # 1. Apply Hardcoded Network Defaults FIRST (Base Layer)
    if model_type == "sdxl":
        if is_style:
            if model_name in network_config_style:
                 network_id = network_config_style[model_name]
                 if network_id in config_mapping:
                    network_config = config_mapping[network_id]
                    config["network_dim"] = network_config["network_dim"]
                    config["network_alpha"] = network_config["network_alpha"]
                    config["network_args"] = network_config["network_args"]
        else:
            if model_name in network_config_person:
                 network_id = network_config_person[model_name]
                 if network_id in config_mapping:
                    network_config = config_mapping[network_id]
                    config["network_dim"] = network_config["network_dim"]
                    config["network_alpha"] = network_config["network_alpha"]
                    config["network_args"] = network_config["network_args"]

    # 2. Apply LRS Overrides (Top Layer) - The "Universal Patcher"
    lrs_settings = None
    source_file = None
    model_hash = hash_model(model_name)
    print(f"🔍 Calculated Model Hash: {model_hash}")
    
    # Determine config library path
    config_dir = os.path.join(script_dir, "lrs")
    
    # Priority List for Cross-Lookup
    files_to_check = []
    if model_type == "flux":
        files_to_check = ["flux.json"]
    else:
        primary = "style_config.json" if is_style else "person_config.json"
        secondary = "person_config.json" if is_style else "style_config.json"
        files_to_check = [primary, secondary]

    # --- PHASE 1: SEARCH SPECIFIC MATCH ---
    for filename in files_to_check:
        config_path = os.path.join(config_dir, filename)
        if not os.path.exists(config_path):
            continue
            
        try:
            with open(config_path, 'r') as f:
                current_lrs = json.load(f)
            
            # Try specific Hash match first (Specific Only = No Default fallback yet)
            match = get_config_for_model(current_lrs, model_hash, specific_only=True)
            if match:
                lrs_settings = match
                source_file = filename
                print(f"✅ Found Specific Hash match in {filename}!")
                break
                
            # Try specific Repo Name match second
            if expected_repo_name:
                match = get_config_for_model(current_lrs, expected_repo_name, specific_only=True)
                if match:
                    lrs_settings = match
                    source_file = filename
                    print(f"✅ Found Specific Repo Name match in {filename}!")
                    break
        except Exception as e:
            print(f"⚠️ Error reading {filename}: {e}")
            continue

    # --- PHASE 2: FALLBACK TO PRIMARY DEFAULT ---
    if not lrs_settings:
        primary_filename = files_to_check[0]
        config_path = os.path.join(config_dir, primary_filename)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    primary_lrs = json.load(f)
                lrs_settings = primary_lrs.get("default", {})
                source_file = primary_filename
                if lrs_settings:
                    print(f"ℹ️ No specific hash found. Using Defaults from {primary_filename}.")
            except: pass

    # --- PHASE 3: APPLY OVERRIDES ---
    if lrs_settings:
        print(f"🚀 APPLYING LRS OVERRIDES from {source_file}...")
        for key, value in lrs_settings.items():
            config[key] = value
            print(f"   -> [OVERRIDE] {key}: {value}")
    else:
        print("⚠️ Warning: No LRS configuration (Specific or Default) could be applied.")

    # Update config
    config["pretrained_model_name_or_path"] = model_path
    config["train_data_dir"] = train_data_dir
    output_dir = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    config["output_dir"] = output_dir

    # Save config to file
    config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
    save_config_toml(config, config_path)
    print(f"config is {config}", flush=True)
    print(f"Created config at {config_path}", flush=True)
    return config_path


def run_training(model_type, config_path):
    print(f"Starting training with config: {config_path}", flush=True)

    if model_type == "sdxl":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-script/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    elif model_type == "flux":
        training_command = [
            "accelerate", "launch",
            "--dynamo_backend", "no",
            "--dynamo_mode", "default",
            "--mixed_precision", "bf16",
            "--num_processes", "1",
            "--num_machines", "1",
            "--num_cpu_threads_per_process", "2",
            f"/app/sd-scripts/{model_type}_train_network.py",
            "--config_file", config_path
        ]
    elif model_type in ["z-image", "qwen-image"]:
        training_command = [
            "python3",
            "/app/ai-toolkit/run.py",
            config_path
        ]

    try:
        print("Starting training subprocess...\n", flush=True)
        process = subprocess.Popen(
            training_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            print(line, end="", flush=True)

        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, training_command)

        print("Training subprocess completed successfully.", flush=True)

    except subprocess.CalledProcessError as e:
        print("Training subprocess failed!", flush=True)
        print(f"Exit Code: {e.returncode}", flush=True)
        print(f"Command: {' '.join(e.cmd) if isinstance(e.cmd, list) else e.cmd}", flush=True)
        raise RuntimeError(f"Training subprocess failed with exit code {e.returncode}")

def hash_model(model: str) -> str:
    model_bytes = model.encode('utf-8')
    hashed = hashlib.sha256(model_bytes).hexdigest()
    return hashed 

async def main():
    print("---STARTING IMAGE TRAINING SCRIPT---", flush=True)
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--dataset-zip", required=True, help="Link to dataset zip file")
    parser.add_argument("--model-type", required=True, choices=["sdxl", "flux", "z-image", "qwen-image"], help="Model type")
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument("--hours-to-complete", type=float, required=True, help="Number of hours to complete the task")
    parser.add_argument("--trigger-word", help="Trigger word for training")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)

    model_path = train_paths.get_image_base_model_path(args.model)

    # Prepare dataset
    print("Preparing dataset...", flush=True)

    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == ImageModelType.SDXL.value else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=args.trigger_word if args.trigger_word else cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=args.task_id,
        output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    # Create config file
    config_path = create_config(
        args.task_id,
        model_path,
        args.model,
        args.model_type,
        args.expected_repo_name,
        trigger_word=args.trigger_word
    )

    # Run training
    run_training(args.model_type, config_path)


if __name__ == "__main__":
    asyncio.run(main())
