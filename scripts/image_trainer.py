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





def get_model_path(path: str) -> str:
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        if len(files) == 1 and files[0].endswith(".safetensors"):
            return os.path.join(path, files[0])
    return path
def calculate_adaptive_steps(train_data_dir, is_style):
    """
    Autopilot V6 (Step-Based Precision):
    Calculates exact Target Steps instead of Epochs.
    This bypasses the 'Hidden Repeats' multiplier problem in datasets.
    """
    try:
        # 1. Count images Recursively (Fix for subfolders)
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        num_images = 0
        for root, dirs, files in os.walk(train_data_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    num_images += 1
        
        if num_images == 0:
            print(f"⚠️ Autopilot found 0 images in {train_data_dir}. Using Default 1000 Steps.")
            return 1000, 1, 0 # Steps, BS, Epochs(Placeholder)

        # 2. DEFINE BASELINE EXPOSURE (Step Logic)
        # Target Steps = Num_Images * Exposure_Per_Image
        
        if num_images < 15:
            # Micro Dataset (<15 Img)
            # Autopilot V7: Target ~240 Steps (Champion Standard)
            # 9 images * 35 exp = 315 / BS 1 -> Wait, let's target 240.
            # 240 steps / 9 img = ~26. 
            rec_batch_size = 1
            exposure = 30 # Rollback to safe baseline
            hard_cap = 650 
        elif num_images < 50:
            # Small Dataset
            rec_batch_size = 2
            exposure = 60 # Strictly back to V6 baseline
            hard_cap = 1000 # Strictly back to V6 baseline
        else:
            # Large Dataset
            rec_batch_size = 4
            exposure = 40
            hard_cap = 1600
            
        # 3. ENTROPY MODIFIER
        if is_style:
            # Style task needs LESS focus per image to avoid style rigidity
            exposure = int(exposure * 0.8) # 30 * 0.8 = 24. 9 * 24 = 216 Steps. Perfect.
            hard_cap = int(hard_cap * 0.8)
            
        # 4. GRADIENT STABILITY (BS Adjustment)
        # Surgical Fix: Only drop to BS 1 if images per batch is truly too low (< 6)
        # This keeps 18 images at BS 2 (18/2=9) but Visionix 45 images (45/2=22) remains untouched.
        if (num_images // rec_batch_size) < 6 and rec_batch_size > 1:
            rec_batch_size = max(1, rec_batch_size - 1)

        # 5. FINAL CALCULATION
        ideal_steps = num_images * exposure
        
        # Adjust for Batch Size (Steps = Total_Img_Exposure / BS)
        # Exposure is "How many times we process the image".
        # If BS=1, Steps = Num * Exp.
        # If BS=2, Steps = (Num * Exp) / 2.
        final_steps = int(ideal_steps / rec_batch_size)
        
        # Apply Hard Cap
        final_steps = min(final_steps, hard_cap)
        
        # Enforce Minimum (Warmup)
        final_steps = max(final_steps, 200)

        print(f"[Autopilot V6] Data: {num_images} img (Recursive) | Type: {'Style' if is_style else 'Person'}")
        print(f"[Autopilot V6] Plan: BS {rec_batch_size} | Exposure {exposure}x | Target Steps {final_steps} (Cap {hard_cap})")
        
        return final_steps, rec_batch_size, num_images
        
    except Exception as e:
        print(f"[Autopilot] Error: {e}. Using Default 1000 Steps.")
        return 1000, 1, 0


def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word: str | None = None):
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
                    # Qwen needs the directory to find config.json, Z-image needs the file
                    if model_type == "qwen-image":
                        process['model']['name_or_path'] = model_path
                    else:
                        process['model']['name_or_path'] = get_model_path(model_path)
                    
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
                
                # FINAL PROTECTOR: Fix Assistant LoRA Path (Inside Loop)
                if 'model' in process and process['model'].get('assistant_lora_path'):
                    lora_path = process['model']['assistant_lora_path']
                    if not os.path.exists(lora_path):
                        print(f"🔍 [AI-Toolkit] Assistant LoRA not found at {lora_path}. Searching alternatives...")
                        # Search in the same directory as the model
                        full_model_path = get_model_path(model_path)
                        model_dir = full_model_path if os.path.isdir(full_model_path) else os.path.dirname(full_model_path)
                        alt_path = os.path.join(model_dir, os.path.basename(lora_path))
                        
                        if os.path.exists(alt_path):
                            process['model']['assistant_lora_path'] = alt_path
                            print(f"🎯 [AI-Toolkit] Redirected Assistant LoRA to: {alt_path}")
                        else:
                            # If still not found, search any .safetensors in that folder
                            found = False
                            for f in os.listdir(model_dir):
                                if f.endswith(".safetensors") and ("adapter" in f.lower() or "lora" in f.lower()):
                                    process['model']['assistant_lora_path'] = os.path.join(model_dir, f)
                                    print(f"✨ [AI-Toolkit] Found alternative adapter: {process['model']['assistant_lora_path']}")
                                    found = True
                                    break
                            
                            if not found:
                                print(f"⚠️ [AI-Toolkit] NO ADAPTER FOUND. Removing key to prevent crash.")
                                process['model'].pop('assistant_lora_path', None)
        
                # Qwen Quantization Fail-safe
                if model_type == "qwen-image" and 'model' in process:
                    qtype = process['model'].get('qtype', '')
                    if '|' in qtype:
                        quant_file = qtype.split('|')[1]
                        if not os.path.exists(quant_file):
                            print(f"⚠️ [AI-Toolkit] Quantization file {quant_file} not found. FALLING BACK to standard BF16 mode.")
                            process['model']['quantize'] = False
                            process['model']['quantize_te'] = False
                            process['model'].pop('qtype', None)
        
        # 2. LRS Override System (The "Universal Patcher" for Toolkit)
        model_hash = hash_model(model_name)
        print(f"🔍 [AI-Toolkit] Calculated Model Hash: {model_hash}")
        
        config_dir = os.path.join(script_dir, "lrs")
        files_to_check = ["flux.json", "person_config.json", "style_config.json"] # Broad look for toolkit
        lrs_settings = None

        for filename in files_to_check:
            lrs_path = os.path.join(config_dir, filename)
            if os.path.exists(lrs_path):
                try:
                    with open(lrs_path, 'r') as f:
                        current_lrs = json.load(f)
                    match = get_config_for_model(current_lrs, model_hash, specific_only=True)
                    if not match and expected_repo_name:
                        match = get_config_for_model(current_lrs, expected_repo_name, specific_only=True)
                    
                    if match:
                        lrs_settings = match
                        print(f"✅ [AI-Toolkit] Found Specific Overrides in {filename}!")
                        break
                except: continue

        # Apply Overrides to YAML structure
        if lrs_settings:
            print(f"🚀 [AI-Toolkit] APPLYING OVERRIDES...")
            
            # Map common LRS keys to Toolkit keys
            key_map = {
                "unet_lr": "lr",
                "max_train_steps": "steps",
                "train_batch_size": "batch_size",
                "max_train_epochs": "epochs"
            }
            
            final_overrides = lrs_settings.copy()
            for old_key, new_key in key_map.items():
                if old_key in lrs_settings:
                    final_overrides[new_key] = lrs_settings[old_key]

            # Patcher: Search and replace common keys in YAML
            def patch_toolkit_config(obj, overrides):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in overrides:
                            obj[k] = overrides[k]
                            print(f"   -> [PATCHED] {k}: {overrides[k]}")
                        patch_toolkit_config(v, overrides)
                elif isinstance(obj, list):
                    for item in obj:
                        patch_toolkit_config(item, overrides)

            patch_toolkit_config(config, final_overrides)
        
        config_path = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, config_path)
        
        # DEBUG: Show final yaml content
        print("\n--- FINAL YAML CONFIG PREVIEW ---")
        print(yaml.dump(config, default_flow_style=False))
        print("---------------------------------\n")
        
        return config_path


    else:
        with open(config_template_path, "r") as file:
            config = toml.load(file)



        # Initialize network_id with safe default (Artistic/General)
        network_id = 99

        # Define network configurations
        # --- LAYER 2: MODEL CLASSIFICATION ---
        network_config_person = {
        # ANIME & ILLUSTRATION (ID: 9)
        "zenless-lab/sdxl-aam-xl-anime-mix": 9,
        "John6666/nova-anime-xl-pony-v5-sdxl": 9,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 9,
        "cagliostrolab/animagine-xl-4.0": 9,
        "zenless-lab/sdxl-anything-xl": 9,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 9,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 9,
        "KBlueLeaf/Kohaku-XL-Zeta": 9,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 9,

        # PHOTOREALISTIC (ID: 69)
        "misri/leosamsHelloworldXL_helloworldXL70": 69,
        "GraydientPlatformAPI/albedobase2-xl": 69,
        "femboysLover/RealisticStockPhoto-fp16": 69,
        "ifmain/UltraReal_Fine-Tune": 69,
        "GraydientPlatformAPI/realism-engine2-xl": 69,
        "SG161222/RealVisXL_V4.0": 69,

        # ARTISTIC / 2.5D / GENERALIST (ID: 99)
        "dataautogpt3/CALAMITY": 99,
        "recoilme/colorfulxl": 99,
        "dataautogpt3/ProteusV0.5": 99,
        "fluently/Fluently-XL-Final": 99,
        "stabilityai/stable-diffusion-xl-base-1.0": 99,
        "openart-custom/DynaVisionXL": 99,
        "Lykon/dreamshaper-xl-1-0": 99,
        "dataautogpt3/ProteusSigma": 99,
        "mann-e/Mann-E_Dreams": 99,
        "Corcelio/mobius": 99,
        "ehristoforu/Visionix-alpha": 99,
        "Lykon/art-diffusion-xl-0.9": 99,
        "stablediffusionapi/omnium-sdxl": 99,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 99,
        "misri/zavychromaxl_v90": 99,
        "stablediffusionapi/protovision-xl-v6.6": 99,
        "dataautogpt3/TempestV0.1": 99,
        "bghira/terminus-xl-velocity-v2": 99
    }

    network_config_style = {
        # ANIME & ILLUSTRATION (ID: 8) - Same models, different ID for Style tasks
        "zenless-lab/sdxl-aam-xl-anime-mix": 8,
        "John6666/nova-anime-xl-pony-v5-sdxl": 8,
        "zenless-lab/sdxl-anima-pencil-xl-v5": 8,
        "cagliostrolab/animagine-xl-4.0": 8,
        "zenless-lab/sdxl-anything-xl": 8,
        "OnomaAIResearch/Illustrious-xl-early-release-v0": 8,
        "John6666/hassaku-xl-illustrious-v10style-sdxl": 8,
        "KBlueLeaf/Kohaku-XL-Zeta": 8,
        "zenless-lab/sdxl-blue-pencil-xl-v7": 8,

        # PHOTOREALISTIC (ID: 78)
        "misri/leosamsHelloworldXL_helloworldXL70": 78,
        "GraydientPlatformAPI/albedobase2-xl": 78,
        "femboysLover/RealisticStockPhoto-fp16": 78,
        "ifmain/UltraReal_Fine-Tune": 78,
        "GraydientPlatformAPI/realism-engine2-xl": 78,
        "SG161222/RealVisXL_V4.0": 78,

        # ARTISTIC / 2.5D / GENERALIST (ID: 118)
        "dataautogpt3/CALAMITY": 118,
        "recoilme/colorfulxl": 118,
        "dataautogpt3/ProteusV0.5": 118,
        "fluently/Fluently-XL-Final": 118,
        "stabilityai/stable-diffusion-xl-base-1.0": 118,
        "openart-custom/DynaVisionXL": 118,
        "Lykon/dreamshaper-xl-1-0": 118,
        "dataautogpt3/ProteusSigma": 118,
        "mann-e/Mann-E_Dreams": 118,
        "Corcelio/mobius": 118,
        "ehristoforu/Visionix-alpha": 118,
        "Lykon/art-diffusion-xl-0.9": 118,
        "stablediffusionapi/omnium-sdxl": 118,
        "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 118,
        "misri/zavychromaxl_v90": 118,
        "stablediffusionapi/protovision-xl-v6.6": 118,
        "dataautogpt3/TempestV0.1": 118,
        "bghira/terminus-xl-velocity-v2": 118
    }

    # --- LAYER 3: CONFIG MAPPING (ARCHITECTURE PARAMS) ---
    config_mapping = {
        # PERSON: ANIME (ID: 9)
        9: {
            "network_dim": 128,          # High Capacity for Anime Details
            "network_alpha": 64,         # Stable Alpha
            "network_args": ["conv_dim=8", "conv_alpha=4", "dropout=null"],
            "clip_skip": 2,              # WAJIB untuk Anime
            "noise_offset": 0.0357       # Standard
        },
        # PERSON: REALIS (ID: 69)
        69: {
            "network_dim": 64,           # Medium Capacity for Realism
            "network_alpha": 32,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"],
            "clip_skip": 1,              # Realism need accurate semantics
            "noise_offset": 0.02         # Low Noise for Clean Texture
        },
        # PERSON: ARTISTIC (ID: 99)
        99: {
            "network_dim": 64,           # Balanced Capacity
            "network_alpha": 32,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"],
            "clip_skip": 1,
            "noise_offset": 0.0357
        },

        # STYLE: ANIME (ID: 8)
        8: {
            "network_dim": 128,
            "network_alpha": 32,         # Lower Alpha for Style (Less overfit)
            "network_args": ["conv_dim=8", "conv_alpha=4", "dropout=null"],
            "clip_skip": 2,
            "noise_offset": 0.0357
        },
        # STYLE: REALIS (ID: 78)
        78: {
            "network_dim": 64,
            "network_alpha": 32,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"],
            "clip_skip": 1,
            "noise_offset": 0.02
        },
        # STYLE: ARTISTIC (ID: 118)
        118: {
            "network_dim": 64,
            "network_alpha": 32,
            "network_args": ["conv_dim=4", "conv_alpha=4", "dropout=null"],
            "clip_skip": 1,
            "noise_offset": 0.035         # Balanced Noise for Style
        }
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

    # 2. Apply LRS Strategy (Category Based)
    lrs_settings = None
    source_file = None
    
    # Determine config library path
    config_dir = os.path.join(script_dir, "lrs")
    
    # Select Primary Config File
    primary_filename = "flux.json" if model_type == "flux" else ("style_config.json" if is_style else "person_config.json")
    config_path = os.path.join(config_dir, primary_filename)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                primary_lrs = json.load(f)
            
            # Determine Category Key based on Config ID
            target_default_key = "default_artistic" # Safe fallback
            
            if network_id in [9, 8]:
                target_default_key = "default_anime"
            elif network_id in [69, 78]:
                target_default_key = "default_realis"
            
            # Flux Handling (Single Default for now)
            if model_type == "flux":
                 target_default_key = "default"

            lrs_settings = primary_lrs.get(target_default_key, {})
            
            # Fallback to generic 'default' if specific category missing
            if not lrs_settings:
                    lrs_settings = primary_lrs.get("default", {})

            source_file = primary_filename
            if lrs_settings:
                print(f"✅ [LRS Strategy] Loaded '{target_default_key}' from {primary_filename}.")
        except Exception as e:
            print(f"⚠️ Error loading strategy from {primary_filename}: {e}")

    # --- PHASE 3: APPLY SETTINGS TO CONFIG ---
    print(f"🚀 [Config Logic] Calculating Autopilot V6 (Step-Based)...")
    target_steps, rec_batch_size, num_images_found = calculate_adaptive_steps(train_data_dir, is_style)

    # 1. Steps Strategy (Primary Control)
    override_steps = lrs_settings.get("max_train_steps") if lrs_settings else None
    
    if override_steps:
         config["max_train_steps"] = override_steps
         config.pop("max_train_epochs", None) # Remove Epochs to avoid conflict
         print(f"[Config] Using MANUAL overrides for Steps: {override_steps} (Epochs Removed)")
    else:
         config["max_train_steps"] = target_steps
         config.pop("max_train_epochs", None) # Remove Epochs to avoid conflict
         print(f"[Config] Using AUTOPILOT V6 Steps: {target_steps} (Epochs Removed)")

    # 2. Batch Size Strategy
    override_bs = lrs_settings.get("train_batch_size") if lrs_settings else None
    
    if override_bs:
         config["train_batch_size"] = override_bs
         print(f"[Config] Using MANUAL overrides for Batch Size: {override_bs}")
    else:
         config["train_batch_size"] = rec_batch_size
         print(f"[Config] Using AUTOPILOT V6 Batch Size: {rec_batch_size}")

    # 3. Apply other LRS settings
    if lrs_settings:
        for key, value in lrs_settings.items():
            if key not in ["max_train_epochs", "train_batch_size", "network_args"]:
                config[key] = value
                print(f"   -> [OVERRIDE] {key}: {value}")

    # 4. Apply Network Args (Special Handling)
    if lrs_settings and "network_args" in lrs_settings:
         config["network_args"] = lrs_settings["network_args"]
    else:
         # Default Network Args if none provided (Standard 128/64/Conv16/8)
         config["network_args"] = [ "conv_dim=16", "conv_alpha=8", "dropout=null" ]
         
    print(f"✅ Final Configuration Applied. Steps: {config.get('max_train_steps')}, BS: {config['train_batch_size']}")


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
            bufsize=1,
            cwd="/app/ai-toolkit" if model_type in ["z-image", "qwen-image"] else None
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

    model_path = train_paths.get_image_base_model_path(args.model, args.model_type)

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
