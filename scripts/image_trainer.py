#!/usr/bin/env python3
"""
Playground - Mr. Toothless
Tournament Edition - Final FIXED & STABLE (Clean Version)
"""

import argparse
import asyncio
import hashlib
import json
import yaml
import os
import subprocess
import sys
import toml

# Project Path Setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import core.constants as cst
import trainer.constants as train_cst
import trainer.utils.training_paths as train_paths
from core.config.config_handler import save_config, save_config_toml
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import ImageModelType

# --- HELPERS ---

def get_model_path(path: str) -> str:
    """Finds the actual .safetensors file ONLY for single-file models"""
    if os.path.isdir(path):
        files = [f for f in os.listdir(path) if f.endswith(".safetensors")]
        if len(files) == 1: return os.path.join(path, files[0])
    return path

def hash_model(model_name: str) -> str:
    return hashlib.sha256(model_name.encode()).hexdigest()

def get_config_for_model(config_dict, model_id, specific_only=False):
    if model_id in config_dict: return config_dict[model_id]
    return None if specific_only else config_dict.get("default")

def patch_toolkit(obj, overrides):
    """Recursively applies overrides to nested dictionaries (AI-Toolkit)"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in overrides:
                obj[k] = overrides[k]
                print(f"   -> [PATCH] {k}: {overrides[k]}")
            patch_toolkit(v, overrides)
    elif isinstance(obj, list):
        for item in obj: patch_toolkit(item, overrides)

# --- CORE LOGIC ---

def calculate_adaptive_steps(train_data_dir, is_style, model_type="sdxl"):
    try:
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        num_images = sum(1 for r, d, files in os.walk(train_data_dir) for f in files if os.path.splitext(f)[1].lower() in exts)
        if num_images == 0: return 1000, 1, 0

        if model_type == "flux":
            bs = 4 if num_images > 20 else 2
            if num_images < 15: bs = 1
            exposure, hard_cap = 120, 1200
        elif num_images < 15:
            bs, exposure, hard_cap = 1, 30, 650
        elif num_images < 50:
            bs, exposure, hard_cap = 2, 60, 1000
        else:
            bs, exposure, hard_cap = 4, 40, 1600
            
        if is_style:
            exposure = int(exposure * 0.8)
            hard_cap = int(hard_cap * 0.8)
            
        if (num_images // bs) < 6 and bs > 1: bs = max(1, bs - 1)
        final_steps = max(200, min(int((num_images * exposure) / bs), hard_cap))
        
        print(f"[Autopilot] Data: {num_images} | BS: {bs} | Exposure: {exposure}x | Steps: {final_steps}")
        return final_steps, bs, num_images
    except: return 1000, 1, 0

def create_config(task_id, model_path, model_name, model_type, expected_repo_name, trigger_word=None):
    train_data_dir = train_paths.get_image_training_images_dir(task_id)
    tmpl_p, is_style = train_paths.get_image_training_config_template_path(model_type, train_data_dir)
    is_toolkit = model_type in ["z-image", "qwen-image"]

    if is_toolkit:
        with open(tmpl_p, "r") as f: config = yaml.safe_load(f)
        for proc in config.get('config', {}).get('process', []):
            if 'model' in proc:
                proc['model']['name_or_path'] = model_path
                
                if 'training_folder' in proc:
                    out = train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
                    os.makedirs(out, exist_ok=True)
                    proc['training_folder'] = out
                
                # AGGRESSIVE ADAPTER SEARCH (Fix for Z-Image)
                if proc['model'].get('assistant_lora_path'):
                    lora_path = proc['model']['assistant_lora_path']
                    if not os.path.exists(lora_path):
                        print(f"[AI-Toolkit] Assistant LoRA not found at {lora_path}. Searching...")
                        m_file = get_model_path(model_path)
                        m_dir = m_file if os.path.isdir(m_file) else os.path.dirname(m_file)
                        
                        # Try exact name in model dir
                        alt = os.path.join(m_dir, os.path.basename(lora_path))
                        if os.path.exists(alt):
                            proc['model']['assistant_lora_path'] = alt
                            print(f"[AI-Toolkit] Found adapter: {alt}")
                        else:
                            # Try any .safetensors in model dir
                            found = False
                            for f in os.listdir(m_dir):
                                if f.endswith(".safetensors") and ("adapter" in f.lower() or "lora" in f.lower()):
                                    proc['model']['assistant_lora_path'] = os.path.join(m_dir, f)
                                    print(f"[AI-Toolkit] Found alternative adapter: {proc['model']['assistant_lora_path']}")
                                    found = True
                                    break
                            if not found:
                                print(f"[WARNING] [AI-Toolkit] NO ADAPTER FOUND. Removing key to prevent crash.")
                                proc['model'].pop('assistant_lora_path', None)
        
            if 'datasets' in proc:
                for ds in proc['datasets']: ds['folder_path'] = train_data_dir
            if trigger_word: proc['trigger_word'] = trigger_word

        m_hash = hash_model(model_name)
        cfg_dir = os.path.join(script_dir, "lrs")
        for fn in ["flux.json", "person_config.json", "style_config.json"]:
            lrs_p = os.path.join(cfg_dir, fn)
            if os.path.exists(lrs_p):
                with open(lrs_p, 'r') as f: lrs_lib = json.load(f)
                match = get_config_for_model(lrs_lib, m_hash, True) or get_config_for_model(lrs_lib, expected_repo_name, True)
                if match:
                    print(f"[AI-Toolkit] Applying overrides from {fn}")
                    key_map = {"unet_lr": "lr", "max_train_steps": "steps", "train_batch_size": "batch_size"}
                    final_ovr = {key_map.get(k, k): v for k, v in match.items()}
                    patch_toolkit(config, final_ovr)
                    break
        
        save_p = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.yaml")
        save_config(config, save_p)
        return save_p

    else:
        with open(tmpl_p, "r") as f: config = toml.load(f)
        sdxl_person_map = {
            "zenless-lab/sdxl-aam-xl-anime-mix": 9, "John6666/nova-anime-xl-pony-v5-sdxl": 9,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 9, "cagliostrolab/animagine-xl-4.0": 9,
            "zenless-lab/sdxl-anything-xl": 9, "OnomaAIResearch/Illustrious-xl-early-release-v0": 9,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 9, "KBlueLeaf/Kohaku-XL-Zeta": 9,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 9, 
            "misri/leosamsHelloworldXL_helloworldXL70": 69, "GraydientPlatformAPI/albedobase2-xl": 69, 
            "femboysLover/RealisticStockPhoto-fp16": 69, "ifmain/UltraReal_Fine-Tune": 69, 
            "GraydientPlatformAPI/realism-engine2-xl": 69, "SG161222/RealVisXL_V4.0": 69,
            "dataautogpt3/CALAMITY": 99, "recoilme/colorfulxl": 99, "dataautogpt3/ProteusV0.5": 99,
            "fluently/Fluently-XL-Final": 99, "stabilityai/stable-diffusion-xl-base-1.0": 99,
            "openart-custom/DynaVisionXL": 99, "Lykon/dreamshaper-xl-1-0": 99, "dataautogpt3/ProteusSigma": 99,
            "mann-e/Mann-E_Dreams": 99, "Corcelio/mobius": 99, "ehristoforu/Visionix-alpha": 99,
            "Lykon/art-diffusion-xl-0.9": 99, "stablediffusionapi/omnium-sdxl": 99,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 99, "misri/zavychromaxl_v90": 99,
            "stablediffusionapi/protovision-xl-v6.6": 99, "dataautogpt3/TempestV0.1": 99,
            "bghira/terminus-xl-velocity-v2": 99
        }
        sdxl_style_map = {
            "zenless-lab/sdxl-aam-xl-anime-mix": 8, "John6666/nova-anime-xl-pony-v5-sdxl": 8,
            "zenless-lab/sdxl-anima-pencil-xl-v5": 8, "cagliostrolab/animagine-xl-4.0": 8,
            "zenless-lab/sdxl-anything-xl": 8, "OnomaAIResearch/Illustrious-xl-early-release-v0": 8,
            "John6666/hassaku-xl-illustrious-v10style-sdxl": 8, "KBlueLeaf/Kohaku-XL-Zeta": 8,
            "zenless-lab/sdxl-blue-pencil-xl-v7": 8, 
            "misri/leosamsHelloworldXL_helloworldXL70": 78, "GraydientPlatformAPI/albedobase2-xl": 78, 
            "femboysLover/RealisticStockPhoto-fp16": 78, "ifmain/UltraReal_Fine-Tune": 78, 
            "GraydientPlatformAPI/realism-engine2-xl": 78, "SG161222/RealVisXL_V4.0": 78,
            "dataautogpt3/CALAMITY": 118, "recoilme/colorfulxl": 118, "dataautogpt3/ProteusV0.5": 118,
            "fluently/Fluently-XL-Final": 118, "stabilityai/stable-diffusion-xl-base-1.0": 118,
            "openart-custom/DynaVisionXL": 118, "Lykon/dreamshaper-xl-1-0": 118, "dataautogpt3/ProteusSigma": 118,
            "mann-e/Mann-E_Dreams": 118, "Corcelio/mobius": 118, "ehristoforu/Visionix-alpha": 118,
            "Lykon/art-diffusion-xl-0.9": 118, "stablediffusionapi/omnium-sdxl": 118,
            "GHArt/Lah_Mysterious_SDXL_V4.0_xl_fp16": 118, "misri/zavychromaxl_v90": 118,
            "stablediffusionapi/protovision-xl-v6.6": 118, "dataautogpt3/TempestV0.1": 118,
            "bghira/terminus-xl-velocity-v2": 118
        }
        
        lrs_key = "default"
        if model_type == "sdxl":
            nid = sdxl_style_map.get(model_name, 118) if is_style else sdxl_person_map.get(model_name, 99)
            archs = {
                8:  {"dim": 128, "alpha": 32, "args": ["conv_dim=8", "conv_alpha=4", "dropout=null"]},
                9:  {"dim": 128, "alpha": 64, "args": ["conv_dim=8", "conv_alpha=4", "dropout=null"]},
                69: {"dim": 64,  "alpha": 32, "args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]},
                78: {"dim": 64,  "alpha": 32, "args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]},
                99: {"dim": 64,  "alpha": 32, "args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]},
                118: {"dim": 64, "alpha": 32, "args": ["conv_dim=4", "conv_alpha=4", "dropout=null"]}
            }
            arch = archs.get(nid, archs[99])
            config.update({"network_dim": arch["dim"], "network_alpha": arch["alpha"], "network_args": arch["args"]})
            
            if nid in [8, 9]: lrs_key = "default_anime"
            elif nid in [69, 78]: lrs_key = "default_realis"
            elif nid in [99, 118]: lrs_key = "default_artistic"

        lrs_fn = "flux.json" if model_type == "flux" else ("style_config.json" if is_style else "person_config.json")
        lrs_p = os.path.join(script_dir, "lrs", lrs_fn)
        lrs_settings = {}
        if os.path.exists(lrs_p):
            with open(lrs_p, 'r') as f:
                lib = json.load(f)
                m_hash = hash_model(model_name)
                lrs_settings = get_config_for_model(lib, m_hash, True) or get_config_for_model(lib, expected_repo_name, True) or lib.get(lrs_key, {}) or lib.get("default", {})
                print(f"[LRS Strategy] Applied settings from {lrs_fn}")

        steps, bs, _ = calculate_adaptive_steps(train_data_dir, is_style, model_type)
        config.update({
            "max_train_steps": lrs_settings.get("max_train_steps", steps),
            "train_batch_size": lrs_settings.get("train_batch_size", bs),
            "pretrained_model_name_or_path": model_path if model_type == "flux" else get_model_path(model_path),
            "train_data_dir": train_data_dir,
            "output_dir": train_paths.get_checkpoints_output_path(task_id, expected_repo_name)
        })
        config.pop("max_train_epochs", None)
        for k, v in lrs_settings.items():
            if k not in ["max_train_steps", "train_batch_size"]: config[k] = v

        save_p = os.path.join(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, f"{task_id}.toml")
        save_config_toml(config, save_p)
        return save_p

def run_training(model_type, config_path):
    print(f"Starting {model_type.upper()} training...")
    if model_type in ["sdxl", "flux"]:
        script = "flux_train_network.py" if model_type == "flux" else "sdxl_train_network.py"
        cmd = ["accelerate", "launch", "--mixed_precision", "bf16", "--num_cpu_threads_per_process", "2", f"/app/sd-script/{script}", "--config_file", config_path]
    else:
        cmd = ["python3", "/app/ai-toolkit/run.py", config_path]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd="/app/ai-toolkit" if model_type in ["z-image", "qwen-image"] else None)
        for line in proc.stdout: print(line, end="", flush=True)
        if proc.wait() != 0: raise RuntimeError("Training failed")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True); parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-zip", required=True); parser.add_argument("--model-type", required=True)
    parser.add_argument("--expected-repo-name"); parser.add_argument("--hours-to-complete", type=float, required=True); parser.add_argument("--trigger-word")
    args = parser.parse_args()

    os.makedirs(train_cst.IMAGE_CONTAINER_CONFIG_SAVE_PATH, exist_ok=True)
    os.makedirs(train_cst.IMAGE_CONTAINER_IMAGES_PATH, exist_ok=True)
    model_path = train_paths.get_image_base_model_path(args.model, args.model_type)

    print("Preparing Dataset Environment...")
    prepare_dataset(
        training_images_zip_path=train_paths.get_image_training_zip_save_path(args.task_id),
        training_images_repeat=cst.DIFFUSION_SDXL_REPEATS if args.model_type == "sdxl" else cst.DIFFUSION_FLUX_REPEATS,
        instance_prompt=args.trigger_word or cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT, job_id=args.task_id, output_dir=train_cst.IMAGE_CONTAINER_IMAGES_PATH
    )

    cfg_p = create_config(args.task_id, model_path, args.model, args.model_type, args.expected_repo_name, args.trigger_word)
    run_training(args.model_type, cfg_p)

if __name__ == "__main__": asyncio.run(main())
