
import json
import os

def upgrade_lrs_config(file_path, is_style=False):
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    # Standard settings based on champion research
    target_epochs = 22 if is_style else 21
    target_snr = 8 if is_style else 7
    target_prior_loss = 1.0 if is_style else 0.8
    
    if "data" in config:
        for hash_id, settings in config["data"].items():
            print(f"Upgrading hash: {hash_id}")
            # Capacity Upgrade
            settings["network_dim"] = 128
            settings["network_alpha"] = 64 # Smooth Beast mode
            settings["network_args"] = ["conv_dim=16", "conv_alpha=8", "dropout=null"]
            
            # Stability Upgrade
            settings["max_train_epochs"] = target_epochs
            settings["min_snr_gamma"] = target_snr
            settings["prior_loss_weight"] = target_prior_loss
            
            # Optional: Keep caption_dropout if exists, else default to 0.05
            if "caption_dropout_rate" not in settings:
                settings["caption_dropout_rate"] = 0.05
                
            # Clean up old keys that might conflict
            settings.pop("noise_offset", None) # Follow default or set if specific needed
            
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Successfully upgraded {file_path}")

# Paths
person_path = r"c:\56\image-in\scripts\lrs\person_config.json"
style_path = r"c:\56\image-in\scripts\lrs\style_config.json"

upgrade_lrs_config(person_path, is_style=False)
upgrade_lrs_config(style_path, is_style=True)
