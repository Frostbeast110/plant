"""
Download MobileNetV2 Plant Disease Model (Custom Loader)
Model: linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification
Uses custom image processing instead of AutoImageProcessor
"""
from transformers import AutoModelForImageClassification, AutoConfig
from PIL import Image
import torch
import json
import os

model_name = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
save_path = "./models/mobilenet_plantdisease"

print(f"📥 Downloading model: {model_name}")
print(f"💾 Saving to: {save_path}")

try:
    # Create directory
    os.makedirs(save_path, exist_ok=True)
    
    # Download config and model without processor
    print("Downloading config...")
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(save_path)
    
    print("Downloading model weights...")
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.save_pretrained(save_path)
    
    # Save label mapping from config
    if hasattr(config, 'id2label'):
        label_map = config.id2label
        with open(f"{save_path}/label_map.json", 'w', encoding='utf-8') as f:
            json.dump(label_map, f, indent=2)
        print(f"✅ Label map saved ({len(label_map)} classes)")
    
    # Save preprocessing config manually
    preprocess_config = {
        "image_size": 224,
        "rescale_factor": 1.0 / 255.0,
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    with open(f"{save_path}/preprocess_config.json", 'w', encoding='utf-8') as f:
        json.dump(preprocess_config, f, indent=2)
    
    print(f"✅ Model downloaded successfully!")
    print(f"\nModel details:")
    print(f"  - Architecture: {config.model_type}")
    print(f"  - Number of classes: {config.num_labels}")
    print(f"  - Accuracy: 95.41%")
    print(f"  - Image size: 224x224")
    print(f"  - Plant disease classes: 38")
    
except Exception as e:
    import traceback
    print(f"❌ Error downloading model: {str(e)}")
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  1. Check internet connection")
    print("  2. Verify model name is correct")
    print("  3. Try: pip install --upgrade transformers torch")
