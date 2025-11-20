import os
import torch
import glob
from src.train import train_model

# =====================
# CONFIGURATION
# =====================

DATA_DIR = "data/naip_128"
MODEL_SAVE_PATH = "models/cnn_autoencoder_s_grade_final.pth"

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 32     # RTX 2050 Safe
TOTAL_EPOCHS = 120
LEARNING_RATE = 1e-4
LATENT_CHANNELS = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Overlap Settings
OVERLAP_ENABLE = True
OVERLAP_PX = 16
LOG_INTERVAL = 200

def get_latest_checkpoint(model_path):
    """Auto-detects the latest checkpoint to resume training."""
    # Check for intermediate checkpoints first: 'model_epoch10.pth'
    base_name = model_path.replace(".pth", "")
    checkpoints = glob.glob(f"{base_name}_epoch*.pth")
    
    if checkpoints:
        # Sort by epoch number
        latest = max(checkpoints, key=lambda x: int(x.split("_epoch")[1].split(".")[0]))
        return latest
    
    # If no intermediate, check for the final model
    if os.path.exists(model_path):
        return model_path
        
    return None

def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or "models", exist_ok=True)

    # --- Auto-Resume Logic ---
    resume_path = get_latest_checkpoint(MODEL_SAVE_PATH)
    
    print("\n===== CNN TRAINING LAUNCH =====")
    print(f"Dataset:       {DATA_DIR}")
    print(f"Target Model:  {MODEL_SAVE_PATH}")
    print(f"Resume From:   {resume_path if resume_path else 'Scratch (Start New)'}")
    print(f"Epochs:        {TOTAL_EPOCHS}")
    print(f"Batch Size:    {BATCH_SIZE}")
    print(f"Device:        {DEVICE}")
    print("================================\n")

    train_model(
        data_dir=DATA_DIR,
        model_path=MODEL_SAVE_PATH,
        epochs=TOTAL_EPOCHS,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        latent_channels=LATENT_CHANNELS,
        lr=LEARNING_RATE,
        device=DEVICE,
        resume=resume_path,  # Passes the detected path
        overlap=OVERLAP_ENABLE,
        overlap_px=OVERLAP_PX,
        log_interval=LOG_INTERVAL,
    )

if __name__ == "__main__":
    main()