import os
import time
import torch
from torch import optim
from torch.amp import autocast, GradScaler
# Scheduler removed for stability
# from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src_v2.dataset import get_balanced_dataloader
from src_v2.model_v2_6 import V2Autoencoder
from src_v2.utils_v2 import charbonnier_loss, sobel_loss, focal_frequency_loss

# --- CONFIGURATION ---
DATA_DIR = "data/naip_128"
MODELS_DIR = "models_v2.6"
MODEL_LATEST = os.path.join(MODELS_DIR, "v2_6_latest.pth")
BATCH_SIZE = 16
ACCUM_STEPS = 4

# <--- THE LONG RUN CONFIG --->
EPOCHS = 120    # Extended for full convergence
LR = 1e-4       # Fixed Low Rate for polishing
# <--------------------------->

def train():
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not found.")
    
    device = torch.device("cuda")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"--- Initializing V2.6 on {device} ---")
    
    loader = get_balanced_dataloader(DATA_DIR, batch_size=BATCH_SIZE, num_workers=2)
    if loader is None: return

    model = V2Autoencoder().to(device)
    
    # Optimizer initializes with target LR
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # SCHEDULER REMOVED: We want constant LR for grid removal
    # scheduler = CosineAnnealingWarmRestarts(...) 
    
    scaler = GradScaler('cuda')
    
    start_epoch = 1

    # --- RESUME LOGIC ---
    if os.path.exists(MODEL_LATEST):
        print(f"🔄 Checking for checkpoint: {MODEL_LATEST}")
        try:
            ckpt = torch.load(MODEL_LATEST, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            opt.load_state_dict(ckpt['optimizer_state_dict'])
            # Scheduler load removed
            scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            
            # <--- FORCE CONSTANT LR --->
            # This overrides whatever "jitter" was saved in the checkpoint
            print(f"⬇️ LOCKING Learning Rate to {LR} (Scheduler Disabled)")
            for param_group in opt.param_groups:
                param_group['lr'] = LR
            # <------------------------->
            
            print(f"✅ Resumed from Epoch {start_epoch}")
        except:
            print("⚠️ Resume failed. Starting fresh.")

    print(f"🚀 Training V2.6 | Batch: {BATCH_SIZE} | Accum: {ACCUM_STEPS} | Target: 120 Epochs")
    
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        run_loss = 0
        opt.zero_grad()
        
        for i, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            
            # Autocast
            with autocast('cuda'):
                recon, vq_loss, _ = model(imgs)
            
            # Float32 Loss Calculation
            recon_f32 = recon.float()
            imgs_f32 = imgs.float()
            vq_loss_f32 = vq_loss.float()
            
            l_char = charbonnier_loss(recon_f32, imgs_f32)
            l_edge = sobel_loss(recon_f32, imgs_f32) * 0.25
            l_fft  = focal_frequency_loss(recon_f32, imgs_f32) * 0.1
            
            loss = l_char + vq_loss_f32 + l_edge + l_fft
            loss = loss / ACCUM_STEPS 
            
            # Backward
            scaler.scale(loss).backward()
            
            if (i + 1) % ACCUM_STEPS == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            # Track Scaled Loss (Visual only)
            current_loss = loss.item() * ACCUM_STEPS
            run_loss += current_loss
            
            # Heartbeat
            if i % 200 == 0:
                print(f"Step {i}/{len(loader)} | Loss: {current_loss:.4f}")
            
        # NO SCHEDULER STEP HERE
        
        avg_loss = run_loss / len(loader)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.5f}")
        
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            # 'scheduler_state_dict': No scheduler to save
            'scaler_state_dict': scaler.state_dict()
        }
        torch.save(state, MODEL_LATEST)
        
        if epoch % 10 == 0:
            torch.save(state, os.path.join(MODELS_DIR, f"v2_6_epoch{epoch}.pth"))

if __name__ == "__main__":
    train()