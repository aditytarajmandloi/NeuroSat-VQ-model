import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import time
import torch

from src.utils import load_s_grade_model, save_kmeans_model, load_kmeans_model
from src.dataset import get_random_subset_dataloader
from src.vq_utils import chunk_latents_from_z

# Config
MODEL_PATH = "models/cnn_autoencoder_s_grade_final_epoch90.pth"
LATENT_CHANNELS = 80
DATA_DIR = "data/naip_128" 
VECTOR_DIM = 4
NUM_CLUSTERS = 4096
CODEBOOK_PATH = "models/vq_codebook_80_4096.npy"
KMEANS_CHECKPOINT_PATH = "models/kmeans_4096_checkpoint.joblib"

SAMPLE_SIZE = 30000
BATCH_SIZE = 32
KMEANS_BATCH_SIZE = 1024
N_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_codebook():
    # 1. Load CNN
    try:
        model = load_s_grade_model(MODEL_PATH, LATENT_CHANNELS, DEVICE)
    except FileNotFoundError:
        print("❌ Model not found. Train CNN first!")
        return

    # 2. Load Data
    # Note: This uses 'get_random_subset_dataloader', which picks a NEW random subset 
    # every time you run the script, ensuring diversity if you restart multiple times.
    loader = get_random_subset_dataloader(DATA_DIR, 128, BATCH_SIZE, SAMPLE_SIZE, num_workers=2)
    if loader is None: return

    # 3. Load/Init KMeans with Epoch State
    ckpt = load_kmeans_model(KMEANS_CHECKPOINT_PATH)
    
    start_epoch = 1
    km = None

    if ckpt is not None:
        # Check if it's our new dict format or legacy raw model
        if isinstance(ckpt, dict) and 'model' in ckpt:
            km = ckpt['model']
            start_epoch = ckpt['epoch'] + 1
            print(f"🔄 Resuming KMeans from Epoch {start_epoch}")
        else:
            # Legacy support or raw model
            km = ckpt
            print("🔄 Resuming from raw KMeans model (Epoch unknown, starting at 1)")
            start_epoch = 1
    else:
        print("✨ Initializing new KMeans...")
        km = MiniBatchKMeans(
            n_clusters=NUM_CLUSTERS, 
            batch_size=KMEANS_BATCH_SIZE, 
            random_state=42,
            max_iter=1, # Important for partial_fit loop control
            compute_labels=False,
            tol=0.0
        )

    if start_epoch > N_EPOCHS:
        print(f"✅ Training already complete ({start_epoch-1}/{N_EPOCHS}).")
        # Ensure we save the numpy export even if finished
        np.save(CODEBOOK_PATH, km.cluster_centers_.astype(np.float32))
        return

    print(f"--- Starting VQ Training (Epochs {start_epoch} to {N_EPOCHS}) ---")
    
    for epoch in range(start_epoch, N_EPOCHS + 1):
        t0 = time.time()
        batch_vecs_list = []
        n_vectors = 0
        
        for imgs, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs = imgs.to(DEVICE)
            with torch.no_grad():
                _, z = model(imgs) 
            
            # Process batch
            z_np = z.cpu().numpy()
            for i in range(z_np.shape[0]):
                v, _ = chunk_latents_from_z(z_np[i], VECTOR_DIM)
                if v.shape[0] > 0:
                    batch_vecs_list.append(v)
                    n_vectors += v.shape[0]
        
        if batch_vecs_list:
            # Fit
            big_batch = np.vstack(batch_vecs_list)
            km.partial_fit(big_batch)
            
            # Save Checkpoint with Epoch info
            checkpoint_data = {
                'model': km,
                'epoch': epoch
            }
            save_kmeans_model(checkpoint_data, KMEANS_CHECKPOINT_PATH)
            
            dt = (time.time()-t0)/60
            print(f"Epoch {epoch} done. Fitted {n_vectors} vectors. Time: {dt:.2f} min.")
        else:
            print("Warning: No vectors generated this epoch.")

    # Final Save of the raw codebook (numpy) for the Quantizer to use
    np.save(CODEBOOK_PATH, km.cluster_centers_.astype(np.float32))
    print(f"✅ Final Codebook saved: {CODEBOOK_PATH}")

if __name__ == "__main__":
    build_codebook()