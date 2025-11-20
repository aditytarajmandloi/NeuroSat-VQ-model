import argparse
import torch
import numpy as np
import pickle
import rasterio
from src_v2.model_v2_6 import V2Autoencoder
from src_v2.utils_v2 import pack_indices, rle_encode
from src.overlap_utils import compute_tile_grid

TILE_SIZE = 128
OVERLAP = 16

def pad_image(image, tile_size, overlap):
    h, w, c = image.shape
    xs, pad_w = compute_tile_grid(w, tile_size, overlap)
    ys, pad_h = compute_tile_grid(h, tile_size, overlap)
    padded = np.zeros((pad_h, pad_w, c), dtype=np.uint8)
    padded[:h, :w] = image
    return padded, (w, h), (pad_w, pad_h), xs, ys

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Model: {args.model}")
    model = V2Autoencoder().to(device)
    # Load just the model weights (ignoring optimizer state)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"Processing: {args.input}")
    with rasterio.open(args.input) as src:
        geo_profile = src.profile
        img_data = src.read().transpose(1, 2, 0) # H,W,C
    
    # Handle Alpha
    alpha_rle = None
    if img_data.shape[2] == 4:
        print("ℹ️ Compressing Alpha Channel (RLE)...")
        alpha_rle = rle_encode(img_data[:, :, 3])
        img_rgb = img_data[:, :, :3]
    else:
        img_rgb = img_data
        
    # Pad & Tile
    padded, (orig_w, orig_h), (pad_w, pad_h), xs, ys = pad_image(img_rgb, TILE_SIZE, OVERLAP)
    all_indices = []
    
    # Inference
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
                tile_t = torch.from_numpy(tile.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
                
                # Get Indices Directly
                _, _, indices = model(tile_t)
                all_indices.append(indices.cpu().numpy().flatten())

    # Pack
    full_indices = np.concatenate(all_indices).astype(np.uint16)
    compressed_blob = pack_indices(full_indices)
    
    header = {
        "geo_profile": geo_profile,
        "orig_size": (orig_w, orig_h),
        "pad_size": (pad_w, pad_h),
        "grid": (len(xs), len(ys)),
        "num_indices": len(full_indices),
        "has_alpha": alpha_rle is not None,
        "alpha_rle": alpha_rle
    }
    
    with open(args.output, "wb") as f:
        pickle.dump({"header": header, "blob": compressed_blob}, f)
        
    print(f"✅ Saved: {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-m", "--model", required=True)
    main(p.parse_args())