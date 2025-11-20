import os
import argparse
import torch
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from src.utils import load_s_grade_model
from src.vq_quantizer import VQQuantizer
import src.packer as packer
from src.overlap_utils import compute_tile_grid

# Defaults
MODEL_PATH = "models/cnn_autoencoder_s_grade_final_epoch10.pth"
CODEBOOK_PATH = "models/vq_codebook_80_4096.npy"
LATENT_CHANNELS = 80
TILE_SIZE = 128
OVERLAP = 16

def pad_image_for_overlap(image, tile_size, overlap):
    h, w, _ = image.shape
    xs, pad_w = compute_tile_grid(w, tile_size, overlap)
    ys, pad_h = compute_tile_grid(h, tile_size, overlap)

    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded[:h, :w] = image
    return padded, (w, h), (pad_w, pad_h), xs, ys

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model: {args.model}...")
    model = load_s_grade_model(args.model, args.latent, device)
    quantizer = VQQuantizer(args.codebook)

    # Read Image
    img_bgr = cv2.imread(args.image)
    if img_bgr is None: raise FileNotFoundError(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Pad
    padded, (orig_w, orig_h), (pad_w, pad_h), xs, ys = pad_image_for_overlap(img_rgb, TILE_SIZE, OVERLAP)

    all_indices = []
    tile_meta = []

    print(f"Compressing... Grid: {len(xs)}x{len(ys)}")
    
    for y in tqdm(ys):
        for x in xs:
            tile = padded[y:y+TILE_SIZE, x:x+TILE_SIZE]
            # Normalize to 0-1 float
            tile_t = torch.from_numpy(tile.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)

            with torch.no_grad():
                _, z = model(tile_t) # (recon, z)

            indices, shape_meta = quantizer.quantize(z)
            all_indices.extend(indices.tolist())
            tile_meta.append({"x": int(x), "y": int(y), "shape": shape_meta})

    # Huffman Packing
    compressed_data, codec = packer.compress_indices(all_indices)
    codec_blob = pickle.dumps(codec)

    header = {
        "original_size": (orig_w, orig_h),
        "padded_size": (pad_w, pad_h),
        "grid_counts": (len(xs), len(ys)),
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "latent_channels": args.latent,
        "vector_dim": quantizer.vector_dim,
        "num_clusters": quantizer.num_clusters,
        "tile_meta": tile_meta
    }

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"header": header, "compressed": compressed_data, "codec": codec_blob}, f)
    
    size_kb = os.path.getsize(args.output) / 1024
    print(f"✅ Saved {args.output} ({size_kb:.2f} KB)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--image", required=True)
    p.add_argument("-o", "--output", default="results/compressed.bin")
    p.add_argument("-m", "--model", default=MODEL_PATH)
    p.add_argument("-c", "--codebook", default=CODEBOOK_PATH)
    p.add_argument("-l", "--latent", type=int, default=LATENT_CHANNELS)
    main(p.parse_args())