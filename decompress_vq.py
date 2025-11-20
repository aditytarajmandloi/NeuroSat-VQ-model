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
from src.overlap_utils import hann_2d

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Reading {args.input}...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    header = data["header"]
    compressed = data["compressed"]
    codec = pickle.loads(data["codec"])

    # Unpack
    orig_w, orig_h = header["original_size"]
    pad_w, pad_h = header["padded_size"]
    tile_meta = header["tile_meta"]
    tile_size = header["tile_size"]

    # Load Model
    model = load_s_grade_model(args.model, header["latent_channels"], device)
    quantizer = VQQuantizer(args.codebook)

    # Decompress Huffman
    indices = np.array(packer.decompress_indices(compressed, codec), dtype=np.int64)

    # Canvas
    acc = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    weights = np.zeros((pad_h, pad_w), dtype=np.float32)
    window = hann_2d(tile_size).astype(np.float32)

    ptr = 0
    for meta in tqdm(tile_meta, desc="Decompressing"):
        x, y = meta["x"], meta["y"]
        shape_meta = meta["shape"]
        B, C, H, W, usable = shape_meta

        vectors_needed = usable // header["vector_dim"]
        tile_indices = indices[ptr : ptr + vectors_needed]
        ptr += vectors_needed

        z = quantizer.dequantize(tile_indices, shape_meta).to(device)
        
        with torch.no_grad():
            # Decoder output (B, 3, H, W)
            recon = model.decoder(z).clamp(0, 1)
            tile = recon.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Accumulate with Window
        acc[y:y+tile_size, x:x+tile_size] += tile * window[:, :, None]
        weights[y:y+tile_size, x:x+tile_size] += window

    # Normalize
    weights[weights < 1e-8] = 1.0 # Prevent div zero
    final = (acc / weights[:, :, None]).clip(0, 1)
    final = (final * 255).astype(np.uint8)

    # Crop
    final = final[:orig_h, :orig_w]

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))
    print(f"✅ Reconstructed: {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", default="results/reconstructed.png")
    p.add_argument("-m", "--model", required=True)
    p.add_argument("-c", "--codebook", required=True)
    main(p.parse_args())