import argparse
import torch
import torch.nn.functional as F # Added for F.embedding
import numpy as np
import pickle
import rasterio
from src_v2.model_v2_6 import V2Autoencoder, LATENT_CHANNELS, SLICE_DIM
from src_v2.utils_v2 import unpack_indices, rle_decode
from src_v2.overlap_utils import hann_2d, compute_tile_grid

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Reading {args.input}...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    header = data["header"]
    blob = data["blob"]
    
    # Load Model
    model = V2Autoencoder().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    # Reconstruct RGB
    all_indices = unpack_indices(blob, header["num_indices"])
    pad_w, pad_h = header["pad_size"]
    acc = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    weights = np.zeros((pad_h, pad_w), dtype=np.float32)
    window = hann_2d(128).astype(np.float32)
    
    indices_per_tile = 8 * 8 * (LATENT_CHANNELS // SLICE_DIM)
    ptr = 0
    xs, _ = compute_tile_grid(pad_w, 128, 16)
    ys, _ = compute_tile_grid(pad_h, 128, 16)
    
    print("Reconstructing RGB...")
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile_idx = all_indices[ptr : ptr + indices_per_tile]
                ptr += indices_per_tile
                
                idx_tensor = torch.from_numpy(tile_idx.astype(np.int64)).to(device)
                
                # --- FIX: Use Functional Embedding Lookup ---
                # Old (Error): model.quantizer.embedding(idx)
                # New (Correct): F.embedding(idx, weight)
                codes = F.embedding(idx_tensor, model.quantizer.embedding)
                # --------------------------------------------

                # Reshape: [N, 4] -> [Slices, 8, 8, 4] -> [1, 96, 8, 8]
                z_sliced = codes.view(24, 8, 8, 4).permute(0, 3, 1, 2)
                z = z_sliced.reshape(1, 96, 8, 8)
                
                # Decoder Pass
                recon = model.dec_ps1(model.dec_conv1(z))
                recon = model.dec_res1(recon)
                recon = model.dec_ps2(model.dec_conv2(recon))
                recon = model.dec_res2(recon)
                recon = model.dec_ps3(model.dec_conv3(recon))
                recon = model.dec_res3(recon)
                recon = model.dec_ps4(model.dec_conv4(recon))
                recon = model.dec_res4(recon)
                out = torch.sigmoid(model.final_conv(recon))
                
                tile = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                acc[y:y+128, x:x+128] += tile * window[:, :, None]
                weights[y:y+128, x:x+128] += window

    final_rgb = (acc / (weights[:, :, None] + 1e-8)).clip(0, 1)
    final_rgb = (final_rgb * 255).astype(np.uint8)
    
    orig_w, orig_h = header["orig_size"]
    final_rgb = final_rgb[:orig_h, :orig_w]
    
    # Re-attach Alpha
    if header["has_alpha"]:
        print("Applying Alpha...")
        alpha = rle_decode(header["alpha_rle"], (orig_h, orig_w))
        final_img = np.dstack((final_rgb, alpha))
        count = 4
    else:
        final_img = final_rgb
        count = 3

    # Save GeoTIFF
    prof = header["geo_profile"]
    prof.update(driver="GTiff", count=count, dtype="uint8")
    
    # Rasterio needs (Count, H, W)
    to_save = final_img.transpose(2, 0, 1)
    
    with rasterio.open(args.output, "w", **prof) as dst:
        dst.write(to_save)
        
    print(f"✅ Done: {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-m", "--model", required=True)
    main(p.parse_args())