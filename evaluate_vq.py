import argparse
import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import os

def calculate_psnr(img1, img2):
    # img1 and img2 must be matching float32 or uint8
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 255.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

def evaluate(orig_path, recon_path):
    if not os.path.exists(orig_path): raise FileNotFoundError(orig_path)
    if not os.path.exists(recon_path): raise FileNotFoundError(recon_path)

    # Load as standard BGR (OpenCV default)
    a = cv2.imread(orig_path)
    b = cv2.imread(recon_path)
    
    # Ensure dimensions match (Crop reconstruction if needed)
    h, w, _ = a.shape
    b = b[:h, :w]

    # 1. MSE (Pixel Accuracy)
    mse = np.mean((a - b) ** 2)

    # 2. PSNR (Engineering Standard)
    psnr = calculate_psnr(a, b)

    # 3. SSIM (Structural Accuracy) - needs Grayscale
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ssim_score = ssim(a_gray, b_gray, data_range=255)

    return mse, psnr, ssim_score

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--orig", required=True)
    p.add_argument("--recon", required=True)
    args = p.parse_args()

    mse, psnr, ssim_val = evaluate(args.orig, args.recon)
    
    print("="*30)
    print("   EVALUATION REPORT   ")
    print("="*30)
    print(f"MSE:   {mse:.4f}  (Lower is better)")
    print(f"PSNR:  {psnr:.2f} dB (Higher is better, >30 is good)")
    print(f"SSIM:  {ssim_val:.4f}  (Higher is better, 1.0 is max)")
    print("="*30)