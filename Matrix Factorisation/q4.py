import os, math
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

OUT_DIR = "outputs_q4"
os.makedirs(OUT_DIR, exist_ok=True)

# image for reconstruction
IMAGE_PATH = r"Image & Patches/The Ambassadors Large.jpg"

# three patches for compression
PATCH_PATHS = [
    r"Image & Patches/Original 1-color patch.jpg",
    r"Image & Patches/Original 2-3 colors patch.jpg",
    r"Image & Patches/Original Multi-color patch.jpg"
]
PATCH_NAMES = ["1-colour", "2-3 colour", "Multi-colour"]

def mse(a, b):
    return ((a - b) ** 2).mean()

def rmse(a, b):
    return float(torch.sqrt(mse(a, b)).item())

def psnr(a, b, data_range=1.0):
    m = float(mse(a, b).item())
    if m == 0: return float('inf')
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(m)

def save_tensor(img_t, path):
    """Save torch tensor HxWxC [0,1] as image."""
    arr = (img_t.detach().clamp(0,1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)

def show_and_save(orig, masked, recon, filename, title=None):
    """Show original, masked, reconstructed side by side."""
    fig, axes = plt.subplots(1,3, figsize=(12,4))
    for ax, img, label in zip(axes,
                              [orig, masked, recon],
                              ['Original', 'Masked', 'Reconstructed']):
        ax.imshow(img.detach().clamp(0,1).cpu().numpy())
        ax.set_title(label)
        ax.axis('off')
    if title: fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)

# Load main image (for reconstruction)
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((400, 400), Image.LANCZOS)
img_t = torch.from_numpy(np.array(img).astype(np.float32)/255.0)
H, W, C = img_t.shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mask generators
def mask_rectangular(H, W, block=30):
    mask = torch.ones((H, W), dtype=torch.bool)
    top = (H-block)//2
    left = (W-block)//2
    mask[top:top+block, left:left+block] = False
    return mask

def mask_random(H, W, missing=900, seed=0):
    rng = np.random.default_rng(seed)
    mask = torch.ones((H, W), dtype=torch.bool)
    idx = rng.choice(H*W, missing, replace=False)
    ys, xs = idx//W, idx%W
    mask[ys, xs] = False
    return mask


# Gradient Descent Reconstruction
def reconstruct_gd(img_t, mask, rank, lr=1e-2, iters=2000):
    H, W, C = img_t.shape
    mask_f = mask.to(device)
    recon = torch.zeros_like(img_t, device=device)
    for ch in range(C):
        M = img_t[:,:,ch].to(device)
        U = nn.Parameter(torch.randn(H, rank, device=device)*0.01)
        V = nn.Parameter(torch.randn(rank, W, device=device)*0.01)
        opt = torch.optim.Adam([U,V], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            pred = U@V
            loss = ((pred - M)*mask_f).pow(2).mean()
            loss.backward()
            opt.step()
        recon[:,:,ch] = (U@V).clamp(0,1)
    return recon.cpu()

# Alternating Least Squares Reconstruction
def reconstruct_als(img_t, mask, rank, iters=30):
    H, W, C = img_t.shape
    mask_f = mask.to(device)
    recon = torch.zeros_like(img_t, device=device)
    for ch in range(C):
        M = img_t[:,:,ch].to(device)
        U = torch.randn(H, rank, device=device)*0.01
        V = torch.randn(rank, W, device=device)*0.01
        for _ in range(iters):
            # Solve for V
            V_new = torch.zeros_like(V)
            for j in range(W):
                obs = mask_f[:, j].nonzero().squeeze()
                if obs.numel() == 0: continue
                A = U[obs,:]; b = M[obs,j]
                sol = torch.linalg.lstsq(A, b).solution
                V_new[:,j] = sol
            V = V_new
            # Solve for U
            U_new = torch.zeros_like(U)
            for i in range(H):
                obs = mask_f[i,:].nonzero().squeeze()
                if obs.numel() == 0: continue
                A = V[:,obs].T; b = M[i,obs]
                sol = torch.linalg.lstsq(A,b).solution
                U_new[i,:] = sol
            U = U_new
        recon[:,:,ch] = (U@V).clamp(0,1)
    return recon.cpu()

# (a) Image Reconstruction
mask_rect = mask_rectangular(H, W)
mask_rand = mask_random(H, W)

masked_rect = img_t.clone(); masked_rect[~mask_rect]=0.5
masked_rand = img_t.clone(); masked_rand[~mask_rand]=0.5

rank = 30
start_gd = time.time()
print("\nRunning Gradient Descent reconstructions...")
gd_rect = reconstruct_gd(img_t, mask_rect, rank)
gd_rand = reconstruct_gd(img_t, mask_rand, rank)

rmse_gd_rect, psnr_gd_rect = rmse(gd_rect,img_t), psnr(gd_rect,img_t)
rmse_gd_rand, psnr_gd_rand = rmse(gd_rand,img_t), psnr(gd_rand,img_t)
print(f"GD Rect: RMSE={rmse_gd_rect:.4f}, PSNR={psnr_gd_rect:.2f}")
print(f"GD Rand: RMSE={rmse_gd_rand:.4f}, PSNR={psnr_gd_rand:.2f}")

show_and_save(img_t, masked_rect, gd_rect,
              f"{OUT_DIR}/GD Reconstruction - Rectangular Block Missing.png")
show_and_save(img_t, masked_rand, gd_rand,
              f"{OUT_DIR}/GD Reconstruction - Random 900 px Missing.png")
end_gd = time.time()
print("Time taken for GD reconstructions: " + f"{end_gd - start_gd:.2f} seconds")


start_als = time.time()
print("\nRunning ALS reconstructions...")
als_rect = reconstruct_als(img_t, mask_rect, rank)
als_rand = reconstruct_als(img_t, mask_rand, rank)

rmse_als_rect, psnr_als_rect = rmse(als_rect,img_t), psnr(als_rect,img_t)
rmse_als_rand, psnr_als_rand = rmse(als_rand,img_t), psnr(als_rand,img_t)
print(f"ALS Rect: RMSE={rmse_als_rect:.4f}, PSNR={psnr_als_rect:.2f}")
print(f"ALS Rand: RMSE={rmse_als_rand:.4f}, PSNR={psnr_als_rand:.2f}")

show_and_save(img_t, masked_rect, als_rect,
              f"{OUT_DIR}/ALS Reconstruction - Rectangular Block missing.png")
show_and_save(img_t, masked_rand, als_rand,
              f"{OUT_DIR}/ALS Reconstruction - Random 900 px missing.png")

end_als = time.time()
print("Time taken for ALS reconstructions: " + f"{end_als - start_als:.2f} seconds")

# (b) Data Compression (3 patches)
def factorize_patch(mat, r, lr=1e-2, iters=1200):
    H, W = mat.shape
    U = nn.Parameter(torch.randn(H,r)*0.01)
    V = nn.Parameter(torch.randn(r,W)*0.01)
    opt = torch.optim.Adam([U,V], lr=lr)
    M = torch.tensor(mat)
    for _ in range(iters):
        opt.zero_grad()
        pred = U@V
        loss = ((pred - M)**2).mean()
        loss.backward(); opt.step()
    return (U@V).clamp(0,1).detach().numpy()

ranks = [5,10,25,50]
N = 50

for path, name in zip(PATCH_PATHS, PATCH_NAMES):
    patch = Image.open(path).convert("RGB").resize((N,N), Image.LANCZOS)
    patch_np = np.array(patch).astype(np.float32)/255.0
    Image.fromarray((patch_np*255).astype(np.uint8)).save(
        f"{OUT_DIR}/{name} patch original.png")
    print(f"\nCompressing patch: {name}")
    for r in ranks:
        recon = np.zeros_like(patch_np)
        for c in range(3):
            recon[:,:,c] = factorize_patch(patch_np[:,:,c], r)
        out_path = f"{OUT_DIR}/{name} patch reconstructed (r={r}).png"
        Image.fromarray((np.clip(recon,0,1)*255).astype(np.uint8)).save(out_path)

print("\nAll outputs saved to:", OUT_DIR)
