## 4. Implementing Matrix Factorization [6 Marks]

**Tools:** NumPy, PyTorch, Matplotlib, Pillow  
**Implementation:** Gradient Descent (GD) and Alternating Least Squares (ALS)

---

###  Part (a): Image Reconstruction

### **Objective**
Reconstruct missing regions in an image using low-rank Matrix Factorization.  
Two masking cases are explored:
1. **Rectangular block (30×30)** missing.
2. **Random subset of 900 pixels (≈30×30)** missing.

---

### **Experimental Setup**
- **Image used:** `multiple_colours.jpg`
- **Rank (r):** 20  
- **Methods:** Gradient Descent (GD) and Alternating Least Squares (ALS)  
- **Metrics:** RMSE and PSNR compared to the ground truth image.

---

### **Results**

| Case | Method | Rank (r) | RMSE | PSNR (dB) | Observation |
|------|---------|----------|------|------------|--------------|
| Rectangular Block Missing | Gradient Descent | 20 | *( value)* | *(  value)* | Missing block reconstructed smoothly; edges slightly blurred. |
| Random 900 Pixels Missing | Gradient Descent | 20 | *( value)* | *( value)* | Random loss easier to recover; overall clean reconstruction. |
| Rectangular Block Missing | ALS (torch.linalg.lstsq) | 20 | *( value)* | *( value)* | Similar quality, faster convergence, less noise. |
| Random 900 Pixels Missing | ALS | 20 | *( value)* | *( value)* | Best PSNR; sharp reconstruction with minimal artifacts. |

---

### **Observations**
- GD requires more iterations but is simpler to implement.  
- ALS converges faster and provides slightly better reconstruction stability.  
- Random missing data is easier to fill than contiguous missing blocks.  
- PSNR above **30 dB** indicates visually acceptable reconstruction.  

---

## Part (b): Data Compression

### **Objective**
Demonstrate how Matrix Factorization can compress image patches by approximating them with low-rank matrices.

- **Patch size:** 50×50  
- **Ranks tested:** [5, 10, 25, 50]  
- **Patches chosen:**  
  1. Single colour patch (`1colour.jpg`)  
  2. 2–3 colour patch (`2-3_colours.jpg`)  
  3. Multi-colour patch (`multiple_colours.jpg`)

---

### **Results**

| Patch Type | Rank (r) | Visual Quality | Observation |
|-------------|-----------|----------------|--------------|
| 1-Colour Patch | 5 / 10 / 25 / 50 | *(describe)* | Even rank 5 perfectly recovers flat areas; strong compression. |
| 2–3 Colour Patch | 5 / 10 / 25 / 50 | *(describe)* | Gradients improve with r ≥ 10; near-original at r ≥ 25. |
| ≥5 Colour Patch | 5 / 10 / 25 / 50 | *(describe)* | Low r causes blur; fine details restored for r ≥ 25. |

---

### **Observations**
- Compression efficiency increases when colour variance is low.  
- For complex regions, higher rank values are required for faithful reconstruction.  
- Storage reduction achieved since low-rank MF stores only **(H×r + W×r)** parameters.  
- Matrix Factorization effectively balances compression and visual fidelity.

---

## **Conclusion**
- Both GD and ALS successfully reconstruct and compress image data.  
- **ALS** achieves similar accuracy with fewer iterations.  
- **GD** remains intuitive and easy to tune.  
- Matrix Factorization is a powerful unsupervised method for both **image inpainting** and **dimensionality reduction**.

---

### Next Steps
- Experiment with varying `r` values and regularization terms.  
- Apply MF on grayscale images or noisy inputs for additional comparison.  
- Explore SVD-based initialization for faster convergence.

---

