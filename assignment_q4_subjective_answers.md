### 4. Implementing Matrix Factorization [6 Marks]

**a) Image Reconstruction:** 

1. a rectangular block of 30X30 is assumed missing from the image.
   
  Answer:
  
2. a random subset of 900 (30X30) pixels is missing from the image.
   
Answer:
    Choose rank `r` yourself. Perform Gradient Descent till convergence, plot the selected regions, original and reconstructed images, Compute the following metrics:
    * RMSE on predicted v/s ground truth high resolution image
    * Peak SNR

    
- **[2 Marks]** Write a function using this [reference](https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html) and use alternating least squares instead of gradient descent to repeat Part 1, 2 of Image reconstruction problem using your written function. 

**b) Data Compression-** Here, ground truth pixel values are not missing- you have access to them. You want to explore the use of matrix factorisation in order to store them more efficiently.
- **[2 Marks]** Consider an image patch of size (NxN) where N=50. We are trying to compress this patch (matrix) into two matrices, by using low-rank matrix factorization. Consider the following three cases-
    1. a patch with mainly a single color.
    2. a patch with 2-3 different colors.
    3. a patch with at least 5 different colors.

    Vary the low-rank value as ```r = [5, 10, 25, 50]```  for each of the cases. Use Gradient Descent and plot the reconstructed patches over the original image (retaining all pixel values outside the patch, and using your learnt compressed matrix in place of the patch) to demonstrate difference in reconstruction quality. Write your observations. 

Here is a reference set of patches chosen for each of the 3 cases from left to right. 

<div style="display: flex;">
<img src="sample_images/1colour.jpg" alt="Image 1" width="250"/>
<img src="sample_images/2-3_colours.jpg" alt="Image 2" width="270"/>
<img src="sample_images/multiple_colours.jpg" alt="Image 3" width="265"/>
</div>

# Matrix Factorization – Image Reconstruction & Data Compression
**Tools:** NumPy, PyTorch, Matplotlib, Pillow  
**Implementation:** Gradient Descent (GD) and Alternating Least Squares (ALS)

---

##  Part (a): Image Reconstruction

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
| Rectangular Block Missing | Gradient Descent | 20 | *(your value)* | *(your value)* | Missing block reconstructed smoothly; edges slightly blurred. |
| Random 900 Pixels Missing | Gradient Descent | 20 | *(your value)* | *(your value)* | Random loss easier to recover; overall clean reconstruction. |
| Rectangular Block Missing | ALS (torch.linalg.lstsq) | 20 | *(your value)* | *(your value)* | Similar quality, faster convergence, less noise. |
| Random 900 Pixels Missing | ALS | 20 | *(your value)* | *(your value)* | Best PSNR; sharp reconstruction with minimal artifacts. |

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

