## 4. Implementing Matrix Factorization [6 Marks]

###  Part (a): Image Reconstruction

Reconstruct missing regions in an image using low-rank Matrix Factorization.  
Two masking cases are explored:  30x30 Rectangular Block missing and Random 900 px missing

- **Image used:** [The Ambassadors (Hoblein)](https://en.wikipedia.org/wiki/The_Ambassadors_(Holbein)#/media/File:Hans_Holbein_the_Younger_-_The_Ambassadors_-_Google_Art_Project.jpg) from Wikipedia
- **Rank (r):** 30 
- **Methods:** Gradient Descent (GD) and Alternating Least Squares (ALS)  
- **Metrics:** RMSE and PSNR compared to the ground truth image.

Gradient Descent - **Rectangular block (30×30)** missing.
<div style="display: flex;">
<img src="outputs_q4/GD Reconstruction - Rectangular Block Missing.png" alt="Image 2" width="1200"/>
</div>

Gradient Descent - **Random subset of 900 pixels (≈30×30)** missing.
<div style="display:flex;">
  <img src="outputs_q4/GD Reconstruction - Random 900 px Missing.png" alt="Image 1" width="1200"/>
</div>

ALS - **Rectangular block (30×30)** missing.
<div style="display: flex;">
<img src="outputs_q4/ALS Reconstruction - Rectangular Block missing.png" alt="Image 1" width="1200"/>
</div>

ALS - **Random subset of 900 pixels (≈30×30)** missing.
<div style="display: flex;">
<img src="outputs_q4/ALS Reconstruction - Random 900 px missing.png" alt="Image 2" width="1200"/>
</div>

| Case | Method | Rank (r) | RMSE | PSNR (dB) | Observation |
|------|---------|----------|------|------------|--------------|
| Rectangular Block Missing | Gradient Descent | 30 | *0.0675* | *23.42* | Missing block reconstructed smoothly; edges slightly blurred. |
| Random 900 Pixels Missing | Gradient Descent | 30 | *0.0673* | *23.44* | Random loss easier to recover; overall clean reconstruction. Time taken: 60.98 s |
| Rectangular Block Missing | ALS (torch.linalg.lstsq) | 30 | *23.42* | *26.18* | Similar quality, faster convergence, less noise.  |
| Random 900 Pixels Missing | ALS | 30 | *0.0673* | *23.44* | Best PSNR; sharp reconstruction with minimal artifacts. Time taken: 76.18 s |



### **Observations**
- GD requires more iterations but is simpler to implement.  
- ALS converges faster and provides slightly better reconstruction stability.  
- Random missing data is easier to fill than contiguous missing blocks.  
- PSNR above **30 dB** indicates visually acceptable reconstruction.  

---

## Part (b): Data Compression

Demonstrate how Matrix Factorization can compress image patches by approximating them with low-rank matrices.

- **Patch size:** 50×50  
- **Ranks tested:** [5, 10, 25, 50]  

1-Colour Patch 
<div style="display: flex;">
<img src="outputs_q4/1-colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/1-colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/1-colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/1-colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/1-colour patch original.png" alt="Image 2" width="200"/>
</div>

2-3 Colour Patch
<div style="display: flex;">
<img src="outputs_q4/2-3 colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/2-3 colour patch reconstructed (r=5).png" alt="Image 2" width="200"/>
<img src="outputs_q4/2-3 colour patch reconstructed (r=10).png" alt="Image 2" width="200"/>
<img src="outputs_q4/2-3 colour patch reconstructed (r=25).png" alt="Image 2" width="200"/>
<img src="outputs_q4/2-3 colour patch reconstructed (r=50).png" alt="Image 2" width="200"/>
</div>

≥5 Colour Patch
<div style="display: flex;">
<img src="outputs_q4/Multi-colour patch original.png" alt="Image 2" width="200"/>
<img src="outputs_q4/Multi-colour patch reconstructed (r=5).png" alt="Image 2" width="200"/>
<img src="outputs_q4/Multi-colour patch reconstructed (r=10).png" alt="Image 2" width="200"/>
<img src="outputs_q4/Multi-colour patch reconstructed (r=25).png" alt="Image 2" width="200"/>
<img src="outputs_q4/Multi-colour patch reconstructed (r=50).png" alt="Image 2" width="200"/>
</div>

| Patch Type | Rank (r) | Visual Quality | Observation |
|-------------|-----------|----------------|--------------|
| 1-Colour Patch | 5 / 10 / 25 / 50 | visible blur, loss of details/ smooth colours, minor distortion/ near original, crisp reconstruction/ identical to original visually | Even rank 5 perfectly recovers flat areas; strong compression. |
| 2–3 Colour Patch | 5 / 10 / 25 / 50 | visible blur, loss of details/ smooth colours, minor distortion/ near original, crisp reconstruction/ identical to original visually | Gradients improve with r ≥ 10; near-original at r ≥ 25. |
| ≥5 Colour Patch | 5 / 10 / 25 / 50 | visible blur, loss of details/ smooth colours, minor distortion/ near original, crisp reconstruction/ identical to original visually | Low r causes blur; fine details restored for r ≥ 25. |



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


