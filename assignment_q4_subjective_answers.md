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
