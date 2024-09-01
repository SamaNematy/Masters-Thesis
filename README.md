# Development of seamless algorithm and robust to vignetting artifact in histological images obtained from whole slide imaging technique
Whole-slide imaging is a technique that generates high-resolution digital images by integrating a sequence of microscopic images. However, shading distortion or vignetting—caused by non-uniform illumination—can create artifacts like black checkered patterns in the final image, reducing visual quality and causing errors in quantitative analyses.

Traditional methods to correct these issues are either time-consuming or ineffective. To address this, my research focuses on using deep neural networks, specifically a type of Generative Adversarial Network (GAN) known as pix2pix, to correct non-uniform brightness in microscopic images. The proposed method allows for the quick, independent correction of each image without the need for a reference image, producing clearer results than previous approaches.

![Methods](https://github.com/user-attachments/assets/94df79d2-cb3e-40a8-99bc-776d0cfe03bb)
Figure 1: Overview of methods used for shading correction in whole-slide images.
