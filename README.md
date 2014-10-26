#COMP 598 (Fall 2014) - Mini-Project #3

##Data

The training set consists of 50 000 labelled examples.

The test set consists of 20 000 unlabelled examples, whose category must be predicted.

The dataset is based on the classic MNIST dataset, with each image modified using the following transformations:
- Embossing of the digt.
- Rotation (by a random angle, sampled from [0, 360deg].
- Rescaling (from 28x28 pixels to 48x48 pixels).
- Texture pattern (randomly selected, and overlayed on background)

To create texture on an image, a random 48 x 48 patch was located on a randomly selected texture from the Brodatz dataset (Brodatz, 1966) and overlaid beneath the MNIST digit. For every pixel position in the resulting image, the pixel from the MNIST digit is used if its intensity is higher than 0.1, otherwise the pixel from the texture is used.

To perform the embossing operation, the light source ange is constant throughout the dataset (The specific angle used is not given).

(Brodatz, P. (1966). Textures: A Photographic Album for Artists and Designers. Dover photgraphy collections. Dover Publications.)

