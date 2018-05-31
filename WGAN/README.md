# IFT-6266 Class Project

## Conditional Image Generation - Inpainting with MSCOCO

### Introduction

I took a pretty standard approach to solving the problem of inpainting an image. Given a 64 x 64 image from MSCOCO, with it's center (32 x 32) masked out, I built a fully convolutional architecture that attempts to predict the center with an L2 reconstruction loss + a Wasserstein GAN (WGAN) objective. Some relevant literature that I used when building this model was

* Convolutional autoencoders with an L2 reconstruction objective - https://pdfs.semanticscholar.org/1c6d/990c80e60aa0b0059415444cdf94b3574f0f.pdf
* Mixing an L2 reconstruction objective with a GAN objective was presented in Context Encoders: Feature Learning by Inpainting - https://arxiv.org/abs/1604.07379. 
* Wasserstein GANs - https://arxiv.org/abs/1701.07875
* Pix2Pix (Image-to-Image Translation with Conditional Adversarial Networks) - Fully convolutional model with an adverserial objective - https://arxiv.org/abs/1611.07004
* DCGAN (Deep Convolutional Generative Adverserial Networks) - https://arxiv.org/abs/1511.06434
* DenseNet (Denseley Connected Convolutional Networks) - https://arxiv.org/abs/1608.06993
* UNet - https://arxiv.org/abs/1505.04597
* Sentence Embeddings (Simple but tough to beat baseline for sentence embeddings) - https://openreview.net/pdf?id=SyK00v5xx
* Adaptation of Word2Vec (Two/Too Simple Adaptations of Word2Vec for Syntax Problems) - http://www.cs.cmu.edu/~lingwang/papers/naacl2015.pdf

An example of masked image  and it's corresponding ground-truth is shown in the image below.

![Task](/images/lamb.png)

### Architecture

I used a fully convolutional architecture for the generator, that is very similar to the DCGAN generator with strided convolution and transpose convolution layers, batch-normalization and ReLU activations. The difference between my generator and the DCGAN generator is that I don't have the initial linear transformation that they use to reshape the noise prior. The discriminator is also similar to the DCGAN discriminator with strided convolutions, batch normalization and ReLUs.

As in the pix2pix paper, I added skip connections between the feature maps produced by the regular convolutions and the transpose convoltion feature maps. The feature maps are concatenated along the channel axis.

#### Generator

| Layer | Filters | Input Shape | Output Shape | Kernel Size |
| ------------- | ------------- | ------------- | ------------ | ------------ |
| Conv1 | 32 | 3 x 64 x 64 | 32 x 32 x 32 | 4 x 4 |
| Conv2 | 64 | 32 x 32 x 32 | 64 x 16 x 16 | 4 x 4 |
| Conv3 | 96 | 64 x 16 x 16 | 96 x 8 x 8 | 4 x 4 |
| Conv4 | 128 | 96 x 8 x 8 | 128 x 4 x 4 | 4 x 4 |
| TConv1 | 96 | 128 x 4 x 4 | 96 x 8 x 8 | 4 x 4 |
| TConv2 | 64 | (96 + 96) x 8 x 8 | 64 x 16 x 16 | 4 x 4 |
| TConv3 | 1 | (64 + 64) x 16 x 16 | 1 x 32 x 32 | 4 x 4 |

The table above presents the generator's architecture where Conv* refers to a regular convolution with a stride of (2, 2) and TConv* refers to a transpose convolution, also with a stride of (2, 2).

#### Discriminator

| Layer | Filters | Input Shape | Output Shape | Kernel Size |
| ------------- | ------------- | ------------- | ------------ | ------------ |
| Conv1 | 32 | 3 x 32 x 32 | 32 x 16 x 16 | 4 x 4 |
| Conv2 | 64 | 32 x 16 x 16 | 64 x 8 x 8 | 4 x 4 |
| Conv3 | 96 | 64 x 8 x 8 | 96 x 4 x 4 | 4 x 4 |
| Conv4 | 128 | 96 x 4 x 4 | 128 x 2 x 2 | 4 x 4 |
| Conv5 | 256 | 128 x 2 x 2 | 256 x 1 x 1 | 2 x 2 |


### Training

As evident from the Generator architecture, the model takes a 3 channel 64 x 64 image with the center masked out and tries to predict the center 32 x 32 square. The discriminator (when using the adverserial objective) takes "fake" 32 x 32 samples from our Generator and "real" 32 x 32 samples from our training data and is trained to distinguish between them. The generator is trained to fool the discriminator. This is formulated as a minimax game as described in Ian Goodfellow's original GAN paper. The L2 reconstruction objective tries to minimize the squared error between the predicted image center and the ground truth. The L2 + GAN objective simply adds these two losses together and backpropagates it through the network. In this work, I used a WGAN instead of a regular GAN which from an implementation perspective, simply removes `log` from the GAN objective and clamps the weights of the discriminator.

### Adding Image Captions

The MSCOCO dataset contains several human written captions of each image. These written descriptions could improve the ability of our model to predict the masked out center. For example, in the image presented above, the sheep are masked out leaving only the green fields in the background visible. The model could therefore inpaint in many possible ways - with an empty field, a deer or a bear etc. However, if the model was present with a caption of the image saying "Three sheep in a field of grass", this could reduce it's search space.

In this project, I explored using a fixed representation of a caption using the "Simple but tough to beat baseline for sentence embeddings". In a nutshell, the paper uses a weighted average of the individual word embeddings for each word in the caption. The word embeddings for each word are trained on a large external corpus. In this paper, I used an adapted word2vec implementation (Two/Too Simple Adaptations of Word2Vec for Syntax Problems) trained on the English Gigaword 5 corpus. The weights of each word are inversely proportional to their word frequency in the text8 corpus. I then used the approach described in the paper to extract the first principal component on the sentence embeddings of the MSCOCO training captions and applied the described transformation on the training and dev set.

When using captions, I modified my generator to downsample all the way to 1 x 1 and then contatenated the embedding of the image's caption before upsampling with transpose convolutions.

#### Caption Generator

| Layer | Filters | Input Shape | Output Shape | Kernel Size |
| ------------- | ------------- | ------------- | ------------ | ------------ |
| Conv1 | 32 | 3 x 64 x 64 | 32 x 32 x 32 | 4 x 4 |
| Conv2 | 64 | 32 x 32 x 32 | 64 x 16 x 16 | 4 x 4 |
| Conv3 | 96 | 64 x 16 x 16 | 96 x 8 x 8 | 4 x 4 |
| Conv4 | 128 | 96 x 8 x 8 | 128 x 4 x 4 | 4 x 4 |
| Conv5 | 256 | 128 x 4 x 4 | 256 x 2 x 2 | 4 x 4 |
| Conv6 | 256 | 256 x 2 x 2 | 256 x 1 x 1 | 2 x 2 |
| TConv1 | 128 | (256 + 300emb) x 1 x 1 | 128 x 2 x 2 | 4 x 4 |
| TConv2 | 96 | (128 + 256) x 2 x 2 | 96 x 4 x 4 | 4 x 4 |
| TConv3 | 64 | (96 + 128) x 4 x 4 | 64 x 8 x 8 | 4 x 4 |
| TConv4 | 32 | (64 + 96) x 8 x 8 | 32 x 16 x 16 | 4 x 4 |
| TConv5 | 1 | (32 + 64) x 16 x 16 | 1 x 32 x 32 | 4 x 4 |

### Hyperparameters 

- Optimizer - ADAM for both the generator and discriminator with a learning rate of 2e-3
- Discriminator weight clamping -  (-0.03, 0.03) if discriminator is trained 5 times for every generator update else (-0.05, 0.05). 
- Batch size - 32

### Results

In this section, I will present some of my cherry-picked inpainted images. All samples are inpaintings of the MSCOCO dev set.

#### L2

![sample_1](/images/l2_epoch_10_samples.png)
![sample_2](/images/l2_epoch_13_samples.png)
![sample_3](/images/l2_epoch_27_samples.png)
![sample_4](/images/l2_epoch_30_samples.png)
![sample_5](/images/l2_epoch_35_samples.png)
![sample_6](/images/l2_epoch_37_samples.png)

#### L2 + WGAN

![sample_1](/images/gan_epoch_22_samples.png)
![sample_2](/images/gan_epoch_27_samples.png)
![sample_3](/images/gan_epoch_50_samples.png)
![sample_4](/images/gan_epoch_54_samples.png)
![sample_5](/images/gan_epoch_63_samples.png)
![sample_6](/images/gan_epoch_57_samples.png)

#### L2 + WGAN + Caption

![sample_1](/images/gan_caption_epoch_25_samples.png)
![sample_2](/images/gan_caption_epoch_32_samples.png)
![sample_3](/images/gan_caption_epoch_34_samples.png)
![sample_4](/images/gan_caption_epoch_35_samples.png)
![sample_5](/images/gan_caption_epoch_36_samples.png)
![sample_6](/images/gan_caption_epoch_17_samples.png)

### Conclusion & Discussion

I experimented with a simple method to solve the inpainting problem using a convolutional autoencoder with a WGAN objective. I also added information from image captions using a simple baseline for sentence embeddings.

The convolutional autoencoder with an L2 reconstruction objective yields reasonable results and appears to mostly predict the center as a smoothened interpolation of its surrounding. Adding the WGAN objective appeared to yield some structure to the predictions especially in the images that required inpainting the foreground and not just the background. Further, adding captions, seemed to help a little more in the quality of samples and also accelerated training (50-60 epochs for L2 + WGAN vs 30-40 epochs with captions). It also appeared to focus on drawing actual objects instead of just smoothing images with captions where a woman's face (first row 5th column) can be seen (if you squint really hard).






