![Collage](readme_imgs/CollageFig.png)

# Differentiable Network Bending

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kSpzhY55ugPKeIdqHN85cM0wwjeK3ZRj?usp=sharing)

Official implementation of our paper **Hacking Generative Models with Differentiable Network Bending**([Website](https://galdegheri.github.io/diffbending/)/[Arxiv](https://arxiv.org/abs/2310.04816)).<br>

In this work, we propose a method to 'hack' generative models, pushing their outputs away from the original training distribution towards a new objective. We inject a small-scale trainable module between the intermediate layers of the model and train it for a low number of iterations, keeping the rest of the network frozen. The resulting output images display an uncanny quality, given by the tension between the original and new objectives that can be exploited for artistic purposes.