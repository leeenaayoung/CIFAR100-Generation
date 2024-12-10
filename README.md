# **CIFAR100-Generation**
## Image Generation with CIFAR100
We present a deep learning challenge focused on image generation using the CIFAR100 dataset. This CIFAR100 Generation challenge targets conditional generation on superclasses (20 classes for CIFAR100), aiming to create high-fidelity images while maintaining class-specific features.

## Envirionment
* Python 3.11.4
* Pytorch 2.1.2+cu121
* wandb 0.18.5(optional)
* NVIDIA RTX A4000 GPU

If your virtual environment meets upper conditions, you can use
```bash
pip install -r requirements.txt
```
OR
```bash
pip install matplotlib numpy scikit-learn scipy stylegan2_pytorch tqdm wandb
```
But you should be careful your CUDA Version is same with ours.

## Restrictions
- 50,000 training data
- No outsource
- Training time (~72 hours)
- Can use single gpu

## Table of Usage
- [Data Preprocessing and Augmentation](#Data-preprocessing-and-augmenataion)
- [Run wandb(optional)](#Run-wandb-optional)
- [Model Structure](#Model-structure)
- [Results](#Results)

## Data Preprocessing and Augmentation
* Resize(32)
* ToTensor()
* RandomHorizontalFlip()
* Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
* DiffAug(color,translation)

# Regularization
* Discriminator: hinge loss + gradient penalty
* Generator: hinge loss + Path Length Regularization

## Training
```bash
python main_train.py
```
OR
```bash
python3 main_train.py
```

## Run wandb(optional)
Install wandb
```bash
pip install wandb
```
```python
import wandb
wandb.init(project="CIFAR-100_Generation", entity=args.wandb_entity, name=run_name, config=vars(args)
```
If you want to use wandb, remove wandb annotation

## Model Structure
![제목 없는 프레젠테이션](https://github.com/user-attachments/assets/4e561b07-e6e8-410f-8479-83d17b2edfeb)
The model we implemented for this project is StyleGAN2, an enhanced version of StyleGAN, which consists of three key components: mapping network, synthesis network, and style injection mechanism. The diagram above illustrates the architecture of our implemented model.
The mapping network, located on the left, transforms a latent vector z into an intermediate latent vector w. This network comprises an L2 normalization layer and 8 fully connected layers, each activated with Leaky ReLU. We incorporated label embedding to enable the model to reflect class information effectively.
The synthesis network, positioned in the center, serves as the core network for image generation. Starting with a 4x4 resolution, it progressively increases the image size through generator blocks, each incorporating upsampling, convolution, noise addition, and style injection processes. We implemented Conv2dModulation and ToRGB layers to effectively capture fine details and gradually generate color information, even with CIFAR100's small image dimensions.
The discriminator, shown on the right, evaluates the authenticity of generated images. We enabled conditional generation based on superclasses through label embedding, with each discriminator block consisting of 3x3 convolution layers and downsampling. The final determination of image authenticity is made through flatten and linear layers.
This architecture was specifically designed to handle the unique characteristics of the CIFAR100 dataset while maintaining high-quality image generation capabilities.

## Results

* Model: StyleGAN2

| Metrics | Score |
|---------|-------|
| IS | 6.51±0.11 |
| FID | 20.88 |
| Intra-FID | 56.83 |
| Runtime | 2d 2h 59m 28s |
| seed | 0 |

## Git Commit Rules
| Tag Name           | Description                                               |
|--------------------|-----------------------------------------------------------|
| **Feat**           | Adds a new feature                                      |
| **Fix**            | Fixes a bug                                              |
| **!HOTFIX**        | Urgently fixes a critical bug                     |
| **!BREAKING CHANGE**| Introduces significant API changes                                |
| **Style**          | Code format changes, missing semicolons, no logic changes      |
| **Refactor**       | Refactors production code                                     |
| **Comment**        | Adds or updates necessary comments                                   |
| **Docs**           | Documentation changes                                                  |
| **Test**           | Adds or refactors test code, no changes to production code |
| **Chore**          | Updates build tasks, package manager configs, no changes to production code |
| **Rename**         | Renames or moves files or directories only         |
| **Remove**         | Removes files only                         |
