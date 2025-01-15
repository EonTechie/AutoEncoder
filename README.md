
# Autoencoder Project

## Overview

This project implements an Autoencoder using PyTorch for the FashionMNIST dataset. The Autoencoder consists of an Encoder and Decoder for compressing and reconstructing images. Additionally, it supports visualization of original, reconstructed, and generated samples.

## Requirements

- Python 3.12.2
- PyTorch
- torchvision
- matplotlib

Install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Code Structure

- **Encoder**: Compresses input images to a latent space representation.
- **Decoder**: Reconstructs images from their latent space representation.
- **Autoencoder**: Combines the Encoder and Decoder.
- **Dataset and DataLoaders**: Prepares the FashionMNIST dataset and splits it into training, validation, and test sets.
- **Training and Validation**: Trains the Autoencoder and evaluates its performance on the validation set.
- **Testing**: Computes the reconstruction loss on the test set.

## Usage

1. **Run the training script**:
   ```bash
   python main.py
   ```

   This trains the Autoencoder and saves the model parameters in `autoencoder.pth`.

2. **Visualize Results**:
   The script includes functions for visualizing the original, reconstructed, and generated samples.

   - Combined visualization of original, reconstructed, and generated images is saved as `output/original_reconstructed_generated_samples.png`.
   - Only generated samples are saved as `output/only_generated_samples.png`.

## Implementation Details

- **Latent Size**: 64
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with a learning rate of 0.001
- **Batch Size**: 64
- **Epochs**: 20

## Visualizations

The project provides two main visualizations:
1. **Original, Reconstructed, and Generated Samples**:
   Displays the input images, their reconstructions, and newly generated samples.
2. **Generated Samples**:
   Displays only new samples generated from random latent vectors.

Both visualizations are saved in the `output` folder.

## Author

It showcases a project personally developed by EonTechie as part of the BLG 527E - Machine Learning course at Istanbul Technical University (ITU). The work reflects individual effort and serves as a testament to the developer's learning journey and dedication to the subject.
