# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import random

# Encoder class: Compresses input images to latent space
class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flattens 28x28 images to 784 vector
            nn.Linear(input_size, 128),  # First dense layer (784 -> 128)
            nn.ReLU(),  # Activation function
            nn.Linear(128, latent_size),  # Compress to latent size
            nn.ReLU()  # Activation function
        )

    def forward(self, x):
        return self.encoder(x)

# Decoder class: Reconstructs input images from latent space
class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),  # Expands latent size to 128
            nn.ReLU(),  # Activation function
            nn.Linear(128, output_size),  # Reconstructs to original size (784)
            nn.Sigmoid()  # Maps output values between 0 and 1
        )

    def forward(self, x):
        return self.decoder(x)

# Autoencoder class: Combines Encoder and Decoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_size)
        self.decoder = Decoder(latent_size, input_size)

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# Prepare the dataset and define data loaders
transform = transforms.Compose([transforms.ToTensor()])

# Load dataset
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split dataset into training and validation sets (80% train, 20% validation)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define model, loss function, and optimizer
input_size = 28 * 28  # 784 pixels
latent_size = 64
model = Autoencoder(input_size, latent_size)
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and Validation loop
epochs = 20
for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0
    for images, _ in train_loader:
        images = images.view(-1, input_size)  # Flatten images
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "autoencoder.pth")
print("Training completed and model saved!")

# Function to visualize original, reconstructed, and generated samples
def visualize_combined(model, test_loader, decoder, latent_size, num_samples=10):
    model.eval()
    decoder.eval()
    with torch.no_grad():
        # Get original and reconstructed images
        for images, _ in test_loader:
            images = images.view(images.size(0), -1)
            reconstructed = model(images)
            break

        # Generate new samples
        latent_vectors = torch.randn(num_samples, latent_size)  # Gaussian noise
        generated_images = decoder(latent_vectors).view(-1, 28, 28)

        # Combine all visuals in one frame
        fig, axes = plt.subplots(3, 10, figsize=(15, 6))  # 3 rows: Original, Reconstructed, Generated

        for i in range(10):
            # Original images
            axes[0, i].imshow(images[i].view(28, 28).numpy(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title("Original")

            # Reconstructed images
            axes[1, i].imshow(reconstructed[i].view(28, 28).numpy(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed")

            # Generated images
            axes[2, i].imshow(generated_images[i].numpy(), cmap='gray')
            axes[2, i].axis('off')
            axes[2, i].set_title("Generated")

        plt.tight_layout()
        plt.show()

# Test the model and visualize
state_dict = torch.load("autoencoder.pth")
model.load_state_dict(state_dict)
visualize_combined(model, test_loader, model.decoder, latent_size=64, num_samples=10)
