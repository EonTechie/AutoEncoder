# GitHub: EonTechie

# Import required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os 

# Set a fixed random seed
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
# Create an output folder for saving visualizations
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)  # Ensure the folder exists

# Encoder class: Compresses input images to a latent space representation
class Encoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Encoder, self).__init__()
        # Define the sequential layers of the encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),  # Flattens input images (28x28) to a 1D vector of size 784
            nn.Linear(input_size, 128),  # Fully connected layer: input_size (784) -> 128
            nn.ReLU(),  # Activation function to introduce non-linearity
            nn.Linear(128, latent_size),  # Fully connected layer: 128 -> latent_size (64)
            nn.ReLU()  # Another ReLU activation
        )

    def forward(self, x):
        return self.encoder(x)  # Forward pass through the encoder

# Decoder class: Reconstructs images from their latent space representation
class Decoder(nn.Module):
    def __init__(self, latent_size, output_size):
        super(Decoder, self).__init__()
        # Define the sequential layers of the decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),  # Fully connected layer: latent_size (64) -> 128
            nn.ReLU(),  # Activation function
            nn.Linear(128, output_size),  # Fully connected layer: 128 -> output_size (784)
            nn.Sigmoid()  # Maps the output values to the range [0, 1]
        )

    def forward(self, x):
        return self.decoder(x)  # Forward pass through the decoder

# Autoencoder class: Combines the Encoder and Decoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, latent_size)  # Encoder instance
        self.decoder = Decoder(latent_size, input_size)  # Decoder instance

    def forward(self, x):
        latent = self.encoder(x)  # Pass input through the encoder
        reconstructed = self.decoder(latent)  # Reconstruct the input from latent representation
        return reconstructed

# Prepare the dataset and define data loaders
# Apply transformations to the dataset )
transform = transforms.Compose([transforms.ToTensor()])

# Load the FashionMNIST dataset
full_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split the dataset into training (80%) and validation (20%) sets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders for batching and shuffling the data
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # Shuffle for generalization
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  # No shuffle for validation
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # Test set remains unchanged

# Define model, loss function, and optimizer
input_size = 28 * 28  # Input size: 784 (flattened image)
latent_size = 64  # Latent space size
model = Autoencoder(input_size, latent_size)  # Instantiate the Autoencoder model
criterion = nn.MSELoss()  # Loss function: Mean Squared Error (used for reconstruction tasks)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer: Adam with learning rate of 0.001

# Training and Validation loop
epochs = 20  # Number of training epochs
for epoch in range(epochs):
    # Training phase
    model.train()  # Set the model to training mode
    train_loss = 0
    for images, _ in train_loader:
        images = images.view(-1, input_size)  # Flatten images
        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, images)  # Compute reconstruction loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        train_loss += loss.item()  # Accumulate training loss

    train_loss /= len(train_loader)  # Compute average training loss

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.view(-1, input_size)
            outputs = model(images)
            loss = criterion(outputs, images)
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Compute average validation loss
    print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model parameters
torch.save(model.state_dict(), "autoencoder.pth")
print("Training completed and model saved!")

# Calculate reconstruction loss on the test set
model.eval()
test_loss = 0
with torch.no_grad():
    for images, _ in test_loader:
        images = images.view(-1, input_size)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Final Test Reconstruction Loss: {test_loss:.4f}")

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

        # Generate new samples from the latent space
        latent_vectors = torch.randn(num_samples, latent_size)  # Random sampling from Gaussian distribution
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
        output_path = os.path.join(output_folder, "original_reconstructed_generated_samples.png")
        plt.savefig(output_path, dpi=300)  # Save the visualization
        plt.show()
        plt.close()

# Test the model and visualize the results
state_dict = torch.load("autoencoder.pth")
model.load_state_dict(state_dict)
visualize_combined(model, test_loader, model.decoder, latent_size=64, num_samples=10)

# Function to visualize only generated samples
def visualize_generated_samples(decoder, latent_size, num_samples=10):
    decoder.eval()
    with torch.no_grad():
        latent_vectors = torch.randn(num_samples, latent_size)  # Random sampling from Gaussian distribution
        generated_images = decoder(latent_vectors).view(-1, 28, 28)

    # Plot generated samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))  # Single row of generated samples
    for i in range(num_samples):
        axes[i].imshow(generated_images[i].numpy(), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title("Generated")
    plt.tight_layout()
    output_path = os.path.join(output_folder, "only_generated_samples.png")
    plt.savefig(output_path, dpi=300)  # Save the visualization
    plt.show()
    plt.close()

visualize_generated_samples(model.decoder, latent_size=64, num_samples=10)
