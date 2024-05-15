import joblib
import torch
from generator import Generator
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.append("src/")

from utils import device_init, load

# Initialize the device
device = device_init(device="mps")

# Load the generator model
state = torch.load(
    "/Users/shahmuhammadraditrahman/Desktop/DualGAN/checkpoints/best_model/best_model.pth"
)
netG_XtoY = Generator(in_channels=3).to(device)
netG_YtoX = Generator(in_channels=3).to(device)

netG_XtoY.load_state_dict(state["netG_XtoY"])
netG_YtoX.load_state_dict(state["netG_YtoX"])

# Load the data
data_loader = load(
    filename="/Users/shahmuhammadraditrahman/Desktop/DualGAN/data/processed/test_dataloader.pkl"
)
X, y = next(iter(data_loader))

# Generate output using the generator
with torch.no_grad():
    netG_XtoY.eval()
    generated_y = netG_XtoY(X.to(device))
    reconstructed_X = netG_YtoX(generated_y)

# Convert tensor to numpy array for plotting
generated_y_np = generated_y.cpu().detach().numpy()
generated_y_np = np.transpose(
    generated_y_np, (0, 2, 3, 1)
)  # Convert from (N, C, H, W) to (N, H, W, C)

reconstructed_X_np = reconstructed_X.cpu().detach().numpy()
reconstructed_X_np = np.transpose(
    reconstructed_X_np, (0, 2, 3, 1)
)  # Convert from (N, C, H, W) to (N, H, W, C)


# Normalize the images to [0, 1] range
def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val)


# Determine the number of images to plot
num_images = min(8, X.size(0))

# Plot the first num_images images
plt.figure(figsize=(20, 10))
for i in range(num_images):
    # Original image
    plt.subplot(4, num_images, i + 1)
    plt.title("Original")
    original_image = X[i].cpu().detach().numpy()
    original_image = np.transpose(
        original_image, (1, 2, 0)
    )  # Convert from (C, H, W) to (H, W, C)
    original_image = normalize_image(original_image)
    plt.imshow(original_image, cmap="gray")
    plt.axis("off")

    # Generated image
    plt.subplot(4, num_images, i + num_images + 1)
    plt.title("Generated")
    generated_image = generated_y_np[i]
    generated_image = normalize_image(generated_image)
    plt.imshow(generated_image)
    plt.axis("off")

    # Reconstructed image
    plt.subplot(4, num_images, i + 2 * num_images + 1)
    plt.title("Reconstructed")
    reconstructed_image = reconstructed_X_np[i]
    reconstructed_image = normalize_image(reconstructed_image)
    plt.imshow(reconstructed_image, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()
