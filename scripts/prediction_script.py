import torch
import rasterio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the U-Net architecture (same as in training)
class UNet(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        self.dec4 = self.conv_block(1024 + 512, 512)
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))

        return self.output_layer(dec1)

# Load and preprocess the raster
def load_raster(raster_path):
    """
    Load a raster file, normalize the bands, and convert to float32.
    """
    with rasterio.open(raster_path) as src:
        data = src.read()  # Shape: [channels, height, width]
        profile = src.profile

    # Convert to float32 and normalize
    data = data.astype(np.float32)
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return data, profile

# Predict the classification map
def predict(model, raster, device, patch_size=128):
    """
    Predict the classification for a given raster using the trained model.
    Ensures the output map matches the size of the input raster.
    """
    model.eval()
    height, width = raster.shape[1], raster.shape[2]
    output_map = np.zeros((height, width), dtype=np.uint8)

    # Sliding window for patch prediction
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            # Dynamically adjust patch size for edge cases
            patch_height = min(patch_size, height - i)
            patch_width = min(patch_size, width - j)
            
            # Extract the patch
            patch = raster[:, i:i+patch_height, j:j+patch_width]

            # Pad the patch to match the model's input size
            padded_patch = np.zeros((raster.shape[0], patch_size, patch_size), dtype=np.float32)
            padded_patch[:, :patch.shape[1], :patch.shape[2]] = patch

            # Convert patch to tensor and run prediction
            patch_tensor = torch.tensor(padded_patch).unsqueeze(0).to(device)  # Shape: [1, channels, height, width]
            with torch.no_grad():
                output = model(patch_tensor)
                predicted_patch = torch.argmax(output, dim=1).cpu().numpy()  # Shape: [1, height, width]

            # Extract only the valid region (without padding) and place it in the output map
            output_map[i:i+patch_height, j:j+patch_width] = predicted_patch[0, :patch_height, :patch_width]

    return output_map



# Visualize and save the output
# Visualize and inspect predicted classes
def visualize_and_save(output_map, profile, output_path):
    """
    Visualize the predicted classification map and save as a GeoTIFF.
    """
    unique_classes = np.unique(output_map)
    print(f"Unique classes in the prediction: {unique_classes}")  # Debug output

    plt.imshow(output_map, cmap='tab10')  # Use a colormap that supports distinct classes
    plt.colorbar()
    plt.title("Predicted Classification Map")
    plt.show()

    # Save the classification map
    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress='lzw'
    )
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(output_map, 1)


# Main function
if __name__ == "__main__":
    # Paths
    model_path = "../models/unet_model.pth"
    raster_path = "../outputs/stacked_raster.tif"
    output_path = "../outputs/predicted_classification.tif"

    # Parameters
    input_channels = 10  # Number of Sentinel-2 bands
    num_classes = 4      # Number of classes from training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the raster
    raster, profile = load_raster(raster_path)

    # Load the model
    model = UNet(input_channels=input_channels, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Run prediction
    output_map = predict(model, raster, device)
    # classif_map = "../outputs/recolored_classification.tif"

    # Debugging step to validate the output size
    print(f"Input raster dimensions: {raster.shape[1]} x {raster.shape[2]}")
    print(f"Output map dimensions: {output_map.shape}")

    # Visualize and save the result
    visualize_and_save(output_map, profile, output_path)
    print(f"Predicted classification map saved at: {output_path}")