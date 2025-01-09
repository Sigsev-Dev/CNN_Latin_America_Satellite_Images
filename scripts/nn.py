import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np

# Neural Network with Semantic Distinction
class SemanticDistinctionNet(nn.Module):
    """
    A U-Net-based neural network for semantic segmentation with morphological semantic distinction.
    """
    def __init__(self, input_channels, num_classes):
        super(SemanticDistinctionNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.dec3 = self.conv_block(512 + 256, 256)
        self.dec2 = self.conv_block(256 + 128, 128)
        self.dec1 = self.conv_block(128 + 64, 64)

        # Output layer
        self.output_layer = nn.Conv2d(64, num_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        Convolutional block: Conv -> ReLU -> Conv -> ReLU
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # Encoding path
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(F.max_pool2d(enc1_out, kernel_size=2))
        enc3_out = self.enc3(F.max_pool2d(enc2_out, kernel_size=2))

        # Bottleneck
        bottleneck_out = self.bottleneck(F.max_pool2d(enc3_out, kernel_size=2))

        # Decoding path
        dec3_out = self.dec3(torch.cat([F.interpolate(bottleneck_out, scale_factor=2, mode='bilinear', align_corners=False), enc3_out], dim=1))
        dec2_out = self.dec2(torch.cat([F.interpolate(dec3_out, scale_factor=2, mode='bilinear', align_corners=False), enc2_out], dim=1))
        dec1_out = self.dec1(torch.cat([F.interpolate(dec2_out, scale_factor=2, mode='bilinear', align_corners=False), enc1_out], dim=1))

        # Output layer
        output = self.output_layer(dec1_out)
        return output

# Custom Dataset
class RasterDataset(Dataset):
    def __init__(self, raster_path, labels_path, patch_size=32):
        """
        Dataset for handling raster input and classified labels.
        
        Parameters:
            raster_path (str): Path to the input raster (stacked bands).
            labels_path (str): Path to the classified raster.
            patch_size (int): Size of patches to extract from the rasters.
        """
        with rasterio.open(raster_path) as src:
            self.data = src.read()  # [channels, height, width]
        with rasterio.open(labels_path) as src:
            self.labels = src.read(1)  # [height, width]

        # Ensure data and labels have the same spatial dimensions
        if self.data.shape[1:] != self.labels.shape:
            raise ValueError("Mismatch between input raster and label dimensions.")

        self.patch_size = patch_size
        self.num_patches_x = self.data.shape[2] // self.patch_size
        self.num_patches_y = self.data.shape[1] // self.patch_size

    def __len__(self):
        return self.num_patches_x * self.num_patches_y  # Total number of patches

    def __getitem__(self, idx):
        """
        Returns a patch of data and its corresponding labels.
        """
        patch_x = (idx % self.num_patches_x) * self.patch_size
        patch_y = (idx // self.num_patches_x) * self.patch_size

        data_patch = self.data[:, patch_y:patch_y+self.patch_size, patch_x:patch_x+self.patch_size]
        label_patch = self.labels[patch_y:patch_y+self.patch_size, patch_x:patch_x+self.patch_size]

        return (
            torch.tensor(data_patch, dtype=torch.float32),  # [channels, patch_size, patch_size]
            torch.tensor(label_patch, dtype=torch.long)    # [patch_size, patch_size]
        )

# Training Script
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for data, labels in dataloader:
            # Ensure data is 4D [batch, channels, height, width]
            data = data.to(device)  # Already in [batch, channels, height, width] format
            labels = labels.to(device)  # [batch, height, width]
            
            optimizer.zero_grad()
            outputs = model(data)  # [batch, num_classes, height, width]

            # CrossEntropyLoss expects class indices, not one-hot encoding
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    # Paths
    raster_path = "../outputs/stacked_image_clipped.tif"
    labels_path = "../outputs/recolored_classification.tif"

    # Prepare dataset and dataloader
    patch_size = 32  # Extract 32x32 patches
    dataset = RasterDataset(raster_path, labels_path, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # 8 patches per batch

    # Initialize model
    input_channels = 4  # Stacked Sentinel-2 bands
    num_classes = 5  # Roads, Paved, Houses, Vegetation, Water
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SemanticDistinctionNet(input_channels, num_classes).to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # For multi-class classification
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, dataloader, criterion, optimizer, num_epochs=20)

    # Save model
    torch.save(model.state_dict(), "../models/semantic_distinction_model.pth")
    print("Model trained and saved successfully.")
