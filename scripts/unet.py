import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
import numpy as np
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
#10000*10000

def calculate_class_weights(label_path):
    """
    Calculate class weights based on pixel counts in the label raster.
    """
    with rasterio.open(label_path) as src:
        labels = src.read(1)
    unique_classes, counts = np.unique(labels, return_counts=True)
    total_pixels = np.sum(counts)
    weights = {cls: total_pixels / count for cls, count in zip(unique_classes, counts)}
    return torch.tensor([weights[i] for i in range(len(unique_classes))], dtype=torch.float)


def analyze_labels(label_path):
    """
    Analyze the labels to ensure all classes are present.
    Args:
        label_path (str): Path to the label raster file.
    """
    with rasterio.open(label_path) as src:
        labels = src.read(1)  # Read the label data (first band assumed)
    
    unique_classes, counts = np.unique(labels, return_counts=True)
    print("Unique Classes and Counts in Labels:")
    for cls, count in zip(unique_classes, counts):
        print(f"Class {cls}: {count} pixels")

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
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        dec4 = self.dec4(torch.cat([F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True), enc4], dim=1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True), enc3], dim=1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True), enc2], dim=1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True), enc1], dim=1))

        # Output Layer
        return self.output_layer(dec1)

# Dataset Class
class RasterDataset(Dataset):
    def __init__(self, raster_path, label_path, patch_size=128):
        """
        Args:
            raster_path: Path to the stacked raster file (.tif).
            label_path: Path to the unsupervised classification raster (.tif).
            patch_size: Size of the patches to extract.
        """
        with rasterio.open(raster_path) as src:
            self.data = src.read()  # Shape: [channels, height, width]
            labels=self.data
        print("Unique Classes in Labels Before Training:", np.unique(labels))
        with rasterio.open(label_path) as src:
            self.labels = src.read(1)  # Shape: [height, width]

        self.patch_size = patch_size
        self.num_patches_x = self.data.shape[2] // patch_size
        self.num_patches_y = self.data.shape[1] // patch_size

    def __len__(self):
        return self.num_patches_x * self.num_patches_y

    def __getitem__(self, idx):
        patch_x = (idx % self.num_patches_x) * self.patch_size
        patch_y = (idx // self.num_patches_x) * self.patch_size

        data_patch = self.data[:, patch_y:patch_y+self.patch_size, patch_x:patch_x+self.patch_size]
        label_patch = self.labels[patch_y:patch_y+self.patch_size, patch_x:patch_x+self.patch_size]

        return (
            torch.tensor(data_patch, dtype=torch.float32),  # [channels, patch_size, patch_size]
            torch.tensor(label_patch, dtype=torch.long)    # [patch_size, patch_size]
        )

# Training Loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data, labels in dataloader:
            data = data.to(device)  # Shape: [batch, channels, height, width]
            labels = labels.to(device)  # Shape: [batch, height, width]

            optimizer.zero_grad()
            outputs = model(data)  # Shape: [batch, num_classes, height, width]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # with torch.no_grad():
            #     preds = torch.argmax(outputs, dim=1).cpu().numpy()
            #     print("Predicted Classes:", np.unique(preds))
            #     print("Ground Truth Classes:", np.unique(labels.cpu().numpy()))
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Main Function
if __name__ == "__main__":
    # File paths
    raster_path = "../outputs/stacked_raster.tif"
    label_path = "../outputs/recolored_classification.tif"

    analyze_labels(label_path)

    # Parameters
    input_channels = 10  # Number of Sentinel-2 bands
    num_classes = 4  # Number of classes in the unsupervised map
    patch_size = 128
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 30

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and DataLoader
    dataset = RasterDataset(raster_path, label_path, patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = UNet(input_channels, num_classes).to(device)

    weights = calculate_class_weights("../outputs/recolored_classification.tif")
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the Model
    train_model(model, dataloader, criterion, optimizer, num_epochs)

    # Save the Model
    torch.save(model.state_dict(), "../models/unet_model.pth")
    print("Model training complete and saved.")
