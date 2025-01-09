import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os

# Dataset class
class SentinelDataset(Dataset):
    def __init__(self, stacked_raster, classified_raster, patch_size=128):
        self.stacked_raster = stacked_raster
        self.classified_raster = classified_raster
        self.patch_size = patch_size
        self.height, self.width = classified_raster.shape

    def __len__(self):
        return (self.height // self.patch_size) * (self.width // self.patch_size)

    def __getitem__(self, idx):
        h = (idx // (self.width // self.patch_size)) * self.patch_size
        w = (idx % (self.width // self.patch_size)) * self.patch_size
        input_patch = self.stacked_raster[:, h:h + self.patch_size, w:w + self.patch_size]
        label_patch = self.classified_raster[h:h + self.patch_size, w:w + self.patch_size]
        return torch.tensor(input_patch, dtype=torch.float32), torch.tensor(label_patch, dtype=torch.long)

def crop_to_match(input_tensor, reference_tensor):
    _, _, h, w = reference_tensor.size()
    return input_tensor[:, :, :h, :w]

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(input_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.bottleneck = self.conv_block(512, 1024)

        self.dec4 = self.upconv_block(1024 + 512, 512)
        self.dec3 = self.upconv_block(512 + 256, 256)
        self.dec2 = self.upconv_block(256 + 128, 128)
        self.dec1 = self.upconv_block(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, output_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        enc4 = crop_to_match(enc4, bottleneck)
        dec4 = self.dec4(torch.cat((bottleneck, enc4), dim=1))
        enc3 = crop_to_match(enc3, dec4)
        dec3 = self.dec3(torch.cat((dec4, enc3), dim=1))
        enc2 = crop_to_match(enc2, dec3)
        dec2 = self.dec2(torch.cat((dec3, enc2), dim=1))
        enc1 = crop_to_match(enc1, dec2)
        dec1 = self.dec1(torch.cat((dec2, enc1), dim=1))

        return self.final_conv(dec1)

# Training function
def train_model(model, dataloader, criterion, optimizer, num_epochs, device, class_colors):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")
        
        # # Visualize training progress on one batch
        # with torch.no_grad():
        #     inputs, labels = next(iter(dataloader))
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     outputs = model(inputs)
        #     predictions = outputs.argmax(dim=1).cpu().numpy()[0]
        #     visualize_classification(predictions, class_colors)
    print("Training complete.")
    return model

# Visualization function
def visualize_classification(prediction, class_colors):
    color_map = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)
    for class_value, color in class_colors.items():
        color_map[prediction == class_value] = color
    plt.figure(figsize=(8, 8))
    plt.title("Training Classification Preview")
    plt.imshow(color_map)
    plt.axis("off")
    plt.show()

# Prediction function
def predict(model, raster, patch_size, device):
    model.eval()
    _, height, width = raster.shape
    prediction = np.zeros((height, width), dtype=np.uint8)

    with torch.no_grad():
        for i in range(0, height, patch_size):
            for j in range(0, width, patch_size):
                patch = raster[:, i:i + patch_size, j:j + patch_size]
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    continue
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).to(device)
                output = model(patch_tensor)
                predicted_patch = output.argmax(dim=1).squeeze(0).cpu().numpy()
                prediction[i:i + patch_size, j:j + patch_size] = predicted_patch
    return prediction

# Main script
if __name__ == "__main__":
    stacked_raster_path = "../outputs/stacked_raster.tif"
    classified_raster_path = "../outputs/recolored_classification.tif"
    patch_size = 128
    num_epochs = 100

    # Define colors for classes
    class_colors = {
        0: [0, 0, 0],        # Black
        1: [255, 0, 0],      # Red
        2: [0, 255, 0],      # Green
        3: [0, 0, 255]       # Blue
    }

    with rasterio.open(stacked_raster_path) as src:
        stacked_raster = src.read()
    with rasterio.open(classified_raster_path) as src:
        classified_raster = src.read(1)

    dataset = SentinelDataset(stacked_raster, classified_raster, patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = UNet(input_channels=stacked_raster.shape[0], output_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # directly load the saved model
    model_path="unet_model.pth" # To change the Path name, replace all in red with new path inside " "
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # model = train_model(model, dataloader, criterion, optimizer, num_epochs, device, class_colors)

    # torch.save(model.state_dict(), "unet_model.pth")

    prediction = predict(model, stacked_raster, patch_size, device)

    output_path = "predicted_map_2000epochs_Teresina.tif"
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=prediction.shape[0],
        width=prediction.shape[1],
        count=1,
        dtype=rasterio.uint8,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(prediction, 1)

    plt.imshow(prediction, cmap="tab20")
    plt.title("Predicted Classification Map")
    plt.colorbar()
    plt.show()
    print(f"Prediction saved at {output_path}")
