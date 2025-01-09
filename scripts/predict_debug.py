import numpy as np
import matplotlib.pyplot as plt
import torch

# Load the model
model_path = "../models/unet_model.pth"
model = UNet(input_channels=10, num_classes=4)  # Adjust for your case
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Load a small subset of the raster
raster, _ = load_raster("../data/stacked_raster.tif")
subset = raster[:, :256, :256]  # Small tile for testing
subset_tensor = torch.tensor(subset).unsqueeze(0)  # Add batch dimension

# Predict
with torch.no_grad():
    output = model(subset_tensor)
    predicted = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

# Check unique classes in the output
unique_classes = np.unique(predicted)
print(f"Unique predicted classes: {unique_classes}")

# Visualize
plt.imshow(predicted, cmap='tab10')
plt.colorbar()
plt.title("Predicted Classes on Subset")
plt.show()
