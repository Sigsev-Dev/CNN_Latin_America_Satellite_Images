import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# Function to load and align bands
def load_bands(data_dir):
    bands = {}
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            band_name = os.path.splitext(file)[0]
            with rasterio.open(os.path.join(data_dir, file)) as src:
                bands[band_name] = {
                    "data": src.read(1, resampling=Resampling.bilinear),
                    "meta": src.meta
                }
    return bands

# Function to stack bands
def stack_bands(bands, output_path):
    # Extract metadata from the first band
    first_band_meta = next(iter(bands.values()))["meta"]
    height, width = first_band_meta['height'], first_band_meta['width']

    # Create an empty array for stacking
    stacked_array = np.zeros((len(bands), height, width), dtype=first_band_meta['dtype'])

    # Stack bands
    for idx, band_name in enumerate(bands.keys()):
        stacked_array[idx] = bands[band_name]["data"]

    # Update metadata for multichannel image
    stacked_meta = first_band_meta.copy()
    stacked_meta.update({
        "count": len(bands),
        "dtype": first_band_meta['dtype']
    })

    # Save stacked image
    with rasterio.open(output_path, "w", **stacked_meta) as dst:
        for idx in range(len(bands)):
            dst.write(stacked_array[idx], idx + 1)
    print(f"Stacked image saved at: {output_path}")

    return stacked_array

# Function to visualize RGB composite
def visualize_rgb(stacked_array, bands, rgb_indices=(0, 1, 2), scale_factor=0.0001):
    rgb_array = np.stack([stacked_array[idx] for idx in rgb_indices], axis=-1)
    rgb_array = np.clip(rgb_array * scale_factor, 0, 1)  # Normalize for display

    plt.figure(figsize=(10, 10))
    plt.imshow(rgb_array)
    plt.title(f"RGB Composite: {bands[rgb_indices[0]]}, {bands[rgb_indices[1]]}, {bands[rgb_indices[2]]}")
    plt.axis('off')
    plt.show()

# Main execution
if __name__ == "__main__":
    data_dir = "../data"
    output_path = "../outputs/xxTeresina_Stacked.tif"

    # Load and stack bands
    bands = load_bands(data_dir)
    stacked_array = stack_bands(bands, output_path)

    # Band names for reference
    band_names = list(bands.keys())
    print(f"Available bands: {band_names}")

    # Visualize RGB combinations
    print("Choose RGB band combinations from available bands (e.g., 0 for first band, 1 for second).")
    print(f"Available bands: {band_names}")
    try:
        r = int(input(f"Enter the index for Red band (0-{len(band_names)-1}): "))
        g = int(input(f"Enter the index for Green band (0-{len(band_names)-1}): "))
        b = int(input(f"Enter the index for Blue band (0-{len(band_names)-1}): "))
        visualize_rgb(stacked_array, band_names, rgb_indices=(r, g, b))
    except ValueError:
        print("Invalid input for band indices. Please use numeric values.")
        