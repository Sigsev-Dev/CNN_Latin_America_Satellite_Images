import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def reclassify(predicted_raster, class_mapping):
    """
    Reclassifies the predicted raster into the desired classes based on a mapping.

    Args:
        predicted_raster (numpy.ndarray): Predicted raster from the U-Net model.
        class_mapping (dict): Mapping of original classes to new classes.

    Returns:
        numpy.ndarray: Reclassified raster.
    """
    reclassified = np.zeros_like(predicted_raster, dtype=np.uint8)
    for orig_class, new_class in class_mapping.items():
        reclassified[predicted_raster == orig_class] = new_class
    return reclassified

def save_raster_with_labels(reclassified_raster, class_labels, class_colors, output_path, src):
    """
    Saves the reclassified raster with class labels and colors.

    Args:
        reclassified_raster (numpy.ndarray): Reclassified raster.
        class_labels (dict): Dictionary mapping class values to labels.
        class_colors (dict): Dictionary mapping class values to RGB colors.
        output_path (str): Path to save the output raster.
        src (rasterio.DatasetReader): Original rasterio source file for metadata.
    """
    height, width = reclassified_raster.shape
    color_map = np.zeros((height, width, 3), dtype=np.uint8)

    # Assign colors to the classes
    for class_value, color in class_colors.items():
        color_map[reclassified_raster == class_value] = color

    # Save as a georeferenced raster with RGB bands
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,  # RGB bands
        dtype=rasterio.uint8,
        crs=src.crs,
        transform=src.transform,
    ) as dst:
        dst.write(color_map[:, :, 0], 1)  # Red band
        dst.write(color_map[:, :, 1], 2)  # Green band
        dst.write(color_map[:, :, 2], 3)  # Blue band

    print(f"Reclassified raster saved at: {output_path}")

def visualize_reclassified_map(reclassified_raster, class_labels, class_colors):
    """
    Visualizes the reclassified raster with the correct colors and labels.

    Args:
        reclassified_raster (numpy.ndarray): Reclassified raster.
        class_labels (dict): Dictionary mapping class values to labels.
        class_colors (dict): Dictionary mapping class values to RGB colors.
    """
    # Define a custom colormap with the specified colors
    colors = [tuple(np.array(class_colors[c]) / 255.0) for c in sorted(class_colors.keys())]
    cmap = ListedColormap(colors)

    # Plot the reclassified raster
    plt.figure(figsize=(10, 10))
    plt.title("Reclassified Map with 4 Classes")
    plt.imshow(reclassified_raster, cmap=cmap, vmin=0, vmax=len(class_colors) - 1)
    cbar = plt.colorbar(ticks=range(len(class_colors)), label="Classes")
    cbar.ax.set_yticklabels([f"{k}: {v}" for k, v in class_labels.items()])
    plt.show()

# Main script
if __name__ == "__main__":
    # Paths to input predicted raster and output raster
    predicted_raster_path = "predicted_map_2000epochs_Teresina.tif"  # Path to predicted classified raster
    output_raster_path = "../outputs/predicted_map_2000epochs_Teresina_with_labels.tif"  # Path to save reclassified raster

    # Define the class mapping
    class_mapping = {
        0: 0,  # Original Class 0 -> New Class 0
        1: 1,  # Original Class 1 -> New Class 1
        2: 2,  # Original Class 2 -> New Class 2 
        3: 3   # Original Class 3 -> New Class 3 
    }

    # Define class labels and colors
    class_labels = {
        0: "Houses",
        1: "Water",
        2: "Vegetation",
        3: "Paved"
    }
    class_colors = {
        0: [255, 255, 0], 
        1: [0, 0, 255],
        2: [0, 255, 0], 
        3: [204, 153, 0]       
    }

    # Load the predicted raster
    with rasterio.open(predicted_raster_path) as src:
        predicted_raster = src.read(1)  # Read the first band (classification map)

        # Reclassify the raster
        reclassified_raster = reclassify(predicted_raster, class_mapping)

        # Save the reclassified raster with labels and colors
        save_raster_with_labels(reclassified_raster, class_labels, class_colors, output_raster_path, src)

        # Visualize the reclassified raster
        visualize_reclassified_map(reclassified_raster, class_labels, class_colors)
