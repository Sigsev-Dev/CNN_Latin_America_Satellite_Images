import rasterio
import matplotlib.pyplot as plt

def read_raster(raster_path):
    """
    Read a raster file.
    Parameters:
        raster_path (str): Path to the raster file.
    Returns:
        np.ndarray: Raster data as a NumPy array.
        dict: Raster metadata.
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band
        meta = src.meta
    return data, meta

def visualize_raster(data, cmap="tab20", title="Raster Visualization"):
    """
    Visualize a raster file.
    Parameters:
        data (np.ndarray): 2D array of raster data.
        cmap (str): Color map for visualization.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Path to the raster file
    raster_path = "../outputs/predicted_map_2000epochs_Teresina_with_labels.tif"  # Update this path if needed

    # Step 1: Read the raster
    raster_data, raster_meta = read_raster(raster_path)

    # Step 2: Visualize the raster
    visualize_raster(raster_data, title="Stacked image clipped")
