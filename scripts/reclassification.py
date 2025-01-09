import numpy as np
import rasterio
import matplotlib.pyplot as plt
import json
from matplotlib.colors import ListedColormap

def read_raster(raster_path):
    """
    Read a raster file and return its data and metadata.
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

def visualize_raster_with_colors(data, colors, title="Raster Data"):
    """
    Visualize a raster with specified colors.
    Parameters:
        data (np.ndarray): 2D array of raster data.
        colors (list): List of hex codes for the colors.
        title (str): Title for the plot.
    """
    cmap = ListedColormap(colors)
    plt.figure(figsize=(8, 8))
    plt.imshow(data, cmap=cmap, interpolation="nearest")
    plt.colorbar(ticks=range(len(colors)))
    plt.title(title)
    plt.axis("off")
    plt.show()

def get_class_info(unique_labels):
    """
    Prompt the user for class names and colors for each label.
    Parameters:
        unique_labels (np.ndarray): Array of unique numeric labels.
    Returns:
        dict: Mapping of labels to names and colors.
    """
    class_info = {}
    for label in unique_labels:
        print(f"Label: {label}")
        name = input(f"Enter a name for class {label}: ")
        current_color = input(f"Enter a hex color for {name} (default #FFFFFF): ") or "#FFFFFF"
        class_info[label] = {"name": name, "color": current_color}
    return class_info

def recolor_classification(data, class_info):
    """
    Recolor a classification map based on user-defined colors.
    Parameters:
        data (np.ndarray): 2D array of classification data.
        class_info (dict): Mapping of labels to names and colors.
    Returns:
        np.ndarray: Recolored classification map.
    """
    recolored_map = np.zeros_like(data, dtype=np.uint8)
    for label, info in class_info.items():
        recolored_map[data == label] = label
    return recolored_map

def save_recolored_map(output_path, recolored_map, meta):
    """
    Save the recolored classification map.
    Parameters:
        output_path (str): Path to save the recolored map.
        recolored_map (np.ndarray): Recolored classification map.
        meta (dict): Raster metadata.
    """
    meta.update({"dtype": "uint8"})
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(recolored_map, 1)
    print(f"Recolored classification map saved at: {output_path}")

def save_class_info(class_info, output_path):
    """
    Save the class names and colors to a JSON file.
    Parameters:
        class_info (dict): Mapping of labels to names and colors.
        output_path (str): Path to save the JSON file.
    """
    # Convert keys to int to ensure JSON compatibility
    class_info_serializable = {int(key): value for key, value in class_info.items()}
    
    with open(output_path, "w") as f:
        json.dump(class_info_serializable, f, indent=4)
    
    print(f"Class info saved at: {output_path}")


# Main execution
if __name__ == "__main__":
    # Paths
    classified_raster_path = "../outputs/classified_image.tif"
    recolored_raster_path = "../outputs/recolored_classification.tif"
    class_info_path = "../outputs/class_info.json"

    # Step 1: Read the classified raster
    classified_data, classified_meta = read_raster(classified_raster_path)
    print(f"Classified raster shape: {classified_data.shape}")

    # Step 2: Visualize current classified raster
    unique_labels = np.unique(classified_data)
    print("Detected classes (numeric labels):", unique_labels)
    visualize_raster_with_colors(classified_data, ["#FFFFFF"] * len(unique_labels), title="Current Classified Raster")

    # Step 3: Prompt for class names and colors
    class_info = get_class_info(unique_labels)

    # Step 4: Recolor the classification map
    recolored_map = recolor_classification(classified_data, class_info)
    visualize_raster_with_colors(recolored_map, [info["color"] for info in class_info.values()],
                                 title="Recolored Classification Map")

    # Step 5: Save the recolored map and class info
    save_recolored_map(recolored_raster_path, recolored_map, classified_meta)
    save_class_info(class_info, class_info_path)

    # Step 6: Save one-hot encoded labels
    print("Preparing one-hot encoded labels...")
    one_hot_labels = {}
    for label, info in class_info.items():
        one_hot_labels[info["name"]] = (classified_data == label).astype(np.uint8)
        json_path = f"../outputs/one_hot_{info['name']}.json"
        with open(json_path, "w") as f:
            json.dump(one_hot_labels[info["name"]].tolist(), f)
        print(f"One-hot label for class '{info['name']}' saved at: {json_path}")
