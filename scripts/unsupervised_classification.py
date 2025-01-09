import numpy as np
import rasterio
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def read_stacked_image(stacked_path):
    """
    Read a stacked multi-band raster image.
    Parameters:
        stacked_path (str): Path to the stacked raster file.
    Returns:
        np.ndarray: Multi-band image as a NumPy array.
        dict: Raster metadata.
    """
    with rasterio.open(stacked_path) as src:
        stacked_data = src.read()  # Shape: (Bands, Height, Width)
        meta = src.meta
    return stacked_data, meta

def reshape_for_clustering(image_data):
    """
    Reshape multi-band raster data for clustering.
    Parameters:
        image_data (np.ndarray): Multi-band image array (Bands, Height, Width).
    Returns:
        np.ndarray: Reshaped array for clustering (Pixels, Bands).
    """
    bands, height, width = image_data.shape
    reshaped_data = image_data.reshape(bands, height * width).T  # Shape: (Pixels, Bands)
    return reshaped_data, height, width

def kmeans_clustering(reshaped_data, n_clusters):
    """
    Perform K-Means clustering on the reshaped data.
    Parameters:
        reshaped_data (np.ndarray): Reshaped array for clustering (Pixels, Bands).
        n_clusters (int): Number of clusters.
    Returns:
        np.ndarray: Cluster labels for each pixel.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reshaped_data)
    return cluster_labels

def reshape_to_image(cluster_labels, height, width):
    """
    Reshape the cluster labels back to the original image dimensions.
    Parameters:
        cluster_labels (np.ndarray): Cluster labels (Pixels).
        height (int): Height of the original image.
        width (int): Width of the original image.
    Returns:
        np.ndarray: Cluster labels reshaped to (Height, Width).
    """
    return cluster_labels.reshape(height, width)

def save_classification(output_path, classification, meta):
    """
    Save the classification result as a GeoTIFF.
    Parameters:
        output_path (str): Path to save the classified image.
        classification (np.ndarray): Classified image (Height, Width).
        meta (dict): Original raster metadata.
    """
    meta.update({
        "count": 1,  # Single band for classification
        "dtype": "uint8"
    })
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(classification.astype("uint8"), 1)
    print(f"Classification saved at: {output_path}")

def visualize_classification(classification, n_clusters):
    """
    Visualize the classification result.
    Parameters:
        classification (np.ndarray): Classified image (Height, Width).
        n_clusters (int): Number of clusters.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(classification, cmap="tab20", interpolation="nearest")
    plt.colorbar(ticks=range(n_clusters))
    plt.title(f"Unsupervised Classification - {n_clusters} Clusters")
    plt.axis("off")
    plt.show()

# Main execution
if __name__ == "__main__":
    # Input and output paths
    stacked_path = "../outputs/stacked_image_clipped.tif"
    output_path = "../outputs/classified_image.tif"
    n_clusters = 4

    # Step 1: Read the stacked image
    stacked_data, meta = read_stacked_image(stacked_path)

    # Step 2: Reshape the data for clustering
    reshaped_data, height, width = reshape_for_clustering(stacked_data)

    # Step 3: Perform K-Means clustering
    cluster_labels = kmeans_clustering(reshaped_data, n_clusters)

    # Step 4: Reshape back to image dimensions
    classified_image = reshape_to_image(cluster_labels, height, width)

    # Step 5: Save the classification result
    save_classification(output_path, classified_image, meta)

    # Step 6: Visualize the classification
    visualize_classification(classified_image, n_clusters)
