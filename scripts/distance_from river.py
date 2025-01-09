import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt

def calculate_distance_analysis(raster_path, class_mapping, house_class=0, river_class=1, distance_bin=100):
    """
    Analyzes the percentage of house pixels based on their distance from river pixels.

    Args:
        raster_path (str): Path to the classified raster.
        class_mapping (dict): Mapping of classes to labels.
        house_class (int): Class value representing houses.
        river_class (int): Class value representing river.
        distance_bin (int): Bin size for distance in meters (or raster units).

    Returns:
        dict: Distance bins with percentages of house pixels.
    """
    with rasterio.open(raster_path) as src:
        classified_raster = src.read(1)  # Read classification map

    # Create binary masks for houses and rivers
    house_mask = (classified_raster == house_class)
    river_mask = (classified_raster == river_class)

    # Calculate distances of every pixel to the nearest river pixel
    distances = distance_transform_edt(~river_mask)  # Distance from river pixels

    # Group house pixels by distance bins
    house_distances = distances[house_mask]
    max_distance = np.max(house_distances)
    bins = np.arange(0, max_distance + distance_bin, distance_bin)
    hist, bin_edges = np.histogram(house_distances, bins=bins)

    # Calculate percentage of house pixels in each distance bin
    total_houses = len(house_distances)
    percentages = (hist / total_houses) * 100

    # Create a dictionary of distance bins and percentages
    distance_analysis = {
        f"{int(bin_edges[i])}-{int(bin_edges[i + 1])} meters": percentages[i]
        for i in range(len(percentages))
    }

    # Plot the distance analysis
    plt.figure(figsize=(10, 6))
    plt.bar(distance_analysis.keys(), percentages, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Distance from River (meters)')
    plt.ylabel('Percentage of Houses')
    plt.title('Percentage of Houses by Distance from River')
    plt.tight_layout()
    plt.show()

    return distance_analysis

# Main script
if __name__ == "__main__":
    classified_raster_path = "../outputs/predicted_map_2000epochs_Teresina_with_labels.tif"  # Path to classified raster

    class_mapping = {
        0: "Houses",
        1: "Water",
        2: "Vegetation",
        3: "Paved",
    }

    # Analyze distance
    distance_analysis = calculate_distance_analysis(
        classified_raster_path,
        class_mapping,
        house_class=0,  # Houses class
        river_class=1,  # River class (Water)
        distance_bin=200  # 1 mile was approx. 1609 meters, going out of city so used 200 meters.
    )

    # Print the analysis
    print("Distance Analysis:")
    for distance_bin, percentage in distance_analysis.items():
        print(f"{distance_bin}: {percentage:.2f}%")
