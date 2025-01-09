import numpy as np
from scipy.ndimage import distance_transform_edt, label, find_objects

def compute_spatial_features(labels):
    """
    Compute spatial features like distance to houses, rectangularity, and object dimensions.
    
    Parameters:
        labels (np.ndarray): Classification labels from unsupervised output.
    
    Returns:
        dict: Dictionary containing spatial features.
    """
    features = {}

    # Distance to houses
    house_mask = (labels == 0)  # Assuming '0' is houses. Have to cross-check the indexing
    features["distance_to_houses"] = distance_transform_edt(house_mask)

    # Rectangularity and object dimensions for roads/paved areas
    road_mask = (labels == 3)  # Assuming '3' is roads
    labeled, num_features = label(road_mask)

    rectangularity = np.zeros_like(labels, dtype=np.float32)
    length_width_ratio = np.zeros_like(labels, dtype=np.float32)

    for region in find_objects(labeled):
        if region:
            h, w = region[0].stop - region[0].start, region[1].stop - region[1].start
            rectangularity[region] = min(h, w) / max(h, w)
            length_width_ratio[region] = max(h, w) / min(h, w)

    features["rectangularity"] = rectangularity
    features["length_width_ratio"] = length_width_ratio

    return features

if __name__ == "__main__":
    labels_path = "../outputs/recolored_classification.tif"
    with rasterio.open(labels_path) as src:
        labels = src.read(1)
    
    spatial_features = compute_spatial_features(labels)
    np.savez("../outputs/spatial_features.npz", **spatial_features)
    print("Spatial features saved.")
