import rasterio
import os

def load_bands(data_dir):
    bands = {}
    for file in os.listdir(data_dir):
        if file.endswith('.tif'):
            band_name = os.path.splitext(file)[0]
            bands[band_name] = os.path.join(data_dir, file)
    return bands

def read_band(band_path):
    with rasterio.open(band_path) as src:
        band_data = src.read(1)
        meta = src.meta
    return band_data, meta

# Example Usage
if __name__ == "__main__":
    data_dir = "../data"
    bands = load_bands(data_dir)
    print("Bands found:", bands)

    # Load one band as an example
    band1_data, band1_meta = read_band(bands['T23MQQ_20231212T131239_B02_10m.jp2'])
    print("Band 1 Shape:", band1_data.shape)
