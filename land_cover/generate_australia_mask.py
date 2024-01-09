import geopandas as gpd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os
from utils.io import gdal, save_array_as_tif

PROBABILITY_CONSTANT = 10000  # fixed in raw input
NO_DATA_CONSTANT = 32767  # fixed in raw input

if __name__ == '__main__':

    # Australia mask is based on 
    # https://www.agriculture.gov.au/sites/default/files/documents/nlum_250m_descriptivemetadata_20220622.pdf
    parent_dir = "./AgProbabilitySurfaces_2015_16"
    input_files = [os.path.join(parent_dir, i) for i in os.listdir(parent_dir)]
    output_epsg = 'EPSG:4326'
    output_size = (2160, 4320)
    pixel_size = 0.083333333333333333333
    x_min = -180
    y_max = 90
    output_affine = rasterio.Affine(pixel_size, 0, x_min, 0, -pixel_size, y_max)

    # Cropland mask - all types (non-agland) except probabilitySurface_2016_210_1_GRAZ_NOTIMBNP.tif
    # Pasture mask - all types (non-agland)
    prob_threshold = 0
    output_cropland_mask = np.zeros(output_size, dtype=np.int16)
    output_pasture_mask = np.zeros(output_size, dtype=np.int16)

    # Open the GeoTIFF file
    for input_file in input_files:
        if "probabilitySurface" in input_file:
            with rasterio.open(input_file) as src:
                # Calculate the transform and dimensions for the reprojected data
                transform, width, height = calculate_default_transform(
                    src.crs, output_epsg, src.width, src.height, *src.bounds
                )

                # Create an array to store the reprojected data
                current_output_data = np.empty(output_size, dtype=src.read(1).dtype)

                # Reproject the data
                reproject(
                    source=rasterio.band(src, 1),
                    destination=current_output_data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=output_affine,
                    dst_crs=output_epsg,
                    resampling=Resampling.nearest
                )

            # Any prob value greater or equal to prob_threshold will be set to 1
            import matplotlib.pyplot as plt
            plt.imsave("current_output_data.png", current_output_data, cmap='gray')
            current_output_data[current_output_data == NO_DATA_CONSTANT] = 0
            current_output_data[current_output_data <= prob_threshold*PROBABILITY_CONSTANT] = 0
            current_output_data[current_output_data > 0] = 1

            # Final mask needs to have all exclusion to be 0 and inclusion to be 1
            if 'GRAZ' not in input_file:
                output_cropland_mask = np.logical_or.reduce([output_cropland_mask.astype(np.int16), current_output_data], axis=0)
            output_pasture_mask = np.logical_or.reduce([output_pasture_mask.astype(np.int16), current_output_data], axis=0)

        # Crop Australia regional mask on top
        australia_mask = rasterio.open("./australia_mask.tif").read(1)
        output_cropland_mask = np.logical_or.reduce([output_cropland_mask.astype(np.int16), australia_mask], axis=0).astype(np.uint16)
        output_pasture_mask = np.logical_or.reduce([output_pasture_mask.astype(np.int16), australia_mask], axis=0).astype(np.uint16)

    save_array_as_tif("./AgProbabilitySurfaces_2015_16/australia_cropland_mask.tif", 
                    output_cropland_mask, 
                    x_min=-180, y_max=90,
                    pixel_size=0.083333333333333333333,
                    epsg=4326,
                    no_data_value=-1,
                    dtype=gdal.GDT_UInt16)

    # save_array_as_tif("./AgProbabilitySurfaces_2015_16/australia_pasture_mask.tif", 
    #                 output_pasture_mask, 
    #                 x_min=-180, y_max=90,
    #                 pixel_size=0.083333333333333333333,
    #                 epsg=4326,
    #                 no_data_value=-1,
    #                 dtype=gdal.GDT_UInt16)



        