import numpy as np
from osgeo import gdal
import cv2
import rasterio
import matplotlib.pyplot as plt

# ====================================================
# Implement Customized gdd_crop_criteria algorithms here
# ====================================================

# def gdd_crop_criteria(gdd_map, lat, lon):
#     """
#     Customized criteria to set boolean matrix based on GDD
#     Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GB002952 (Section 2.1)
#     - Above 50 deg in y axis, with GDD < 1000 not included

#     Args:
#         gdd_map (np.ndarray): 2D GDD map
#         lat (np.ndarray): latitude grid for GDD map 1D
#         lon (np.ndarray): longitude grid for GDD map 1D

#     Returns: (tuple array) (x_idx array, y_idx array)
#     """
#     # Above 50 deg in y axis, with GDD < 1000 not included
#     mask_index = np.where(gdd_map[np.where(np.flip(lat) >= 50)] < 1000)
#     return mask_index


def gdd_crop_criteria(gdd_map, lat, lon, land_cover_dir, gdd_regions):
    """
    Customized criteria to set boolean matrix based on GDD
    Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GB002952 (Section 2.1)
    - Above 50 deg in y axis, with GDD < 1500 not included

    Args:
        gdd_map (np.ndarray): 2D GDD map
        lat (np.ndarray): latitude grid for GDD map 1D
        lon (np.ndarray): longitude grid for GDD map 1D

    Returns: (tuple array) (x_idx array, y_idx array)
    """

    # # Load land cover map and convert it into a binary mask using category 
    # # 12: 'CROPLAND' and 14: 'CROPLAND-NATURAL VEGETATION MOSAICS', then resize
    # land_cover_map = np.array(gdal.Open(land_cover_dir).ReadAsArray())
    # land_cover_map_cropland_mask = np.zeros(land_cover_map.shape)
    # land_cover_map_cropland_mask[np.where((land_cover_map == 12) | (land_cover_map == 14))] = 1

    # # Resize land_cover_map to gdd_map using nearest neighbor
    # if land_cover_map_cropland_mask.shape != gdd_map.shape:
    #     land_cover_map_cropland_mask = cv2.resize(land_cover_map_cropland_mask,
    #                                               gdd_map.shape[::-1], 
    #                                               interpolation=cv2.INTER_NEAREST)

    # # Above 50 deg in y axis, with GDD < 1500 not included
    # # On top of that, any cropland shown in land cover map should be excluded 
    # mask_index = np.where(np.logical_and(gdd_map[np.where(np.flip(lat) >= 50)] < 1500, \
    #     land_cover_map_cropland_mask[np.where(np.flip(lat) >= 50)] == 0))


    with rasterio.open(land_cover_dir) as dataset:
        land_cover_map_cropland_mask = dataset.read()[0, :, :]

    # land_cover_map_cropland_mask = np.zeros(land_cover_map.shape)
    # land_cover_map_cropland_mask[np.where((land_cover_map == 12) | (land_cover_map == 14))] = 1

    # Resize land_cover_map to gdd_map using nearest neighbor
    # if land_cover_map_cropland_mask.shape != gdd_map.shape:
    #     land_cover_map_cropland_mask = cv2.resize(land_cover_map_cropland_mask,
    #                                               gdd_map.shape[::-1], 
    #                                               interpolation=cv2.INTER_NEAREST)

    # On top of that, any cropland shown in land cover map should be excluded
    mask_index = np.where(
        np.logical_or((
            np.logical_and(gdd_map[np.where(np.flip(lat) >= 50)] < 1500, \
                           land_cover_map_cropland_mask[np.where(np.flip(lat) >= 50)] == 0)
        ), 
            gdd_regions[np.where(np.flip(lat) >= 50)] == 0
        )
        )

    return mask_index
