import numpy as np


# ====================================================
# Implement Customized gdd_crop_criteria algorithms here
# ====================================================

def gdd_crop_criteria(gdd_map, lat, lon):
    """
    Customized criteria to set boolean matrix based on GDD
    Source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GB002952 (Section 2.1)
    - Above 50 deg in y axis, with GDD < 1000 not included

    Args:
        gdd_map (2d array): GDD map
        lat (1d array): latitude grid for GDD map
        lon (1d array): longitude grid for GDD map

    Returns: (tuple array) (x_idx array, y_idx array)
    """
    # Above 50 deg in y axis, with GDD < 1000 not included
    mask_index = np.where(gdd_map[np.where(np.flip(lat) >= 50)] < 1000)
    return mask_index
