from ..dataset import *
from ..agland_map import *
from rasterio.mask import mask
from tqdm import tqdm
from utils.tools.geo import get_border


def apply_back_correction_to_agland_map(input_dataset, agland_map, correction_method='scale'):
    """
    Back correct the input AglandMap obj to match the state-level samples in input_dataset.
    This process does not guarantee a perfect match, as the outputs will break the probability
    distribution after each iteration of correction. Then correction_method is called to
    force each modified values in the 3 agland map to probability distribution

    Args:
        input_dataset (Dataset): input census dataset to be matched to
        agland_map (AglandMap): input agland_map to be corrected

    Returns: (AglandMap)
    """
    cropland_map = agland_map.get_cropland().copy()
    pasture_map = agland_map.get_pasture().copy()
    other_map = agland_map.get_other().copy()

    # Iterate over each state level sample in the census table
    for i in tqdm(range(len(input_dataset.census_table))):

        # Crop intermediate samples with nodata to be -1
        out_cropland, _ = mask(cropland_map, get_border(i, input_dataset.census_table),
                               crop=False, nodata=-1)
        out_pasture, _ = mask(pasture_map, get_border(i, input_dataset.census_table),
                              crop=False, nodata=-1)
        out_other, _ = mask(other_map, get_border(i, input_dataset.census_table),
                            crop=False, nodata=-1)
        out_cropland = out_cropland[0]
        out_pasture = out_pasture[0]
        out_other = out_other[0]

        # Get the back correction factor from average values in the state
        ground_truth_cropland = input_dataset.census_table.iloc[i]['CROPLAND_PER']
        ground_truth_pasture = input_dataset.census_table.iloc[i]['PASTURE_PER']
        ground_truth_other = input_dataset.census_table.iloc[i]['OTHER_PER']

        mask_index_cropland = np.where(out_cropland != -1)
        mask_index_pasture = np.where(out_pasture != -1)
        mask_index_other = np.where(out_other != -1)

        mean_pred_cropland = np.mean(out_cropland[mask_index_cropland])
        mean_pred_pasture = np.mean(out_pasture[mask_index_pasture])
        mean_pred_other = np.mean(out_other[mask_index_other])

        # If average values is found to be 0 that means the state level is not
        # presented in agland map. This is due to the change in resolution from census_table
        # to agland map (high res -> low res). For these cases, factor is set to
        # be 1
        if mean_pred_cropland != 0:
            back_correction_factor_cropland = ground_truth_cropland / mean_pred_cropland
        else:
            back_correction_factor_cropland = 1

        if mean_pred_pasture != 0:
            back_correction_factor_pasture = ground_truth_pasture / mean_pred_pasture
        else:
            back_correction_factor_pasture = 1

        if mean_pred_other != 0:
            back_correction_factor_other = ground_truth_other / mean_pred_other
        else:
            back_correction_factor_other = 1

        agland_map.apply_factor(mask_index_cropland, mask_index_pasture, mask_index_other,
                                back_correction_factor_cropland, back_correction_factor_pasture,
                                back_correction_factor_other, correction_method=correction_method)

    return agland_map
