import rasterio.io

from ..dataset import *
from ..agland_map import *
from tqdm import tqdm
from utils.io import *
from models import gbt
from utils.tools.census_core import load_census_table_pkl
from utils.tools.geo import crop_intermediate_state


def make_nonagricultural_mask(water_body_mask_dir, gdd_filter_map_dir, shape):
    """
    Generate a non-agricultural boolean mask by merging water_body_mask and gdd_filter_map,
    both mask shall indicate 0 as non-agricultural regions and 1 otherwise

    Args:
        water_body_mask_dir (str): path directory to water body mask tif file
        gdd_filter_map_dir (str): path directory to gdd filter map tif file
        shape (tuple): (height, width) of the output mask shape

    Returns: (np.array) 2D boolean mask matrix
    """
    # Load maps
    water_body_mask_map = rasterio.open(water_body_mask_dir).read(1)
    gdd_filter_map = rasterio.open(gdd_filter_map_dir).read(1)

    # Resize two maps to match the input shape
    # Use nearest neighbors as interpolation method
    water_body_mask_map_scaled = cv2.resize(water_body_mask_map, dsize=(shape[1], shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
    gdd_filter_map_scaled = cv2.resize(gdd_filter_map, dsize=(shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    return np.multiply(water_body_mask_map_scaled, gdd_filter_map_scaled)


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
        out_cropland = crop_intermediate_state(cropland_map, agland_map.affine, input_dataset, i)
        out_pasture = crop_intermediate_state(pasture_map, agland_map.affine, input_dataset, i)
        out_other = crop_intermediate_state(other_map, agland_map.affine, input_dataset, i)

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


def pipeline(deploy_setting_cfg, land_cover_cfg, training_cfg):
    # Load land cover counts histogram map
    land_cover_counts = load_pkl(land_cover_cfg['path_dir']['pred_input_map'][:-len('.pkl')])
    output_height, output_width = int(max(land_cover_counts.census_table['ROW_IDX']) + 1), \
                                  int(max(land_cover_counts.census_table['COL_IDX']) + 1),

    # Load model
    prob_est = gbt.GradientBoostingTree(ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
                                        max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
                                        nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'],
                                        distribution=training_cfg['model']['gradient_boosting_tree']['distribution'])
    try:
        prob_est.load(deploy_setting_cfg['path_dir']['model'])
        print('Model loaded from {}'.format(deploy_setting_cfg['path_dir']['model']))
    except h2o.exceptions.H2OResponseError:
        raise h2o.exceptions.H2OResponseError(
            'File {} is not valid model path.'.format(deploy_setting_cfg['path_dir']['model']))

    # Initial deployment
    output_prob = prob_est.predict(land_cover_counts).to_numpy()
    initial_agland_map = AglandMap(output_prob[:, 0].reshape(output_height, output_width),
                                   output_prob[:, 1].reshape(output_height, output_width),
                                   output_prob[:, 2].reshape(output_height, output_width), force_load=True)

    # Save initial results
    initial_agland_map.save_as_tif(deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')]
                                   + '_0' + '.tif')

    # Back correct outputs for n times
    input_dataset = Dataset(
        census_table=load_census_table_pkl(deploy_setting_cfg['path_dir']['census_table_input']),
        land_cover_code=land_cover_cfg['code']['MCD12Q1'],
        remove_land_cover_feature_index=deploy_setting_cfg['feature_remove'])

    for i in range(deploy_setting_cfg['post_process']['correction']['itr']):
        # Load previous agland map
        print('Back Correction itr: {}/{}'.format(i, deploy_setting_cfg['post_process']['correction']['itr']))
        intermediate_agland_map = load_tif_as_AglandMap(
            (deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')] + '_{}' + '.tif').format(str(i)),
            force_load=True)

        # Do back correction
        intermediate_agland_map = apply_back_correction_to_agland_map(
            input_dataset,
            intermediate_agland_map,
            deploy_setting_cfg['post_process']['correction']['method'])

        # Save current intermediate results
        intermediate_agland_map.save_as_tif((deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')]
                                             + '_{}' + '.tif').format(i + 1))
