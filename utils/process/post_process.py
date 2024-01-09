import rasterio.io
import os
from ..dataset import *
from ..agland_map import *
from tqdm import tqdm
from utils.io import *
from models import gbt
from utils.tools.census_core import load_census_table_pkl
from utils.tools.geo import crop_intermediate_state
from utils.tools.pycno_interp import rasterio, rasterize, pycno
from utils.constants import *

BIAS_CORRECTION_ATTRIBUTES = ['BC_CROP', 'BC_PAST', 'BC_OTHE']

def is_bool(a):
    """ Check if input is bool """
    return isinstance(a, bool)


def is_list(a):
    """ Check if input is a list """
    return isinstance(a, list)


def make_nonagricultural_mask(shape, mask_dir_list=[]):
    """
    Generate a non-agricultural boolean mask by merging a list of mask tif files, each 
    mask shall indicate 0 as non-agricultural regions and 1 otherwise

    Args:
        shape (tuple): (height, width) of the output mask shape
        mask_dir_list (list of str): list of path directories to nonagricultural mask tif files (Default: [])

    Returns: (np.ndarray) 2D boolean mask matrix
    """
    # Load maps
    nonagricultural_mask = np.ones(shape)
    if mask_dir_list:

        # merge each mask
        for mask_dir in mask_dir_list:
            if len(mask_dir) != 0: 
                current_mask = rasterio.open(mask_dir).read(1)

                # resize map to match the input shape
                # use nearest neighbors as interpolation method
                current_mask_scaled = cv2.resize(current_mask,
                                                dsize=(shape[1], shape[0]),
                                                interpolation=cv2.INTER_NEAREST)

                # merge
                nonagricultural_mask = np.multiply(nonagricultural_mask, 
                                                   current_mask_scaled)

    return nonagricultural_mask


def check_weights_exists(deploy_setting_cfg, iter):
    """
    Check if bias correction weights arrays for iter exist in the path defined in 
    deploy_setting_cfg['path_dir']['base']

    Args:
        deploy_setting_cfg (dict): deploy settings from yaml
        iter (int): iter index
    
    Returns: (boolean)
    """
    base_path = deploy_setting_cfg['path_dir']['base']
    for attribute in BIAS_CORRECTION_ATTRIBUTES:
        if not os.path.exists(
                os.path.join(base_path,
                             attribute + '_' + str(int(iter)) + '.npy')):
            return False

    return True


def load_weights_array(deploy_setting_cfg, iter):
    """
    Load bias correction weights arrays for iter specified in the path defined in 
    deploy_setting_cfg['path_dir']['base']

    Args:
        deploy_setting_cfg (dict): deploy settings from yaml
        iter (int): iter index

    Returns: (tuple) 2D weights arrays tuple, (crop, past, other)
    """
    base_path = deploy_setting_cfg['path_dir']['base']
    weight_array_list = []
    for attribute in BIAS_CORRECTION_ATTRIBUTES:
        weight_array_list.append(
            np.load(
                os.path.join(base_path,
                             attribute + '_' + str(int(iter)) + '.npy')))

    return (*weight_array_list, )


def convert_weights_table_to_raster_array(gdf, value_field, x_min, y_max,
                                          pixel_size):
    """
    Directly convert a weight table to rasterized array. value_field is the specified 
    column to fill values in the map

    Args:
        gdf (geopandas.geodataframe.GeoDataFrame): Input GeoDataFrame.
        value_field (str): Field name of values to be used to produce pycnophylactic surface
        x_min (float): x min in geo transform
        y_max (float): y max in geo transform
        pixel_size (float):  pixel size in geo transform

    Returns: (np.ndarray) weight matrix
    """
    nodata = 1
    x_max = -x_min
    y_min = -y_max
    xres = int((x_max - x_min) / pixel_size)
    yres = int((y_max - y_min) / pixel_size)

    # Work out transform so that we rasterize the area where the data are!
    trans = rasterio.Affine.from_gdal(x_min, pixel_size, 0, y_max, 0,
                                      -pixel_size)

    shapes = ((geom, value)
              for geom, value in zip(gdf.geometry, gdf[value_field]))
    # burn the features into a raster array
    feature_array = rasterize(shapes=shapes,
                              fill=nodata,
                              out_shape=(yres, xres),
                              transform=trans)

    return feature_array


def generate_weights_array(deploy_setting_cfg,
                           input_dataset,
                           agland_map,
                           iter=0, 
                           save=False):
    """
    Generate bias correction weights arrays by back correcting input agland_map to 
    match the input_dataset, followed by a probability distribution fix (scale), and 
    a pycno interpolation based on settings in deploy_setting_cfg['post_process']['interpolation'] 
    for smoothing boundary effects

    Args:
        deploy_setting_cfg (dict): deploy settings from yaml
        input_dataset (Dataset): input dataset for training
        agland_map (AglandMap): input agland_map to be corrected
        iter (int): iter index. Default: 0
        save (bool): save weights array or not. Default: False

    Returns: (tuple) 2D weights arrays tuple, (crop, past, other)
    """
    grid_size = agland_map.affine[0]
    x_min = agland_map.affine[2]
    y_max = agland_map.affine[5]

    cropland_map = agland_map.get_cropland().copy()
    pasture_map = agland_map.get_pasture().copy()
    other_map = agland_map.get_other().copy()

    bc_factor_cropland = np.zeros((len(input_dataset.census_table)))
    bc_factor_pasture = np.zeros((len(input_dataset.census_table)))
    bc_factor_other = np.zeros((len(input_dataset.census_table)))

    global_area_map = rasterio.open(
        './land_cover/global_area_2160x4320.tif').read(1)

    for i in tqdm(range(len(input_dataset.census_table))):
        # Crop intermediate samples with nodata to be -1
        out_cropland = crop_intermediate_state(cropland_map, agland_map.affine,
                                               input_dataset.census_table, i)
        out_pasture = crop_intermediate_state(pasture_map, agland_map.affine,
                                              input_dataset.census_table, i)
        out_other = crop_intermediate_state(other_map, agland_map.affine,
                                            input_dataset.census_table, i)
        out_area = crop_intermediate_state(global_area_map, agland_map.affine,
                                           input_dataset.census_table, i)

        ground_truth_cropland = input_dataset.census_table.iloc[i][
            'CROPLAND_PER']
        ground_truth_pasture = input_dataset.census_table.iloc[i][
            'PASTURE_PER']
        ground_truth_other = input_dataset.census_table.iloc[i]['OTHER_PER']
        ground_truth_area = input_dataset.census_table.iloc[i]['AREA']

        # Average % = sum(C*A)/sum(A)
        out_area[np.where(out_area == -1)] = 0
        mean_pred_cropland = np.sum(out_cropland * out_area) / np.sum(out_area)
        mean_pred_pasture = np.sum(out_pasture * out_area) / np.sum(out_area)
        mean_pred_other = np.sum(out_other * out_area) / np.sum(out_area)

        # If criteria is based on total areial agland
        if deploy_setting_cfg['post_process']['correction']['criteria'] == 'area':
            ground_truth_cropland *= ground_truth_area*KHA_TO_KM2
            ground_truth_pasture *= ground_truth_area*KHA_TO_KM2
            ground_truth_other *= ground_truth_area*KHA_TO_KM2
            mean_pred_cropland *= np.sum(out_area)
            mean_pred_pasture *= np.sum(out_area)
            mean_pred_other *= np.sum(out_area)

        # If average values is found to be 0 that means the state level is not
        # presented in agland map. This is due to the change in resolution from census_table
        # to agland map (high res -> low res). For these cases, factor is set to
        # be 1
        if mean_pred_cropland != 0:
            bias_correction_factor_cropland = ground_truth_cropland / mean_pred_cropland
        else:
            bias_correction_factor_cropland = 1

        if mean_pred_pasture != 0:
            bias_correction_factor_pasture = ground_truth_pasture / mean_pred_pasture
        else:
            bias_correction_factor_pasture = 1

        if mean_pred_other != 0:
            bias_correction_factor_other = ground_truth_other / mean_pred_other
        else:
            bias_correction_factor_other = 1

        bc_factor_cropland[i] = bias_correction_factor_cropland
        bc_factor_pasture[i] = bias_correction_factor_pasture
        bc_factor_other[i] = bias_correction_factor_other

    # Add bc_factors to census table as weights table
    census_table = GeoDataFrame(input_dataset.census_table, crs=4326)
    weights_table = census_table[[
        'STATE', 'GID_0', 'REGIONS', 'geometry', 'AREA'
    ]].copy()

    weights_table[BIAS_CORRECTION_ATTRIBUTES[0]] = list(bc_factor_cropland)
    weights_table[BIAS_CORRECTION_ATTRIBUTES[1]] = list(bc_factor_pasture)
    weights_table[BIAS_CORRECTION_ATTRIBUTES[2]] = list(bc_factor_other)
    weights_table = weights_table.fillna(
        1)  # replace any nan in weights table by 1

    # Apply pycno interpolation over weights arrays
    weight_array_list = []
    for attribute in BIAS_CORRECTION_ATTRIBUTES:

        if deploy_setting_cfg['post_process']['disable_pycno']:
            # Directly get rasterized array from table
            weights_array = convert_weights_table_to_raster_array(
                gdf=weights_table,
                value_field=attribute,
                x_min=x_min,
                y_max=y_max,
                pixel_size=grid_size)
        else:
            # Use pycno interpolation to smooth weights array
            weights_array = pycno(
                gdf=weights_table,
                value_field=attribute,
                x_min=x_min,
                y_max=y_max,
                pixel_size=grid_size,
                converge=deploy_setting_cfg['post_process']['interpolation']
                ['converge'],
                r=deploy_setting_cfg['post_process']['interpolation']['r'],
                seperable_filter=deploy_setting_cfg['post_process']
                ['interpolation']['seperable_filter'],
                verbose=True)
            weights_array = weights_array[0]

        weight_array_list.append(weights_array)
        if save:
            weights_file_dir = os.path.join(
                deploy_setting_cfg['path_dir']['base'],
                attribute + '_' + str(int(iter)) + '.npy')
            np.save(weights_file_dir, weights_array)
            print('{} saved'.format(weights_file_dir))

    return (*weight_array_list, )


def apply_bias_correction_to_agland_map(agland_map,
                                        bc_crop,
                                        bc_past,
                                        bc_other,
                                        force_zero=False,
                                        threshold=0.01,
                                        correction_method='scale',
                                        iter=0):
    """
    Bias correct the input AglandMap obj to match the state-level samples in input_dataset.
    This process does not guarantee a perfect match, as the outputs will break the probability
    distribution after each iteration of correction. Then correction_method is called to
    force each modified values in the 3 agland map to probability distribution

    Args:
        agland_map (AglandMap): input agland_map to be corrected
        bc_crop (np.ndarray): weights for cropland
        bc_past (np.ndarray): weights for pasture
        bc_other (np.ndarray): weights for other
        force_zero (bool): if True, agland map with < threshold will turn into 0% before correction
        threshold (float): threshold in force_zero
        correction_method (str): 'scale' ('softmax' does not provide good results)
        iter (int): iter index

    Returns: (AglandMap)
    """
    cropland_map = agland_map.get_cropland()
    pasture_map = agland_map.get_pasture()
    other_map = agland_map.get_other()

    if force_zero:
        cropland_map[np.where(cropland_map < threshold)] = 0.01
        pasture_map[np.where(pasture_map < threshold)] = 0.01
        other_map[np.where(other_map < threshold)] = 0.01

    return AglandMap(cropland_map * bc_crop,
                     pasture_map * bc_past,
                     other_map * bc_other,
                     force_load=True)


def pipeline(deploy_setting_cfg, land_cover_cfg, training_cfg):
    """
    Deploy pipeline:
    1. Load pre-trained model, land cover inputs, inputs census data
    2. Run prediction on pre-trained model weights on inputs to get an initial agland map
    3. Apply bias correction and pycno interpolation on weights iteratively

    Args:
        deploy_setting_cfg (dict): deploy settings from yaml
        land_cover_cfg (dict): land cover settings from yaml
        training_cfg (dict): training settings from yaml
    """
    # Load land cover counts histogram map
    land_cover_counts = load_pkl(
        land_cover_cfg['path_dir']['pred_input_map'][:-len('.pkl')])
    output_height, output_width = int(max(land_cover_counts.census_table['ROW_IDX']) + 1), \
                                  int(max(land_cover_counts.census_table['COL_IDX']) + 1),

    # Load model
    # prob_est = gbt.OvRBernoulliGradientBoostingTree(
    #     ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
    #     max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
    #     nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'])
    prob_est = gbt.MultinomialGradientBoostingTree(
            ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
            max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
            nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'], 
            min_rows=training_cfg['model']['gradient_boosting_tree']['min_rows'], 
            learn_rate=training_cfg['model']['gradient_boosting_tree']['learn_rate'], 
            sample_rate=training_cfg['model']['gradient_boosting_tree']['sample_rate'], 
            col_sample_rate=training_cfg['model']['gradient_boosting_tree']['col_sample_rate']
            )
    try:
        prob_est.load(deploy_setting_cfg['path_dir']['model'])
        print('Model loaded from {}'.format(
            deploy_setting_cfg['path_dir']['model']))
    except h2o.exceptions.H2OResponseError:
        raise h2o.exceptions.H2OResponseError(
            'File {} is not valid model path.'.format(
                deploy_setting_cfg['path_dir']['model']))

    # Initial deployment
    output_prob = prob_est.predict(land_cover_counts).to_numpy()
    initial_agland_map = AglandMap(
        output_prob[:, 0].reshape(output_height, output_width),
        output_prob[:, 1].reshape(output_height, output_width),
        output_prob[:, 2].reshape(output_height, output_width),
        force_load=True)

    # Save initial results
    initial_agland_map.save_as_tif(
        deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')] +
        '_0' + '.tif')

    # Load input dataset for bias correction step
    input_dataset = Dataset(
        census_table=load_census_table_pkl(
            deploy_setting_cfg['path_dir']['census_table_input']),
        land_cover_code=land_cover_cfg['code']['MCD12Q1'],
        remove_land_cover_feature_index=deploy_setting_cfg['feature_remove'],
        invalid_data=training_cfg['invalid_data_handle'])

    # Prepare masks
    cropland_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask', 'australia_cropland_mask']
    pasture_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask', 'australia_pasture_mask']

    cropland_mask = make_nonagricultural_mask(
        shape=(initial_agland_map.height, initial_agland_map.width),
        mask_dir_list=[deploy_setting_cfg['post_process']['correction']['mask'][m] for m in cropland_mask_list])
    pasture_mask = make_nonagricultural_mask(
        shape=(initial_agland_map.height, initial_agland_map.width),
        mask_dir_list=[deploy_setting_cfg['post_process']['correction']['mask'][m] for m in pasture_mask_list])

    # Bias correction iterator
    for i in range(deploy_setting_cfg['post_process']['correction']['itr']):
        # Load previous agland map
        print('Bias Correction itr: {}/{}'.format(
            i, deploy_setting_cfg['post_process']['correction']['itr']))
        intermediate_agland_map = load_tif_as_AglandMap(
            (deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')]
             + '_{}' + '.tif').format(str(i)),
            force_load=True)

        # Apply masks
        if is_list(deploy_setting_cfg['post_process']['correction']['mask']['switch']):
            apply_mask = deploy_setting_cfg['post_process']['correction']['mask']['switch'][i]
        elif deploy_setting_cfg['post_process']['correction']['mask']['switch'] is None:
            apply_mask = False
        else:
            apply_mask = deploy_setting_cfg['post_process']['correction']['mask']['switch']
        if apply_mask:
            intermediate_agland_map.apply_mask([cropland_mask, pasture_mask], value=0)

        # Do bias correction
        intermediate_agland_map.fill_nan(value=0)
        if check_weights_exists(deploy_setting_cfg, i):
            print('Bias correction weights loaded')
            bc_crop, bc_past, bc_other = load_weights_array(
                deploy_setting_cfg, i)
        else:
            print('Generate new bias correction weights')
            bc_crop, bc_past, bc_other = generate_weights_array(
                deploy_setting_cfg, input_dataset,
                intermediate_agland_map, iter=i, save=True)

        if is_list(deploy_setting_cfg['post_process']['correction']['force_zero']['switch']):
            force_zero = deploy_setting_cfg['post_process']['correction']['force_zero']['switch'][i]
        elif deploy_setting_cfg['post_process']['correction']['force_zero']['switch'] is None:
            force_zero = False
        else:
            force_zero = deploy_setting_cfg['post_process']['correction']['force_zero']['switch']

        intermediate_agland_map = apply_bias_correction_to_agland_map(
            intermediate_agland_map, bc_crop, bc_past, bc_other,
            force_zero,
            deploy_setting_cfg['post_process']['correction']['force_zero']['threshold'],
            deploy_setting_cfg['post_process']['correction']['method'], i)

        # Save current intermediate results
        intermediate_agland_map.save_as_tif(
            (deploy_setting_cfg['path_dir']['agland_map_output'][:-len('.tif')]
             + '_{}' + '.tif').format(i + 1))
