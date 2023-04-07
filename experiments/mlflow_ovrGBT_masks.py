import mlflow
import multiprocessing
import itertools
from experiments.helpers import *


EXPERIMENT_CFG = io.load_yaml_config('experiments/all_correct_to_FAO_scale_itr3_fr_0/configs/experiment_cfg_masks.yaml')
LAND_COVER_COUNTS = load_pkl(EXPERIMENT_CFG['pred_input_map'][:-len('.pkl')])
INPUT_DATASET = Dataset(
    census_table=load_census_table_pkl(EXPERIMENT_CFG['census_table_input']), 
    land_cover_code=EXPERIMENT_CFG['code']['MCD12Q1'],
    remove_land_cover_feature_index=EXPERIMENT_CFG['feature_remove'],
    invalid_data=EXPERIMENT_CFG['invalid_data_handle'])

MASK_APPLY_ORDER = ['after']
# THRESHOLD_AI = [0.01, 0.02, 0.03, 0.04, 0.05]
# THRESHOLD_AEI = [0.01, 100, 1000, 5000, 10000]

THRESHOLD_AI = [0.03]
THRESHOLD_AEI = [0.01]


def do_bias_correction(mask_apply_order, threshold_AI, threshold_AEI):
    """
    Apply bias correction to raw prediction agland map

    Args:
        mask_apply_order (str): order to apply mask
        threshold_AI (float): threshold values
        threshold_AEI (float): threshold values

    Returns: (AglandMap) AglandMap obj after bias correction 
    """
    assert (mask_apply_order in ['before', 'after']), "mask_apply_order must be either before bias correction or after"

    # Bias correction iterator
    try:
        intermediate_agland_map = load_tif_as_AglandMap(os.path.join(EXPERIMENT_CFG['agland_map_output'], 'agland_map_output_0.tif'), force_load=True)
    except FileNotFoundError:
        print ('Could not find initial agland map in {}. Run initialization step.'.format(EXPERIMENT_CFG['agland_map_output']))

    deploy_settings = {
        'post_process': {
            'disable_pycno': EXPERIMENT_CFG['post_process']['disable_pycno'], 
            'interpolation': {
                'seperable_filter': EXPERIMENT_CFG['post_process']['interpolation']['seperable_filter'], 
                'converge': EXPERIMENT_CFG['post_process']['interpolation']['converge'], 
                'r': EXPERIMENT_CFG['post_process']['interpolation']['r']
            }
        }
    }
    
    # Prepare masks
    cropland_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask_{}_{}'.format(str(threshold_AEI), str(threshold_AI).replace('.', ''))]
    pasture_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask_{}_{}'.format(str(threshold_AEI), str(threshold_AI).replace('.', ''))]
    cropland_mask = make_nonagricultural_mask(
        shape=(intermediate_agland_map.height, intermediate_agland_map.width),
        mask_dir_list=[EXPERIMENT_CFG['mask'][m] for m in cropland_mask_list])
    pasture_mask = make_nonagricultural_mask(
        shape=(intermediate_agland_map.height, intermediate_agland_map.width),
        mask_dir_list=[EXPERIMENT_CFG['mask'][m] for m in pasture_mask_list])

    # Apply masks before bias correction 
    if mask_apply_order == 'before':
        intermediate_agland_map.apply_mask([cropland_mask, pasture_mask], value=0)

    # Bias correction step
    for i in range(EXPERIMENT_CFG['post_process']['correction']['itr']):
        bc_crop, bc_past, bc_other = generate_weights_array(deploy_settings, INPUT_DATASET, intermediate_agland_map, iter=i, save=False)

        intermediate_agland_map = apply_bias_correction_to_agland_map(
            intermediate_agland_map, bc_crop, bc_past, bc_other,
            False,
            0,
            EXPERIMENT_CFG['post_process']['correction']['method'], i)

    # Apply masks after bias correction
    if mask_apply_order == 'after':
        intermediate_agland_map.apply_mask([cropland_mask, pasture_mask], value=0)

    return intermediate_agland_map


def new_experiment(args):
    """
    Process of a single experiment on post-processing masks without any thresholding to the 
    agaland map values. Search space includes:
    1. Apply masks before bias correction vs. after bias correction
    2. threshold_AI = [0.01, 0.02, 0.03, 0.04, 0.05]  
    3. threshold_AEI = [100, 500, 1000, 5000, 10000]
    
    To view results, 
    call ``` mlflow ui ``` in terminal for dashboard

    Args:
        args (tuple): mask_apply_order, threshold_AI, threshold_AEI
    """
    mask_apply_order, threshold_AI, threshold_AEI = args

    # Bias correction
    agland_map_to_test = do_bias_correction(mask_apply_order, threshold_AI, threshold_AEI)

    # Log metrics
    with mlflow.start_run(nested=True, run_name=f'mask_apply_{mask_apply_order}_threshold_AI_{threshold_AI}_threshold_AEI_{threshold_AEI}'):
        
        mlflow.log_params({
            'mask_apply_order': mask_apply_order, 
            'threshold_AI': threshold_AI, 
            'threshold_AEI': threshold_AEI
        })

        mlflow_id = mlflow.active_run().info.run_id

        cropland_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask_{}_{}'.format(str(threshold_AEI), str(threshold_AI).replace('.', ''))]
        pasture_mask_list = ['water_body_mask', 'gdd_filter_mask', 'antarctica_mask', 'aridity_mask_{}_{}'.format(str(threshold_AEI), str(threshold_AI).replace('.', ''))]
        metrics_results = compute_metrics(EXPERIMENT_CFG, INPUT_DATASET, agland_map_to_test, 
                                          cropland_mask_list, pasture_mask_list, mlflow_id)

        mlflow.log_metrics(metrics_results)
        
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'aggregated_gt_comparsion.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'maryland_diff_hist.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'geowiki_diff_hist.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'cropland.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'pasture.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'other.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'masked_cropland.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'masked_pasture.png'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id, 'masked_other.png'))

        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'agland_map_output_{}.tif'.format(EXPERIMENT_CFG['post_process']['correction']['itr'])))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'masked_agland_map_output_{}.tif'.format(EXPERIMENT_CFG['post_process']['correction']['itr'])))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'cropland.tif'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'pasture.tif'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'other.tif'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'masked_cropland.tif'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'masked_pasture.tif'))
        mlflow.log_artifact(os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id, 'masked_other.tif'))


def run():
    initialize_iter0_output(EXPERIMENT_CFG, LAND_COVER_COUNTS)
    # with mlflow.start_run(run_name='exp_ovrGBT_masks'):
    #     pool = multiprocessing.Pool(processes=3)
    #     pool.map(new_experiment, itertools.product(MASK_APPLY_ORDER, THRESHOLD_AI, THRESHOLD_AEI))


if __name__ == '__main__':
    run()

