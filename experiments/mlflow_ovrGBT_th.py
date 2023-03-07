import mlflow
import multiprocessing
from experiments.helpers import *


EXPERIMENT_CFG = io.load_yaml_config('experiments/all_correct_to_FAO_scale_itr3_fr_0/configs/experiment_cfg_th.yaml')
LAND_COVER_COUNTS = load_pkl(EXPERIMENT_CFG['pred_input_map'][:-len('.pkl')])
INPUT_DATASET = Dataset(
    census_table=load_census_table_pkl(EXPERIMENT_CFG['census_table_input']), 
    land_cover_code=EXPERIMENT_CFG['code']['MCD12Q1'],
    remove_land_cover_feature_index=EXPERIMENT_CFG['feature_remove'],
    invalid_data=EXPERIMENT_CFG['invalid_data_handle'])

THRESHOLD_SPACE = [i / 100 for i in range(0, 10)]   # <- Change this for search space 


def do_bias_correction(threshold):
    """
    Apply bias correction to raw prediction agland map

    Args:
        threshold (float): threshold values

    Returns: (AglandMap) AglandMap obj after bias correction 
    """
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

    for i in range(EXPERIMENT_CFG['post_process']['correction']['itr']):
        bc_crop, bc_past, bc_other = generate_weights_array(deploy_settings, INPUT_DATASET, intermediate_agland_map, iter=i, save=False)

        if is_list(EXPERIMENT_CFG['post_process']['correction']['force_zero']):
            force_zero = EXPERIMENT_CFG['post_process']['correction']['force_zero'][i]
        else:
            force_zero = EXPERIMENT_CFG['post_process']['correction']['force_zero']

        intermediate_agland_map = apply_bias_correction_to_agland_map(
            intermediate_agland_map, bc_crop, bc_past, bc_other,
            force_zero,
            threshold,
            EXPERIMENT_CFG['post_process']['correction']['method'], i)

    return intermediate_agland_map


def new_experiment(threshold):
    """
    Process of a single experiment on an input threshold value. To view results, 
    call ``` mlflow ui ``` in terminal for dashboard

    Args:
        threshold (float): threshold used for bias correction step
    """
    # Bias correction
    agland_map_to_test = do_bias_correction(threshold)

    # Log metrics
    with mlflow.start_run(nested=True, run_name=f'th_{threshold}'):
        
        mlflow.log_params({
            'bc_threshold': threshold
        })

        mlflow_id = mlflow.active_run().info.run_id
        metrics_results = compute_metrics(EXPERIMENT_CFG, INPUT_DATASET, agland_map_to_test, mlflow_id)

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
    with mlflow.start_run(run_name='exp_ovrGBT_th'):
        pool = multiprocessing.Pool(processes=3)
        pool.map(new_experiment, THRESHOLD_SPACE)


if __name__ == '__main__':
    run()

