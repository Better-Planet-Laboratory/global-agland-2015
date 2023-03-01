import mlflow
from utils import io
from utils.tools.census_core import load_census_table_pkl
from utils.dataset import *
from utils.tools.visualizer import *
from utils import constants
from models.gbt import *
from utils.agland_map import *
from utils.process.post_process import *
from rasterio import warp
import os, copy
import multiprocessing
from evaluation.cropland_eval_geowiki import parse_geowiki_cropland, reproject_geowiki_to_index_coord
from evaluation.FAO_total_comp_by_continent import pack_continent_counts_in_table

EXPERIMENT_CFG = io.load_yaml_config('experiments/all_correct_to_FAO_scale_itr3_fr_0/configs/experiment_cfg.yaml')
LAND_COVER_COUNTS = load_pkl(EXPERIMENT_CFG['pred_input_map'][:-len('.pkl')])
INPUT_DATASET = Dataset(
    census_table=load_census_table_pkl(EXPERIMENT_CFG['census_table_input']), 
    land_cover_code=EXPERIMENT_CFG['code']['MCD12Q1'],
    remove_land_cover_feature_index=EXPERIMENT_CFG['feature_remove'],
    invalid_data=EXPERIMENT_CFG['invalid_data_handle'])

THRESHOLD_SPACE = [i / 100 for i in range(0, 10)]


def initialize_iter0_output():
    """
    Initialize raw prediction output from the model as iteration 0. Save an AglandMap tif file to 
    the directory specified in experiment configs    
    """
    # Initialize model and save the first iteration map
    output_height, output_width = int(max(LAND_COVER_COUNTS.census_table['ROW_IDX']) + 1), \
                                  int(max(LAND_COVER_COUNTS.census_table['COL_IDX']) + 1),

    # Load model
    prob_est = OvRBernoulliGradientBoostingTree(
        ntrees=EXPERIMENT_CFG['model']['gradient_boosting_tree']['ntrees'],
        max_depth=EXPERIMENT_CFG['model']['gradient_boosting_tree']['max_depth'],
        nfolds=EXPERIMENT_CFG['model']['gradient_boosting_tree']['nfolds'])
    try:
        prob_est.load(EXPERIMENT_CFG['model']['path_dir'])
        print('Model loaded from {}'.format(EXPERIMENT_CFG['model']['path_dir']))
    except h2o.exceptions.H2OResponseError:
        raise h2o.exceptions.H2OResponseError(
            'File {} is not valid model path.'.format(EXPERIMENT_CFG['model']['path_dir']))

    # Initial deployment
    output_prob = prob_est.predict(LAND_COVER_COUNTS).to_numpy()
    initial_agland_map = AglandMap(
        output_prob[:, 0].reshape(output_height, output_width),
        output_prob[:, 1].reshape(output_height, output_width),
        output_prob[:, 2].reshape(output_height, output_width),
        force_load=True)

    # Save initial results
    initial_agland_map.save_as_tif(os.path.join(EXPERIMENT_CFG['agland_map_output'], 'agland_map_output_0.tif'))


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

    for i in range(EXPERIMENT_CFG['post_process']['correction']['itr']):
        bc_crop, bc_past, bc_other = generate_weights_array(
            {
                'post_process': {
                    'disable_pycno': EXPERIMENT_CFG['post_process']['disable_pycno'], 
                    'interpolation': {
                        'seperable_filter': EXPERIMENT_CFG['post_process']['interpolation']['seperable_filter'], 
                        'converge': EXPERIMENT_CFG['post_process']['interpolation']['converge'], 
                        'r': EXPERIMENT_CFG['post_process']['interpolation']['r']
                    }
                }
            }, INPUT_DATASET, 
            intermediate_agland_map, iter=i, save=False)

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


def compute_metrics(agland_map_output, mlflow_id=None):
    # Metrics:
    #   1. RMSE and R2 for cropland and pasture at iteration 3
    #   2. mu, sigma, RMSE for maryland and geowiki 
    #   3. cropland and pasture % difference FAO continent table with cropland masked and pasture unmasked (but need to include water body masks)
    def aggregated_gt_comparsion(output_path):
        ground_truth_collection, pred_collection = agland_map_output.extract_state_level_data(
            INPUT_DATASET, 
            rasterio.open(EXPERIMENT_CFG['global_area_2160x4320_map']).read(1))
        
        rmse_cropland = rmse(pred_collection[:, 0], ground_truth_collection[:, 0])
        rmse_pasture = rmse(pred_collection[:, 1], ground_truth_collection[:, 1])
        rmse_other = rmse(pred_collection[:, 2], ground_truth_collection[:, 2])

        r2_cropland = r2(pred_collection[:, 0], ground_truth_collection[:, 0])
        r2_pasture = r2(pred_collection[:, 1], ground_truth_collection[:, 1])
        r2_other = r2(pred_collection[:, 2], ground_truth_collection[:, 2])

        plot_agland_pred_vs_ground_truth(
            EXPERIMENT_CFG['post_process']['correction']['itr'], {
                EXPERIMENT_CFG['post_process']['correction']['itr']: {
                    'ground_truth_collection': ground_truth_collection,
                    'pred_collection': pred_collection
                }
            }, 
            output_dir=os.path.join(output_path, 'aggregated_gt_comparsion.png'))
        
        return {
            'rmse_cropland': rmse_cropland, 
            'rmse_pasture': rmse_pasture, 
            'rmse_other': rmse_other, 
            'r2_cropland': r2_cropland, 
            'r2_pasture': r2_pasture, 
            'r2_other': r2_other
        }

    def maryland_comparsion(output_path, agland_map):
        maryland_cropland = rasterio.open(EXPERIMENT_CFG['maryland_cropland_dir'])
        maryland_cropland_reproj = np.empty((agland_map.height, agland_map.width),
                                    dtype=np.uint8)
        warp.reproject(maryland_cropland.read(1),
                    maryland_cropland_reproj,
                    src_transform=maryland_cropland.transform,
                    src_crs=maryland_cropland.crs,
                    dst_transform=agland_map.affine,
                    dst_crs='EPSG:4326')

        pred_map = agland_map.get_cropland()
        diff = pred_map - maryland_cropland_reproj / 100
        diff *= 100  # use percentage
        rmse_error = np.sqrt(np.nanmean(diff**2))

        plot_histogram_diff_maryland_pred_cropland(
            maryland_cropland_reproj, pred_map, 
            os.path.join(output_path, 'maryland_diff_hist.png'))

        return {
            'maryland_mu': np.nanmean(diff), 
            'maryland_sigma': np.nanstd(diff), 
            'maryland_rmse': rmse_error
        }

    def geowiki_comparsion(output_path, agland_map):
        geowiki_cropland = parse_geowiki_cropland(EXPERIMENT_CFG['geowiki_cropland_dir'])
        geowiki_cropland_by_index = reproject_geowiki_to_index_coord(
            geowiki_cropland, agland_map.affine)
        pred = agland_map.get_cropland()[(
            (geowiki_cropland_by_index[:, 0]).astype(int),
            (geowiki_cropland_by_index[:, 1]).astype(int))]

        # Compute difference map between Geowiki and pred
        nan_index = np.isnan(pred)
        diff = pred[~nan_index] - geowiki_cropland_by_index[~nan_index, 2] / 100
        diff *= 100  # use percentage
        rmse_error = np.sqrt(np.mean(diff**2))

        plot_histogram_diff_geowiki_pred_cropland(
            geowiki_cropland_by_index, pred,
            os.path.join(output_path, 'geowiki_diff_hist.png'))

        return {
            'maryland_mu': np.mean(diff), 
            'maryland_sigma': np.std(diff), 
            'maryland_rmse': rmse_error
        }
    
    def fao_comparsion(agland_map):
        global_area_map = rasterio.open(EXPERIMENT_CFG['global_area_2160x4320_map']).read()[0]
        assert (global_area_map.shape == (
            agland_map.height,
            agland_map.width)), "Input global area map must match agland map"
        
        global_census_table = load_census_table_pkl(EXPERIMENT_CFG['census_table_input'])
        num_states = len(global_census_table)

        continent_list = np.unique(np.asarray(list(
            global_census_table['REGIONS'])))
        FAO_continent_count = {i: np.zeros(2) for i in continent_list}
        pred_continent_count = {i: np.zeros(2) for i in continent_list}
        for i, _ in tqdm(enumerate(range(num_states)), total=num_states):

            out_cropland = np.nan_to_num(
                crop_intermediate_state(agland_map.get_cropland(), agland_map.affine,
                                        global_census_table, i))
            out_pasture = np.nan_to_num(
                crop_intermediate_state(agland_map.get_pasture(), agland_map.affine,
                                        global_census_table, i))
            area_map = crop_intermediate_state(global_area_map, agland_map.affine,
                                            global_census_table, i)

            out_cropland[out_cropland < 0] = 0
            out_pasture[out_pasture < 0] = 0
            area_map[area_map < 0] = 0

            out_cropland_area = out_cropland * area_map
            out_pasture_area = out_pasture * area_map

            pred_continent_count[
                global_census_table.iloc[i]['REGIONS']] += np.asarray([
                    np.sum(out_cropland_area) / constants.KHA_TO_KM2,
                    np.sum(out_pasture_area) / constants.KHA_TO_KM2
                ])

            FAO_continent_count[global_census_table.iloc[i]
                                ['REGIONS']] += np.nan_to_num(
                                    np.asarray([
                                        global_census_table.iloc[i]['CROPLAND'],
                                        global_census_table.iloc[i]['PASTURE']
                                    ]), 0)
        comparison_table = pack_continent_counts_in_table(pred_continent_count, FAO_continent_count)

        results = {}
        for i in range(len(comparison_table.index)):
            current_continent = comparison_table.iloc[i]['Continent']
            current_cropland_pred = comparison_table.iloc[i]['Pred_Crop (kHa)']
            current_pasture_pred = comparison_table.iloc[i]['Pred_Past (kHa)']
            current_cropland_fao = comparison_table.iloc[i]['FAO_Crop (kHa)']
            current_pasture_fao = comparison_table.iloc[i]['FAO_Past (kHa)']

            results['{}_cropland_per_diff'.format(current_continent)] = 100*(current_cropland_pred - current_cropland_fao)/current_cropland_fao
            results['{}_pasture_per_diff'.format(current_continent)] = 100*(current_pasture_pred - current_pasture_fao)/current_pasture_fao
        
        return results

    def save_visual_outputs(output_img_path, output_tif_path, unmasked_agland_map, masked_agland_map):
        def save_seperate_agland(img_path, tif_path, agland_map):
            save_array_as_tif(tif_path + 'cropland.tif',
                    agland_map.get_cropland(),
                    x_min=-180,
                    y_max=90,
                    pixel_size=abs(-180) * 2 / agland_map.width,
                    epsg=4326,
                    no_data_value=255,
                    dtype=gdal.GDT_Float64)
            save_array_as_tif(tif_path + 'pasture.tif',
                            agland_map.get_pasture(),
                            x_min=-180,
                            y_max=90,
                            pixel_size=abs(-180) * 2 / agland_map.width,
                            epsg=4326,
                            no_data_value=255,
                            dtype=gdal.GDT_Float64)
            save_array_as_tif(tif_path + 'other.tif',
                            agland_map.get_other(),
                            x_min=-180,
                            y_max=90,
                            pixel_size=abs(-180) * 2 / agland_map.width,
                            epsg=4326,
                            no_data_value=255,
                            dtype=gdal.GDT_Float64)
            plot_agland_map_tif(tif_path + 'cropland.tif',
                            type='cropland',
                            global_boundary_shp=EXPERIMENT_CFG['global_boundary_shp'],
                            output_dir=img_path + 'cropland.png')
            plot_agland_map_tif(tif_path + 'pasture.tif',
                            type='pasture',
                            global_boundary_shp=EXPERIMENT_CFG['global_boundary_shp'],
                            output_dir=img_path + 'pasture.png')
            plot_agland_map_tif(tif_path + 'other.tif',
                            type='other',
                            global_boundary_shp=EXPERIMENT_CFG['global_boundary_shp'],
                            output_dir=img_path + 'other.png')

        # Save agland tif outputs as visual metrics
        unmasked_agland_map.save_as_tif(os.path.join(output_tif_path, 'agland_map_output_{}.tif'.format(EXPERIMENT_CFG['post_process']['correction']['itr'])))
        masked_agland_map.save_as_tif(os.path.join(output_tif_path, 'masked_agland_map_output_{}.tif'.format(EXPERIMENT_CFG['post_process']['correction']['itr'])))
        save_seperate_agland(output_img_path + '/', output_tif_path + '/', unmasked_agland_map)
        save_seperate_agland(output_img_path + '/masked_', output_tif_path + '/masked_', masked_agland_map)
        
    # Create output dir
    img_output_path = os.path.join(EXPERIMENT_CFG['img_path'], mlflow_id)
    agland_output_path = os.path.join(EXPERIMENT_CFG['agland_map_output'], mlflow_id)
    if not os.path.exists(img_output_path):
        print(f'Create directory {img_output_path}')
        os.mkdir(img_output_path)
    if not os.path.exists(agland_output_path):
        print(f'Create directory {agland_output_path}')
        os.mkdir(agland_output_path)

    # Get masked results
    cropland_mask = make_nonagricultural_mask(
        shape=(agland_map_output.height, agland_map_output.width),
        mask_dir_list=[EXPERIMENT_CFG['mask']['water_body_mask'], EXPERIMENT_CFG['mask']['gdd_filter_mask']])
    pasture_mask = make_nonagricultural_mask(
        shape=(agland_map_output.height, agland_map_output.width),
        mask_dir_list=[EXPERIMENT_CFG['mask']['water_body_mask']])
    masked_agland_map_output = copy.deepcopy(agland_map_output)
    masked_agland_map_output.apply_mask([cropland_mask, pasture_mask])

    # Visual outputs
    save_visual_outputs(img_output_path, agland_output_path, agland_map_output, masked_agland_map_output)

    # Pack metrics
    results = {}
    for r in [aggregated_gt_comparsion(img_output_path), \
              maryland_comparsion(img_output_path, masked_agland_map_output), \
              geowiki_comparsion(img_output_path, masked_agland_map_output), \
              fao_comparsion(masked_agland_map_output)]:
        results.update(r)

    return results


def new_experiment(threshold):

    # Bias correction
    agland_map_to_test = do_bias_correction(threshold)

    # Log metrics
    with mlflow.start_run(nested=True, run_name=f'th_{threshold}'):
        
        mlflow.log_params({
            'bc_threshold': threshold
        })

        mlflow_id = mlflow.active_run().info.run_id
        metrics_results = compute_metrics(agland_map_to_test, mlflow_id)

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
    # initialize_iter0_output()
    with mlflow.start_run(run_name='exp_ovrGBT_th'):
        pool = multiprocessing.Pool(processes=4)
        pool.map(new_experiment, THRESHOLD_SPACE)


if __name__ == '__main__':
    run()

