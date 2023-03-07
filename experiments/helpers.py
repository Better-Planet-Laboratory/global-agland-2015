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
from evaluation.cropland_eval_geowiki import parse_geowiki_cropland, reproject_geowiki_to_index_coord
from evaluation.FAO_total_comp_by_continent import pack_continent_counts_in_table


def initialize_iter0_output(experiment_cfg, land_cover_counts):
    """
    Initialize raw prediction output from the model as iteration 0. Save an AglandMap tif file to 
    the directory specified in experiment configs    
    """
    # Initialize model and save the first iteration map
    output_height, output_width = int(max(land_cover_counts.census_table['ROW_IDX']) + 1), \
                                  int(max(land_cover_counts.census_table['COL_IDX']) + 1),

    # Load model
    prob_est = OvRBernoulliGradientBoostingTree(
        ntrees=experiment_cfg['model']['gradient_boosting_tree']['ntrees'],
        max_depth=experiment_cfg['model']['gradient_boosting_tree']['max_depth'],
        nfolds=experiment_cfg['model']['gradient_boosting_tree']['nfolds'])
    try:
        prob_est.load(experiment_cfg['model']['path_dir'])
        print('Model loaded from {}'.format(experiment_cfg['model']['path_dir']))
    except h2o.exceptions.H2OResponseError:
        raise h2o.exceptions.H2OResponseError(
            'File {} is not valid model path.'.format(experiment_cfg['model']['path_dir']))

    # Initial deployment
    output_prob = prob_est.predict(land_cover_counts).to_numpy()
    initial_agland_map = AglandMap(
        output_prob[:, 0].reshape(output_height, output_width),
        output_prob[:, 1].reshape(output_height, output_width),
        output_prob[:, 2].reshape(output_height, output_width),
        force_load=True)

    # Save initial results
    initial_agland_map.save_as_tif(os.path.join(experiment_cfg['agland_map_output'], 'agland_map_output_0.tif'))


def compute_metrics(experiment_cfg, input_dataset, agland_map_output, mlflow_id='000'):
    """
    Compute metrics for input agland_map_output. Metrics include:
    1. Comparison against input census dataset
        'rmse_cropland', 
        'rmse_pasture', 
        'rmse_other', 
        'r2_cropland', 
        'r2_pasture', 
        'r2_other'
    2. Comparison against Maryland (in %)
        'maryland_mu', 
        'maryland_sigma', 
        'maryland_rmse'
    3. Comparison against GeoWiki (in %)
        'geowiki_mu', 
        'geowiki_sigma', 
        'geowiki_rmse'
    4. Comparison against FAO sum grouped over CONTINENT (in %)
        '%CONTINENT%_cropland_diff', 
        '%CONTINENT%_pasture_diff', 

    Args:
        agland_map_output (AglandMap): input AglandMap obj to be tested
        mlflow_id (str, optional): mlflow experiment id. Defaults to '000'.

    Returns: (dict): metrics results
    """
    def aggregated_gt_comparsion(output_path):
        ground_truth_collection, pred_collection = agland_map_output.extract_state_level_data(
            input_dataset, 
            rasterio.open(experiment_cfg['global_area_2160x4320_map']).read(1))
        
        rmse_cropland = rmse(pred_collection[:, 0], ground_truth_collection[:, 0])
        rmse_pasture = rmse(pred_collection[:, 1], ground_truth_collection[:, 1])
        rmse_other = rmse(pred_collection[:, 2], ground_truth_collection[:, 2])

        r2_cropland = r2(pred_collection[:, 0], ground_truth_collection[:, 0])
        r2_pasture = r2(pred_collection[:, 1], ground_truth_collection[:, 1])
        r2_other = r2(pred_collection[:, 2], ground_truth_collection[:, 2])

        plot_agland_pred_vs_ground_truth(
            experiment_cfg['post_process']['correction']['itr'], {
                experiment_cfg['post_process']['correction']['itr']: {
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
        maryland_cropland = rasterio.open(experiment_cfg['maryland_cropland_dir'])
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
        geowiki_cropland = parse_geowiki_cropland(experiment_cfg['geowiki_cropland_dir'])
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
            'geowiki_mu': np.mean(diff), 
            'geowiki_sigma': np.std(diff), 
            'geowiki_rmse': rmse_error
        }
    
    def fao_comparsion(agland_map):
        global_area_map = rasterio.open(experiment_cfg['global_area_2160x4320_map']).read()[0]
        assert (global_area_map.shape == (
            agland_map.height,
            agland_map.width)), "Input global area map must match agland map"
        
        global_census_table = load_census_table_pkl(experiment_cfg['census_table_input'])
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

            results['{}_cropland_diff'.format(current_continent)] = 100*(current_cropland_pred - current_cropland_fao)/current_cropland_fao
            results['{}_pasture_diff'.format(current_continent)] = 100*(current_pasture_pred - current_pasture_fao)/current_pasture_fao
        
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
                            global_boundary_shp=experiment_cfg['global_boundary_shp'],
                            output_dir=img_path + 'cropland.png')
            plot_agland_map_tif(tif_path + 'pasture.tif',
                            type='pasture',
                            global_boundary_shp=experiment_cfg['global_boundary_shp'],
                            output_dir=img_path + 'pasture.png')
            plot_agland_map_tif(tif_path + 'other.tif',
                            type='other',
                            global_boundary_shp=experiment_cfg['global_boundary_shp'],
                            output_dir=img_path + 'other.png')

        # Save agland tif outputs as visual metrics
        unmasked_agland_map.save_as_tif(os.path.join(output_tif_path, 'agland_map_output_{}.tif'.format(experiment_cfg['post_process']['correction']['itr'])))
        masked_agland_map.save_as_tif(os.path.join(output_tif_path, 'masked_agland_map_output_{}.tif'.format(experiment_cfg['post_process']['correction']['itr'])))
        save_seperate_agland(output_img_path + '/', output_tif_path + '/', unmasked_agland_map)
        save_seperate_agland(output_img_path + '/masked_', output_tif_path + '/masked_', masked_agland_map)
        
    # Create output dir
    img_output_path = os.path.join(experiment_cfg['img_path'], mlflow_id)
    agland_output_path = os.path.join(experiment_cfg['agland_map_output'], mlflow_id)
    if not os.path.exists(img_output_path):
        print(f'Create directory {img_output_path}')
        os.mkdir(img_output_path)
    if not os.path.exists(agland_output_path):
        print(f'Create directory {agland_output_path}')
        os.mkdir(agland_output_path)

    # Get masked results
    cropland_mask = make_nonagricultural_mask(
        shape=(agland_map_output.height, agland_map_output.width),
        mask_dir_list=[experiment_cfg['mask']['water_body_mask'], experiment_cfg['mask']['gdd_filter_mask'], experiment_cfg['mask']['antarctica_mask']])
    pasture_mask = make_nonagricultural_mask(
        shape=(agland_map_output.height, agland_map_output.width),
        mask_dir_list=[experiment_cfg['mask']['water_body_mask'], experiment_cfg['mask']['antarctica_mask']])
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

