import argparse
from utils.agland_map import *
from utils.process.post_process import *
import os

LAND_COVER_CFG = load_yaml_config('../../../configs/land_cover_cfg.yaml')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--itr", type=int, default=1,
                        help="iteration of back correction to be used")
    parser.add_argument("--output_dir", type=str, default='../_static/img/model_outputs/',
                        help="path dir to outputs")
    parser.add_argument("--water_body_dir", type=str, default='../../../land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir", type=str, default='../../../gdd/gdd_filter_map_360x720.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--census_table_dir", type=str, default='../../../outputs/processed_census.pkl',
                        help="path dir to processed census table pkl")
    args = parser.parse_args()
    print(args)

    # Set default path directory=
    agland_map_dir = '../../../outputs/agland_map_output_{}.tif'.format(args.itr)
    output_map_dir = args.output_dir + 'output_{}_'.format(args.itr)
    output_pred_vs_ground_truth_fig_dir = args.output_dir + 'pred_vs_ground_truth_fig_{}.png'.format(args.itr)
    output_pred_vs_ground_truth_data_dir = args.output_dir + 'pred_vs_ground_truth_data_{}.csv'.format(args.itr)

    # Load input dataset and agalnd map
    input_dataset = Dataset(
        census_table=load_census_table_pkl(args.census_table_dir),
        land_cover_code=LAND_COVER_CFG['code']['MCD12Q1'],
        remove_land_cover_feature_index=[])

    agland_map = load_tif_as_AglandMap(agland_map_dir, force_load=True)

    # Find prediction vs. ground truth before cropping out any region
    if not os.path.exists(output_pred_vs_ground_truth_data_dir):
        ground_truth_collection, pred_collection = agland_map.extract_state_level_data(input_dataset)
        np.savetxt(output_pred_vs_ground_truth_data_dir,
                   np.hstack((ground_truth_collection, pred_collection)),
                   delimiter=',')
    else:
        load_results = np.loadtxt(output_pred_vs_ground_truth_data_dir, delimiter=',')
        ground_truth_collection = load_results[:, 0:3]
        pred_collection = load_results[:, 3:]

    # Apply water body mask and gdd filter mask on top of agland map
    mask = make_nonagricultural_mask(args.water_body_dir,
                                     args.gdd_filter_map_dir,
                                     shape=(agland_map.height, agland_map.width))
    agland_map.apply_mask(mask)

    # Plot
    agland_map.plot(output_map_dir)
    plot_agland_pred_vs_ground_truth(ground_truth_collection, pred_collection,
                                     output_dir=output_pred_vs_ground_truth_fig_dir)


if __name__ == '__main__':
    main()
