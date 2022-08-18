import argparse
from mimetypes import suffix_map
from utils.agland_map import *
from utils.process.post_process import *
import os

LAND_COVER_CFG = load_yaml_config('../../../configs/land_cover_cfg.yaml')


def pack_agland_maps(input_dir):
    """
    Pack agland map files 'agland_map_output_*.tif' exist in input_dir into 
    lookup table, with itr "*" as keys

    Args:
        input_dir (str): path dir that contains 'agland_map_output_*.tif'

    Return: (dict) agland maps table
    """
    prefix = 'agland_map_output_'
    suffix = '.tif'
    agland_maps_table = {}
    agland_maps_path = [
        i for i in os.listdir(input_dir) if (prefix in i) and (suffix in i)
    ]
    if len(agland_maps_path) == 0:
        return agland_maps_table
    else:
        for path in agland_maps_path:
            itr = int(path[len(prefix):-len(suffix)])
            agland_maps_table[itr] = load_tif_as_AglandMap(os.path.join(
                input_dir, path),
                                                           force_load=True)

    return agland_maps_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=
        '../_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/',
        help="path dir to save outputs")
    parser.add_argument(
        "--results_dir",
        type=str,
        default='../../../outputs/all_correct_to_FAO_scale_itr3_fr_0/',
        help="path dir to generated results for evaluation")
    parser.add_argument("--water_body_dir",
                        type=str,
                        default='../../../land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir",
                        type=str,
                        default='../../../gdd/gdd_filter_map_360x720.tif',
                        help="path dir to gdd filter map tif")
    args = parser.parse_args()
    print(args)

    # Set default path directory
    census_table_dir = os.path.join(args.results_dir, 'processed_census.pkl')

    agland_maps_table = pack_agland_maps(args.results_dir)
    assert (len(agland_maps_table) >
            0), "results_dir must contain at least 1 agland map tif file"

    # Load input dataset (agland maps generated from same configs but different iteration
    # share the same input dataset)
    input_dataset = Dataset(
        census_table=load_census_table_pkl(census_table_dir),
        land_cover_code=LAND_COVER_CFG['code']['MCD12Q1'],
        remove_land_cover_feature_index=[])

    # Find prediction vs. ground truth before cropping out any region
    output_pred_vs_ground_truth_data_collection = {}
    for itr, agland_map in agland_maps_table.items():
        output_pred_vs_ground_truth_data_dir = os.path.join(
            args.output_dir,
            'pred_vs_ground_truth_data_{}.csv'.format(str(itr)))

        if not os.path.exists(output_pred_vs_ground_truth_data_dir):
            print("Generate new output_pred_vs_ground_truth_data for itr:{}".
                  format(str(itr)))
            ground_truth_collection, pred_collection = agland_map.extract_state_level_data(
                input_dataset)
            np.savetxt(output_pred_vs_ground_truth_data_dir,
                       np.hstack((ground_truth_collection, pred_collection)),
                       delimiter=',')
        else:
            print("Load output_pred_vs_ground_truth_data for itr:{}".format(
                str(itr)))
            load_results = np.loadtxt(output_pred_vs_ground_truth_data_dir,
                                      delimiter=',')
            ground_truth_collection = load_results[:, 0:3]
            pred_collection = load_results[:, 3:]

        output_pred_vs_ground_truth_data_collection[itr] = {
            'ground_truth_collection': ground_truth_collection,
            'pred_collection': pred_collection
        }

    # Update agland maps with water-body masked maps
    # Save map plots
    map_height, map_width = agland_maps_table[next(iter(agland_maps_table))].height, \
                            agland_maps_table[next(iter(agland_maps_table))].width
    mask = make_nonagricultural_mask(args.water_body_dir,
                                     args.gdd_filter_map_dir,
                                     shape=(map_height, map_width))
    for itr, agland_map in agland_maps_table.items():
        output_map_dir = os.path.join(args.output_dir,
                                      'output_{}_'.format(str(itr)))
        agland_maps_table[itr] = agland_map.apply_mask(mask)
        agland_map.plot(output_map_dir)

    # Make agland pred vs. ground truth plots
    plot_agland_pred_vs_ground_truth(
        output_pred_vs_ground_truth_data_collection,
        output_dir=os.path.join(args.output_dir,
                                'pred_vs_ground_truth_fig.png'))


if __name__ == '__main__':
    main()
