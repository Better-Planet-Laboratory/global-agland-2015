import argparse
import rasterio
from utils.tools.visualizer import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pasture_diff_map_dir",
        type=str,
        default=
        # './all_correct_to_FAO_scale_itr3_fr_0/australia/agland_map_output_3_australia_diff_map.tif',
        # './all_correct_to_FAO_scale_itr3_fr_0/brazil/agland_map_output_3_brazil_diff_map.tif',
        # './all_correct_to_FAO_scale_itr3_fr_0/europe/agland_map_output_3_eu_diff_map.tif',
        './all_correct_to_FAO_scale_itr3_fr_0/hilda/agland_map_output_3_diff_map.tif',
        # './all_correct_to_FAO_scale_itr3_fr_0/hyde/agland_map_output_3_diff_map.tif',
        # './all_correct_to_FAO_scale_itr3_fr_0/usa/agland_map_output_3_usa_diff_map.tif',
        help="path dir to pasture eval map tif")
    parser.add_argument("--global_boundary_shp",
                    type=str,
                    default='../shapefile/ne_10m_land/ne_10m_land.shp',
                    help="path dir to global boundary shp")

    args = parser.parse_args()
    print(args)

    output_diff_map_dir = args.pasture_diff_map_dir[:-len('.tif')] + '.png'
    output_histogram_map_dir = args.pasture_diff_map_dir[:-len('map.tif'
                                                               )] + 'hist.png'

    pasture_diff_map = rasterio.open(args.pasture_diff_map_dir).read(1)
    reference_name = args.pasture_diff_map_dir.split('/')[2]

    global_boundary_shp = args.global_boundary_shp if "hilda" in args.pasture_diff_map_dir or "hyde" in args.pasture_diff_map_dir else None
    plot_diff_pred_pasture(pasture_diff_map, global_boundary_shp=global_boundary_shp, output_dir=output_diff_map_dir)
    plot_histogram_diff_pred_pasture(pasture_diff_map,
                                     reference_name.capitalize(), 
                                     output_histogram_map_dir)


if __name__ == '__main__':
    main()
