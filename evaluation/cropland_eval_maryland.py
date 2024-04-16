from rasterio import warp
import argparse
from utils.process.post_process import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maryland_cropland_dir",
        type=str,
        default='./Global_cropland_3km_2015.tif',
        help="path dir to marland Global_cropland_3km_2015.tif")
    parser.add_argument(
        "--agland_map_dir",
        type=str,
        default=
        '../outputs/all_correct_to_FAO_scale_itr3_fr_0/agland_map_output_3.tif',
        help="path dir to agland map dir to be evaluated")
    parser.add_argument("--water_body_dir",
                        type=str,
                        default='../land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir",
                        type=str,
                        default='../gdd/gdd_filter_map_21600x43200.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--output_dir",
                        type=str,
                        default='./',
                        help="path dir to output evaluation figs")

    args = parser.parse_args()
    print(args)

    # Get input agland map filename
    input_filename = args.agland_map_dir.split('/')[-1][:-len('.tif')]
    output_maryland_filename = 'maryland.png'
    output_diff_map_filename = input_filename + '_maryland_diff_map.png'
    output_diff_hist_filename = input_filename + '_maryland_diff_hist.png'

    # Load data and pred results
    maryland_cropland = rasterio.open(args.maryland_cropland_dir)
    agland_map = load_tif_as_AglandMap(args.agland_map_dir, force_load=True)

    # Reproject maryland to match transform of agland map
    maryland_cropland_reproj = np.empty((agland_map.height, agland_map.width),
                                        dtype=np.uint8)
    warp.reproject(maryland_cropland.read(1),
                   maryland_cropland_reproj,
                   src_transform=maryland_cropland.transform,
                   src_crs=maryland_cropland.crs,
                   dst_transform=agland_map.affine,
                   dst_crs='EPSG:4326')

    # Apply water body and gdd masks
    mask = make_nonagricultural_mask(
        shape=(agland_map.height, agland_map.width),
        mask_dir_list=[args.water_body_dir, args.gdd_filter_map_dir])
    agland_map.apply_mask(mask)

    # Figure 1. Maryland cropland plot
    # plot_agland_map_slice(maryland_cropland_reproj / 100, 'cropland',
    #                       args.output_dir + output_maryland_filename)

    # Figure 2. Difference between Maryland cropland and pred cropland
    plot_diff_maryland_pred_cropland(
        maryland_cropland_reproj, agland_map.get_cropland(),
        args.output_dir + output_diff_map_filename)

    # Figure 3. Histogram of difference map
    plot_histogram_diff_maryland_pred_cropland(
        maryland_cropland_reproj, agland_map.get_cropland(),
        args.output_dir + output_diff_hist_filename)


if __name__ == '__main__':
    main()
