import argparse
import rasterio
from utils.tools.visualizer import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pasture_diff_map_dir",
        type=str,
        default=
        './all_correct_to_subnation_scale_itr3_fr_0/australia/agland_map_output_3_australia_diff_map.tif',
        help="path dir to pasture eval map tif")

    args = parser.parse_args()
    print(args)

    output_diff_map_dir = args.pasture_diff_map_dir[:-len('.tif')] + '.png'
    output_histogram_map_dir = args.pasture_diff_map_dir[:-len('map.tif'
                                                               )] + 'hist.png'

    pasture_diff_map = rasterio.open(args.pasture_diff_map_dir).read(1)

    # ============= Delete lines artifact for Brazil =============
    height, width = pasture_diff_map.shape
    i_list, j_list = [], []
    for i in range(height):
        if np.unique(pasture_diff_map[i, :]).size == 1:
            i_list.append(i)
    for j in range(width):
        if np.unique(pasture_diff_map[:, j]).size == 1:
            j_list.append(j)
    pasture_diff_map = np.delete(pasture_diff_map, i_list, axis=0)
    pasture_diff_map = np.delete(pasture_diff_map, j_list, axis=1)
    # =============================================================

    plot_diff_pred_pasture(pasture_diff_map, output_diff_map_dir)
    plot_histogram_diff_pred_pasture(pasture_diff_map,
                                     output_histogram_map_dir)


if __name__ == '__main__':
    main()
