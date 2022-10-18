import argparse
import pandas as pd
import numpy as np
from utils.agland_map import *
from utils.tools.visualizer import *
from utils.process.post_process import make_nonagricultural_mask


def parse_geowiki_cropland(file_dir):
    """
    Load Geowiki cropland data as np.array with columns centroid_x, centroid_y, sumcrop
    Geowiki cropland data: https://doi.pangaea.de/10.1594/PANGAEA.873912

    Args:
        file_dir (str): path directory to "loc_all_2.txt"

    Returns: (np.array)
    """
    try:
        cropland_ref = pd.read_csv(file_dir, sep='\t')
    except FileNotFoundError:
        raise "Please download loc_all_2.txt from https://doi.pangaea.de/10.1594/PANGAEA.873912"

    # local_all_2.txt has records of the same location id over a span of time stamps
    # Group to take the average sum cropland values
    cropland_ref = cropland_ref.groupby(['location_id']).mean()

    return cropland_ref[['loc_cent_X', 'loc_cent_Y', 'sumcrop']].to_numpy()


def reproject_geowiki_to_index_coord(geowiki_data, affine):
    """
    Reproject geowiki data from latitude and longitude coordinate system to grid
    indices by nearest neighbors

    Args:
        geowiki_data (np.array): 2D array with columns centroid_x, centroid_y, data
        affine (affine.Affine): transform

    Returns: (np.array)
    """
    assert (geowiki_data.shape[1] == 3), \
        "Input geowiki must have columns columns centroid_x, centroid_y, data"

    # Latitude and Longitude
    #       +90
    #   -180  +180
    #       -90
    num_samples = geowiki_data.shape[0]
    grid_size = affine[0]
    min_x = affine[2]
    max_y = affine[5]
    lat = np.flip(
        np.arange(-max_y + (grid_size / 2),
                  max_y - (grid_size / 2) + grid_size, grid_size))
    lon = np.arange(min_x + (grid_size / 2),
                    -min_x - (grid_size / 2) + grid_size, grid_size)

    # Get nearest neighbor indices
    nearest_index = []
    for i in range(num_samples):
        x = np.argmin(np.abs(lat - geowiki_data[i, 1]))
        y = np.argmin(np.abs(lon - geowiki_data[i, 0]))
        nearest_index.append([x, y])

    geowiki_data_by_index = geowiki_data.copy()
    geowiki_data_by_index[:, 0:2] = np.asarray(nearest_index)

    return geowiki_data_by_index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--geowiki_cropland_dir",
                        type=str,
                        default='./loc_all_2.txt',
                        help="path dir to geowiki loc_all_2.txt")
    parser.add_argument(
        "--agland_map_dir",
        type=str,
        default=
        '../outputs/all_correct_to_subnation_scale_itr3_fr_0/agland_map_output_3.tif',
        help="path dir to agland map dir to be evaluated")
    parser.add_argument("--water_body_dir",
                        type=str,
                        default='../land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir",
                        type=str,
                        default='../gdd/gdd_filter_map_360x720.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--output_dir",
                        type=str,
                        default='./',
                        help="path dir to output evaluation figs")

    args = parser.parse_args()
    print(args)

    # Get input agland map filename
    input_filename = args.agland_map_dir.split('/')[-1][:-len('.tif')]
    output_geowiki_filename = 'geowiki_scatter.png'
    output_nan_filename = 'mask_removed_points.png'
    output_diff_map_filename = input_filename + '_geowiki_diff_map.png'
    output_diff_hist_filename = input_filename + '_geowiki_diff_hist.png'

    # Load data and pred results
    geowiki_cropland = parse_geowiki_cropland(args.geowiki_cropland_dir)
    agland_map = load_tif_as_AglandMap(args.agland_map_dir, force_load=True)
    geowiki_cropland_by_index = reproject_geowiki_to_index_coord(
        geowiki_cropland, agland_map.affine)

    # Apply water body and gdd masks
    mask = make_nonagricultural_mask(
        shape=(agland_map.height, agland_map.width),
        mask_dir_list=[args.water_body_dir, args.gdd_filter_map_dir])
    agland_map.apply_mask(mask)

    # Extract prediction values
    pred = agland_map.get_cropland()[(
        (geowiki_cropland_by_index[:, 0]).astype(int),
        (geowiki_cropland_by_index[:, 1]).astype(int))]

    # Figure 1. Geowiki cropland scatter plot
    plot_geowiki_cropland(geowiki_cropland_by_index,
                          args.output_dir + output_geowiki_filename)

    # Figure 2. nan scatter plots (due to masking)
    plot_geowiki_cropland(geowiki_cropland_by_index[np.isnan(pred), :],
                          args.output_dir + output_nan_filename)

    # Figure 3. Difference between pred cropland and Geowiki cropland
    plot_diff_geowiki_pred_cropland(geowiki_cropland_by_index, pred,
                                    args.output_dir + output_diff_map_filename)

    # Figure 4. Histogram of difference map
    plot_histogram_diff_geowiki_pred_cropland(
        geowiki_cropland_by_index, pred,
        args.output_dir + output_diff_hist_filename)


if __name__ == '__main__':
    main()
