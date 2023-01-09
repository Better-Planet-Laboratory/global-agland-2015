import argparse
from utils.tools.census_core import *
from utils.tools.visualizer import *
from utils import io
from gdd.gdd_criteria import gdd_crop_criteria

# Only include nan_filter regions as grey (with gdd_filter on top of the raster)
MARKER = {'nan_filter': -1, 'gdd_mask': -2}

ROOT = './'  # run from root
CENSUS_SETTING_CFG = io.load_yaml_config(ROOT +
                                         'configs/census_setting_cfg.yaml')
GDD_CFG = io.load_yaml_config(ROOT + 'configs/gdd_cfg.yaml')
SUBNATIONAL_STATS_CFG = io.load_yaml_config(
    ROOT + 'configs/subnational_stats_cfg.yaml')
SHAPEFILE_CFG = io.load_yaml_config(ROOT + 'configs/shapefile_cfg.yaml')

SUBNATIONAL_CENSUS = load_pkl(ROOT + 'outputs/SUBNATIONAL_CENSUS')
WORLD_CENSUS = load_pkl(ROOT + 'outputs/WORLD_CENSUS')


def mark_nan_filter(census, marker):
    """
    Mark samples with either CROPLAND or PASTURE has nan values

    Args:
        census (pd): census table
        marker (int or float): marker indicator

    Returns: (pd) processed census table (new copy)
    """
    assert (census_has(census,
                       'CROPLAND')), 'census must have attribute CROPLAND'
    assert (census_has(census,
                       'PASTURE')), 'census must have attribute PASTURE'
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

    # Get nan indices in CROPLAND and PASTURE
    census_copy = census.copy()
    cropland_nan_index = set(
        census_copy['CROPLAND'].index[census_copy['CROPLAND'].apply(np.isnan)])
    pasture_nan_index = set(
        census_copy['PASTURE'].index[census_copy['PASTURE'].apply(np.isnan)])
    nan_index = cropland_nan_index.union(pasture_nan_index)

    nan_state = census_copy['STATE'][nan_index].to_list()
    print(
        'The following STATE has missing values in CROPLAND or PASTURE: \n{}'.
        format(nan_state))

    # Instead of filtering, set cropland and pasture values as marker
    census_copy.loc[nan_index, 'CROPLAND'] = marker
    census_copy.loc[nan_index, 'PASTURE'] = marker

    return census_copy


def mark_GDD_filter(census, gdd_config, accept_ratio, marker,
                    gdd_crop_criteria, *args):
    """
    Mark samples based on GDD and its criteria, samples with
    GDD included region / total region < accept_ratio will be filtered from dataset

    Args:
        census (pd): census table
        gdd_config (dict): GDD settings from yaml
        accept_ratio (float): threshold
        marker (int or float): marker indicator
        gdd_crop_criteria (func): criteria func
        *args: arguments for gdd_crop_criteria

    Returns: (pd) processed census table (new copy)
    """
    assert (0 < accept_ratio <= 1), "accept_ratio must be in (0, 1]"
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

    # Check if GDD filter map exists
    if not os.path.exists(gdd_config['path_dir']['GDD_filter_map']):
        print('File {} not found. Generating new GDD filter map'.format(
            gdd_config['path_dir']['GDD_filter_map']))
        generate_GDD_filter(gdd_config['path_dir']['GDD_xyz'],
                            tuple(gdd_config['setting']['GDD_xyz']['shape']),
                            gdd_config['setting']['GDD_xyz']['grid_size'],
                            gdd_config['path_dir']['GDD_filter_map'],
                            gdd_config['setting']['GDD_filter_map']['x_min'],
                            gdd_config['setting']['GDD_filter_map']['y_max'],
                            gdd_config['setting']['GDD_filter_map']['epsg'],
                            gdd_crop_criteria, *args)

    # Iterate over all samples geometry
    index_list = []  # index to be filtered
    with rasterio.open(gdd_config['path_dir']['GDD_filter_map']) as src:

        for i in range(len(census)):
            out_image, _ = mask(src,
                                get_border(i, census),
                                crop=True,
                                nodata=255)
            out_image = out_image[0]
            num_exclude = np.count_nonzero(out_image == 0)
            num_include = np.count_nonzero(out_image == 1)

            try:
                ratio = num_include / (num_exclude + num_include)
            except ZeroDivisionError:
                # Soft alert
                # No changes need to be made in this situation
                print("{} not found on GDD map. "
                      "This is likely caused by extremely small geometry "
                      "that does not fit on GDD resolution (not filtered)".
                      format(census.iloc[i]['STATE']))
                ratio = 1

            if ratio < accept_ratio:
                index_list.append(i)

    GDD_ex_state = census['STATE'][index_list].to_list()
    print('The following STATE is excluded by GDD criteria: \n{}'.format(
        GDD_ex_state))

    # Instead of filtering, set cropland and pasture values as marker
    census_copy = census.copy()
    census_copy.loc[index_list, 'CROPLAND'] = marker
    census_copy.loc[index_list, 'PASTURE'] = marker

    return census_copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default=ROOT + 'docs/source/_static/img/census/',
        help="path dir output cropland and pasture from merged census")

    args = parser.parse_args()
    print(args)

    # Merge FAO with subnational census
    merged_census = merge_subnation_to_world(WORLD_CENSUS, SUBNATIONAL_CENSUS,
                                             CENSUS_SETTING_CFG)
    print('Total Initial Number of samples: {}'.format(len(merged_census)))

    # Mark nan filter
    merged_census = mark_nan_filter(merged_census, MARKER['nan_filter'])
    print('After nan filter: {}'.format(
        len(merged_census) -
        list(merged_census['CROPLAND']).count(MARKER['nan_filter'])))

    # Mark gdd filter
    GDD_CFG['path_dir'][
        'GDD_filter_map'] = ROOT + GDD_CFG['path_dir']['GDD_filter_map']
    # merged_census = mark_GDD_filter(merged_census, GDD_CFG, CENSUS_SETTING_CFG['GDD_filter']['accept_ratio'],
    #                                 MARKER['gdd_filter'], gdd_crop_criteria)
    # print(
    #     'After gdd filter: {}'.format(len(merged_census) - list(merged_census['CROPLAND']).count(MARKER['gdd_filter'])
    #                                   - list(merged_census['CROPLAND']).count(MARKER['nan_filter'])))

    # Generate raster map of the final input to the model (in Kha not percentage)
    WORLD_CENSUS.assign_census_table(merged_census)
    plot_merged_census(WORLD_CENSUS.census_table,
                       MARKER,
                       GDD_CFG,
                       output_dir=args.output_dir)


if __name__ == '__main__':
    main()
