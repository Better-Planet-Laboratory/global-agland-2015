import pandas as pd
import os
from utils.tools.gdd_core import *
from utils.tools.geo import get_border
import rasterio
from rasterio.mask import mask
import warnings


def count_census_states(subnational_census):
    """
    Count number of states available in each Country in subnational_census

    Args:
        subnational_census (dict): country name (str) -> (Country)

    Returns: (dict) country name (str) -> number of states in census (int)
    """
    states_count_table = {}
    for country, census in subnational_census.items():
        states_count_table[country] = len(census.merge_census_to_spatial()['STATE'].to_list())
    return states_count_table


def write_census_states_count_table(states_count_table, file_name):
    """
    Write states_count_table to cvs with attributes,
    ['Country', 'states_count'] where states_count represents the total
    number of states included in the census for that country

    Args:
        subnational_census (dict): country name (str) -> (Country)
        file_name (str): output directory
    """
    country_list = []
    count_list = []
    for country, count in states_count_table.items():
        country_list.append(country)
        count_list.append(count)
    pd.DataFrame({'Country': country_list,
                  'states_count': count_list}).to_csv(file_name)


def get_bias_factors_table(subnational_census):
    """
    Get bias_correction factor tuples (cropland, pasture) for each country in
    subnational_census

    Args:
        subnational_census (dict): country name (str) -> (Country)

    Returns: (dict) country name (str) -> (bias_cropland, bias_pasture) (float, float)
    """
    bias_factors_table = {}
    for country, census in subnational_census.items():
        bias_factors_table[country] = census.get_bias_factor()
    return bias_factors_table


def write_bias_factors_table_to_csv(bias_factors_table, file_name):
    """
    Write bias_factors_table to csv with attributes,
    ['Country', 'bias_cropland', 'bias_pasture']

    Args:
        bias_factors_table (dict):  country name (str) -> (float, float)
        file_name (str): output directory
    """
    country_list = []
    bias_cropland_list = []
    bias_pasture_list = []
    for country, (bias_cropland, bias_pasture) in bias_factors_table.items():
        country_list.append(country)
        bias_cropland_list.append(bias_cropland)
        bias_pasture_list.append(bias_pasture)
    pd.DataFrame({'Country': country_list,
                  'bias_cropland': bias_cropland_list,
                  'bias_pasture': bias_pasture_list}).to_csv(file_name)


def merge_subnation_to_world(world_census, subnational_census, bias_correct):
    """
    world_census contains global record from FAOSTAT, and subnational_census contains
    states level data for some countries. This function merge the two census sources

    Args:
        world_census (World): World object
        subnational_census (dict): country name (str) -> (Country)
        bias_correct (dict): country name (str) -> (bool) for bias correction

    Returns: (pd) processed table
    """
    assert (set(subnational_census.keys()) == set(bias_correct.keys())), \
        'bias_correct must contain all countries in subnational_census'
    return world_census.replace_subnation(subnational_census, bias_correct, inplace=False)


def apply_nan_filter(census):
    """
    Filter samples with either CROPLAND or PASTURE has nan values

    Args:
        census (pd): census table

    Returns: (pd) processed census table (new copy)
    """
    # Get nan indices in CROPLAND and PASTURE
    census_copy = census.copy()
    cropland_nan_index = set(census_copy['CROPLAND'].index[census_copy['CROPLAND'].apply(np.isnan)])
    pasture_nan_index = set(census_copy['PASTURE'].index[census_copy['PASTURE'].apply(np.isnan)])
    nan_index = cropland_nan_index.union(pasture_nan_index)

    nan_state = census_copy['STATE'][nan_index].to_list()
    print('The following STATE has missing values in CROPLAND or PASTURE: \n{}'.format(nan_state))

    # Filter nan samples
    census_copy = census_copy.drop(nan_index)
    census_copy = census_copy.reset_index()

    return census_copy


def apply_GDD_filter(census, gdd_config, accept_ratio, gdd_crop_criteria, *args):
    """
    Filter samples based on GDD and its criteria, samples with
    GDD included region / total region < accept_ratio will be filtered from dataset

    Args:
        census (pd): census table
        gdd_config (dict): GDD settings from yaml
        accept_ratio (float): threshold
        gdd_crop_criteria (func): criteria func
        *args: arguments for gdd_crop_criteria

    Returns: (pd) processed census table (new copy)
    """
    assert(0 < accept_ratio <= 1), "accept_ratio must be in (0, 1]"

    # Check if GDD filter map exists
    if not os.path.exists(gdd_config['path_dir']['GDD_filter_map']):
        print('File {} not found. Generating new GDD filter map'.
              format(gdd_config['path_dir']['GDD_filter_map']))
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
            out_image, _ = mask(src, get_border(i, census), crop=True, nodata=255)
            out_image = out_image[0]
            num_exclude = np.count_nonzero(out_image == 0)
            num_include = np.count_nonzero(out_image == 1)

            try:
                ratio = num_include / (num_exclude + num_include)
            except ZeroDivisionError:
                warnings.warn("{} not found on GDD map. "
                              "This is likely caused by extremely small geometry "
                              "that does not fit on GDD resolution (not filtered)".
                              format(census.iloc[i]['STATE']))
                ratio = 1

            if ratio < accept_ratio:
                index_list.append(i)

    GDD_ex_state = census['STATE'][index_list].to_list()
    print('The following STATE is excluded by GDD criteria: \n{}'.format(GDD_ex_state))

    # Filter GDD excluded samples
    census_copy = census.copy()
    census_copy = census_copy.drop(index_list)
    census_copy = census_copy.reset_index()

    return census_copy
