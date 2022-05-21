import numpy as np
import pandas as pd
import os
from utils.tools.gdd_core import *
from utils.tools.geo import get_border
from utils.io import *
import rasterio
from rasterio.mask import mask
import warnings
from utils.constants import *
from gdd.gdd_criteria import gdd_crop_criteria


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
    print('File {} generated'.format(file_name))


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
    print('File {} generated'.format(file_name))


def write_census_table_to_pkl(census, file_name):
    """
    Write census table to pkl file

    Args:
        census (pd): census table
        file_name (str): output directory
    """
    assert (file_name.endswith('.pkl')), "file_name has to be .pkl"

    save_pkl(census, file_name[:-len('.pkl')])
    print('File {} generated'.format(file_name))


def load_census_table_pkl(census_path):
    """
    Load census table pkl file as pd.DataFrame

    Args:
        census_path (str): census table path dir
    """
    assert (census_path.endswith('.pkl')), "file_name has to be .pkl"
    return load_pkl(census_path[:-len('.pkl')])


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


def census_has(census, attribute_name):
    """
    Return True if attribute_name is in census attributes, otherwise False

    Args:
        census (pd): census table
        attribute_name (num or str) attribute name to check

    Returns: (bool)
    """
    return attribute_name in census.columns.values


def apply_nan_filter(census):
    """
    Filter samples with either CROPLAND or PASTURE has nan values

    Args:
        census (pd): census table

    Returns: (pd) processed census table (new copy)
    """
    assert (census_has(census, 'CROPLAND')), 'census must have attribute CROPLAND'
    assert (census_has(census, 'PASTURE')), 'census must have attribute PASTURE'
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

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
    assert (0 < accept_ratio <= 1), "accept_ratio must be in (0, 1]"
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

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
    print('The following STATE is excluded by GDD criteria: \n{}'.format(GDD_ex_state))

    # Filter GDD excluded samples
    census_copy = census.copy()
    census_copy = census_copy.drop(index_list)
    census_copy = census_copy.reset_index()

    return census_copy


def add_land_cover_percentage(census, land_cover_dir, land_cover_code, land_cover_null):
    """
    Based on input land cover GeoTIFF file (normally a MCD12** product), add percentage
    of each key types in land_cover_code to the input census. Return a new processed
    census without modifying the original input

    Args:
        census (pd): census table
        land_cover_dir (str): path dir to GeoTIFF file (normally a MCD12** product)
        land_cover_code (dict): class types(int) -> (str)
        land_cover_null (int): null value representation in land cover GeoTIFF

    Returns: (pd) processed census table (new copy)
    """
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

    # Load land_cover_map
    land_cover_map = rasterio.open(land_cover_dir)

    # Placeholder for percentage data
    nodata = 255  # change this if conflict with land_cover_code keys
    remove_index_list = []
    land_cover_percentage = {i: [] for i in land_cover_code.keys()}
    for i in range(len(census)):

        nan_flag = False
        out_image, _ = mask(land_cover_map, get_border(i, census), crop=True, nodata=nodata)
        out_image = out_image[0]

        # Only include pixels inside geometry and not null values in land cover map
        total_pixels = np.count_nonzero((out_image != nodata) & (out_image != land_cover_null))

        if total_pixels == 0:
            print("{} not found on GDD map. "
                  "This is likely caused by extremely small geometry "
                  "that does not fit on land_cover_map resolution (removed)".
                  format(census.iloc[i]['STATE']))
            remove_index_list.append(i)
            nan_flag = True

        for key in land_cover_percentage.keys():
            if nan_flag:
                land_cover_percentage[key].append(np.nan)
            else:
                land_cover_percentage[key].append(
                    np.count_nonzero(out_image == key) / total_pixels)

    # Add land_cover_percentage and remove invalid entries
    census_copy = census.copy()
    for i in land_cover_percentage.keys():
        census_copy[i] = land_cover_percentage[i]
    census_copy = census_copy.drop(remove_index_list)
    census_copy = census_copy.reset_index(drop=True)

    return census_copy


def add_state_area(census, global_area_dir, input_unit='M2'):
    """
    Use input global_area_map GeoTIFF to assign total area of each state in census
    in Kha (to match the cropland and pasture census data units from cascaded sources). A
    conversion of unit from input_unit to Kha is operated in the function. Return
    a new processed census without modifying the original input

    Note:
        Options for input_unit include:
            'Arc', 'Ha', 'Kha', 'Donum', 'Km2', 'M2', 'Mha'

    Args:
        census (pd): census table
        global_area_dir (str): path dir to GeoTIFF file
        input_unit (str): unit used in global_area_map (Default: 'M2')

    Returns: (pd) processed census table (new copy)
    """
    assert (not census_has(census, 'AREA')), 'census already have attribute AREA'
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'

    # Load global_area_map
    global_area_map = rasterio.open(global_area_dir)
    assert (global_area_map.dtypes[0] in [rasterio.dtypes.float32, rasterio.dtypes.float64]), \
        "Input global area map must be either float32 or float64"
    assert (input_unit in UNIT_LOOKUP.keys()), "Unrecognized input unit type: {}".format(input_unit)

    # Placeholder for total area data
    nodata = -1
    remove_index_list = []
    state_area = []
    for i in range(len(census)):

        out_image, _ = mask(global_area_map, get_border(i, census), crop=True, nodata=nodata)
        out_image = out_image[0]

        area = np.sum(out_image[out_image != nodata])

        if area == 0:
            print("{} not found on global_area_map. "
                  "This is likely caused by extremely small geometry "
                  "that does not fit on global_area_map resolution (removed)".
                  format(census.iloc[i]['STATE']))
            remove_index_list.append(i)

        state_area.append(area)

    # Add state_area and remove invalid entries
    census_copy = census.copy()
    state_area = [i for i in state_area if i != 0]  # remove 0 area in area list
    census_copy = census_copy.drop(remove_index_list)  # remove 0 area in census
    census_copy = census_copy.reset_index(drop=True)

    # Unit Conversion to KHa
    census_copy['AREA'] = [i / UNIT_LOOKUP[input_unit] for i in state_area]

    return census_copy


def add_agland_percentage(census):
    """
    Add CROPLAND_PER,  PASTURE_PER and OTHER_PER to the input census. Input census shall contain
    attributes CROPLAND, PASTURE and AREA (in Kha).  Return a new processed census
    without modifying the original input

    Note:
        CROPLAND_PER = CROPLAND / AREA
        PASTURE_PER = PASTURE / AREA
        OTHER_PER = 1 - (CROPLAND_PER + PASTURE_PER)

    Args:
        census (pd): census table

    Returns: (pd) processed census table (new copy)
    """
    assert (census_has(census, 'CROPLAND')), 'census must have attribute CROPLAND'
    assert (census_has(census, 'PASTURE')), 'census must have attribute PASTURE'
    assert (census_has(census, 'STATE')), 'census must have attribute STATE'
    assert (census_has(census, 'AREA')), 'census must have attribute AREA'

    # Get current agland data
    cropland_kha = census['CROPLAND'].to_list()
    pasture_kha = census['PASTURE'].to_list()
    area_kha = census['AREA'].to_list()

    cropland_per = [i / area_kha[idx] if area_kha[idx] != 0 else np.nan for idx, i in enumerate(cropland_kha)]
    pasture_per = [i / area_kha[idx] if area_kha[idx] != 0 else np.nan for idx, i in enumerate(pasture_kha)]
    other_per = [1 - (cropland_per[idx] + pasture_per[idx]) for idx, i in enumerate(pasture_kha)]

    # Add CROPLAND_PER and PASTURE_PER
    census_copy = census.copy()
    census_copy['CROPLAND_PER'] = cropland_per
    census_copy['PASTURE_PER'] = pasture_per
    census_copy['OTHER_PER'] = other_per

    return census_copy


def pipeline(world_census, subnational_census, census_setting_cfg, gdd_cfg, land_cover_cfg):
    """
    Census processing pipelien:
    1. Merge WORLD with SUBNATIONAL
    2. Apply 2 filters - nan_filter, GDD_filter
    3. Add land_cover percentage features
    4. Add state area in kHa
    5. Add agland percentage

    Args:
        world_census (World): World object
        subnational_census (dict): country name (str) -> (Country)
        census_setting_cfg (dict): census settings from yaml
        gdd_cfg (dict): GDD settings from yaml
        land_cover_cfg (dict): land cover settings from yaml

    Returns: (pd) processed census table
    """
    # Save intermediate outputs
    # bias_factors_table - bias correction factors for each census sample
    # census_states_count_table - number of states count for each country in census
    bias_factors_table_dir, \
    census_states_count_table_dir = census_setting_cfg['path_dir']['outputs']['bias_factors_table'], \
                                    census_setting_cfg['path_dir']['outputs']['census_states_count_table']
    if bias_factors_table_dir is not None:
        write_bias_factors_table_to_csv(get_bias_factors_table(subnational_census),
                                        bias_factors_table_dir)
    if census_states_count_table_dir is not None:
        write_census_states_count_table(count_census_states(subnational_census),
                                        census_states_count_table_dir)

    # Merge WORLD with SUBNATIONAL
    merged_census = merge_subnation_to_world(world_census, subnational_census, census_setting_cfg['bias_correct'])
    print('Merge WORLD census with SUBNATIONAL census. Total samples: {}'.format(len(merged_census)))

    # Apply 2 filters
    # 1. nan_filter: nan in either CROPLAND or PASTURE
    # 2. GDD_filter: GDD exclude: above 50d north with < 1000
    #                GDD include / (GDD exclude + GDD include) < accept_ratio
    merged_census = apply_nan_filter(merged_census)
    merged_census = apply_GDD_filter(merged_census, gdd_cfg, census_setting_cfg['GDD_filter']['accept_ratio'],
                                     gdd_crop_criteria)
    print('Apply nan_filter and GDD_filter. Total samples: {}'.format(len(merged_census)))

    # Add land_cover percentage features to census table
    merged_census = add_land_cover_percentage(merged_census, land_cover_cfg['path_dir']['land_cover_map'],
                                              land_cover_cfg['code']['MCD12Q1'],
                                              land_cover_cfg['null_value'])
    print('Add land_cover percentage to census. Total samples: {}'.format(len(merged_census)))

    # Add state area in kHa for each sample
    merged_census = add_state_area(merged_census, land_cover_cfg['path_dir']['global_area_map'],
                                   land_cover_cfg['area_unit'])
    print('Add state area to census. Total samples: {}'.format(len(merged_census)))

    # Add agland percentage for each sample
    merged_census = add_agland_percentage(merged_census)
    print('Add cropland/pasture/other percentage to census. Total samples: {}'.format(len(merged_census)))

    # Save final census table as pkl
    processed_census_table_dir = census_setting_cfg['path_dir']['outputs']['processed_census_table']
    if processed_census_table_dir is not None:
        write_census_table_to_pkl(merged_census, processed_census_table_dir)

    return merged_census
