from ..tools.census_core import *
from gdd.gdd_criteria import gdd_crop_criteria


def pipeline(world_census, subnational_census, census_setting_cfg, gdd_cfg,
             land_cover_cfg):
    """
    Census processing pipeline:
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
    # calibration_factors_table - FAO calibration factors for each census sample
    # census_states_count_table - number of states count for each country in census
    calibration_factors_table_dir, \
    census_states_count_table_dir = census_setting_cfg['path_dir']['outputs']['calibration_factors_table'], \
                                    census_setting_cfg['path_dir']['outputs']['census_states_count_table']
    if calibration_factors_table_dir is not None:
        write_calibration_factors_table_to_csv(
            get_calibration_factors_table(subnational_census),
            calibration_factors_table_dir)
    if census_states_count_table_dir is not None:
        write_census_states_count_table(
            count_census_states(subnational_census),
            census_states_count_table_dir)

    # Merge WORLD with SUBNATIONAL
    merged_census = merge_subnation_to_world(world_census, subnational_census,
                                             census_setting_cfg['calibrate'])
    print(
        'Merge WORLD census with SUBNATIONAL census. Total samples: {}'.format(
            len(merged_census)))

    # Apply 2 filters
    # 1. nan_filter: nan in either CROPLAND or PASTURE
    # 2. GDD_filter: GDD exclude: above 50d north with < 1000
    #                GDD include / (GDD exclude + GDD include) < accept_ratio
    merged_census = apply_nan_filter(merged_census)
    merged_census = apply_GDD_filter(
        merged_census, gdd_cfg,
        census_setting_cfg['GDD_filter']['accept_ratio'], gdd_crop_criteria)
    print('Apply nan_filter and GDD_filter. Total samples: {}'.format(
        len(merged_census)))

    # Add land_cover percentage features to census table
    merged_census = add_land_cover_percentage(
        merged_census, land_cover_cfg['path_dir']['land_cover_map'],
        land_cover_cfg['code']['MCD12Q1'], land_cover_cfg['null_value'])
    print('Add land_cover percentage to census. Total samples: {}'.format(
        len(merged_census)))

    # Add state area in kHa for each sample
    merged_census = add_state_area(
        merged_census, land_cover_cfg['path_dir']['global_area_map'],
        land_cover_cfg['area_unit'])
    print('Add state area to census. Total samples: {}'.format(
        len(merged_census)))

    # Add agland percentage for each sample
    merged_census = add_agland_percentage(merged_census)
    print('Add cropland/pasture/other percentage to census. Total samples: {}'.
          format(len(merged_census)))

    # Save final census table as pkl
    processed_census_table_dir = census_setting_cfg['path_dir']['outputs'][
        'processed_census_table']
    if processed_census_table_dir is not None:
        write_census_table_to_pkl(merged_census, processed_census_table_dir)

    return merged_census
