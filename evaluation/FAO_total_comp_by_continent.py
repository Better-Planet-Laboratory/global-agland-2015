import argparse
import numpy as np
from utils import io
from utils import constants
import rasterio
from tqdm import tqdm
from census_processor import World
from utils.tools.geo import crop_intermediate_state
from utils.agland_map import *
from utils.process.post_process import make_nonagricultural_mask
from census_processor import Argentina, Australia, Austria, Belgium, Brazil, \
    Bulgaria, Canada, China, Croatia, Cyprus, Czechia, \
    Denmark, Estonia, Ethiopia, Finland, France, Germany, \
    Greece, Hungary, India, Indonesia, Ireland, Italy, Kazakhstan, \
    Latvia, Lithuania, Luxembourg, Malta, Mexico, Mongolia, \
    Mozambique, Namibia, Netherlands, Pakistan, Poland, Portugal, \
    Romania, Russia, SaudiArabia, Slovakia, Slovenia, SouthAfrica, \
    Spain, Sweden, Tanzania, Turkey, Uganda, UK, Ukraine, USA, World


# FULL FAO Continent Sum 
# Continent	FAO_Crop (kHa)	FAO_Past (kHa)
# AFRICA	271775.916	821897.464
# ASIA	570438.492	1073859.12
# EUROPE	259329.318	139709.782
# LATIN AMERICA	170832.408	535083.352
# NORTHERN AMERICA	159720.744	245195.559
# OCEANIA	31950.6	340640


CENSUS_SETTING_CFG = io.load_yaml_config('configs/census_setting_cfg.yaml')
SHAPEFILE_CFG = io.load_yaml_config('configs/shapefile_cfg.yaml')
SUBNATIONAL_STATS_CFG = io.load_yaml_config(
    'configs/subnational_stats_cfg.yaml')


def pack_continent_counts_in_table(pred_continent_count, FAO_continent_count):
    """
    Pack pred_continent_count and into table format

    Args:
        pred_continent_count (dict): continent -> (cropland, pasture)
        FAO_continent_count (dict): continent -> (cropland, pasture)

    Return: (pd.DataFrame)
    """

    def pack_single(continent_count, column_names):
        assert (len(column_names) == 3), "column_names must have 3"
        count_table = {i: [] for i in column_names}
        for continent, count in continent_count.items():
            count_table[column_names[0]].append(continent)
            count_table[column_names[1]].append(count[0])
            count_table[column_names[2]].append(count[1])
        return count_table

    pred_continent_count_table = pd.DataFrame.from_dict(
        pack_single(pred_continent_count,
                    ['Continent', 'Pred_Crop (kHa)', 'Pred_Past (kHa)']))
    FAO_continent_count_table = pd.DataFrame.from_dict(
        pack_single(FAO_continent_count,
                    ['Continent', 'FAO_Crop (kHa)', 'FAO_Past (kHa)']))

    return pred_continent_count_table.join(
        FAO_continent_count_table.set_index('Continent'),
        on='Continent',
        how='left')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agland_map_dir",
        type=str,
        default=
        'outputs/all_correct_to_FAO_scale_itr3_fr_0/agland_map_output_3.tif',
        help="path dir to agland_map for evaluation")
    parser.add_argument("--water_body_dir",
                        type=str,
                        default='land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir",
                        type=str,
                        default='gdd/gdd_filter_map_21600x43200.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--global_area_map_dir",
                        type=str,
                        default='land_cover/global_area_2160x4320.tif',
                        help="path dir to global area map tif")
    parser.add_argument("--add_mask", type=bool, default=False, help="apply mask on agland map")
    args = parser.parse_args()
    print(args)

    # ==================== [SELECT FAO / SUBNATIONAL TOTAL] ====================
    # Load world census table from FAOSTAT
    # global_census_table = WORLD_CENSUS.census_table

    # Load world census table from subnational
    # CENSUS_SETTING_CFG['calibrate'] should be all False, otherwise will be the same as FAOSTAT
    # global_census_table = WORLD_CENSUS.replace_subnation(
    #     SUBNATIONAL_CENSUS, CENSUS_SETTING_CFG['calibrate'])
    global_census_table = io.load_pkl('./outputs/all_correct_to_FAO_scale_itr3_fr_0/processed_census')
    # ==========================================================================

    num_states = len(global_census_table)

    # Load agland map and apply masks
    agland_map = load_tif_as_AglandMap(args.agland_map_dir, force_load=True)
    nonagricultural_mask = make_nonagricultural_mask(
        shape=(agland_map.height, agland_map.width),
        mask_dir_list=[args.water_body_dir, args.gdd_filter_map_dir])

    # ======= [SELECT WITH / WITHTOUT GDD+WATERBODY MASK ON AGLAND MAP] =======
    # if args.add_mask:
    #     agland_map.apply_mask(nonagricultural_mask)
    # ==========================================================================

    cropland_map = agland_map.get_cropland().copy()
    pasture_map = agland_map.get_pasture().copy()

    # ======================= TO BE DELETED =======================
    # cropland_map[cropland_map <= 0.03] = 0
    # pasture_map[pasture_map <= 0.03] = 0
    # other_copy = np.ones_like(cropland_map) - cropland_map - pasture_map
    # agland_map.data = np.zeros((agland_map.height, agland_map.width, 3))
    # agland_map.data[:, :, 0] = cropland_map
    # agland_map.data[:, :, 1] = pasture_map
    # agland_map.data[:, :, 2] = other_copy
    # =============================================================

    # Load global area map
    # Note: Area map is aggregatable, therefore when agland map size is
    #       changed, new global area map needs to be generated (or
    #       upsample/downsample via averaging/aggregating
    global_area_map = rasterio.open(args.global_area_map_dir).read()[0]
    assert (global_area_map.shape == (
        agland_map.height,
        agland_map.width)), "Input global area map must match agland map"

    # Initialize continent counts (in kHa)
    continent_list = np.unique(np.asarray(list(
        global_census_table['REGIONS'])))
    FAO_continent_count = {i: np.zeros(2) for i in continent_list}
    pred_continent_count = {i: np.zeros(2) for i in continent_list}

    # continent: (cropland, pasture)
    for i, _ in tqdm(enumerate(range(num_states)), total=num_states):

        out_cropland = np.nan_to_num(
            crop_intermediate_state(cropland_map, agland_map.affine,
                                    global_census_table, i))
        out_pasture = np.nan_to_num(
            crop_intermediate_state(pasture_map, agland_map.affine,
                                    global_census_table, i))
        area_map = crop_intermediate_state(global_area_map, agland_map.affine,
                                        global_census_table, i)

        out_cropland[out_cropland < 0] = 0
        out_pasture[out_pasture < 0] = 0
        area_map[area_map < 0] = 0

        out_cropland_area = out_cropland * area_map
        out_pasture_area = out_pasture * area_map

        pred_continent_count[
            global_census_table.iloc[i]['REGIONS']] += np.asarray([
                np.sum(out_cropland_area) / constants.KHA_TO_KM2,
                np.sum(out_pasture_area) / constants.KHA_TO_KM2
            ])

        FAO_continent_count[global_census_table.iloc[i]
                            ['REGIONS']] += np.nan_to_num(
                                np.asarray([
                                    global_census_table.iloc[i]['CROPLAND'],
                                    global_census_table.iloc[i]['PASTURE']
                                ]), 0)

    comparison_table = pack_continent_counts_in_table(pred_continent_count,
                                                      FAO_continent_count)
    comparison_table.to_csv('./evaluation/FAO_total_comp.csv')


if __name__ == '__main__':
    main()
