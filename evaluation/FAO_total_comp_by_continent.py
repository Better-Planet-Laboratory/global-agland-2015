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

WORLD_CENSUS = World('../shapefile/World/gadm36_0.shp',
                     '../FAOSTAT_data/FAOSTAT_data_11-14-2020.csv',
                     '../FAOSTAT_data/FAOcountryProfileUTF8_withregions.csv')


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
        '../outputs/all_correct_to_FAO_scale_itr3_fr_0/agland_map_output_3.tif',
        help="path dir to agland_map for evaluation")
    parser.add_argument("--water_body_dir",
                        type=str,
                        default='../land_cover/water_body_mask.tif',
                        help="path dir to water body mask tif")
    parser.add_argument("--gdd_filter_map_dir",
                        type=str,
                        default='../gdd/gdd_filter_map_360x720.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--global_area_map_dir",
                        type=str,
                        default='../land_cover/global_area_2160x4320.tif',
                        help="path dir to global area map tif")
    args = parser.parse_args()
    print(args)

    # Load world census table from FAOSTAT
    global_census_table = WORLD_CENSUS.merge_all()
    num_states = len(global_census_table)

    # Load agland map and apply masks
    agland_map = load_tif_as_AglandMap(args.agland_map_dir, force_load=True)
    nonagricultural_mask = make_nonagricultural_mask(args.water_body_dir,
                                                     args.gdd_filter_map_dir,
                                                     shape=(agland_map.height,
                                                            agland_map.width))
    # Our mask also include GDD - try not to apply mask
    # agland_map.apply_mask(nonagricultural_mask)
    cropland_map = agland_map.get_cropland().copy()
    pasture_map = agland_map.get_pasture().copy()

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
                            ['REGIONS']] += np.asarray([
                                global_census_table.iloc[i]['CROPLAND'],
                                global_census_table.iloc[i]['PASTURE']
                            ])

    comparison_table = pack_continent_counts_in_table(pred_continent_count,
                                                      FAO_continent_count)
    comparison_table.to_csv('./FAO_total_comp.csv')


if __name__ == '__main__':
    main()
