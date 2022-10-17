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

CENSUS_SETTING_CFG = io.load_yaml_config('configs/census_setting_cfg.yaml')
SHAPEFILE_CFG = io.load_yaml_config('configs/shapefile_cfg.yaml')
SUBNATIONAL_STATS_CFG = io.load_yaml_config(
    'configs/subnational_stats_cfg.yaml')

SUBNATIONAL_CENSUS = {
    'Argentina':
    Argentina(SHAPEFILE_CFG['path_dir']['Argentina'],
              SUBNATIONAL_STATS_CFG['path_dir']['Argentina'],
              CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Australia':
    Australia(SHAPEFILE_CFG['path_dir']['Australia'],
              SUBNATIONAL_STATS_CFG['path_dir']['Australia'],
              CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Austria':
    Austria(SHAPEFILE_CFG['path_dir']['Austria'],
            SUBNATIONAL_STATS_CFG['path_dir']['Austria'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Belgium':
    Belgium(SHAPEFILE_CFG['path_dir']['Belgium'],
            SUBNATIONAL_STATS_CFG['path_dir']['Belgium'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Brazil':
    Brazil(SHAPEFILE_CFG['path_dir']['Brazil'],
           SUBNATIONAL_STATS_CFG['path_dir']['Brazil'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Bulgaria':
    Bulgaria(SHAPEFILE_CFG['path_dir']['Bulgaria'],
             SUBNATIONAL_STATS_CFG['path_dir']['Bulgaria'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Canada':
    Canada(SHAPEFILE_CFG['path_dir']['Canada'],
           SUBNATIONAL_STATS_CFG['path_dir']['Canada'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'China':
    China(SHAPEFILE_CFG['path_dir']['China'],
          SUBNATIONAL_STATS_CFG['path_dir']['China'],
          CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Croatia':
    Croatia(SHAPEFILE_CFG['path_dir']['Croatia'],
            SUBNATIONAL_STATS_CFG['path_dir']['Croatia'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Cyprus':
    Cyprus(SHAPEFILE_CFG['path_dir']['Cyprus'],
           SUBNATIONAL_STATS_CFG['path_dir']['Cyprus'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Czechia':
    Czechia(SHAPEFILE_CFG['path_dir']['Czechia'],
            SUBNATIONAL_STATS_CFG['path_dir']['Czechia'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Denmark':
    Denmark(SHAPEFILE_CFG['path_dir']['Denmark'],
            SUBNATIONAL_STATS_CFG['path_dir']['Denmark'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Estonia':
    Estonia(SHAPEFILE_CFG['path_dir']['Estonia'],
            SUBNATIONAL_STATS_CFG['path_dir']['Estonia'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ethiopia':
    Ethiopia(SHAPEFILE_CFG['path_dir']['Ethiopia'],
             SUBNATIONAL_STATS_CFG['path_dir']['Ethiopia'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Finland':
    Finland(SHAPEFILE_CFG['path_dir']['Finland'],
            SUBNATIONAL_STATS_CFG['path_dir']['Finland'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'France':
    France(SHAPEFILE_CFG['path_dir']['France'],
           SUBNATIONAL_STATS_CFG['path_dir']['France'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Germany':
    Germany(SHAPEFILE_CFG['path_dir']['Germany'],
            SUBNATIONAL_STATS_CFG['path_dir']['Germany'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Greece':
    Greece(SHAPEFILE_CFG['path_dir']['Greece'],
           SUBNATIONAL_STATS_CFG['path_dir']['Greece'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Hungary':
    Hungary(SHAPEFILE_CFG['path_dir']['Hungary'],
            SUBNATIONAL_STATS_CFG['path_dir']['Hungary'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'India':
    India(SHAPEFILE_CFG['path_dir']['India'],
          SUBNATIONAL_STATS_CFG['path_dir']['India'],
          CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Indonesia':
    Indonesia(SHAPEFILE_CFG['path_dir']['Indonesia'],
              SUBNATIONAL_STATS_CFG['path_dir']['Indonesia'],
              CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ireland':
    Ireland(SHAPEFILE_CFG['path_dir']['Ireland'],
            SUBNATIONAL_STATS_CFG['path_dir']['Ireland'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Italy':
    Italy(SHAPEFILE_CFG['path_dir']['Italy'],
          SUBNATIONAL_STATS_CFG['path_dir']['Italy'],
          CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Kazakhstan':
    Kazakhstan(SHAPEFILE_CFG['path_dir']['Kazakhstan'],
               SUBNATIONAL_STATS_CFG['path_dir']['Kazakhstan'],
               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Latvia':
    Latvia(SHAPEFILE_CFG['path_dir']['Latvia'],
           SUBNATIONAL_STATS_CFG['path_dir']['Latvia'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Lithuania':
    Lithuania(SHAPEFILE_CFG['path_dir']['Lithuania'],
              SUBNATIONAL_STATS_CFG['path_dir']['Lithuania'],
              CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Luxembourg':
    Luxembourg(SHAPEFILE_CFG['path_dir']['Luxembourg'],
               SUBNATIONAL_STATS_CFG['path_dir']['Luxembourg'],
               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Malta':
    Malta(SHAPEFILE_CFG['path_dir']['Malta'],
          SUBNATIONAL_STATS_CFG['path_dir']['Malta'],
          CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mexico':
    Mexico(SHAPEFILE_CFG['path_dir']['Mexico'],
           SUBNATIONAL_STATS_CFG['path_dir']['Mexico'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mongolia':
    Mongolia(SHAPEFILE_CFG['path_dir']['Mongolia'],
             SUBNATIONAL_STATS_CFG['path_dir']['Mongolia'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mozambique':
    Mozambique(SHAPEFILE_CFG['path_dir']['Mozambique'],
               SUBNATIONAL_STATS_CFG['path_dir']['Mozambique'],
               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Namibia':
    Namibia(SHAPEFILE_CFG['path_dir']['Namibia'],
            SUBNATIONAL_STATS_CFG['path_dir']['Namibia'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Netherlands':
    Netherlands(SHAPEFILE_CFG['path_dir']['Netherlands'],
                SUBNATIONAL_STATS_CFG['path_dir']['Netherlands'],
                CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Pakistan':
    Pakistan(SHAPEFILE_CFG['path_dir']['Pakistan'],
             SUBNATIONAL_STATS_CFG['path_dir']['Pakistan'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Poland':
    Poland(SHAPEFILE_CFG['path_dir']['Poland'],
           SUBNATIONAL_STATS_CFG['path_dir']['Poland'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Portugal':
    Portugal(SHAPEFILE_CFG['path_dir']['Portugal'],
             SUBNATIONAL_STATS_CFG['path_dir']['Portugal'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Romania':
    Romania(SHAPEFILE_CFG['path_dir']['Romania'],
            SUBNATIONAL_STATS_CFG['path_dir']['Romania'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Russia':
    Russia(SHAPEFILE_CFG['path_dir']['Russia'],
           SUBNATIONAL_STATS_CFG['path_dir']['Russia'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'SaudiArabia':
    SaudiArabia(SHAPEFILE_CFG['path_dir']['SaudiArabia'],
                SUBNATIONAL_STATS_CFG['path_dir']['SaudiArabia'],
                CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Slovakia':
    Slovakia(SHAPEFILE_CFG['path_dir']['Slovakia'],
             SUBNATIONAL_STATS_CFG['path_dir']['Slovakia'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Slovenia':
    Slovenia(SHAPEFILE_CFG['path_dir']['Slovenia'],
             SUBNATIONAL_STATS_CFG['path_dir']['Slovenia'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'SouthAfrica':
    SouthAfrica(SHAPEFILE_CFG['path_dir']['SouthAfrica'],
                SUBNATIONAL_STATS_CFG['path_dir']['SouthAfrica'],
                CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Spain':
    Spain(SHAPEFILE_CFG['path_dir']['Spain'],
          SUBNATIONAL_STATS_CFG['path_dir']['Spain'],
          CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Sweden':
    Sweden(SHAPEFILE_CFG['path_dir']['Sweden'],
           SUBNATIONAL_STATS_CFG['path_dir']['Sweden'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Tanzania':
    Tanzania(SHAPEFILE_CFG['path_dir']['Tanzania'],
             SUBNATIONAL_STATS_CFG['path_dir']['Tanzania'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Turkey':
    Turkey(SHAPEFILE_CFG['path_dir']['Turkey'],
           SUBNATIONAL_STATS_CFG['path_dir']['Turkey'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Uganda':
    Uganda(SHAPEFILE_CFG['path_dir']['Uganda'],
           SUBNATIONAL_STATS_CFG['path_dir']['Uganda'],
           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'UK':
    UK(SHAPEFILE_CFG['path_dir']['UK'],
       SUBNATIONAL_STATS_CFG['path_dir']['UK'],
       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ukraine':
    Ukraine(SHAPEFILE_CFG['path_dir']['Ukraine'],
            SUBNATIONAL_STATS_CFG['path_dir']['Ukraine'],
            CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'USA':
    USA(SHAPEFILE_CFG['path_dir']['USA'],
        SUBNATIONAL_STATS_CFG['path_dir']['USA'],
        CENSUS_SETTING_CFG['path_dir']['FAOSTAT'])
}

WORLD_CENSUS = World('shapefile/World/gadm36_0.shp',
                     'FAOSTAT_data/FAOSTAT_data_11-14-2020.csv',
                     'FAOSTAT_data/FAOcountryProfileUTF8_withregions.csv')


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
                        default='gdd/gdd_filter_map_360x720.tif',
                        help="path dir to gdd filter map tif")
    parser.add_argument("--global_area_map_dir",
                        type=str,
                        default='land_cover/global_area_2160x4320.tif',
                        help="path dir to global area map tif")
    args = parser.parse_args()
    print(args)

    # ==================== [SELECT FAO / SUBNATIONAL TOTAL] ====================
    # Load world census table from FAOSTAT
    # global_census_table = WORLD_CENSUS.census_table

    # Load world census table from subnational
    # CENSUS_SETTING_CFG['calibrate'] should be all False, otherwise will be the same as FAOSTAT
    global_census_table = WORLD_CENSUS.replace_subnation(
        SUBNATIONAL_CENSUS, CENSUS_SETTING_CFG['calibrate'])
    # ==========================================================================

    num_states = len(global_census_table)

    # Load agland map and apply masks
    agland_map = load_tif_as_AglandMap(args.agland_map_dir, force_load=True)
    nonagricultural_mask = make_nonagricultural_mask(
        shape=(agland_map.height, agland_map.width),
        mask_dir_list=[args.water_body_dir, args.gdd_filter_map_dir])

    # ======= [SELECT WITH / WITHTOUT GDD+WATERBODY MASK ON AGLAND MAP] =======
    # agland_map.apply_mask(nonagricultural_mask)
    # ==========================================================================

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
                            ['REGIONS']] += np.nan_to_num(
                                np.asarray([
                                    global_census_table.iloc[i]['CROPLAND'],
                                    global_census_table.iloc[i]['PASTURE']
                                ]), 0)

    comparison_table = pack_continent_counts_in_table(pred_continent_count,
                                                      FAO_continent_count)
    comparison_table.to_csv('./FAO_total_comp.csv')


if __name__ == '__main__':
    main()
