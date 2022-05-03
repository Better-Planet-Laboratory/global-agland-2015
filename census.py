import os
from helpers.world import *
from helpers.country import *
from helpers.usa import USA
from helpers.australia import Australia
from helpers.china import China
from helpers.canada import Canada
from helpers.mexico import Mexico
from helpers.brazil import Brazil
from helpers.argentina import Argentina
from helpers.india import India
from helpers.south_africa import SouthAfrica
from helpers.mozambique import Mozambique
from helpers.tanzania import Tanzania
from helpers.kazakhstan import Kazakhstan
from helpers.russia import Russia
from helpers.namibia import Namibia
from helpers.saudi_arabia import SaudiArabia
from helpers.ethiopia import Ethiopia
from helpers.utils import *
from land_cover.config import LAND_COVER_CODE
import argparse

SHAPEFILE_DIR = {'Global': 'shapefile/gadm36_0.shp',
                 'USA': 'shapefile/gadm36_USA_1.shp',
                 'Australia': 'shapefile/gadm36_AUS_1.shp',
                 'Canada': 'shapefile/gadm36_CAN_1.shp',
                 'Brazil': 'shapefile/gadm36_BRA_1.shp',
                 'Argentina': 'shapefile/gadm36_ARG_1.shp',
                 'Mexico': 'shapefile/gadm36_MEX_1.shp',
                 'China': 'shapefile/gadm36_CHN_1.shp',
                 'India': 'shapefile/gadm36_IND_1.shp',
                 'South Africa': 'shapefile/gadm36_ZAF_1.shp',
                 'Mozambique': 'shapefile/gadm36_MOZ_1.shp',
                 'Namibia': 'shapefile/gadm36_NAM_1.shp',
                 'Tanzania': 'shapefile/columbia_fewsn_1996_tanzaniaadmn2.shp',
                 'Kazakhstan': 'shapefile/gadm36_KAZ_1.shp',
                 'Russia': 'shapefile/gadm36_RUS_1.shp',
                 'Saudi Arabia': 'shapefile/gadm36_SAU_1.shp',
                 'Ethiopia': 'shapefile/gadm36_ETH_1.shp'
                 }

SUBNATIONAL_DIR = {'USA': 'subnational_stats/USA',
                   'Australia': 'subnational_stats/46270do002_201617.csv',
                   'Canada': 'subnational_stats/32100406.csv',
                   'Brazil': 'subnational_stats/Tabela 6881.xlsx',
                   'Argentina': 'subnational_stats/CNA2018_resultados_preliminares.xls',
                   'Mexico': 'subnational_stats/Tabulado_VIII_CAGyF_2.xls',
                   'China': 'subnational_stats/china',
                   'India': 'subnational_stats/india',
                   'South Africa': 'subnational_stats/south_africa/tabula-Report-11-02-012017.csv',
                   'Mozambique': 'subnational_stats/Mozambique/' +
                                 'tabula-Censo Agro 2013 Pecuario 2009 2013 2010 Resultados Definitivos -2_modified.csv',
                   'Namibia': 'subnational_stats/Namibia/' +
                              'Namibia.csv',
                   'Saudi Arabia': 'subnational_stats/Saudi_Arabia/extracted.xlsx',
                   'Tanzania': 'subnational_stats/Tanzania/' +
                               'tabula-Final Crops National Report 11 JUNE 2012_modified2.csv',
                   'Kazakhstan': 'subnational_stats/Kazakhstan.xlsx',
                   'Russia': 'subnational_stats/Russia/',
                   'Ethiopia': 'subnational_stats/Ethiopia.xlsx'
                   }

FAOSTAT_DIR = 'FAOSTAT_data/FAOSTAT_data_11-14-2020.csv'

FAOSTAT_PROFILE_DIR = 'FAOSTAT_data/FAOcountryProfileUTF8_withregions.csv'


def fuse_census(shapefile_dir, subnational_dir, FAOSTAT_dir,
                FAOSTAT_profile_dir, bias_correct=False,
                threshold=0.5):
    """
    Fuse census data from subnational level and FAOSTAT reports. If bias_correct set True, scale
    subnational level to match FAOSTAT for all country. If the % of unknown (nan) subnational level
    entries in a state level FAOSTAT is greater or equal to input threshold, then keep FAOSTAT records,
    otherwise replacing by subnational level

    Note: The output fused census shapefile has attributes (Cropland and Pasture are in kHa):
    State | geometry | Cropland | Pasture | Region | GID_0

    :param shapefile_dir: dict of directory of shapefile for each country
    :param subnational_dir: dict of directory of subnational records for each country
    :param FAOSTAT_dir: directory of FAOSTAT report
    :param FAOSTAT_profile_dir: directory of FAOSTAT profile
    :param bias_correct: bool
    :param threshold: float in [0, 1]
    :return: fused census shapefile
    """

    # Process data for each country
    world_data = World(shapefile_dir['Global'], FAOSTAT_dir, FAOSTAT_profile_dir)

    usa_data = USA(shapefile_dir['USA'], subnational_dir['USA'], FAOSTAT_dir)
    usa_spatial_map = usa_data.calibrate('Arc', bias_correct)  # pasture data is by default kArc

    australia_data = Australia(shapefile_dir['Australia'], subnational_dir['Australia'], FAOSTAT_dir)
    australia_spatial_map = australia_data.calibrate('Ha', bias_correct)

    canada_data = Canada(shapefile_dir['Canada'], subnational_dir['Canada'], FAOSTAT_dir, 2016)
    canada_spatial_map = canada_data.calibrate('Arc', bias_correct)

    mexico_data = Mexico(shapefile_dir['Mexico'], subnational_dir['Mexico'], FAOSTAT_dir)
    mexico_spatial_map = mexico_data.calibrate('Ha', bias_correct)

    brazil_data = Brazil(shapefile_dir['Brazil'], subnational_dir['Brazil'], FAOSTAT_dir)
    brazil_spatial_map = brazil_data.calibrate('Ha', bias_correct)

    argentina_data = Argentina(shapefile_dir['Argentina'], subnational_dir['Argentina'], FAOSTAT_dir)
    argentina_spatial_map = argentina_data.calibrate('Ha', bias_correct)

    china_data = China(shapefile_dir['China'], subnational_dir['China'], FAOSTAT_dir)
    china_spatial_map = china_data.calibrate('Kha', bias_correct)

    india_data = India(shapefile_dir['India'], subnational_dir['India'], FAOSTAT_dir)
    india_spatial_map = india_data.calibrate('Kha', bias_correct)

    south_africa_data = SouthAfrica(shapefile_dir['South Africa'], subnational_dir['South Africa'], FAOSTAT_dir)
    south_africa_spatial_map = south_africa_data.calibrate('Ha', bias_correct)

    mozambique_data = Mozambique(shapefile_dir['Mozambique'], subnational_dir['Mozambique'], FAOSTAT_dir)
    mozambique_spatial_map = mozambique_data.calibrate('Ha', bias_correct)

    tanzania_data = Tanzania(shapefile_dir['Tanzania'], subnational_dir['Tanzania'], FAOSTAT_dir)
    tanzania_spatial_map = tanzania_data.calibrate('Ha', bias_correct)

    kazakhstan_data = Kazakhstan(shapefile_dir['Kazakhstan'], subnational_dir['Kazakhstan'], FAOSTAT_dir)
    kazakhstan_spatial_map = kazakhstan_data.calibrate('Kha', bias_correct)

    russian_data = Russia(shapefile_dir['Russia'], subnational_dir['Russia'], FAOSTAT_dir)
    russian_spatial_map = russian_data.calibrate('Kha', bias_correct)

    saudi_data = SaudiArabia(shapefile_dir['Saudi Arabia'], subnational_dir['Saudi Arabia'], FAOSTAT_dir)
    saudi_spatial_map = saudi_data.calibrate('Donum', False)  # force bias correct False

    namibia_data = Namibia(shapefile_dir['Namibia'], subnational_dir['Namibia'], FAOSTAT_dir)
    namibia_spatial_map = namibia_data.calibrate('Ha', bias_correct)

    ethiopia_data = Ethiopia(shapefile_dir['Ethiopia'], subnational_dir['Ethiopia'], FAOSTAT_dir)
    ethiopia_spatial_map = ethiopia_data.calibrate('Ha', bias_correct)  # force bias correct False

    # Combine spatial maps
    subnational_census_shp = pd.concat([usa_spatial_map, australia_spatial_map,
                                        canada_spatial_map, mexico_spatial_map, brazil_spatial_map,
                                        argentina_spatial_map, china_spatial_map, india_spatial_map,
                                        south_africa_spatial_map, mozambique_spatial_map, tanzania_spatial_map,
                                        kazakhstan_spatial_map, russian_spatial_map, saudi_spatial_map,
                                        namibia_spatial_map, ethiopia_spatial_map])

    fused_census_shp = world_data.replace_country_by_subnational(subnational_census_shp, threshold)

    return fused_census_shp


def add_land_cover_info_to_census(census_shapefile, global_area_dir, land_cover_dir, no_data_value=255,
                                  items=['area', 'pixels', 'land_cover', 'centroid']):
    """
    Add info for each state into the census shapefile by combining global_area_map and
    land_cover_map. Available items to be added include: ['area', 'pixels', 'land_cover', 'centroid']

    'area': geographical area in km^2
    'pixels': number of pixel counts in polygon given square grid
    'land_cover': land cover counts for all (17) classes in land_cover_map
    'centroid': x and y centroid coordinate at state level

    :param census_shapefile: fused census shapefile (with FAOSTATS and subnational level info)
    :param global_area_dir: directory of global area array on the grid of land cover
    :param land_cover_dir: directory of land cover array (MCD12Q1 product)
    :param no_data_value: no data value indicator (Default: 255)
    :param items: list of items to be added to the input census shapefile
                  (Default: ['area', 'pixels', 'land_cover', 'centroid'])

    :return: modified shapefile with attributes (in order):
             State | geometry | Cropland | Pasture |
             GID_0 | (Land Cover) 1 | 2 | 3 ... 17 | Total Area (in km^2) | Total Pixels |
             Centroid_x | Centroid_y
    """

    def add_centroid_helper(census_shapefile):
        """
        Add Centroid x and y for each state at the end of the census_shapefile

        :return: modified shapefile with attributes: State | geometry | Cropland | Pasture |
                 GID_0 | ... | Centroid_x | Centroid_y
        """
        # Include centroid column for each state
        centroid_list = []

        # census_shapefile shall already be in epsg:4326, but rasterio will throw
        # warning if removing .to_crs(epsg=4326)
        for point in census_shapefile.geometry.to_crs(epsg=4326).centroid:
            centroid_list.append([point.x, point.y])

        centroid_list = np.asarray(centroid_list)
        census_shapefile['Centroid_x'] = centroid_list[:, 0]
        census_shapefile['Centroid_y'] = centroid_list[:, 1]

        return census_shapefile

    # Check input items
    assert (set(items).issubset({'area', 'pixels', 'land_cover', 'centroid'})), \
        "items must be selected from ['area', 'pixels', 'land_cover', 'centroid']"

    # Iterate over each state in the shapefile
    total_area_column = []
    sum_pixels_column = []
    global_area_map_dataset = rasterio.open(global_area_dir)
    land_cover_map_dataset = rasterio.open(land_cover_dir)
    total_num_state = census_shapefile.State.count()

    # Check if shape of global_area_map matches land_cover_map
    assert (global_area_map_dataset.read().shape == land_cover_map_dataset.read().shape), \
        "global_area_map and land_cover_map must have the same shape"

    for current_state_index in range(total_num_state):
        print('Currently processing {} | {} / {}'.format(
            census_shapefile.State[current_state_index], current_state_index, total_num_state - 1))

        if 'pixels' in items or 'land_cover' in items:
            out_image, _ = mask(land_cover_map_dataset, get_border(current_state_index, census_shapefile),
                                crop=True, nodata=no_data_value)

            if 'pixels' in items:
                sum_pixels = np.sum(out_image[0, :, :] != no_data_value)
                sum_pixels_column.append(sum_pixels)  # append sum pixels

            if 'land_cover' in items:
                for land_type_index in LAND_COVER_CODE.keys():
                    census_shapefile.loc[current_state_index, str(land_type_index)] = \
                        (out_image[0, :, :] == land_type_index).sum()

        if 'area' in items:
            out_image, _ = mask(global_area_map_dataset, get_border(current_state_index, census_shapefile),
                                crop=True, nodata=no_data_value)
            state_area = np.sum(out_image[0, out_image[0, :, :] != no_data_value])
            total_area_column.append(state_area)

    # Add Total Area
    if len(total_area_column) != 0:
        census_shapefile['Total Area (km2)'] = total_area_column

    # Add Total Pixels
    if len(sum_pixels_column) != 0:
        census_shapefile['Total Pixels'] = sum_pixels_column

    # Add Centroid
    if 'centroid' in items:
        census_shapefile = add_centroid_helper(census_shapefile)

    return census_shapefile


def generate_complete_census():
    """
    Generate complete census shapefile that combines FAOSTAT, subnational level census data,
    land cover % breakdown, total area, pixels and centroid info for each state

    Note: The output complete census shapefile has attributes (Cropland and Pasture are in kHa):
    State | geometry | Cropland | Pasture | GID_0 | (Land Cover) 1 | 2 | 3 ... 17 |
    Total Area (in km^2) | Total Pixels | Centroid_x | Centroid_y
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--census_data_dir", type=str, default=None,
                        help="directory of fused census shapefile")
    parser.add_argument("--land_cover_dir", type=str, default='land_cover/MCD12Q1_merged.tif',
                        help="directory of merged global MCD12Q1 product")
    parser.add_argument("--global_area_dir", type=str, default='land_cover/global_area.tif',
                        help="directory of global area on square grid")
    parser.add_argument("--output_dir", type=str, default='outputs/',
                        help="directory of output")
    parser.add_argument("--bias_correct", type=bool, default=True,
                        help="bias correction for subnational data to FAOSTAT")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="threshold value for fusion")
    parser.add_argument("--invalid_index", type=int, default=-9999,
                        help="invalid index used in land cover")
    args = parser.parse_args()
    print(args)

    if args.census_data_dir is None:
        print('Start fusing subnational level census and FAOSTAT')
        fused_census_shp = fuse_census(SHAPEFILE_DIR, SUBNATIONAL_DIR,
                                       FAOSTAT_DIR,
                                       FAOSTAT_PROFILE_DIR,
                                       args.bias_correct,
                                       args.threshold)
        save_pkl(fused_census_shp, args.output_dir + 'fused_census_shp')
        print('File fused_census_shp.pkl is saved in {}'.format(args.output_dir))
    else:
        print('Load fused_census_shp pkl file')
        fused_census_shp = load_pkl(args.census_data_dir)

    print('Adding land cover info to census data')
    complete_census = add_land_cover_info_to_census(fused_census_shp,
                                                    args.global_area_dir,
                                                    args.land_cover_dir,
                                                    no_data_value=args.invalid_index,
                                                    items=['area', 'pixels', 'land_cover', 'centroid'])

    save_pkl(complete_census, args.output_dir + 'complete_census_shp')


if __name__ == '__main__':
    """
    ##############################################
    #            FAO Terms Definition            #
    ##############################################
    Full List: http://www.fao.org/ag/agn/nutrition/Indicatorsfiles/Agriculture.pdf

        ## Arable Land ## 
        Land under temporary crops (double-cropped areas are counted only once), temporary meadows for 
        mowing or pasture, land under market and kitchen gardens and land temporarily fallow (less than five years). 
        The abandoned land resulting from shifting cultivation is not included in this category. Data for arable 
        land are not meant to indicate the amount of land that is potentially cultivable

        ## Permanent Crops ## 
        Land cultivated with crops that occupy the land for long periods and need not be replanted 
        after each harvest, such as cocoa, coffee and rubber; this category includes land under flowering shrubs, 
        fruit trees, nut trees and vines, but excludes land under trees grown for wood or timber

        ## Permanent Pastures ## 
        Land used permanently (five years or more) for herbaceous forage crops, either cultivated or growing wild 
        (wild prairie or grazing land)

    ##############################################
    #            List of Data Sources            #
    ##############################################

        ## All Country 2015 ##
        spatial_map: https://gadm.org/download_country_v3.html
        FAOSTAT: http://www.fao.org/faostat/en/#data/RL

        ## USA 2015 ## 
        subnation_data: https://quickstats.nass.usda.gov/results/5DB67F78-A7D7-3DB6-AD1E-82E69F3730B8

        ## Australia 2015 ##
        subnation_data: https://www.abs.gov.au/statistics/industry/agriculture/land-management-and-farming-australia/latest-release

        ## Canada 2015 ##
        subnation_data: https://www150.statcan.gc.ca/n1/tbl/csv/32100406-eng.zip  

        ## Mexico 2007 ##
        subnation data: http://en.www.inegi.org.mx/programas/cagf/2007/default.html#Tabular_data 
                        -> "1. Structure of the production unit"
                        -> "Total area of production units according to land use"

        ## Brazil 2017 ##
        subnation data: https://sidra.ibge.gov.br/tabela/6881
                        -> "Variável": "Área dos estabelecimentos agropecuários (Hectares)"
                        -> "Utilização das terras": "Lavouras - permanentes
                                                     Lavouras - temporárias
                                                     Lavouras - área para cultivo de flores
                                                     Pastagens - naturais
                                                     Pastagens - plantadas em boas condições
                                                     Pastagens - pastagens plantadas em más condições"
                        -> "Unidade Territorial": "Unidade da Federação"

        ## Argentina 2018 ##
        subnation data: https://cna2018.indec.gob.ar/informe-de-resultados.html
                        -> "Ver cuadros estadísticos del CNA 2018" (Section 3.4)

        ## China 2018 ##
        subnation data: http://www.stats.gov.cn/tjsj/ndsj/2018/indexeh.htm
                        -> (Cropland) http://www.stats.gov.cn/tjsj/ndsj/2018/html/EN0823.jpg
                        -> (Pasture) http://www.stats.gov.cn/tjsj/ndsj/2018/html/EN0827.jpg

        ## India 2015 ##
        subnation data: https://eands.dacnet.nic.in/PDF/At%20a%20Glance%202019%20Eng.pdf (Table 13.5)

        ## South Africa 2017 ##
        subnation data: http://www.statssa.gov.za/publications/Report-11-02-01/Report-11-02-012017.pdf
                        -> 2.2 Land use (Table G)

        ## EU 2016 ##
        subnation data: http://appsso.eurostat.ec.europa.eu/nui/show.do?dataset=ef_lus_main&lang=en
                        -> "Main farm land use by NUTS 2 regions" from eurostat

        ## Mozambique ##
        subnation data: Mozambique was a pdf, I used Tabula to extract the table. The extraction was not perfect, 
                        so I modified it directly in Excel (using Data>Text to columns). I have included the pdf, 
                        the raw Tabula extraction and the cleaned up version.

        ## Namibia ##
        subnation data: https://microdata.fao.org/index.php/catalog/940

        ## Tanzania ##
        subnation data: https://www.nbs.go.tz/index.php/en/

        ## Kazakhstan ##
        subnation data: https://stat.gov.kz/census/national/agriculture2006_2007

        ## Russia ##
        subnation data: https://eng.rosstat.gov.ru/

   """
    generate_complete_census()
