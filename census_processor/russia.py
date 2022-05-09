from census_processor.country import *
from utils.tools.geo import polygon_union


def subnational_processor_russia(subnational_dir):
    """
    Customized subnational_stats data processor for Russia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Russia_translated.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    raw_subnation_data = pd.read_csv(subnational_dir, header=2)
    raw_subnation_data = raw_subnation_data[['Federal districts, subjects of the Russian Federation',
                                             'Farmland - total area',
                                             'Farmland - pastures']]

    # Row index to be removed from census (not state level)
    remove_index = [0, 1, 2, 21, 33, 40, 48, 63, 70, 83, 93]
    raw_subnation_data = raw_subnation_data.drop(remove_index)

    # Farmland \- total area - Farmland \- pastures -> Cropland
    # Farmland \- pastures -> Pasture
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['Federal districts, subjects of the Russian Federation']
    subnation_data['CROPLAND'] = raw_subnation_data['Farmland - total area'] - raw_subnation_data['Farmland - pastures']
    subnation_data['PASTURE'] = raw_subnation_data['Farmland - pastures']
    subnation_data = Russia.strip_string(subnation_data)

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Republic of Adygea': 'ADYGEY',
        'Altai Republic': 'ALTAY',
        'Amurskaya Oblast': 'AMUR',
        'Arkhangelsk region': 'ARKHANGEL\'SK',
        'Astrakhan region': 'ASTRAKHAN\'',
        'Republic of Bashkortostan': 'BASHKORTOSTAN',
        'Belgorod region': 'BELGOROD',
        'Bryansk region': 'BRYANSK',
        'The Republic of Buryatia': 'BURYAT',
        'Chechen Republic': 'CHECHNYA',
        'Chelyabinsk region': 'CHELYABINSK',
        'Chukotka JSC': 'CHUKOT',
        'Chuvash Republic': 'CHUVASH',
        'St. Petersburg': 'CITY OF ST. PETERSBURG',
        'The Republic of Dagestan': 'DAGESTAN',
        'The Republic of Ingushetia': 'INGUSH',
        'Irkutsk region': 'IRKUTSK',
        'Ivanovo region': 'IVANOVO',
        'Kabardino-Balkar Republic': 'KABARDIN-BALKAR',
        'Kaliningrad region': 'KALININGRAD',
        'Republic of Kalmykia': 'KALMYK',
        'Kaluga region': 'KALUGA',
        'Kamchatka Krai': 'KAMCHATKA',
        'Karachay-Cherkess Republic': 'KARACHAY-CHERKESS',
        'Republic of Karelia': 'KARELIA',
        'Kemerovo region': 'KEMEROVO',
        'Khabarovsk region': 'KHABAROVSK',
        'The Republic of Khakassia': 'KHAKASS',
        'Khanty-Mansiysk jsc': 'KHANTY-MANSIY',
        'Kirov region': 'KIROV',
        'Komi Republic': 'KOMI',
        'Kostroma region': 'KOSTROMA',
        'Krasnodar region': 'KRASNODAR',
        'Krasnoyarsk region': 'KRASNOYARSK',
        'Kurgan region': 'KURGAN',
        'Kursk region': 'KURSK',
        'Leningrad region': 'LENINGRAD',
        'Lipetsk region': 'LIPETSK',
        'Magadan Region': 'MAGA BURYATDAN',
        'Mari El Republic': 'MARIY-EL',
        'The Republic of Mordovia': 'MORDOVIA',
        'Moscow city': 'MOSCOW CITY',
        'Moscow region': 'MOSKVA',
        'Murmansk region': 'MURMANSK',
        'Nenetsky A.O.': 'NENETS',
        'Nizhny Novgorod Region': 'NIZHEGOROD',
        'Republic of North Ossetia - Alania': 'NORTH OSSETIA',
        'Novgorod region': 'NOVGOROD',
        'Novosibirsk region': 'NOVOSIBIRSK',
        'Omsk region': 'OMSK',
        'Oryol Region': 'OREL',
        'Orenburg region': 'ORENBURG',
        'Penza region': 'PENZA',
        'Perm Territory': 'PERM\'',
        'Primorsky Krai': 'PRIMOR\'YE',
        'Pskov region': 'PSKOV',
        'Rostov region': 'ROSTOV',
        'Ryazan Oblast': 'RYAZAN\'',
        'The Republic of Sakha (Yakutia)': 'SAKHA',
        'Sakhalin Region': 'SAKHALIN',
        'Samara Region': 'SAMARA',
        'Saratov region': 'SARATOV',
        'Smolensk region': 'SMOLENSK',
        'Stavropol region': 'STAVROPOL\'',
        'Sverdlovsk region': 'SVERDLOVSK',
        'Tambov Region': 'TAMBOV',
        'Republic of Tatarstan': 'TATARSTAN',
        'Tomsk region': 'TOMSK',
        'Tula region': 'TULA',
        'Tyva Republic': 'TUVA',
        'Tver region': 'TVER\'',
        'Tyumen region': 'TYUMEN\'',
        'Udmurtia': 'UDMURT',
        'Ulyanovsk region': 'UL\'YANOVSK',
        'Vladimir region': 'VLADIMIR',
        'Volgograd region': 'VOLGOGRAD',
        'Vologodskaya Oblast': 'VOLOGDA',
        'Voronezh region': 'VORONEZH',
        'Yamalo-Nenets JSC': 'YAMAL-NENETS',
        'Yaroslavskaya oblast': 'YAROSLAVL\'',
        'Jewish Auth. region': 'YEVREY',
        'Zabaykalsky Krai': 'ZABAYKAL\'YE'
    }, inplace=False)

    return subnation_data


class Russia(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Kha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Kha)
        """
        assert units in Country.UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_russia, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        spatial_map = Russia.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_1': 'STATE'}, inplace=False)

        # Merge 'ALTAY' and 'GORNO-ALTAY' into ALTAY
        polygon_list = [spatial_map[spatial_map['STATE'] == 'ALTAY']['geometry'].to_list()[0],
                        spatial_map[spatial_map['STATE'] == 'GORNO-ALTAY']['geometry'].to_list()[0]]
        spatial_map.loc[spatial_map['STATE'] == 'ALTAY', 'geometry'] = polygon_union(polygon_list)
        spatial_map = spatial_map.drop(spatial_map.index[spatial_map['STATE'] == 'GORNO-ALTAY'])

        return spatial_map
