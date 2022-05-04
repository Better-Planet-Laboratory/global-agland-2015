from country import *
from utils.tools.geo import polygon_union


def subnational_processor_australia(subnational_dir):
    """
    Customized subnational_stats data processor for Australia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for 46270do002_201617.csv

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    state_list = ['New South Wales and Australian Capital Territory',
                  'Victoria', 'Queensland', 'South Australia', 'Western Australia',
                  'Tasmania', 'Northern Territory']

    # data_item[1] -> Cropland
    # data_item[0] -> Pasture
    data_item = ['Land use - Land mainly used for grazing - Total area (ha) (e)',
                 'Land use - Land mainly used for crops - Area (ha) (d)']

    skip_top_rows = 4  # top rows to skip
    raw_subnation_data = pd.read_csv(subnational_dir,
                                     skiprows=range(0, skip_top_rows),
                                     encoding='cp1252', engine='python')

    # Take Region label that is not NaN
    raw_subnation_data = raw_subnation_data[raw_subnation_data['Region label'].notna()]
    raw_subnation_data = raw_subnation_data[raw_subnation_data['Region label'].isin(state_list) &
                                            raw_subnation_data['Data item description'].isin(data_item)]
    cropland_subset = raw_subnation_data.loc[(raw_subnation_data['Data item description'] ==
                                              data_item[1])][['Region label', 'Estimate']]
    pasture_subset = raw_subnation_data.loc[(raw_subnation_data['Data item description'] ==
                                             data_item[0])][['Region label', 'Estimate']]

    subnation_data = cropland_subset.copy()
    subnation_data = subnation_data.merge(pasture_subset, on='Region label', how='left')

    # Northern Territory has 4 identital entries
    subnation_data = subnation_data.drop([7, 8, 9])
    subnation_data = subnation_data.rename(columns={'Region label': 'STATE',
                                                    'Estimate_x': 'CROPLAND',
                                                    'Estimate_y': 'PASTURE'},
                                           inplace=False)
    subnation_data['CROPLAND'] = Australia.string_to_num(subnation_data['CROPLAND'].to_list())
    subnation_data['PASTURE'] = Australia.string_to_num(subnation_data['PASTURE'].to_list())

    return subnation_data


def spatial_map_processor_australia(shapefile_dir, state_lookup):
    """
    One state level requires a union on GADM, so this function helps
    union the corresponding cities into their state in Australia

    Args:
        shapefile_dir (str): shapefile dir for this country from GADM (level 1)
        state_lookup (dict): key: state name
                             value: list of cities in each state

    Returns: (pd) dataframe that contains state level spatial map for Australia
    """
    # Load original shapefile
    raw_spatial_map = gpd.read_file(shapefile_dir)
    raw_spatial_map = raw_spatial_map.rename(columns={'NAME_1': 'STATE'}, inplace=False)
    raw_spatial_map = Country.switch_case(raw_spatial_map, 'upper')

    # Unioned spatial_map placeholder
    spatial_map = pd.DataFrame(columns=['ID_0', 'COUNTRY', 'STATE', 'geometry'])

    for state, city_list in state_lookup.items():
        # Get current list of polygons
        poly_list = raw_spatial_map[raw_spatial_map['STATE'].isin(city_list)]['geometry'].to_list()
        geo_feature = polygon_union(poly_list)
        spatial_map = spatial_map.append({
            'ID_0': 'AUS',
            'COUNTRY': 'AUSTRALIA',
            'STATE': state,
            'geometry': geo_feature
        }, ignore_index=True)

    return spatial_map


class Australia(Country):
    STATE_LOOKUP = {
        'New South Wales and Australian Capital Territory': ['AUSTRALIAN CAPITAL TERRITORY', 'NEW SOUTH WALES'],
        'Victoria': ['VICTORIA'],
        'Queensland': ['QUEENSLAND'],
        'South Australia': ['SOUTH AUSTRALIA'],
        'Western Australia': ['WESTERN AUSTRALIA'],
        'Tasmania': ['TASMANIA'],
        'Northern Territory': ['NORTHERN TERRITORY'],
    }

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Ha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Ha)
        """
        assert units in Country.UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_australia, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        spatial_map = spatial_map_processor_australia(shapefile_dir, Australia.STATE_LOOKUP)
        spatial_map = Country.switch_case(spatial_map, 'upper')
        return spatial_map
