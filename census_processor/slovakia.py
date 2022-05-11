from census_processor.country import *
from utils.tools.geo import polygon_union


def subnational_processor_slovakia(subnational_dir):
    """
    Customized subnational_stats data processor for Slovakia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for EU country collection .xlsx file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    # cropland - Arable land + Permanent crops
    # pasture - Permanent grassland - outdoor
    raw_subnation_data = pd.read_excel(subnational_dir, header=11, sheet_name='Sheet 1', skipfooter=6)
    raw_subnation_data = raw_subnation_data[
        ['CROPS (Labels)', 'Arable land', 'Permanent crops', 'Permanent grassland - outdoor']]

    # Modify here for EU Countries
    num_states = 4
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'Slovakia'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    return subnation_data


def spatial_map_processor_slovakia(shapefile_dir, state_lookup):
    """
    The state level shapefile could not be found on GADM, so this function helps
    union the corresponding cities into their state in Slovakia

    Args:
        shapefile_dir (str): shapefile dir for this country from GADM (level 1)
        state_lookup (dict): key: state name
                             value: list of cities in each state

    Returns: (pd) dataframe that contains state level spatial map for Slovakia
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
            'ID_0': 'SVK',
            'COUNTRY': 'Slovakia',
            'STATE': state,
            'geometry': geo_feature
        }, ignore_index=True)

    return spatial_map


class Slovakia(Country):
    STATE_LOOKUP = {
        'Bratislavský kraj': ['BRATISLAVSKÝ'],
        'Západné Slovensko': ['NITRIANSKY', 'TRENČIANSKY', 'TRNAVSKÝ'],
        'Stredné Slovensko': ['BANSKOBYSTRICKÝ', 'ŽILINSKÝ'],
        'Východné Slovensko': ['KOŠICKÝ', 'PREŠOVSKÝ']
    }

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Ha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir for EU country
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Ha)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_slovakia, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        spatial_map = spatial_map_processor_slovakia(shapefile_dir, Slovakia.STATE_LOOKUP)
        spatial_map = Country.switch_case(spatial_map, 'upper')
        return spatial_map
