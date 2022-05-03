from country import *


def subnational_processor_belgium(subnational_dir):
    """
    Customized subnational_stats data processor for Belgium files
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
    num_states = 11
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'Belgium'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    # Rename states in census to match GADM
    subnation_data = subnation_data.replace({
        'Région de Bruxelles-Capitale/Brussels Hoofdstedelijk Gewest': 'BRUXELLES',
        'Prov. Antwerpen': 'ANTWERPEN',
        'Prov. Limburg (BE)': 'LIMBURG',
        'Prov. Oost-Vlaanderen': 'OOST-VLAANDEREN',
        'Prov. Vlaams-Brabant': 'VLAAMS BRABANT',
        'Prov. West-Vlaanderen': 'WEST-VLAANDEREN',
        'Prov. Brabant wallon': 'BRABANT WALLON',
        'Prov. Hainaut': 'HAINAUT',
        'Prov. Liège': 'LIÈGE',
        'Prov. Luxembourg (BE)': 'LUXEMBOURG',
        'Prov. Namur': 'NAMUR'
    }, inplace=False)

    return subnation_data


class Belgium(Country):

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
        assert units in Country.UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_belgium, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        """
        Load shapefile from GADM. Replace NAME_2 column name by 'STATE', changing all str to
        upper case

        Args:
            shapefile_dir (str): path dir to shapefile for this country

        Returns: (pd) processed dataframe
        """
        spatial_map = Country.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_2': 'STATE'}, inplace=False)

        return spatial_map
