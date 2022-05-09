from census_processor.country import *


def subnational_processor_finland(subnational_dir):
    """
    Customized subnational_stats data processor for Finland files. The census does not
    contain any info on the provinces in the north part of Finland
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
    num_states = 5
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'Finland'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    # Merge Etelä-Suomi and Helsinki-Uusimaa based on their geographical location
    # Ignore Åland entry since it does not appear in spatial_map shapefile
    subnation_data.iloc[2, :] += subnation_data.iloc[1, :]
    subnation_data = subnation_data.drop([268])
    subnation_data.iloc[1, 0] = 'SOUTHERN FINLAND'
    subnation_data = subnation_data.replace({
        'Länsi-Suomi': 'WESTERN FINLAND',
        'Pohjois- ja Itä-Suomi': 'EASTERN FINLAND'
    }, inplace=False)

    return subnation_data


class Finland(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_finland, *subnational_dir)
        self.units = units
