from country import *


def subnational_processor_france(subnational_dir):
    """
    Customized subnational_stats data processor for France files
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
    num_states = 30
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'France'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    # Merge the following based on geographical location
    # Champagne-Ardenne (NUTS 2013) + Lorraine (NUTS 2013) + Alsace (NUTS 2013) -> GRAND EST
    # Languedoc-Roussillon (NUTS 2013) + Midi-Pyrénées (NUTS 2013) -> OCCITANIE
    subnation_data.iloc[1, :] += subnation_data.iloc[8, :] + subnation_data.iloc[9, :]
    subnation_data.iloc[19, :] += subnation_data.iloc[15, :]
    subnation_data = subnation_data.drop([134, 135, 141])
    subnation_data.iloc[1, 0] = 'GRAND EST'
    subnation_data.iloc[16, 0] = 'OCCITANIE'

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Île de France': 'ÎLE-DE-FRANCE',
        'Haute-Normandie (NUTS 2013)': 'HAUTS-DE-FRANCE',
        'Centre (FR) (NUTS 2013)': 'CENTRE-VAL DE LOIRE',
        'Basse-Normandie (NUTS 2013)': 'NORMANDIE',
        'Bourgogne (NUTS 2013)': 'BOURGOGNE-FRANCHE-COMTÉ',
        'Pays de la Loire (NUTS 2013)': 'PAYS DE LA LOIRE',
        'Bretagne (NUTS 2013)': 'BRETAGNE',
        'Aquitaine (NUTS 2013)': 'NOUVELLE-AQUITAINE',
        'Auvergne (NUTS 2013)': 'AUVERGNE-RHÔNE-ALPES',
        'Provence-Alpes-Côte d\'Azur (NUTS 2013)': 'PROVENCE-ALPES-CÔTE D\'AZUR',
        'Corse (NUTS 2013)': 'CORSE'
    }, inplace=False)

    return subnation_data


class France(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_france, *subnational_dir)
        self.units = units
