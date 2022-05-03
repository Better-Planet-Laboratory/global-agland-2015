from country import *


def subnational_processor_greece(subnational_dir):
    """
    Customized subnational_stats data processor for Greece files
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
    num_states = 22
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'Greece'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    # Remove top 9 rows, these entries are repeatition with - NUTS 2010
    subnation_data = subnation_data.iloc[9:, :]

    # Merge the following based on geographical location
    # Voreio Aigaio + Notio Aigaio -> AEGEAN
    # Anatoliki Makedonia, Thraki + Dytiki Makedonia + Ipeiros -> EPIRUS AND WESTERN MACEDONIA
    # Ionia Nisia + Dytiki Ellada + Peloponnisos -> PELOPONNESE, WESTERN GREECE AND THE IONIAN ISLANDS
    # Thessalia + Sterea Ellada -> THESSALY AND CENTRAL GREECE
    subnation_data.iloc[1, :] += subnation_data.iloc[2, :]
    subnation_data.iloc[4, :] += subnation_data.iloc[6, :] + subnation_data.iloc[7, :]
    subnation_data.iloc[9, :] += subnation_data.iloc[10, :] + subnation_data.iloc[12, :]
    subnation_data.iloc[8, :] += subnation_data.iloc[11, :]
    subnation_data = subnation_data.drop([94, 98, 99, 102, 103, 104])

    subnation_data.iloc[1, 0] = 'AEGEAN'
    subnation_data.iloc[3, 0] = 'EPIRUS AND WESTERN MACEDONIA'
    subnation_data.iloc[5, 0] = 'THESSALY AND CENTRAL GREECE'
    subnation_data.iloc[6, 0] = 'PELOPONNESE, WESTERN GREECE AND THE IONIAN ISLANDS'

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Attiki': 'ATTICA',
        'Kriti': 'CRETE',
        'Kentriki Makedonia': 'MACEDONIA AND THRACE'
    }, inplace=False)

    return subnation_data


class Greece(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_greece, *subnational_dir)
        self.units = units
