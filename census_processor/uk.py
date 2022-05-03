from country import *


def subnational_processor_uk(subnational_dir):
    """
    Customized subnational_stats data processor for UK files
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
    num_states = 42
    start_index = raw_subnation_data.index[raw_subnation_data['CROPS (Labels)'] == 'United Kingdom'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']

    # In the xlsx file, it is possible to have ':' (not available) in the entry
    # to handle str + int issue, we could have set the entry with str to be nan
    # but in the subnational stats file, it seems they set it to be 0 during computation
    subnation_data['CROPLAND'] = UK.string_to_num(raw_subnation_data['Arable land']) + UK.string_to_num(
        raw_subnation_data['Permanent crops'])
    subnation_data['PASTURE'] = UK.string_to_num(raw_subnation_data['Permanent grassland - outdoor'])
    subnation_data = subnation_data.fillna(0)

    # The first 35 cities are all part of NA state
    for i in range(34):
        subnation_data.iloc[0, :] += subnation_data.iloc[i + 1, :]
    subnation_data = subnation_data.drop([i for i in range(291, 325)])
    subnation_data.iloc[0, 0] = 'NA'

    # West Wales and The Valleys + East Wales -> WALES
    subnation_data.iloc[1, :] += subnation_data.iloc[2, :]
    subnation_data = subnation_data.drop([326])
    subnation_data.iloc[1, 0] = 'WALES'

    # Eastern Scotland (NUTS 2013) + South Western Scotland (NUTS 2013) + \
    # North Eastern Scotland + Highlands and Islands -> SCOTLAND
    subnation_data.iloc[2, :] += subnation_data.iloc[3, :] + subnation_data.iloc[4, :] + subnation_data.iloc[5, :]
    subnation_data = subnation_data.drop([328, 329, 330])
    subnation_data.iloc[2, 0] = 'SCOTLAND'

    # Northern Ireland (UK) -> NORTHERN IRELAND
    subnation_data.iloc[3, 0] = 'NORTHERN IRELAND'

    return subnation_data


class UK(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_uk, *subnational_dir)
        self.units = units
