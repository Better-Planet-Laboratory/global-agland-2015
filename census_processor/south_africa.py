from census_processor.country import *


def subnational_processor_south_africa(subnational_dir):
    """
    Customized subnational_stats data processor for South Africa files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for tabula-Report-11-02-012017.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_csv(subnational_dir, header=1)[:-1]  # drop last row which is the sum

    # Arable land -> Cropland
    # Grazing land -> Pasture
    subnation_data = subnation_data.rename(columns={
        subnation_data.columns[0]: 'STATE',
        subnation_data.columns[3]: 'CROPLAND',
        subnation_data.columns[5]: 'PASTURE'
    })
    subnation_data = subnation_data[['STATE', 'CROPLAND', 'PASTURE']]
    subnation_data = Country.strip_string(subnation_data)
    subnation_data['CROPLAND'] = Country.string_to_num(subnation_data['CROPLAND'])
    subnation_data['PASTURE'] = Country.string_to_num(subnation_data['PASTURE'])

    return subnation_data


class SouthAfrica(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_south_africa, *subnational_dir)
        self.units = units
