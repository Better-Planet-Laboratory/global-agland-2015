from census_processor.country import *


def subnational_processor_namibia(subnational_dir):
    """
    Customized subnational_stats data processor for Namibia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Namibia census, Namibia.csv

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_csv(subnational_dir)

    # Rename column names
    subnation_data = subnation_data.rename(columns={'region': 'STATE',
                                                    'CROPLAND': 'CROPLAND',
                                                    'PASTURE': 'PASTURE'}, inplace=False)

    # //Karas -> !KARAS
    subnation_data['STATE'] = subnation_data['STATE'].replace(['//Karas'], '!KARAS')
    subnation_data = Namibia.switch_case(subnation_data, 'upper')

    # Trim off GEO strings
    subnation_data['STATE'] = subnation_data['STATE'].map(lambda x: x.split('[', 1)[0].strip())

    # Convert format to standard
    subnation_data['CROPLAND'] = Namibia.string_to_num(subnation_data['CROPLAND'])
    subnation_data['PASTURE'] = Namibia.string_to_num(subnation_data['PASTURE'])

    return subnation_data


class Namibia(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_namibia, *subnational_dir)
        self.units = units
