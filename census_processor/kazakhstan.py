from census_processor.country import *


def subnational_processor_kazakhstan(subnational_dir):
    """
    Customized subnational_stats data processor for Kazakhstan files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Kazakhstan.xlsx file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    # Agricultural grounds - arable land -> Cropland
    # Agricultural grounds - pastures -> Pasture
    raw_subnation_data = pd.read_excel(subnational_dir).loc[5:20]
    subnation_data = raw_subnation_data[['Land area', 'Unnamed: 7', 'Unnamed: 10']]
    subnation_data = subnation_data.rename(columns={'Land area': 'STATE',
                                                    'Unnamed: 7': 'CROPLAND',
                                                    'Unnamed: 10': 'PASTURE'},
                                           inplace=False)

    # Trim off empty space
    subnation_data = Kazakhstan.strip_string(subnation_data)

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({'Akmola': 'AQMOLA',
                                             'Aktobe': 'AQTÖBE',
                                             'East-Kazakh': 'EAST KAZAKHSTAN',
                                             'Mangystau': 'MANGGHYSTAU',
                                             'North-Kazakh': 'NORTH KAZAKHSTAN',
                                             'Karaganda': 'QARAGHANDY',
                                             'Kostanay': 'QOSTANAY',
                                             'Kyzylordinskaya': 'QYZYLORDA',
                                             'South-Kazakhstan': 'SOUTH KAZAKHSTAN',
                                             'West-Kazakh': 'WEST KAZAKHSTAN'}, inplace=False)

    return subnation_data


class Kazakhstan(Country):

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
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_kazakhstan, *subnational_dir)
        self.units = units
