from census_processor.country import *


def subnational_processor_mozambique(subnational_dir):
    """
    Customized subnational_stats data processor for Mozambique files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for
                               tabula-Censo Agro 2013 Pecuario 2009 2013 2010 Resultados Definitivos -2_modified.csv

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    skip_top_rows = 2
    skip_last_rows = 13  # last rows to skip
    subnation_data = pd.read_csv(subnational_dir)[skip_top_rows:skip_last_rows]

    # Rename attributes
    subnation_data = subnation_data.rename(columns={
        subnation_data.columns[0]: 'STATE',
        subnation_data.columns[4]: 'CROPLAND'
    })

    subnation_data = subnation_data[['STATE', 'CROPLAND']]
    subnation_data['PASTURE'] = np.nan  # Mozambique data does not contain pasture data
    subnation_data = subnation_data.replace({'Niassa': 'NASSA',
                                             'Zambézia': 'ZAMBEZIA',
                                             'Inhambane': 'INHAMBANE',
                                             'Cidade de Maputo': 'MAPUTO CITY'},
                                            inplace=False)

    # Trim off empty space
    subnation_data = Mozambique.strip_string(subnation_data)

    # Convert string to numeric
    subnation_data['CROPLAND'] = Mozambique.string_to_num(subnation_data['CROPLAND'])
    subnation_data['PASTURE'] = Mozambique.string_to_num(subnation_data['PASTURE'])

    return subnation_data


class Mozambique(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_mozambique, *subnational_dir)
        self.units = units
