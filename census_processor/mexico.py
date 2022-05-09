from census_processor.country import *


def subnational_processor_mexico(subnational_dir):
    """
    Customized subnational_stats data processor for Mexico files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Tabulado_VIII_CAGyF_2.xls

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    skip_top_rows = 9  # top rows to skip
    skip_last_rows = 3  # last rows to skip
    subnation_data = pd.read_excel(subnational_dir, skiprows=range(0, skip_top_rows))[:-1 * skip_last_rows][
                         ['Unnamed: 0', 'Unnamed: 2', 'Unnamed: 3']][1:]
    subnation_data = subnation_data.rename(columns=
                                           {'Unnamed: 0': 'STATE',
                                            'Unnamed: 2': 'CROPLAND',
                                            'Unnamed: 3': 'PASTURE'},
                                           inplace=False)
    subnation_data = Mexico.strip_string(subnation_data)

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({'COAHUILA DE ZARAGOZA': 'COAHUILA',
                                             'MICHOACÁN DE OCAMPO': 'MICHOACÁN',
                                             'VERACRUZ LLAVE': 'VERACRUZ'},
                                            inplace=False)

    return subnation_data


class Mexico(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_mexico, *subnational_dir)
        self.units = units
