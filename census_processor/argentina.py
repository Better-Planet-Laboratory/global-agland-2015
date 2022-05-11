from census_processor.country import *


def subnational_processor_argentina(subnational_dir):
    """
    Customized subnational_stats data processor for Argentina files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for CNA2018_resultados_preliminares.xls

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    skip_top_rows = 7  # top rows to skip (must remove 'Total del país')
    skip_last_rows = 6  # last rows to skip
    subnation_data = pd.read_excel(subnational_dir, sheet_name='3.4')
    subnation_data = subnation_data.iloc[skip_top_rows:(-1 * skip_last_rows), [1, 3, 12]]

    # Rename attributes
    subnation_data = subnation_data.rename(columns={
        subnation_data.columns[0]: 'STATE',
        subnation_data.columns[1]: 'CROPLAND',
        subnation_data.columns[2]: 'PASTURE'
    })

    # Trim off empty space
    subnation_data = Argentina.strip_string(subnation_data)

    # Convert string to numeric
    subnation_data['CROPLAND'] = Country.string_to_num(subnation_data['CROPLAND'])
    subnation_data['PASTURE'] = Country.string_to_num(subnation_data['PASTURE'])

    return subnation_data


class Argentina(Country):

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
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_argentina, *subnational_dir)
        self.units = units
