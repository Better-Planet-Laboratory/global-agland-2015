from census_processor.country import *


def subnational_processor_pakistan(subnational_dir):
    """
    Customized subnational_stats data processor for Pakistan files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Pakistan.xlsx file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_excel(subnational_dir, sheet_name='Sheet2').iloc[0:-1, :]
    subnation_data = subnation_data.rename(columns={
        'Province (2015-16)': 'STATE',
        'Cultivated area': 'CROPLAND'
    })
    subnation_data['PASTURE'] = np.nan

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Khyber Pakhtunkhwa': 'KHYBER-PAKHTUNKHWA',
        'Balochistan': 'BALOCHISTAN'},
        inplace=False)

    return subnation_data


class Pakistan(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Mha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Mha)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_pakistan, *subnational_dir)
        self.units = units
