from census_processor.country import *


def subnational_processor_uganda(subnational_dir):
    """
    Customized subnational_stats data processor for Uganda files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Uganda.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    raw_subnation_data = pd.read_csv(subnational_dir).iloc[0:-1][['Area', 'Total crop area - 2018']]

    # Uganda shapefile has many districts, while in census it seems to contain one
    # level up states. Since we do not have Uganda pasture data, we will sum over
    # all state level in census and use level 0 shapefile (or FAO)
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data = subnation_data.append({
        'STATE': 'UGANDA',
        'CROPLAND': sum(raw_subnation_data['Total crop area - 2018'].to_list()),
        'PASTURE': np.nan}, ignore_index=True)
    return subnation_data


class Uganda(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_uganda, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        # We use country level for Uganda
        spatial_map = Country.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'COUNTRY': 'STATE'}, inplace=False)

        return spatial_map
