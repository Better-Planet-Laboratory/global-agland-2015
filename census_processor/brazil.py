from census_processor.country import *


def subnational_processor_brazil(subnational_dir):
    """
    Customized subnational_stats data processor for Brazil files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Tabela 6881.xlsx

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    cropland_item = ['Lavouras - permanentes', 'Lavouras - temporárias',
                     'Lavouras - área para cultivo de flores']

    pasture_item = ['Pastagens - naturais', 'Pastagens - plantadas em boas condições',
                    'Pastagens - pastagens plantadas em más condições']

    skip_top_rows = 8  # top rows to skip
    skip_last_rows = 1  # last rows to skip
    subnation_data = pd.read_excel(subnational_dir, skiprows=range(0, skip_top_rows))[:-1 * skip_last_rows]

    # Convert to string to number
    for _, item_list in enumerate([cropland_item, pasture_item]):
        for _, col_name in enumerate(item_list):
            subnation_data[col_name] = Brazil.string_to_num(subnation_data[col_name])

    # Sum each item list columns (Skip NaN)
    for idx, item in enumerate(cropland_item):
        if idx == 0:
            subnation_data['CROPLAND'] = subnation_data[item].fillna(0)
        else:
            subnation_data['CROPLAND'] = subnation_data['CROPLAND'] + subnation_data[item].fillna(0)

    for idx, item in enumerate(pasture_item):
        if idx == 0:
            subnation_data['PASTURE'] = subnation_data[item].fillna(0)
        else:
            subnation_data['PASTURE'] = subnation_data['PASTURE'] + subnation_data[item].fillna(0)

    # Drop intermediate columns
    subnation_data = subnation_data.drop(columns=(cropland_item + pasture_item))

    # Rename 'Unnamed: 0'-"State"
    subnation_data = subnation_data.rename(columns={'Unnamed: 0': 'STATE'}, inplace=False)
    subnation_data = Brazil.strip_string(subnation_data)

    return subnation_data


class Brazil(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_brazil, *subnational_dir)
        self.units = units


