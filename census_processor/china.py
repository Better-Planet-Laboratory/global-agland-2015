from census_processor.country import *


def subnational_processor_china(cropland_dir, pasture_dir):
    """
    Customized subnational_stats data processor for China files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        cropland_dir (str): path dir for cropland_dir.xlsx
        pasture_dir (str): path dir for pasture_dir.xlsx

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    # China subnational data is split into 2 files using OCR tools to
    # convert img files to xlsx
    assert (os.path.exists(cropland_dir)), "dir does not have cropland.xlsx"
    assert (os.path.exists(pasture_dir)), "dir does not have pasture.xlsx"

    # Merge cropland and pasture data
    subnation_data = pd.read_excel(cropland_dir, header=1)
    subnation_data = subnation_data.merge(pd.read_excel(pasture_dir, header=1),
                                          on='State', how='left')

    # Rename states in census to match GADM
    subnation_data = subnation_data.replace({'Inner Mongolia': 'NEI MONGOL',
                                             'Ningxia': 'NINGXIA HUI',
                                             'Xinjiang': 'XINJIANG UYGUR',
                                             'Tibet': 'XIZANG'}, inplace=False)

    # Rename column names to State | Cropland | PASTURE
    subnation_data = subnation_data.rename(columns={subnation_data.columns[0]: "STATE",
                                                    subnation_data.columns[1]: "CROPLAND",
                                                    subnation_data.columns[2]: "PASTURE"})

    return subnation_data


class China(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Kha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir(s) for China, [cropland_path, pasture_path]
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Kha)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_china, *subnational_dir)
        self.units = units
