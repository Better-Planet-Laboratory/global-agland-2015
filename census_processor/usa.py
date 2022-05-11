from census_processor.country import *


def subnational_processor_usa(cropland_dir, pasture_dir):
    """
    Customized subnational_stats data processor for USA files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        cropland_dir (str): path dir for cropland.csv
        pasture_dir (str): path dir for pasture.xls

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    region_name = ['Northeast', 'Lake States', 'Corn Belt', 'Northern Plains',
                   'Appalachian', 'Southeast', 'Delta States', 'Southern Plains',
                   'Mountain', 'Pacific', '48 States 2/']

    # Process Cropland
    cropland_data = pd.read_csv(cropland_dir)
    cropland_data = cropland_data.loc[(cropland_data['Data Item'] ==
                                       'AG LAND, CROPLAND - ACRES') &
                                      (cropland_data['Domain'] == 'AREA OPERATED')]
    cropland_data = cropland_data.rename(columns={'Value': 'CROPLAND', 'State': 'STATE'}, inplace=False)
    cropland_data['CROPLAND'] = Country.string_to_num(cropland_data['CROPLAND'])
    cropland_data = cropland_data[["STATE", "CROPLAND"]].groupby(['STATE']).sum().reset_index()

    # Process Pasture
    row_index_begin = 6
    row_index_end = 79
    pasture_data = pd.read_excel(pasture_dir)
    pasture_data = pasture_data[row_index_begin:row_index_end].reset_index()
    pasture_data = pd.DataFrame(data={'STATE': pasture_data.iloc[:, 1],
                                      'PASTURE': pasture_data.iloc[:, -1]})

    # Remove empty rows
    pasture_data = pasture_data.dropna(axis=0, how='any',
                                       thresh=None, subset=None, inplace=False)

    # Remove region level data and clean up
    pasture_data = pasture_data[~pasture_data["STATE"].isin(region_name)]
    pasture_data['PASTURE'] = Country.string_to_num(pasture_data['PASTURE'])
    pasture_data['STATE'] = pasture_data['STATE'].str.strip()
    pasture_data["PASTURE"] *= 1000  # Default unit used in our case in pasture is kAcres, convert to Acres
    pasture_data = Country.switch_case(pasture_data, 'upper')  # convert states name to upper case to match cropland

    # Merge pasture and cropland
    subnation_data = cropland_data.copy()
    subnation_data = subnation_data.merge(pasture_data, on='STATE', how='left')

    return subnation_data


class USA(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Arc'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Arc)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_usa, *subnational_dir)
        self.units = units
