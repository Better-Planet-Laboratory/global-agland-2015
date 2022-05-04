from country import *


def subnational_processor_tanzania(subnational_dir):
    """
    Customized subnational_stats data processor for Tanzania files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for tabula-Large Scale Farms Report_modified.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    raw_subnation_data = pd.read_csv(subnational_dir, header=0)

    # cropland_index -> Cropland
    # pasture_index -> Pasture
    cropland_index = [1, 3, 5, 7, 9, 15]
    pasture_index = [11, 13]
    raw_subnation_data = raw_subnation_data.iloc[1:-1, [0] + cropland_index + pasture_index]

    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['Region']

    cropland_subset = raw_subnation_data.iloc[:, 1:1 + len(cropland_index)]
    pasture_subset = raw_subnation_data.iloc[:, (-len(pasture_index)):]

    for i in range(len(cropland_index)):
        # Since we are summing over different types of cropland,
        # treat NaN as 0
        current_cropland_list = Tanzania.string_to_num(cropland_subset.iloc[:, i].to_list())
        current_cropland_list[np.isnan(current_cropland_list)] = 0

        # Sum over columns
        if i == 0:
            cropland_sum = current_cropland_list
        else:
            cropland_sum += current_cropland_list

    for i in range(len(pasture_index)):
        # Since we are summing over different types of pasture,
        # treat NaN as 0
        current_pasture_list = Tanzania.string_to_num(pasture_subset.iloc[:, i].to_list())
        current_pasture_list[np.isnan(current_pasture_list)] = 0

        # Sum over columns
        if i == 0:
            pasture_sum = current_pasture_list
        else:
            pasture_sum += current_pasture_list

    subnation_data['CROPLAND'] = cropland_sum
    subnation_data['PASTURE'] = pasture_sum

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'North Pemba': 'PEMBA NORTH',
        'South Pemba': 'PEMBA SOUTH',
        'North Unguja': 'ZANZIBAR NORTH',
        'South Unguja': 'ZANZIBAR SOUTH AND CENTRAL',
        'Urban West': 'ZANZIBAR WEST'
    }, inplace=False)

    return subnation_data


class Tanzania(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_tanzania, *subnational_dir)
        self.units = units
