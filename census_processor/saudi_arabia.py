from census_processor.country import *


def subnational_processor_saudi_arabia(subnational_dir):
    """
    Customized subnational_stats data processor for Saudi Arabia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for extracted.xlsx file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    raw_subnation_data = pd.read_excel(subnational_dir, skipfooter=4)

    # Temporary Meadows + Open field vegetables + Protected field vegetables + \
    # Grain and Feed + Dates trees + Permanent trees except date trees -> Cropland
    # Permanent meadows -> Pasture
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['Regions']
    subnation_data['PASTURE'] = SaudiArabia.string_to_num(raw_subnation_data.iloc[:, 1].to_list())

    cropland_subset = raw_subnation_data.iloc[:, 2:]
    for i in range(6):
        # Since we are summing over different types of cropland,
        # treat NaN as 0
        current_cropland_list = SaudiArabia.string_to_num(cropland_subset.iloc[:, i].to_list())
        current_cropland_list[np.isnan(current_cropland_list)] = 0

        # Sum over columns
        if i == 0:
            cropland_sum = current_cropland_list
        else:
            cropland_sum += current_cropland_list

    subnation_data['CROPLAND'] = cropland_sum

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({'Ar Riyad': 'AR RIYAD',
                                             'Al Madinah Al Munawwarah': 'AL MADINAH',
                                             'Al Qaseem': 'AL QUASSIM',
                                             'Makkah Al Mukarramah': 'MAKKAH',
                                             'Eastern Region': 'ASH SHARQIYAH',
                                             'Aseer': '`ASIR',
                                             'Tabuk': 'TABUK',
                                             'Hail': 'HA\'IL',
                                             'Northern Borders': 'AL HUDUD ASH SHAMALIYAH',
                                             'Jazan': 'JIZAN',
                                             'Najran': 'NAJRAN',
                                             'Al Bahah': 'AL BAHAH',
                                             'Al Jawf': 'AL JAWF'
                                             },
                                            inplace=False)

    subnation_data = SaudiArabia.strip_string(subnation_data)

    return subnation_data


class SaudiArabia(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Donum'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Donum)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_saudi_arabia, *subnational_dir)
        self.units = units
