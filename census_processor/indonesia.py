from census_processor.country import *


def subnational_processor_indonesia(subnational_dir):
    """
    Customized subnational_stats data processor for Indonesia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Indonesia.xlsx

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_excel(subnational_dir, sheet_name='TOTAL').iloc[1:, 1:]
    subnation_data = subnation_data.rename(columns={'SUM ACROSS ALL TAB TOTALS': 'STATE',
                                                    'Unnamed: 2': 'CROPLAND'},
                                           inplace=False)
    subnation_data['PASTURE'] = np.nan  # Indonesia data does not contain pasture data

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Kep.\xa0Bangka Belitung': 'BANGKA BELITUNG',
        'DKI Jakarta': 'JAKARTA RAYA',
        'West Java': 'JAWA BARAT',
        'Central Java': 'JAWA TENGAH',
        'East Java': 'JAWA TIMUR',
        'West Kalimantan': 'KALIMANTAN BARAT',
        'South Borneo': 'KALIMANTAN SELATAN',
        'Central Kalimantan': 'KALIMANTAN TENGAH',
        'East Kalimantan': 'KALIMANTAN TIMUR',
        'North Kalimantan': 'KALIMANTAN UTARA',
        'Riau islands': 'KEPULAUAN RIAU',
        'North Maluku': 'MALUKU UTARA',
        'West Nusa Tenggara': 'NUSA TENGGARA BARAT',
        'East Nusa Tenggara': 'NUSA TENGGARA TIMUR',
        'West Papua': 'PAPUA BARAT',
        'West Sulawesi': 'SULAWESI BARAT',
        'South Sulawesi': 'SULAWESI SELATAN',
        'Central Sulawesi': 'SULAWESI TENGAH',
        'Southeast Sulawesi': 'SULAWESI TENGGARA',
        'North Sulawesi': 'SULAWESI UTARA',
        'West Sumatra': 'SUMATERA BARAT',
        'South Sumatra': 'SUMATERA SELATAN',
        'North Sumatra': 'SUMATERA UTARA',
        'In Yogyakarta': 'YOGYAKARTA'
    }, inplace=False)

    return subnation_data


class Indonesia(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='M2'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: M2)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_indonesia, *subnational_dir)
        self.units = units
