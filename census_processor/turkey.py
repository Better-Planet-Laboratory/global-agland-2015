from census_processor.country import *


def subnational_processor_turkey(subnational_dir):
    """
    Customized subnational_stats data processor for Turkey files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for table.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_csv(subnational_dir).iloc[0:-5, :][[
        'Region Name', 'Total arable land and land under permanent crops (hectare)']]
    subnation_data = subnation_data.rename(columns={
        'Region Name': 'STATE',
        'Total arable land and land under permanent crops (hectare)': 'CROPLAND'
    })
    subnation_data['PASTURE'] = np.nan

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Afyonkarahisar': 'AFYON',
        'Ağrı': 'AGRI',
        'Eskişehir': 'ESKISEHIR',
        'Gümüşhane': 'GÜMÜSHANE',
        'İstanbul': 'ISTANBUL',
        'İzmir': 'IZMIR',
        'Kahramanmaraş': 'K. MARAS',
        'Kırıkkale': 'KINKKALE',
        'Kırşehir': 'KIRSEHIR',
        'Muğla': 'MUGLA',
        'Muş': 'MUS',
        'Nevşehir': 'NEVSEHIR',
        'Niğde': 'NIGDE',
        'Şanlıurfa': 'SANLIURFA',
        'Şırnak': 'SIRNAK',
        'Tekirdağ': 'TEKIRDAG',
        'Uşak': 'USAK',
        'Zonguldak': 'ZINGULDAK'
    }, inplace=False)

    return subnation_data


class Turkey(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_turkey, *subnational_dir)
        self.units = units
