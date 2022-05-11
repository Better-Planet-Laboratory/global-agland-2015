from census_processor.country import *


def subnational_processor_mongolia(subnational_dir):
    """
    Customized subnational_stats data processor for Mongolia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Mongolia.csv

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    subnation_data = pd.read_csv(subnational_dir, header=0)[['Aimag', '2015']]
    subnation_data = subnation_data.rename(columns={'Aimag': 'STATE',
                                                    '2015': 'CROPLAND'},
                                           inplace=False)
    subnation_data = subnation_data.drop([22])
    subnation_data['PASTURE'] = np.nan
    subnation_data['CROPLAND'] = Mongolia.string_to_num(subnation_data['CROPLAND'])

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Arkhangai': 'ARHANGAY',
        'Bayan-Ulgii': 'BAYAN-ÖLGIY',
        'Bayankhongor': 'BAYANHONGOR',
        'Darkhan-Uul': 'DARHAN-UUL',
        'Zavkhan': 'DZAVHAN',
        'Govi-Altai': 'GOVI-ALTAY',
        'Govisumber': 'GOVISÜMBER',
        'Khentii': 'HENTIY',
        'Khovd': 'HOVD',
        'Khuvsgul': 'HÖVSGÖL',
        'Umnugovi': 'ÖMNÖGOVI',
        'Orkhon': 'ORHON',
        'Uvurkhangai': 'ÖVÖRHANGAY',
        'Sukhbaatar': 'SÜHBAATAR',
        'Tuv': 'TÖV'},
        inplace=False)

    return subnation_data


class Mongolia(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_mongolia, *subnational_dir)
        self.units = units
