from census_processor.country import *


def subnational_processor_ukraine(subnational_dir):
    """
    Customized subnational_stats data processor for Ukraine files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for tabula-Agriculture 2015.csv file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    raw_subnation_data = pd.read_csv(subnational_dir, header=1).iloc[1:]

    # Arable land (thousand hectares) -> Cropland
    # Agricultural land (thousand hectares) - Arable land (thousand hectares) -> Pasture
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['Oblast']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land (thousand hectares)']
    subnation_data['PASTURE'] = raw_subnation_data['Agricultural land (thousand hectares)'] - raw_subnation_data[
        'Arable land (thousand hectares)']

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Autonomous Republic of Crimea': 'CRIMEA',
        'Dnipropetrovsk': 'DNIPROPETROVS\'K',
        'Donetsk': 'DONETS\'K',
        'Ivano-Frankivsk': 'IVANO-FRANKIVS\'K',
        'Khmelnytskiy': 'KHMEL\'NYTS\'KYY',
        'Kyiv': 'KIEV',
        'Lviv': 'L\'VIV',
        'Luhansk': 'LUHANS\'K',
        'Mikolayiv': 'MYKOLAYIV',
        'Odesa': 'ODESSA',
        'Rivne ': 'RIVNE',
        'Stevastopol': 'SEVASTOPOL\'',
        'Ternopil': 'TERNOPIL\'',
        'Vinnytsia': 'VINNYTSYA',
        'Zakarpattya': 'ZAKARPATTIA',
        'Zaporizhzhya': 'ZAPORIZHIA'
    }, inplace=False)

    return subnation_data


class Ukraine(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Kha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Kha)
        """
        assert units in Country.UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_ukraine, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        spatial_map = Country.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_1': 'STATE'}, inplace=False)
        spatial_map = spatial_map.drop([0])  # remove first entry "?"

        return spatial_map
