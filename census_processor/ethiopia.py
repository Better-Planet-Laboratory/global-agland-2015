from census_processor.country import *


def subnational_processor_ethiopia(subnational_dir):
    """
    Customized subnational_stats data processor for Ethiopia files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Ethiopia census

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    l1_state_col = ["Items", "Tigray Region", "Afar Region",
                    "Amhara Region", "Oromia Region", "Somale Region",
                    "Benishangul - Gumuz", "S.N.N.P. Region",
                    "Gambela Region", "Harari Region", "Dire Dawa"]

    raw_subnation_data = pd.read_excel(subnational_dir, 'Merged', skiprows=range(3))

    # Select subdataset based on l1 state level
    raw_subnation_data = raw_subnation_data.rename(columns=lambda x: x.strip())[l1_state_col]

    # Transpose dataframe with states as rows
    raw_subnation_data = raw_subnation_data.T
    raw_subnation_data = raw_subnation_data.rename(columns=raw_subnation_data.iloc[0]).drop(raw_subnation_data.index[0])
    raw_subnation_data.insert(loc=0, column='State', value=raw_subnation_data.index.values)
    raw_subnation_data = raw_subnation_data.reset_index(drop=True)

    raw_subnation_data = raw_subnation_data.replace({'Tigray Region': 'Tigray',
                                                     'Afar Region': 'Afar',
                                                     'Amhara Region': 'Amhara',
                                                     'Oromia Region': 'Oromia',
                                                     'Somale Region': 'Somali',
                                                     'Benishangul - Gumuz': 'Benshangul-Gumaz',
                                                     'S.N.N.P. Region': 'Southern Nations, Nationalities and Peoples',
                                                     'Gambela Region': 'Gambela Peoples',
                                                     'Harari Region': 'Harari People',
                                                     'Dire Dawa': 'Dire Dawa'},
                                                    inplace=False)

    # All crop area + Fallow land -> Cropland
    # Graz`ing land -> Pasture
    raw_subnation_data = raw_subnation_data[['State', 'All crop area', 'Fallow land', 'Graz`ing land']]

    # Correction to Tigray Cropland
    # Though Fallow land in Tigray is NaN, it has a valid All crop area,
    # we shall treat that NaN as 0
    raw_subnation_data.iloc[0, 2] = 0

    # Create subnation data
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['State']
    subnation_data['CROPLAND'] = raw_subnation_data['All crop area'] + raw_subnation_data['Fallow land']
    subnation_data['PASTURE'] = raw_subnation_data['Graz`ing land']

    return subnation_data


class Ethiopia(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_ethiopia, *subnational_dir)
        self.units = units
