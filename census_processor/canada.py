from census_processor.country import *


def subnational_processor_canada(subnational_dir):
    """
    Customized subnational_stats data processor for Canada files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for Canada census

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    year = 2016
    raw_subnation_data = pd.read_csv(subnational_dir)

    # Filter out non-corpland or pasture data (using Arc here)
    # Land in crops (excluding Christmas tree area) + Summerfallow land -> Cropland
    # Natural land for pasture + Tame or seeded pasture -> Pasture
    raw_subnation_data = raw_subnation_data.loc[(raw_subnation_data['Unit of measure'] == 'Acres') &
                                                ((raw_subnation_data[
                                                      'Land use'] == 'Land in crops (excluding Christmas tree area)') |
                                                 (raw_subnation_data['Land use'] == 'Summerfallow land') |
                                                 (raw_subnation_data['Land use'] == 'Natural land for pasture') |
                                                 (raw_subnation_data['Land use'] == 'Tame or seeded pasture')) &
                                                (raw_subnation_data['REF_DATE'] == year)]

    # Filter out non-level 1 regions
    # Trim off GEO strings
    # Convert format to standard
    raw_subnation_data = raw_subnation_data.loc[(raw_subnation_data['GEO'].
                                                 str.contains('\[PR|\[600000000]|\[610000000]', regex=True))]
    raw_subnation_data['GEO'] = raw_subnation_data['GEO'].map(lambda x: x.split('[', 1)[0].strip())
    raw_subnation_data['VALUE'] = Canada.string_to_num(raw_subnation_data['VALUE'])

    # Get subset of cropland
    cropland_subnation = raw_subnation_data.loc[(raw_subnation_data['Land use'] ==
                                                 'Land in crops (excluding Christmas tree area)') |
                                                (raw_subnation_data['Land use'] ==
                                                 'Summerfallow land')]

    cropland_subnation = cropland_subnation[['GEO', 'VALUE']].groupby(by=['GEO']).sum().reset_index()
    cropland_subnation = cropland_subnation.rename(columns=
                                                   {'VALUE': 'CROPLAND',
                                                    'GEO': 'STATE'},
                                                   inplace=False)
    cropland_subnation = Canada.switch_case(cropland_subnation, 'upper')

    # Get subset of pasture
    pasture_subnation = raw_subnation_data.loc[(raw_subnation_data['Land use'] ==
                                                'Natural land for pasture') |
                                               (raw_subnation_data['Land use'] ==
                                                'Tame or seeded pasture')]

    pasture_subnation = pasture_subnation[['GEO', 'VALUE']].groupby(by=['GEO']).sum().reset_index()
    pasture_subnation = pasture_subnation.rename(columns=
                                                 {'VALUE': 'PASTURE',
                                                  'GEO': 'STATE'},
                                                 inplace=False)
    pasture_subnation = Canada.switch_case(pasture_subnation, 'upper')

    # Merge cropland and pasture subset
    subnation_data = cropland_subnation.copy()
    subnation_data = subnation_data.merge(pasture_subnation, on='STATE', how='left')

    return subnation_data


class Canada(Country):

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
        self.subnational_data = self.get_subnational_data(subnational_processor_canada, *subnational_dir)
        self.units = units

    def get_spatial_map(self, shapefile_dir):
        spatial_map = Country.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_1': 'STATE'}, inplace=False)
        spatial_map = spatial_map.replace('QUÉBEC', 'QUEBEC')  # Modify Quebec
        return spatial_map
