from census_processor.country import *


def subnational_processor_india(subnational_dir):
    """
    Customized subnational_stats data processor for India files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for India census, india_subnational.xlsx

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    # Land under Misc. tree crops &groves not incl. in net area sown + \
    # Fallow land (Total) + Net area sown -> Cropland
    # Permanent pastures & other grazing lands -> Pasture
    raw_subnation_data = pd.read_excel(subnational_dir, header=1)
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['State']
    subnation_data['CROPLAND'] = India.string_to_num(
        raw_subnation_data['Land under Misc. tree crops &groves not incl. in net area sown']) + \
                                 India.string_to_num(raw_subnation_data['Fallow land (Total)']) + \
                                 India.string_to_num(raw_subnation_data['Net area sown'])

    subnation_data['PASTURE'] = India.string_to_num(raw_subnation_data['Permanent pastures & other grazing lands'])

    # India data also needs some renaming to match GADM
    subnation_data = subnation_data.replace({'ANDAMAN & NICOBAR ISLAND': 'ANDAMAN AND NICOBAR',
                                             'DADAR & NAGAR HAVELI': 'DADRA AND NAGAR HAVELI',
                                             'DAMAN & DIU': 'DAMAN AND DIU',
                                             'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
                                             'MAHARASHTRA*': 'MAHARASHTRA',
                                             'MANIPUR*': 'MANIPUR',
                                             'DELHI': 'NCT OF DELHI',
                                             'SIKKIM*': 'SIKKIM'}, inplace=False)

    # Trim off empty space
    subnation_data = India.strip_string(subnation_data)

    return subnation_data


class India(Country):

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
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_india, *subnational_dir)
        self.units = units
