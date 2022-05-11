from census_processor.country import *


def subnational_processor_germany(subnational_dir):
    """
    Customized subnational_stats data processor for Germany files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for EU country collection .xlsx file

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    # cropland - Arable land + Permanent crops
    # pasture - Permanent grassland - outdoor
    raw_subnation_data = pd.read_excel(subnational_dir, header=11, sheet_name='Sheet 1', skipfooter=6)
    raw_subnation_data = raw_subnation_data[
        ['CROPS (Labels)', 'Arable land', 'Permanent crops', 'Permanent grassland - outdoor']]

    # Modify here for EU Countries
    num_states = 38
    start_index = raw_subnation_data.index[
                      raw_subnation_data['CROPS (Labels)'] == 'Germany (until 1990 former territory of the FRG)'][0] + 1
    end_index = start_index + num_states

    raw_subnation_data = raw_subnation_data.iloc[start_index:end_index, :]
    subnation_data = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
    subnation_data['STATE'] = raw_subnation_data['CROPS (Labels)']
    subnation_data['CROPLAND'] = raw_subnation_data['Arable land'] + raw_subnation_data['Permanent crops']
    subnation_data['PASTURE'] = raw_subnation_data['Permanent grassland - outdoor']

    # Merge the following based on geographical location
    # Stuttgart + Karlsruhe + Freiburg + Tübingen -> BADEN-WÜRTTEMBERG
    # Oberbayern + Niederbayern + Oberpfalz + Oberfranken + Mittelfranken + Unterfranken + Schwaben -> BAYERN
    # Darmstadt + Gießen + Kassel -> HESSEN
    # Braunschweig + Hannover + Lüneburg + Weser-Ems -> NIEDERSACHSEN
    # Düsseldorf + Köln + Münster + Detmold + Arnsberg -> NORDRHEIN-WESTFALEN
    # Koblenz + Trier + Rheinhessen-Pfalz -> RHEINLAND-PFALZ
    # Dresden + Chemnitz + Leipzig -> SACHSEN
    subnation_data.iloc[0, :] += subnation_data.iloc[1, :] + subnation_data.iloc[2, :] + \
                                 subnation_data.iloc[3, :]
    subnation_data.iloc[4, :] += subnation_data.iloc[5, :] + subnation_data.iloc[6, :] + \
                                 subnation_data.iloc[7, :] + subnation_data.iloc[8, :] + \
                                 subnation_data.iloc[9, :] + subnation_data.iloc[10, :]
    subnation_data.iloc[15, :] += subnation_data.iloc[16, :] + subnation_data.iloc[17, :]
    subnation_data.iloc[19, :] += subnation_data.iloc[20, :] + subnation_data.iloc[21, :] + \
                                  subnation_data.iloc[22, :]
    subnation_data.iloc[23, :] += subnation_data.iloc[24, :] + subnation_data.iloc[25, :] + \
                                  subnation_data.iloc[26, :] + subnation_data.iloc[27, :]
    subnation_data.iloc[28, :] += subnation_data.iloc[29, :] + subnation_data.iloc[30, :]
    subnation_data.iloc[32, :] += subnation_data.iloc[33, :] + subnation_data.iloc[34, :]
    subnation_data = subnation_data.drop(
        [40, 41, 42, 44, 45, 46, 47, 48, 49, 55, 56, 59, 60, 61, 63, 64, 65, 66, 68, 69, 72, 73])

    subnation_data.iloc[0, 0] = 'BADEN-WÜRTTEMBERG'
    subnation_data.iloc[1, 0] = 'BAYERN'
    subnation_data.iloc[6, 0] = 'HESSEN'
    subnation_data.iloc[8, 0] = 'NIEDERSACHSEN'
    subnation_data.iloc[9, 0] = 'NORDRHEIN-WESTFALEN'
    subnation_data.iloc[10, 0] = 'RHEINLAND-PFALZ'
    subnation_data.iloc[12, 0] = 'SACHSEN'

    # Rename state names to match GADM
    subnation_data = subnation_data.replace({
        'Berlin': 'BERLIN',
        'Brandenburg': 'BRANDENBURG',
        'Bremen': 'BREMEN',
        'Hamburg': 'HAMBURG',
        'Mecklenburg-Vorpommern': 'MECKLENBURG-VORPOMMERN',
        'Saarland': 'SAARLAND',
        'Sachsen-Anhalt': 'SACHSEN-ANHALT',
        'Schleswig-Holstein': 'SCHLESWIG-HOLSTEIN',
        'Thüringen': 'THÜRINGEN'
    }, inplace=False)

    return subnation_data


class Germany(Country):

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Ha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir for EU country
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Ha)
        """
        assert units in UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_germany, *subnational_dir)
        self.units = units
