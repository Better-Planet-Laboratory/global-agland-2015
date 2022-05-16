import geopandas as gpd
from utils.tools.fao import *
from utils.tools.visualizer import plot_FAO_census


class World:

    @staticmethod
    def switch_case(df, to_case='upper'):
        """ Convert all elements in df to upper/lower """
        if to_case == 'upper':
            return df.applymap(lambda item: item.upper()
            if type(item) == str else item)
        else:
            return df.applymap(lambda item: item.lower()
            if type(item) == str else item)

    @staticmethod
    def assign_GID_0(census_table, state_gid_table):
        """ Update GID_0 in census_table """
        for state, gid in state_gid_table.items():
            census_table.loc[census_table['STATE'] == state, 'GID_0'] = gid

    @staticmethod
    def assign_REGIONS(census_table, state_regions_table):
        """ Update REGIONS in census_table """
        for state, regions in state_regions_table.items():
            census_table.loc[census_table['STATE'] == state, 'REGIONS'] = regions

    @staticmethod
    def has_duplicates(census_table):
        """ Check if census_table has duplicated STATE """
        return not len(census_table['STATE'].to_list()) == len(set(census_table['STATE']))

    def __init__(self, global_shapefile_dir, FAOSTAT_dir, FAOSTAT_profile_dir):
        """
        Constructor that takes directory of global shapefile (GADM level 0), FAOSTAT
        and FAOSTAT_profile

        Args:
            global_shapefile_dir (str): path dir to gadm36_0.shp
            FAOSTAT_dir (str): path dir to FAOSTAT_data*.csv
            FAOSTAT_profile_dir (str): path dir to FAOcountryProfileUTF9_withregions.csv
        """
        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.FAOSTAT_profile = self.get_FAOSTAT_profile(FAOSTAT_profile_dir)
        self.spatial_map = self.get_spatial_map(global_shapefile_dir)

        self.census_table = self.merge_all()
        assert (not World.has_duplicates(self.census_table)), 'Duplicated STATE found'

    def __len__(self):
        """ Length of World object is the number of unique STATE in census_table """
        return len(self.get_states_list())

    def merge_all(self):
        """
        Merge FAOSTAT.data to FAOSTAT_profile via STATE, then merge the intermediate
        results to global spatial_map via GID_0. The output results contain attributes:
        [STATE, CROPLAND, PASTURE, GID_0, REGIONS, geometry]

        Note: CHANNEL ISLANDS is removed due to missing in GADM. A total of 223 countries
        are included with using FAOSTAT_data_11-14-2020.csv and gadm36_0.shp

        Returns: (pd) dataframe that contains
        [STATE, CROPLAND, PASTURE, GID_0, REGIONS, geometry]
        """
        # Process FAOSTAT data for global
        global_census_table = pd.DataFrame(columns=['STATE', 'CROPLAND', 'PASTURE'])
        for c in set(self.FAOSTAT.data['Area']):
            cropland, pasture = self.FAOSTAT.get_by_country(c).mean()
            global_census_table = global_census_table.append({
                'STATE': c,
                'CROPLAND': cropland,
                'PASTURE': pasture
            }, ignore_index=True)

        global_census_table = World.switch_case(global_census_table, 'upper')

        # Merge FAOSTAT profile with FAOSTAT data via STATE
        global_census_table = global_census_table.merge(self.FAOSTAT_profile, on='STATE', how='left')

        # Rename samples to match GADM, add GID_0 missing
        new_country_name = {
            'SYRIAN ARAB REPUBLIC': 'SYRIA',
            'PALESTINE': 'PALESTINA',
            'RUSSIAN FEDERATION': 'RUSSIA',
            'CZECHIA': 'CZECH REPUBLIC',
            'UNITED STATES OF AMERICA': 'UNITED STATES',
            'LAO PEOPLE\'S DEMOCRATIC REPUBLIC': 'LAOS',
            'UNITED KINGDOM OF GREAT BRITAIN AND NORTHERN IRELAND': 'UNITED KINGDOM',
            'FRENCH GUYANA': 'FRENCH GUIANA',
            'UNITED REPUBLIC OF TANZANIA': 'TANZANIA',
            'NORTH MACEDONIA': 'MACEDONIA',
            'ESWATINI': 'SWAZILAND'
        }

        new_gid_0 = {
            'SYRIA': 'SYR',
            'PALESTINA': 'PSE',
            'NIGER': 'NER',
            'RUSSIA': 'RUS',
            'COMOROS': 'COM',
            'SOUTH SUDAN': 'SSD',
            'BAHAMAS': 'BHS',
            'CZECH REPUBLIC': 'CZE',
            'PHILIPPINES': 'PHL',
            'SUDAN': 'SDN',
            'UNITED STATES': 'USA',
            'LAOS': 'LAO',
            'UNITED KINGDOM': 'GBR',
            'MARSHALL ISLANDS': 'MHL',
            'DEMOCRATIC REPUBLIC OF THE CONGO': 'COD',
            'CHINA': 'CHN',
            'GAMBIA': 'GMB',
            'FRENCH GUIANA': 'GUF',
            'TANZANIA': 'TZA',
            'MACEDONIA': 'MKD',
            'NETHERLANDS': 'NLD',
            'UNITED ARAB EMIRATES': 'ARE',
            'SWAZILAND': 'SWZ',
            'COOK ISLANDS': 'COK',
            'DOMINICAN REPUBLIC': 'DOM'
        }

        new_regions = {
            'SYRIA': 'ASIA',
            'PALESTINA': 'ASIA',
            'NIGER': 'AFRICA',
            'RUSSIA': 'EUROPE',
            'COMOROS': 'AFRICA',
            'BAHAMAS': 'LATIN AMERICA',
            'CZECH REPUBLIC': 'EUROPE',
            'PHILIPPINES': 'ASIA',
            'SUDAN': 'AFRICA',
            'UNITED STATES': 'NORTHERN AMERICA',
            'LAOS': 'ASIA',
            'UNITED KINGDOM': 'EUROPE',
            'MARSHALL ISLANDS': 'ASIA',
            'DEMOCRATIC REPUBLIC OF THE CONGO': 'AFRICA',
            'GAMBIA': 'AFRICA',
            'FRENCH GUIANA': 'LATIN AMERICA',
            'TANZANIA': 'AFRICA',
            'MACEDONIA': 'EUROPE',
            'NETHERLANDS': 'EUROPE',
            'UNITED ARAB EMIRATES': 'ASIA',
            'SWAZILAND': 'AFRICA',
            'COOK ISLANDS': 'ASIA',
            'DOMINICAN REPUBLIC': 'LATIN AMERICA'
        }

        global_census_table = global_census_table.replace(new_country_name, inplace=False)
        World.assign_GID_0(global_census_table, new_gid_0)
        World.assign_REGIONS(global_census_table, new_regions)

        # Remove CHANNEL ISLANDS (geometry not found)
        global_census_table = global_census_table.drop(
            global_census_table.index[
                global_census_table['STATE'] == 'CHANNEL ISLANDS'])
        global_census_table = global_census_table.drop(
            global_census_table.index[
                global_census_table['GID_0'] == 'CHI'])

        # Merge intermediate results with spatial_map via GID_0
        global_census_table = global_census_table.merge(self.spatial_map[['GID_0', 'geometry']], on='GID_0', how='left')

        # Rename country names to match Country keys (settings)
        global_census_table.loc[global_census_table.STATE == 'CZECH REPUBLIC', 'STATE'] = 'CZECHIA'
        global_census_table.loc[global_census_table.STATE == 'SAUDI ARABIA', 'STATE'] = 'SAUDIARABIA'
        global_census_table.loc[global_census_table.STATE == 'SOUTH AFRICA', 'STATE'] = 'SOUTHAFRICA'
        global_census_table.loc[global_census_table.STATE == 'UNITED KINGDOM', 'STATE'] = 'UK'
        global_census_table.loc[global_census_table.STATE == 'UNITED STATES', 'STATE'] = 'USA'

        return global_census_table

    def get_FAOSTAT(self, FAOSTAT_dir):
        """
        Load FAOSTAT file for the whole global

        Args:
            FAOSTAT_dir (str): path dir to FAOSTAT_data*.csv

        Returns: (FAOSTAT)
        """
        return FAOSTAT(FAOSTAT_dir)

    def get_FAOSTAT_profile(self, FAOSTAT_profile_dir):
        """
        Load FAOSTAT profile file

        Args:
            FAOSTAT_profile_dir (str): path dir to FAOcountryProfileUTF9_withregions.csv

        Returns: (pd) processed FAOSTAT_profile dataframe
        """
        FAOSTAT_profile = pd.read_csv(FAOSTAT_profile_dir, encoding='utf-8')[['ISO3_CODE', 'FAO_TABLE_NAME', 'REGIONS']]
        FAOSTAT_profile = FAOSTAT_profile.rename(columns={"ISO3_CODE": "GID_0", "FAO_TABLE_NAME": "STATE"},
                                                 inplace=False)
        FAOSTAT_profile = World.switch_case(FAOSTAT_profile, 'upper')
        FAOSTAT_profile = FAOSTAT_profile.drop_duplicates(subset=['STATE'])  # avoid multiple instances during merge

        return FAOSTAT_profile

    def get_spatial_map(self, global_shapefile_dir):
        """
        Load global shapefile level 0

        Args:
            global_shapefile_dir (str): path dir to gadm36_0.shp

        Returns: (pd) processed spatial_map dataframe
        """
        spatial_map = World.switch_case(gpd.read_file(global_shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_0': 'STATE'}, inplace=False)

        return spatial_map

    def get_states_list(self):
        """
        Get list of states in the World census_table

        Returns: (list) of str with state names
        """
        return self.census_table['STATE'].to_list()

    def assign_census_table(self, census_table):
        """
        Replace the current World census_table by input

        Args:
            census_table (pd): dataframe must contain attributes
                               ['STATE', 'CROPLAND', 'PASTURE', 'GID_0', 'REGIONS', 'geometry']
                               with no duplicated entries
        """
        assert (not World.has_duplicates(census_table)), 'Duplicated STATE found'
        assert (set(census_table.columns.values) ==
                {'CROPLAND', 'GID_0', 'PASTURE', 'REGIONS', 'STATE', 'geometry'}), 'Attributes mismatch'
        self.census_table = census_table

    def replace_subnation(self, subnational_census, census_settings, inplace=False):
        """
        Replace country level entry (FAOSTAT) in the World object census_table attribute table
        by the stats level entries from subnational_census. Input census_settings
        contains info on bias correction bool

        Args:
            subnational_census (dict): country name to be replaced (str) -> census class obj (Country)
            census_settings (dict): country name (str) -> bias correction (bool)
            inplace (bool): if replace World census_table directly (Default: False)

        Returns: (pd) processed census_table dataframe
        """
        census_table_copy = self.census_table.copy()
        if len(subnational_census) != 0:
            for country, census in subnational_census.items():
                index_list = census_table_copy.index[census_table_copy['STATE'] == country.upper()].tolist()

                if len(index_list) == 0:
                    continue
                else:
                    index = index_list[0]

                current_GID = census_table_copy.iloc[index]['GID_0']
                current_REGIONS = census_table_copy.iloc[index]['REGIONS']
                sub_table = census.merge_census_to_spatial(bias_correct=census_settings[country], convert_to_kha=True)
                sub_table['GID_0'] = current_GID
                sub_table['REGIONS'] = current_REGIONS
                sub_table = sub_table[['STATE', 'CROPLAND', 'PASTURE', 'GID_0', 'REGIONS', 'geometry']]

                # Remove country level entry
                census_table_copy = census_table_copy.drop([index], axis=0)

                # Append
                census_table_copy = census_table_copy.append(sub_table)

                # Reset index labels
                census_table_copy = census_table_copy.reset_index(drop=True)

        if inplace:
            self.assign_census_table(census_table_copy)

        return census_table_copy

    def plot_cropland(self, output_dir=None):
        """
        Plot FAO census map for CROPLAND

        Args:
            output_dir (str): output dir (Default: None)
        """
        plot_FAO_census(self.census_table, 'CROPLAND',
                        cmap='Accent', num_bins=8, label='CROPLAND (kHa)',
                        output_dir=output_dir)

    def plot_pasture(self, output_dir=None):
        """
        Plot FAO census map for PASTURE

        Args:
            output_dir (str): output dir (Default: None)
        """
        plot_FAO_census(self.census_table, 'PASTURE',
                        cmap='Dark2', num_bins=8, label='PASTURE (kHa)',
                        output_dir=output_dir)
