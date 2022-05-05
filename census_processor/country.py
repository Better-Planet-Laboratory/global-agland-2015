import geopandas as gpd
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
from utils.io import *
from utils.tools.fao import *


def subnational_processor_country(subnational_dir):
    """
    Customized subnational_stats data processor for this country files
    Note: Output pd.Dataframe must only contain STATE | CROPLAND | PASTURE

    Args:
        subnational_dir (str): path dir for subnational_stats

    Returns: (pd) dataframe that contains only STATE | CROPLAND | PASTURE
    """
    return pd.read_csv(subnational_dir)


class Country:
    # Class Level CONSTANTS --hidden from users
    #                       --maintained by developers

    # Units Convention
    KHA_TO_ARC = 2471.0538
    KHA_TO_HA = 1000
    KHA_TO_KHA = 1
    KHA_TO_DONUM = 10000
    KHA_TO_KM2 = 10
    KHA_TO_M2 = 10000000
    KHA_TO_MHA = 0.001

    UNIT_LOOKUP = {'Arc': KHA_TO_ARC,
                   'Ha': KHA_TO_HA,
                   'Kha': KHA_TO_KHA,
                   'Donum': KHA_TO_DONUM,
                   'Km2': KHA_TO_KM2,
                   'M2': KHA_TO_M2,
                   'Mha': KHA_TO_MHA}

    # For class name -> FAO Convention
    COUNTRY_NAME = {'Country': 'Country',
                    'Argentina': 'Argentina',
                    'Australia': 'Australia',
                    'Brazil': 'Brazil',
                    'Canada': 'Canada',
                    'China': 'China',
                    'Ethiopia': 'Ethiopia',
                    'India': 'India',
                    'Indonesia': 'Indonesia',
                    'Mexico': 'Mexico',
                    'Nigeria': 'Nigeria',
                    'Russia': 'Russian Federation',
                    'SouthAfrica': 'South Africa',
                    'USA': 'United States of America',
                    'Mozambique': 'Mozambique',
                    'Namibia': 'Namibia',
                    'Tanzania': 'United Republic of Tanzania',
                    'Kazakhstan': 'Kazakhstan',
                    'SaudiArabia': 'Saudi Arabia',
                    'Belgium': 'Belgium',
                    'Austria': 'Austria',
                    'Bulgaria': 'Bulgaria',
                    'Croatia': 'Croatia',
                    'Cyprus': 'Cyprus',
                    'Czechia': 'Czechia',
                    'Denmark': 'Denmark',
                    'Estonia': 'Estonia',
                    'Finland': 'Finland',
                    'France': 'France',
                    'Germany': 'Germany',
                    'Greece': 'Greece',
                    'Hungary': 'Hungary',
                    'Ireland': 'Ireland',
                    'Italy': 'Italy',
                    'Latvia': 'Latvia',
                    'Lithuania': 'Lithuania',
                    'Luxembourg': 'Luxembourg',
                    'Malta': 'Malta',
                    'Netherlands': 'Netherlands',
                    'Poland': 'Poland',
                    'Portugal': 'Portugal',
                    'Romania': 'Romania',
                    'Slovakia': 'Slovakia',
                    'Slovenia': 'Slovenia',
                    'Spain': 'Spain',
                    'Sweden': 'Sweden',
                    'UK': 'United Kingdom of Great Britain and Northern Ireland',
                    'Mongolia': 'Mongolia',
                    'Pakistan': 'Pakistan',
                    'Turkey': 'Turkey',
                    'Ukraine': 'Ukraine',
                    'Uganda': 'Uganda'
                    }

    @staticmethod
    def string_to_num(str_array):
        """ Convert 1D string array to numerical """
        num_list = []
        for idx, item in enumerate(str_array):
            try:
                converted_item = float(item.replace(',', '').replace(' ', ''))
            except:
                if isinstance(item, str):
                    converted_item = np.nan
                else:
                    converted_item = item
            num_list.append(converted_item)
        return np.asarray(num_list)

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
    def strip_string(df):
        """ Trim all white space before and after string """
        return df.applymap(lambda item: item.strip() if isinstance(item, str) else item)

    @staticmethod
    def convert_to_kha(data_array, unit):
        """ Convert data_array from input unit to kHA """
        return data_array / Country.UNIT_LOOKUP[unit]

    def __init__(self, shapefile_dir, subnational_dir, FAOSTAT_dir, units='Kha'):
        """
        Constructor that takes directory of the countries' shapefile,
        subnational data and FAOSTAT

        Args:
            shapefile_dir (str): shapefile dir for this country
            subnational_dir (list of str): subnational_stats dir(s) for this country
            FAOSTAT_dir (str): global FAOSTAT dir
            units (str): units used in the subnational file (Default: Kha)
        """
        assert units in Country.UNIT_LOOKUP, \
            "Unit: [" + units + "] is not found in the dict. Consider adding it to the class or check your input"

        self.FAOSTAT = self.get_FAOSTAT(FAOSTAT_dir)
        self.spatial_map = self.get_spatial_map(shapefile_dir)
        self.subnational_data = self.get_subnational_data(subnational_processor_country, *subnational_dir)
        self.units = units

    def get_subnational_data(self, subnational_processor, *args):
        """
        Process subnational_stats file(s) for this country. Output a pd dataframe that only contains
        STATE | CROPLAND | PASTURE
        in its own units. All str converted to UPPER case. Numerical str converted to num

        Args:
            subnational_dir (str): path dir to subnational_stats for this country (Default: csv)

        Returns: (pd) processed dataframe that contains "State | Cropland | Pasture" for this country
        """
        subnational_data = Country.switch_case(subnational_processor(*args), 'upper')
        subnational_data['CROPLAND'] = Country.string_to_num(subnational_data['CROPLAND'])
        subnational_data['PASTURE'] = Country.string_to_num(subnational_data['PASTURE'])
        return subnational_data

    def get_spatial_map(self, shapefile_dir):
        """
        Load shapefile from GADM. Replace NAME_1 column name by 'STATE', changing all str to
        upper case

        Args:
            shapefile_dir (str): path dir to shapefile for this country

        Returns: (pd) processed dataframe
        """
        spatial_map = Country.switch_case(gpd.read_file(shapefile_dir), 'upper')
        spatial_map = spatial_map.rename(columns={'NAME_1': 'STATE'}, inplace=False)

        return spatial_map

    def get_FAOSTAT(self, FAOSTAT_dir):
        """
        Load FAOSTAT file and extract the rows for this country

        Args:
            FAOSTAT_dir (str): path dir to FAOSTAT csv that contains this country info

        Returns: (pd) processed dataframe
        """
        assert(self.__class__.__name__ in list(Country.COUNTRY_NAME.keys())), \
            "{} does not exist in Country class key, add a class name -> FAOSTAT name entry to dict".\
                format(self.__class__.__name__)
        return FAOSTAT(FAOSTAT_dir).get_by_country(Country.COUNTRY_NAME[self.__class__.__name__])

    def get_FAOSTAT_mean(self, cropland_attributes=['Arable land', 'Land under permanent crops'],
                         pasture_attributes=['Land under perm. meadows and pastures']):
        """
        Get the mean from FAOSTAT for this country (averaged over n years)
        By default, we use:
        cropland - Arable land + Land under permanent crops
        pasture - Land under perm. meadows and pastures

        Returns: (tuple) cropland_mean, pasture_mean
        """
        return self.FAOSTAT.mean(cropland_attributes=cropland_attributes,
                                 pasture_attributes=pasture_attributes)

    def get_subnational_cropland(self):
        """
        Get subnational level cropland dataset

        Returns: (pd) dataframe that contains STATE | CROPLAND
        """
        return self.subnational_data[['STATE', 'CROPLAND']]

    def get_subnational_pasture(self):
        """
        Get subnational level pasture dataset

        Returns: (pd) dataframe that contains STATE | PASTURE
        """
        return self.subnational_data[['STATE', 'PASTURE']]

    def get_subnational_cropland_sum(self):
        """
        Sum all subnational level cropland values (in its original unit)

        Returns: (float)
        """
        subnational_cropland = self.get_subnational_cropland()
        return np.nansum(subnational_cropland['CROPLAND'].to_numpy())

    def get_subnational_pasture_sum(self):
        """
        Sum all subnational level pasture values (in its original unit)

        Returns: (float)
        """
        subnational_pasture = self.get_subnational_pasture()
        return np.nansum(subnational_pasture['PASTURE'].to_numpy())

    def get_bias_factor(self):
        """
        Get bias correction factor (FAO / subnational report)

        Returns:
        """
        # Get national and subnational sum for cropland and pasture
        cropland_FAO_sum, pasture_FAO_sum = self.get_FAOSTAT_mean()
        cropland_subnational_sum, pasture_subnational_sum = self.get_subnational_cropland_sum(), \
                                                            self.get_subnational_pasture_sum()

        # Bias Correction (Scaling)
        # Use kHa as the standard unit for the conversion
        if cropland_subnational_sum == 0:
            # Zero division
            if abs(cropland_FAO_sum) <= 0.003:
                print('Zero division encountered. |cropland_FAO_sum| {} <= 0.003. Set bias_factor 1'.
                      format(abs(cropland_FAO_sum)))
                scaling_factor_cropland = 1
            else:
                print('Zero division encountered. |cropland_FAO_sum| {} > 0.003. Set bias_factor np.nan'.
                      format(abs(cropland_FAO_sum)))
                scaling_factor_cropland = np.nan
        else:
            scaling_factor_cropland = cropland_FAO_sum / (
                    cropland_subnational_sum / Country.UNIT_LOOKUP[self.units])

        if pasture_subnational_sum == 0:
            # Zero division
            if abs(pasture_FAO_sum) <= 0.003:
                print('Zero division encountered. |pasture_FAO_sum| {} <= 0.003. Set bias_factor 1'.
                      format(abs(pasture_FAO_sum)))
                scaling_factor_pasture = 1
            else:
                print('Zero division encountered. |pasture_FAO_sum| {} > 0.003. Set bias_factor np.nan'.
                      format(abs(pasture_FAO_sum)))
                scaling_factor_pasture = np.nan
        else:
            scaling_factor_pasture = pasture_FAO_sum / (
                    pasture_subnational_sum / Country.UNIT_LOOKUP[self.units])

        return scaling_factor_cropland, scaling_factor_pasture

    def get_bias_corrected_dataset(self, bias_correct=False, convert_to_kha=True):
        """
        Apply bias correction to the current subnational data in its own units

        Args:
            bias_correct (bool): if True, apply bias_correction, otherwise *1
                                 (Default: False)
            convert_to_kha (bool): if True, also convert to kha, otherwise use
                                   class unit (Default: True)

        Returns: (tuple of pd) dataframe of cropland, dataframe of pasture
        """
        scaling_factor_cropland, \
        scaling_factor_pasture = self.get_bias_factor() if bias_correct else (1, 1)

        # Apply factor to the original attribute
        # use .copy() to avoid SettingWithCopyWarning from pandas
        cropland_data = self.get_subnational_cropland().copy()
        pasture_data = self.get_subnational_pasture().copy()

        cropland_data['CROPLAND'] = cropland_data['CROPLAND'].to_numpy() * scaling_factor_cropland
        pasture_data['PASTURE'] = pasture_data['PASTURE'].to_numpy() * scaling_factor_pasture

        if convert_to_kha:
            cropland_data['CROPLAND'] = Country.convert_to_kha(cropland_data['CROPLAND'], self.units)
            pasture_data['PASTURE'] = Country.convert_to_kha(pasture_data['PASTURE'], self.units)

        return cropland_data, pasture_data

    def merge_census_to_spatial(self, bias_correct=False, convert_to_kha=True):
        """
        Bias and unit corrected (if specified) subnational stats data merged on spatial map in kHa

        Args:
            bias_correct (bool): if True, apply bias_correction, otherwise *1
                                 (Default: False)
            convert_to_kha (bool): if True, also convert to kha, otherwise use
                                   class unit (Default: True)

        Returns: (pd) merged cropland and pasture census (with correction) with spatial_map
        """
        cropland_data, pasture_data = self.get_bias_corrected_dataset(bias_correct, convert_to_kha)
        merged_map = self.spatial_map.copy()
        merged_map = merged_map.merge(cropland_data, on='STATE', how='left')
        merged_map = merged_map.merge(pasture_data, on='STATE', how='left')
        return merged_map
