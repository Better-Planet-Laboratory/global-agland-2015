import numpy as np
import pandas as pd


class FAOSTAT:

    def __init__(self, arg):
        """
        Constructor that loads the FAOSTAT_dir

        Args:
            arg (str): path dir to FAOSTAT file (Default: csv)
                (pd): pd dataframe that contains FAOSTAT data
        """
        if isinstance(arg, str):
            self.data = pd.read_csv(arg)
        elif isinstance(arg, pd.DataFrame):
            self.data = arg
        else:
            raise ValueError("Input arg must be either str or pd.DataFrame")

        self.country_set = set(self.data['Area'])

    def __len__(self):
        return len(self.data.index)

    def get_by_country(self, country_name):
        """
        Get subset of FAOSTAT based on input country_name

        Args:
            country_name (str): country name to be extracted

        Returns: (FAOSTAT) dataframe that only contains country_name rows
        """
        return FAOSTAT(self.data.loc[(self.data['Area'] == country_name)])

    def mean(self, cropland_attributes=['Arable land', 'Land under permanent crops'],
             pasture_attributes=['Land under perm. meadows and pastures']):
        """
        Get the mean in FAOSTAT averaged over n years for all countries

        Args:
            cropland_attributes (list of str): list of attributes str for cropland
            pasture_attributes (list of str): list of attributes str for pasture

        Returns: (tuple) cropland_mean, pasture_mean
        """
        num_year = pd.unique(self.data['Year']).size
        cropland_nation_total = np.nansum(self.data.loc[(self.data['Item'].isin(cropland_attributes))]
                                          ['Value'].to_numpy()) / num_year
        pasture_nation_total = np.nansum(self.data.loc[(self.data['Item'].isin(pasture_attributes))]
                                         ['Value'].to_numpy()) / num_year

        return cropland_nation_total, pasture_nation_total

