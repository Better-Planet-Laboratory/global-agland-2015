import numpy as np
import pandas as pd
import h2o


class Dataset:
    TRAIN_TYPE = 'train'
    DEPLOY_TYPE = 'deploy'

    def __init__(self, census_table, land_cover_code, remove_land_cover_feature_index):
        """
        Constructor that loads pd.DataFrame of census table as dataset ready for training or deploy

        Args:
            census_table (pd.DataFrame): (train set) census table that has attributes
                                             <land cover type>, AREA, CROPLAND_PER,
                                             PASTURE_PER, OTHER_PER, REGIONS, STATE,
                                             geometry
                                         (deploy set) census table that has attributes
                                             ROW_IDX, COL_IDX, <land cover type>, GRID_SIZE
            land_cover_code (dict): class types(int) -> (str)
            remove_land_cover_feature_index (list): index of land cover type code to be
                                                    removed from features
        """
        # Dataset obj could be train set or deploy set
        # train set is defined as census table with CROPLAND / PASTURE / OTHER attributes
        # deploy set only contains <land cover type>
        self.census_table = census_table
        self.land_cover_code = land_cover_code
        if all([i in self.census_table.columns for i in ['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']]):
            # Note: these outliers are the ones that have cropland and pasture % 
            #       sum over 100% due to back correction weights. We decided to 
            #       scale and include them
            # self._remove_outliers() 
            self._scale_outliers() 
            self.type = Dataset.TRAIN_TYPE
        else:
            self.type = Dataset.DEPLOY_TYPE
        self.remove_land_cover_features(set(remove_land_cover_feature_index))

        assert(all([i in self.census_table.columns for i in self.land_cover_code.keys()])), \
            "Input census table does not contain land cover type keys"

    def __len__(self):
        """ Number of samples in the dataset """
        return len(self.census_table)
    
    def _scale_outliers(self):
        """
        Linearly scale outliers from census table. 
        scaling_factor = (Cropland % + Pasture %)
        Cropland % /= scaling_factor
        Pasture % /= scaling_factor
        Other % = 0

        Note:
            samples with OTHER_PER < 0 are considered as outliers. This is likely caused by
            invalid cropland and pasture values
        """
        outliers_index = self.census_table.index[self.census_table['OTHER_PER'] < 0].to_list()
        
        print('Scale outliers: {}'.format(self.census_table.iloc[outliers_index]['STATE'].to_list()))
        scaling_factor = self.census_table.iloc[outliers_index]['CROPLAND_PER']+self.census_table.iloc[outliers_index]['PASTURE_PER']
        self.census_table.loc[outliers_index, 'CROPLAND_PER'] /= scaling_factor
        self.census_table.loc[outliers_index, 'PASTURE_PER'] /= scaling_factor
        self.census_table.loc[outliers_index, 'OTHER_PER'] = 0
        
    def _remove_outliers(self):
        """
        Remove outliers from census table

        Note:
            samples with OTHER_PER < 0 are considered as outliers. This is likely caused by
            invalid cropland and pasture values
        """
        outliers_index = self.census_table.index[self.census_table['OTHER_PER'] < 0].to_list()

        print('Remove outliers: {}'.format(self.census_table.iloc[outliers_index]['STATE'].to_list()))
        self.census_table = self.census_table.drop(outliers_index, inplace=False)
        self.census_table = self.census_table.reset_index(drop=True)

    def remove_land_cover_features(self, rm_feature_index_set):
        """
        Remove land cover features based on input index set from census table. Update
        land_cover_code

        Note:
            Since land cover features are histogram percentage computed using all land cover
            indices, removing a feature (or multiple ones) requires applying a factor of
            1/(1-sum(rm_feature percentage)) for each sample, besides removing the corresponding
            feature columns. For example, suppose we only have 4 land cover features,

            * remove_land_cover_feature_index = [3, 4]

            Current:
            [class 1] 0.1
            [class 2] 0.2
            [class 3] 0.3
            [class 4] 0.4
            ^============ sum up to 1

            After:
            [class 1] 0.1 / (1 - (0.3 + 0.4))
            [class 2] 0.2 / (1 - (0.3 + 0.4))
            [class 3] -
            [class 4] -
            ^============ sum up to 1

        Args:
            rm_feature_index_set (set): index of land cover type code to be
                                        removed from features
        """
        if len(rm_feature_index_set) != 0:
            factor = self.census_table[rm_feature_index_set].sum(axis=1)
            for i in rm_feature_index_set:
                # Remove from land_cover_code
                self.land_cover_code.pop(i, None)

            # Apply factor
            for i in self.land_cover_code.keys():
                self.census_table[i] = self.census_table[i] / (1 - factor)

            # Remove features from census table
            self.census_table = self.census_table.drop(list(rm_feature_index_set), axis=1)

    def to_numpy(self):
        """
        Convert census table from pd.DataFrame to np.array format

        Note:
            (train set) Each sample is converted to a 3xN array stacked on top of one another
                          Input features | Label | Weights
                        <land cover type> | 0 | AREA * CROPLAND_PER
                        <land cover type> | 1 | AREA * PASTURE_PER
                        <land cover type> | 2 | AREA * OTHER_PER
            (deploy set) Each sample only has one record as follows
                         Row index | column index |   Input features  | Grid size
                             X     |       X      | <land cover type> |     X

        Returns: (np.array) dataset
        """
        num_samples = len(self.census_table)
        num_features = len(self.land_cover_code.keys())
        num_labels = 1  # single class output
        num_class = 3  # cropland, pasture, other
        num_weights = 1  # area * percentage

        if self.type == Dataset.TRAIN_TYPE:
            # Input features | Label | Weights
            np_census_data = np.zeros((num_samples * num_class, num_features + num_weights + num_labels))
            for n in range(num_samples):
                np_census_data[n * num_class:(n + 1) * num_class, 0:num_features] = self.census_table.iloc[n][
                    [i for i in list(self.land_cover_code.keys())]].to_numpy()  # feature columns (land cover)
                np_census_data[n * num_class:(n + 1) * num_class, num_features] = np.asarray(
                    [i for i in range(num_class)])  # class label [0,1,2,...]
                np_census_data[n * num_class:(n + 1) * num_class, -1] = np.asarray(
                    [self.census_table.iloc[n]['AREA'] * self.census_table.iloc[n][p] for p in
                     ['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']])  # weight columns

            return np_census_data

        elif self.type == Dataset.DEPLOY_TYPE:
            #  Row index | column index | Input features | Grid size
            return self.census_table.to_numpy()

        else:
            raise ValueError('Unknown Dataset types')

    def to_H2OFrame(self):
        """
        Convert census table from pd.DataFrame to H2OFrame format

        Note:
            (train set) H2OFrame will contain the following attributes, where CATEGORY is
                        categorical label column
                            <land cover type> | CATEGORY | WEIGHTS
            (deploy set) H2OFrame will share the same structure as input census table

        Returns: (h2o.H2OFrame) dataset
        """
        if self.type == Dataset.TRAIN_TYPE:
            np_census_data = self.to_numpy()
            pd_census_data = pd.DataFrame(np_census_data,
                                          columns=list(self.land_cover_code.values()) + ['CATEGORY', 'WEIGHTS'])
            h2o_census_data = h2o.H2OFrame(pd_census_data)
            h2o_census_data['CATEGORY'] = h2o_census_data['CATEGORY'].asfactor()

        elif self.type == Dataset.DEPLOY_TYPE:
            pd_census_data = self.census_table.rename(columns=self.land_cover_code, inplace=False)
            h2o_census_data = h2o.H2OFrame(pd_census_data)

        else:
            raise ValueError('Unknown Dataset types')

        return h2o_census_data
