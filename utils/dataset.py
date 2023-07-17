import numpy as np
import pandas as pd
import h2o
from h2o.estimators import H2OKMeansEstimator
from scipy.spatial.distance import cdist


class Dataset:
    TRAIN_TYPE = 'train'
    DEPLOY_TYPE = 'deploy'

    def __init__(self, census_table, land_cover_code, remove_land_cover_feature_index, invalid_data='remove'):
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
            invalid_data (str): 'scale' or 'remove'. Default: 'remove'
        """
        # Dataset obj could be train set or deploy set
        # train set is defined as census table with CROPLAND / PASTURE / OTHER attributes
        # deploy set only contains <land cover type>
        assert (invalid_data in ['scale', 'remove']), "Unknown invalid_data handler"
        self.census_table = census_table
        self.land_cover_code = land_cover_code
        if all([i in self.census_table.columns for i in ['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']]):
            # Note: these outliers are the ones that have cropland and pasture % 
            #       sum over 100% due to back correction weights. We decided to 
            #       scale and include them
            if invalid_data == 'remove':
                self._remove_outliers() 
            elif invalid_data == 'scale':
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

    def to_multinomial_set(self, shuffle=False):
        """
        Convert census table from pd.DataFrame to np.ndarray format, where the labels are 
        prepared for multinomial classification in one step

        Note:
            (train set) Each sample is converted to a 3xN array stacked on top of one another
                          Input features | Label | Weights
                        <land cover type> | 0 | AREA * CROPLAND_PER
                        <land cover type> | 1 | AREA * PASTURE_PER
                        <land cover type> | 2 | AREA * OTHER_PER
            (deploy set) Each sample only has one record as follows
                         Row index | column index |   Input features  | Grid size
                             X     |       X      | <land cover type> |     X
        
        Args:
            shuffle (bool, optional): shuffle dataset. Defaults to False.

        Returns: (np.ndarray) dataset
        """
        num_samples = len(self.census_table)
        num_features = len(self.land_cover_code.keys())
        num_labels = 1  # single class output
        num_class = 3   # cropland, pasture, other
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

            if shuffle:
                np.random.shuffle(np_census_data)
            return np_census_data

        elif self.type == Dataset.DEPLOY_TYPE:
            #  Row index | column index | Input features | Grid size
            return self.census_table.to_numpy()

        else:
            raise ValueError('Unknown Dataset types')

    def to_bernoulli_set(self, shuffle=False):
        """
        Convert census table from pd.DataFrame to a collection of np.ndarray format, 
        where the labels are prepared for bernoulli classification in one-vs-rest 
        for each class

        Note:
            (train set) Each class and each sample is converted to a 2xN array stacked 
                        on top of one another
                                    'CROPLAND_PER'
                          Input features | Label | Weights
                        <land cover type> | 0 | AREA * CROPLAND_PER
                        <land cover type> | 1 | AREA * (1-CROPLAND_PER)
                                    'PASTURE_PER'
                        <land cover type> | 0 | AREA * PASTURE_PER
                        <land cover type> | 1 | AREA * (1-PASTURE_PER)
                                    'OTHER_PER'
                        <land cover type> | 0 | AREA * OTHER_PER
                        <land cover type> | 1 | AREA * (1-OTHER_PER)
            (deploy set) Each sample only has one record as follows
                         Row index | column index |   Input features  | Grid size
                             X     |       X      | <land cover type> |     X
        
        Args:
            shuffle (bool, optional): shuffle dataset. Defaults to False.

        Returns: (dict) dict of np.ndarray
        """
        num_samples = len(self.census_table)
        num_features = len(self.land_cover_code.keys())
        num_labels = 1  # single class output
        num_class = 2   # one vs the rest
        num_weights = 1  # area * percentage

        if self.type == Dataset.TRAIN_TYPE:
            # Input features | Label | Weights
            dataset = {}
            for _, p in enumerate(['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']):
                np_census_data = np.zeros((num_samples * num_class, num_features + num_weights + num_labels))
                for n in range(num_samples):
                    np_census_data[n * num_class:(n + 1) * num_class, 0:num_features] = self.census_table.iloc[n][
                        [i for i in list(self.land_cover_code.keys())]].to_numpy()  # feature columns (land cover)
                    np_census_data[n * num_class:(n + 1) * num_class, num_features] = np.asarray([0, 1]) # percentage values
                    np_census_data[n * num_class:(n + 1) * num_class, -1] = np.asarray([self.census_table.iloc[n]['AREA'] * self.census_table.iloc[n][p], \
                        self.census_table.iloc[n]['AREA'] * (1 - self.census_table.iloc[n][p])])  # weight columns
                
                if shuffle:
                    np.random.shuffle(np_census_data)
                dataset[p] = np_census_data
            
            return dataset
        
        elif self.type == Dataset.DEPLOY_TYPE:
            #  Row index | column index | Input features | Grid size
            return self.census_table.to_numpy()

        else:
            raise ValueError('Unknown Dataset types')

    def to_percentage_set(self, shuffle=False):
        """
        Convert census table from pd.DataFrame to a collection of np.ndarray format, 
        where the labels are prepared for regression of percentage values in each class

        Note:
            (train set) Each class and each sample is converted to a 2xN array stacked 
                        on top of one another
                                    'CROPLAND_PER'
                          Input features | Label | Weights
                        <land cover type> | 0.XX | AREA * CROPLAND_PER
                                    'PASTURE_PER'
                        <land cover type> | 0.XX | AREA * PASTURE_PER
                                    'OTHER_PER'
                        <land cover type> | 0.XX | AREA * OTHER_PER
            (deploy set) Each sample only has one record as follows
                         Row index | column index |   Input features  | Grid size
                             X     |       X      | <land cover type> |     X

        Args:
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        
        Returns: (dict) dict of np.ndarray
        """
        num_samples = len(self.census_table)
        num_features = len(self.land_cover_code.keys())
        num_labels = 1  # single class output
        num_class = 1   # numeric percentage 
        num_weights = 1  # area * percentage

        if self.type == Dataset.TRAIN_TYPE:
            # Input features | Label | Weights
            dataset = {}
            for _, p in enumerate(['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']):
                np_census_data = np.zeros((num_samples * num_class, num_features + num_weights + num_labels))
                for n in range(num_samples):
                    np_census_data[n * num_class:(n + 1) * num_class, 0:num_features] = self.census_table.iloc[n][
                        [i for i in list(self.land_cover_code.keys())]].to_numpy()  # feature columns (land cover)
                    np_census_data[n * num_class:(n + 1) * num_class, num_features] = self.census_table.iloc[n][p] # percentage values
                    np_census_data[n * num_class:(n + 1) * num_class, -1] = self.census_table.iloc[n]['AREA'] * self.census_table.iloc[n][p]  # weight columns
                
                if shuffle:
                    np.random.shuffle(np_census_data)
                dataset[p] = np_census_data
            
            return dataset

        elif self.type == Dataset.DEPLOY_TYPE:
            #  Row index | column index | Input features | Grid size
            return self.census_table.to_numpy()

        else:
            raise ValueError('Unknown Dataset types')

    def spatial_sampling(self, method, num_samples, masked_indices=[], center_index=None):
        """
        Sample data points in dataset with spatial aspects
        Method #1: 'uniform'
            Uniformly sample data points on the whole spatial region
        Method #2: 'blocked'
            Sample N (including center_index) data points that are closest to center_index in euclidean distances 

        Args:
            method (str): 'uniform' or 'blocked'
            num_samples (int): number of data points to be sampled
            masked_indices (list, optional): indices of data points to be masked out. Defaults to [].
            center_index (int, optional): center index in 'blocked' method. Defaults to None.

        Returns:
            list: list of indices in dataset that are selected
        """
        def get_centroid_coordinates(census_table):
            # Get centorid coordinates np array from census_table with geometry attribute
            centroid_coordinates = np.array(list(
                census_table['geometry'].apply(
                    lambda geom: np.array([geom.centroid.x, geom.centroid.y]))))
            return centroid_coordinates

        def uniform_spatial_sampling(census_table, num_samples, seed=38471):
            # Uniform spatial sampling
            # Cluster centroids in the dataset by N group, where N = num_samples
            # Randomly select a candidate from each group as part of selection
            centroid_coordinates = get_centroid_coordinates(census_table)

            h2o.init()
            h2o_coordinates = h2o.H2OFrame(centroid_coordinates, column_names=['x', 'y'])
            centroid_cluster = H2OKMeansEstimator(k=num_samples, seed=seed)
            centroid_cluster.train(training_frame=h2o_coordinates)

            cluster_assignments = centroid_cluster.predict(h2o_coordinates)
            assignment_class = cluster_assignments.as_data_frame().values
            selected_indices = [census_table.loc[np.random.choice(np.where(assignment_class == i)[0]), 'index_copy'] for i in range(num_samples)]
            return selected_indices

        def blocked_spatial_sampling(census_table, num_samples, center_index):
            # Block spatial sampling
            # Get top N sample points with shortest euclidean distance to center_index
            assert (center_index is not None), "Must specify center_index"
            centroid_coordinates = get_centroid_coordinates(census_table)

            distances = cdist(centroid_coordinates, centroid_coordinates)
            new_center_index = census_table.loc[census_table['index_copy'] == center_index].index[0]
            top_smallest_indices = np.argsort(distances[new_center_index, :])[:num_samples]
            selected_indices = [census_table.loc[i, 'index_copy'] for i in top_smallest_indices.tolist()]
            return selected_indices

        assert (method in ['uniform', 'blocked']), "method can only be either uniform or blocked"
        
        # Get masked census table for processing
        census_table_masked = self.census_table.copy()
        census_table_masked['index_copy'] = np.arange(0, census_table_masked.shape[0])
        census_table_masked = census_table_masked.drop(masked_indices)
        census_table_masked.reset_index(inplace=True, drop=True)
        assert (1 <= num_samples <= census_table_masked.shape[0]), "num_samples must be in [1, size of masked dataset]"

        if method == 'uniform':
            selected_indices = uniform_spatial_sampling(census_table_masked, num_samples)
            
        elif method == 'blocked':
            assert (center_index not in masked_indices), "center_index cannot be included in masked_indices"
            selected_indices = blocked_spatial_sampling(census_table_masked, num_samples, center_index)

        return selected_indices


