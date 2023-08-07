import h2o
import os
from h2o.estimators import H2OGradientBoostingEstimator
from utils.dataset import *
from utils.tools.visualizer import *
from utils.io import load_pkl, save_pkl


class OvRBernoulliGradientBoostingTree: 

    def __init__(self, 
                 ntrees=50, 
                 max_depth=10, 
                 nclasses=3, 
                 nfolds=10):
        """
        Constructor that initializes a OvR bernoulli gradient boosting tree model

        Note:
            A OvR bernoulli gradient boosting tree model is basically K independently 
            trained bernoulli GBT models fit on each of the K classes under one-vs-rest,
            followed by scaling or softmax

        Args:
            ntrees (int): number of trees (Default: 50)
            max_depth (int): max. depth of tree (Default: 10)
            nclasses (int): number of classes for one-vs-rest (Default: 3)
            nfolds (int): number of folds for cross validation (Default: 10)
        """
        os.environ["JAVA_ARGS"] = "-ea:"
        h2o.init(enable_assertions=False)  # to prevent over aggressive warnings
        self.model = {}
        self.model_id = [f'class_{int(i)}' for i in range(nclasses)]
        for p in self.model_id:
            current_model = H2OGradientBoostingEstimator(
                nfolds=nfolds,
                seed=1111,
                keep_cross_validation_predictions=True,
                weights_column='WEIGHTS',
                distribution='bernoulli',
                ntrees=ntrees,
                max_depth=max_depth)
            self.model[p] = current_model

    def _normalize_prediction(self, pred_results, normalization_method='scale'):
        """
        Normalization of the prediction outputs so that the prediction 
        values follow a probability distribution (scale/softmax)

        Args:
            pred_df (dict of np.ndarray): prediction
            normalization_method (str): 'scale' or 'softmax'. Defaults to 'scale'
        """
        for i, id in enumerate(pred_results):
            
            if normalization_method == 'scale':
                # Take the sum 
                if i == 0:
                    normalization_denominator = pred_results[id]  # ref of obj
                else:
                    # Note: must use "A=A+B" in this case, because in the first iteration 
                    #       A is a reference of B, when A gets updated using "A+=B", no
                    #       intermediate obj is created, when ref gets updated, B gets updated 
                    #       too. "A=A+B" creates intermediate new obj
                    normalization_denominator = normalization_denominator +  pred_results[id]
            
            elif normalization_method == 'softmax':
                # Take the sum of exp
                if i == 0:
                    normalization_denominator = np.exp(pred_results[id])
                else:
                    normalization_denominator = normalization_denominator + np.exp(pred_results[id])
        
        if normalization_method == 'scale':
            for i, id in enumerate(pred_results):
                pred_results[id] = pred_results[id] / normalization_denominator
        
        elif normalization_method == 'softmax':
            for i, id in enumerate(pred_results):
                pred_results[id] = np.exp(pred_results[id]) / normalization_denominator

        return pred_results

    def _convert_input_to_H2OFrame(self, census_data, shuffle=False):
        """
        Process input census_data Dataset obj to H2OFrame for model

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.

        Returns: (dict or H2OFrame)
        """
        if census_data.type == Dataset.TRAIN_TYPE:
            np_census_data = census_data.to_bernoulli_set(shuffle)
            h2o_census_data_collection = []
            for d in ['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']:
                pd_census_data = pd.DataFrame(np_census_data[d],
                                              columns=list(census_data.land_cover_code.values()) + ['CATEGORY', 'WEIGHTS'])
                h2o_census_data = h2o.H2OFrame(pd_census_data, column_types={'CATEGORY':'categorical', 'WEIGHTS':'numeric'})
                h2o_census_data['CATEGORY'] = h2o_census_data['CATEGORY'].asfactor()
                
                h2o_census_data_collection.append(h2o_census_data)
            
            return h2o_census_data_collection

        elif census_data.type == Dataset.DEPLOY_TYPE:
            pd_census_data = census_data.census_table.rename(columns=census_data.land_cover_code, inplace=False)
            h2o_census_data = h2o.H2OFrame(pd_census_data)

            return h2o_census_data

        else:
            raise ValueError('Unknown Dataset types')

    def train(self, census_data, shuffle=False):
        """
        Train model on input census_data

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        h2o_frame_input = self._convert_input_to_H2OFrame(census_data, shuffle)
        for i, id in enumerate(self.model_id):
            self.model[id].train(x=list(census_data.land_cover_code.values()),
                    y='CATEGORY',
                    training_frame=h2o_frame_input[i])

    def predict(self, census_data, normalization_method='scale'):
        """
        Predict output probability distribution using model

        Args:
            census_data (Dataset): census dataset
            normalization_method (str): 'scale' or 'softmax'. Defaults to 'scale'

        Returns: (pd.DataFrame) with attributes p0 | p1 | p2
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        assert (normalization_method in ['scale', 'softmax']), "Unknown normalization method"
        pred_results = {}
        h2o_frame_input = self._convert_input_to_H2OFrame(census_data, False)
        for i, id in enumerate(self.model_id):
            pred_results[id] = self.model[id].predict(
                h2o_frame_input).as_data_frame().to_numpy()

        # Take column 1 in each predictor
        for i in pred_results:
            pred_results[i] = pred_results[i][:, 1]
        
        # Normalize output results
        pred_results = self._normalize_prediction(pred_results, normalization_method)

        return pd.DataFrame.from_dict(pred_results)

    def evaluate(self, census_data, normalization_method='scale'):
        """
        Evaluate model on census_data (with ground truth). Output predicition results 
        and ground truth tuple

        Args:
            census_data (Dataset): census dataset
            normalization_method (str): 'scale' or 'softmax'. Defaults to 'scale'
        
        Returns: (tuple of np.ndarray) pred, ground_truth
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        assert (census_data.type == Dataset.TRAIN_TYPE
                ), "Input census_data must have ground truth"

        # pred_outputs = self.predict(census_data).to_numpy()
        pred_outputs = {}
        h2o_frame_input = self._convert_input_to_H2OFrame(census_data, False)
        for i, id in enumerate(self.model_id):
            pred_outputs[id] = self.model[id].predict(
                h2o_frame_input[i]).as_data_frame().to_numpy()

        # Take column 1 in each predictor
        for i in pred_outputs:
            pred_outputs[i] = pred_outputs[i][:, 1]
        
        # Normalize output results
        pred_outputs = self._normalize_prediction(pred_outputs, normalization_method)

        # Note that the ground truth might not be "correct", since
        # there could be invalid data samples where coverage is over 100%
        ground_truth = census_data.census_table[[
            'CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER'
        ]].to_numpy()
        
        pred_results = np.zeros_like(ground_truth)
        for i, id in enumerate(self.model_id):
            pred_results[:, i] = pred_outputs[id][::2]

        return pred_results, ground_truth

    def save(self, output_dir):
        """
        Save GBT model

        Note: 
            A OvRBernoulliGradientBoostingTree obj has multiple models that are saved 
            under the input output_dir, with model_id as child dir and model param saved
            under their child dir

        Args:
            output_dir (str): output dir to save model parameters
        """
        # Need to save K models for K classes
        for i, id in enumerate(self.model_id):
            model_path = h2o.save_model(model=self.model[id],
                                        path=os.path.join(output_dir, '{}'.format(id)),
                                        force=True)
            print('{} is saved'.format(model_path))

    def load(self, model_path):
        """
        Load GBT model parameters (in order with model_path)

        Args:
            model_path (str): model path to be loaded
        """
        self.model_id = [i for i in os.listdir(model_path) if '.' not in i]
        for id in self.model_id:
            self.model[id] = h2o.load_model(os.path.join(model_path, id, os.listdir(os.path.join(model_path, id))[0]))
        print('Model parameters successfully loaded from {}'.format(model_path))


class RegressionGradientBoostingTree:

    def __init__(self, 
                 ntrees=50, 
                 max_depth=10, 
                 nclasses=3, 
                 nfolds=10):
        """
        Constructor that initializes a regression gradient boosting tree model

        Note:
            A regression gradient boosting tree model is a naive classifier that predicts 
            the percentage values in each class directly, followed by clamping the prediction 
            probability to [0, 1] and scaling. This is not a recommanded way to do classification  

        Args:
            ntrees (int): number of trees (Default: 50)
            max_depth (int): max. depth of tree (Default: 10)
            nclasses (int): number of classes for one-vs-rest (Default: 3)
            nfolds (int): number of folds for cross validation (Default: 10)
        """
        os.environ["JAVA_ARGS"] = "-ea:"
        h2o.init(enable_assertions=False)  # to prevent over aggressive warnings
        self.model = {}
        self.model_id = [f'class_{int(i)}' for i in range(nclasses)]
        for p in self.model_id:
            current_model = H2OGradientBoostingEstimator(
                nfolds=nfolds,
                seed=1111,
                keep_cross_validation_predictions=True,
                weights_column='WEIGHTS',
                distribution='AUTO',
                ntrees=ntrees,
                max_depth=max_depth)
            self.model[p] = current_model

    def _normalize_prediction(self, pred_results, normalization_method='scale'):
        """
        Normalization of the prediction outputs so that the prediction 
        values follow a probability distribution (Clamp + scale/softmax)

        Args:
            pred_df (dict of np.ndarray): prediction array
            normalization_method (str): 'scale' or 'softmax'. Defaults to 'scale'
        
        Returns: (dict of np.ndarray)
        """
        for i, id in enumerate(pred_results):
            current_pred_results = pred_results[id]
            current_pred_results[current_pred_results > 1] = 1
            current_pred_results[current_pred_results < 0] = 0
            pred_results[id] = current_pred_results
            
            if normalization_method == 'scale':
                # Take the sum 
                if i == 0:
                    normalization_denominator = pred_results[id]
                else:
                    normalization_denominator = normalization_denominator + pred_results[id]
            
            elif normalization_method == 'softmax':
                # Take the sum of exp
                if i == 0:
                    normalization_denominator = np.exp(pred_results[id])
                else:
                    normalization_denominator = normalization_denominator + np.exp(pred_results[id])
        
        if normalization_method == 'scale':
            for i, id in enumerate(pred_results):
                pred_results[id] = pred_results[id] / normalization_denominator
        
        elif normalization_method == 'softmax':
            for i, id in enumerate(pred_results):
                pred_results[id] = np.exp(pred_results[id]) / normalization_denominator

        return pred_results

    def _convert_input_to_H2OFrame(self, census_data, shuffle=False):
        """
        Process input census_data Dataset obj to H2OFrame for model

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.

        Returns: (dict or H2OFrame)
        """
        if census_data.type == Dataset.TRAIN_TYPE:
            np_census_data = census_data.to_percentage_set(shuffle)
            h2o_census_data_collection = []
            for d in ['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']:
                pd_census_data = pd.DataFrame(np_census_data[d],
                                              columns=list(census_data.land_cover_code.values()) + ['CATEGORY', 'WEIGHTS'])
                h2o_census_data = h2o.H2OFrame(pd_census_data, column_types={'CATEGORY':'numeric', 'WEIGHTS':'numeric'})
                
                h2o_census_data_collection.append(h2o_census_data)
            
            return h2o_census_data_collection

        elif census_data.type == Dataset.DEPLOY_TYPE:
            pd_census_data = census_data.census_table.rename(columns=census_data.land_cover_code, inplace=False)
            h2o_census_data = h2o.H2OFrame(pd_census_data)

            return h2o_census_data

        else:
            raise ValueError('Unknown Dataset types')

    def train(self, census_data, shuffle=False):
        """
        Train model on input census_data

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        h2o_frame_input = self._convert_input_to_H2OFrame(census_data, shuffle)
        for i, id in enumerate(self.model_id):
            self.model[id].train(x=list(census_data.land_cover_code.values()),
                    y='CATEGORY',
                    training_frame=h2o_frame_input[i])

    def predict(self, census_data, normalization_method='scale'):
        """
        Predict output probability distribution using model

        Args:
            census_data (Dataset): census dataset
            normalization_method (str): 'scale' or 'softmax'. Defaults to 'scale'

        Returns: (pd.DataFrame) with attributes p0 | p1 | p2
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        assert (normalization_method in ['scale', 'softmax']), "Unknown normalization method"
        pred_results = {}
        h2o_frame_input = self._convert_input_to_H2OFrame(census_data, False)
        for i, id in enumerate(self.model_id):
            pred_results[id] = self.model[id].predict(
                h2o_frame_input).as_data_frame().to_numpy()

        # Take column 0 in each predictor
        for i in pred_results:
            pred_results[i] = pred_results[i][:, 0]

        # Clamp and normalize output results
        pred_results = self._normalize_prediction(pred_results, normalization_method)

        return pd.DataFrame.from_dict(pred_results)

    def evaluate(self, census_data):
        """
        Evaluate model on census_data (with ground truth). Output predicition results 
        and ground truth tuple

        Args:
            census_data (Dataset): census dataset
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        assert (census_data.type == Dataset.TRAIN_TYPE
                ), "Input census_data must have ground truth"

        pred_outputs = self.predict(census_data).to_numpy()
        ground_truth = census_data.census_table[[
            'CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER'
        ]].to_numpy()
        return pred_outputs, ground_truth

    def save(self, output_dir):
        """
        Save GBT model

        Note: 
            A RegressionGradientBoostingTree obj has multiple models that are saved 
            under the input output_dir, with model_id as child dir and model param saved
            under their child dir

        Args:
            output_dir (str): output dir to save model parameters
        """
        # Need to save K models for K classes
        for i, id in enumerate(self.model_id):
            model_path = h2o.save_model(model=self.model[id],
                                        path=os.path.join(output_dir, '{}'.format(id)),
                                        force=True)
            print('{} is saved'.format(model_path))

    def load(self, model_path):
        """
        Load GBT model parameters (in order with model_path)

        Args:
            model_path (str): model path to be loaded
        """
        self.model_id = [i for i in os.listdir(model_path) if '.' not in i]
        for id in self.model_id:
            self.model[id] = h2o.load_model(os.path.join(model_path, id, os.listdir(os.path.join(model_path, id))[0]))
        print('Model parameters successfully loaded from {}'.format(model_path))


class MultinomialGradientBoostingTree:

    def __init__(self,
                 ntrees=50,
                 max_depth=10,
                 nfolds=10, 
                 min_rows=5, 
                 learn_rate=0.05, 
                 sample_rate=1.0, 
                 col_sample_rate=0.5):
        """
        Constructor that initializes a multinomial gradient boosting tree model

        Note:
            A multinomial gradient boosting tree model is basically K GBT models fit 
            on K classes followed by a softmax to the output predictions 

        Args:
            ntrees (int): number of trees (Default: 50)
            max_depth (int): max. depth of tree (Default: 10)
            nfolds (int): number of folds for cross validation (Default: 10)
            min_rows (int): min rows (Default: 5)
            learn_rate (float): learning rate (Default: 0.05)
            sample_rate (float): sampling rate (Default: 1.0)
            col_sample_rate (float): column sampling rate (Default: 0.5)
        """
        h2o.init()
        self.model = H2OGradientBoostingEstimator(
            nfolds=nfolds,
            seed=1111,
            keep_cross_validation_predictions=True,
            weights_column='WEIGHTS',
            distribution='multinomial',
            ntrees=ntrees,
            max_depth=max_depth, 
            min_rows=min_rows, 
            learn_rate=learn_rate, 
            sample_rate=sample_rate, 
            col_sample_rate=col_sample_rate)

    def _convert_input_to_H2OFrame(self, census_data, shuffle=False):
        """
        Process input census_data Dataset obj to H2OFrame for model

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.

        Returns: (H2OFrame)
        """
        if census_data.type == Dataset.TRAIN_TYPE:
            np_census_data = census_data.to_multinomial_set(shuffle)
            pd_census_data = pd.DataFrame(np_census_data,
                                          columns=list(census_data.land_cover_code.values()) + ['CATEGORY', 'WEIGHTS'])
            h2o_census_data = h2o.H2OFrame(pd_census_data)
            h2o_census_data['CATEGORY'] = h2o_census_data['CATEGORY'].asfactor()

        elif census_data.type == Dataset.DEPLOY_TYPE:
            pd_census_data = census_data.census_table.rename(columns=census_data.land_cover_code, inplace=False)
            h2o_census_data = h2o.H2OFrame(pd_census_data)

        else:
            raise ValueError('Unknown Dataset types')

        return h2o_census_data

    def train(self, census_data, shuffle=False):
        """
        Train model on input census_data

        Args:
            census_data (Dataset): census dataset
            shuffle (bool, optional): shuffle dataset. Defaults to False.
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"

        self.model.train(x=list(census_data.land_cover_code.values()),
                         y='CATEGORY',
                         training_frame=self._convert_input_to_H2OFrame(census_data, shuffle))

    def predict(self, census_data):
        """
        Predict output probability distribution using model

        Args:
            census_data (Dataset): census dataset

        Returns: (pd.DataFrame) with attributes p0 | p1 | p2
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        pred_results = self.model.predict(
            self._convert_input_to_H2OFrame(census_data, False)).as_data_frame()

        # for train set census_data, each input sample has 3 replicated copies
        # for cropland, pasture and other
        if census_data.type == Dataset.TRAIN_TYPE:
            pred_results = pred_results.iloc[::3]

        return pred_results.iloc[:, 1:]

    def evaluate(self, census_data):
        """
        Evaluate model on census_data (with ground truth). Output predicition results 
        and ground truth tuple

        Args:
            census_data (Dataset): census dataset
        """
        assert (isinstance(census_data,
                           Dataset)), "Input census_data must be a Dataset obj"
        assert (census_data.type == Dataset.TRAIN_TYPE
                ), "Input census_data must have ground truth"

        pred_outputs = self.predict(census_data).to_numpy()[:, -3:]
        ground_truth = census_data.census_table[[
            'CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER'
        ]].to_numpy()
        return pred_outputs, ground_truth

    def save(self, output_dir):
        """
        Save GBT model

        Args:
            output_dir (str): output dir to save model parameters
        """
        model_path = h2o.save_model(model=self.model,
                                    path=output_dir,
                                    force=True)
        print('{} is saved'.format(model_path))

    def load(self, model_path):
        """
        Load GBT model parameters

        Args:
            model_path (str): model path to be loaded
        """
        self.model = h2o.load_model(os.path.join(model_path, os.listdir(model_path)[0]))
        print(
            'Model parameters successfully loaded from {}'.format(model_path))


class OvRBernoulliGradientBoostingTreeWithCustomCV(OvRBernoulliGradientBoostingTree):
    def __init__(self, 
                 ntrees=[50]*3, 
                 max_depth=[10]*3, 
                 nclasses=3):
        os.environ["JAVA_ARGS"] = "-ea:"
        h2o.init(enable_assertions=False)  # to prevent over aggressive warnings
        h2o.no_progress()
        self.model = None
        self.reset_hyper_params(ntrees, max_depth, nclasses)
    
    def reset_hyper_params(self, ntrees, max_depth, nclasses):
        assert(len(ntrees) == nclasses), "length of ntrees must match nclasses"
        assert(len(max_depth) == nclasses), "length of max_depth must match nclasses"
        if self.model is not None:
            h2o.remove(self.model)
        self.model = {}
        self.model_id = [f'class_{int(i)}' for i in range(nclasses)]
        for i, p in enumerate(self.model_id):
            current_model = H2OGradientBoostingEstimator(
                nfolds=0,
                weights_column='WEIGHTS',
                distribution='bernoulli',
                ntrees=ntrees[i],
                max_depth=max_depth[i], 
                min_rows=50)
            self.model[p] = current_model


class MultinomialGradientBoostingTreeWithCustomCV(MultinomialGradientBoostingTree):
    def __init__(self, 
                 ntrees=50, 
                 max_depth=10,
                 min_rows=10, 
                 learn_rate=0.3, 
                 sample_rate=1.0, 
                 col_sample_rate=0.5, 
                 balance_classes=False
                 ):
        os.environ["JAVA_ARGS"] = "-ea:"
        h2o.init(enable_assertions=False)  # to prevent over aggressive warnings
        h2o.no_progress()
        self.model = None
        self.reset_hyper_params(
            ntrees,
            max_depth, 
            min_rows, 
            learn_rate, 
            sample_rate, 
            col_sample_rate, 
            balance_classes
            )
    
    def reset_hyper_params(self, 
                           ntrees=50, 
                           max_depth=10,
                           min_rows=10, 
                           learn_rate=0.3, 
                           sample_rate=1.0, 
                           col_sample_rate=0.5, 
                           balance_classes=False
                           ):
        if self.model is not None:
            h2o.remove(self.model)
        self.model = H2OGradientBoostingEstimator(
            nfolds=0,
            seed=1111,
            keep_cross_validation_predictions=False,
            weights_column='WEIGHTS',
            distribution='multinomial',
            ntrees=ntrees,
            max_depth=max_depth, 
            min_rows=min_rows, 
            learn_rate=learn_rate, 
            sample_rate=sample_rate, 
            col_sample_rate=col_sample_rate, 
            balance_classes=balance_classes)

