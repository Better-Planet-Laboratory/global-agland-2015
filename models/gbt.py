import h2o
from h2o.estimators import H2OGradientBoostingEstimator
from utils.dataset import *
from utils.tools.visualizer import *


class GradientBoostingTree:

    def __init__(self, ntrees, max_depth, nfolds=10, distribution='multinomial'):
        """
        Constructor that initializes a gradient boosting tree model

        Args:
            ntrees (int): number of trees
            max_depth (int): max. depth of tree
            nfolds (int): number of folds for cross validation (Default: 10)
            distribution (str): distribution for categorical label (Default: 'multinomial')
        """
        h2o.init()
        self.model = H2OGradientBoostingEstimator(nfolds=nfolds,
                                                  seed=1111,
                                                  keep_cross_validation_predictions=True,
                                                  weights_column='WEIGHTS',
                                                  distribution=distribution,
                                                  ntrees=ntrees,
                                                  max_depth=max_depth
                                                  )

    def train(self, census_data):
        """
        Train model on input census_data

        Args:
            census_data (Dataset): census dataset
        """
        assert (isinstance(census_data, Dataset)), "Input census_data must be a Dataset obj"

        self.model.train(x=list(census_data.land_cover_code.values()),
                         y='CATEGORY',
                         training_frame=census_data.to_H2OFrame())

    def predict(self, census_data):
        """
        Predict output probability distribution using model

        Args:
            census_data (Dataset): census dataset

        Returns: (pd.DataFrame) with attributes p0 | p1 | p2
        """
        assert (isinstance(census_data, Dataset)), "Input census_data must be a Dataset obj"
        pred_results = self.model.predict(census_data.to_H2OFrame()).as_data_frame()

        # for train set census_data, each input sample has 3 replicated copies
        # for cropland, pasture and other
        if census_data.type == Dataset.TRAIN_TYPE:
            pred_results = pred_results.iloc[::3]

        return pred_results

    def evaluate(self, census_data, output_dir=None):
        """
        Evaluate model on census_data (with ground truth). Output pred vs. ground_truth
        plots for CROPLAND, PASTURE, OTHER that are saved in output_dir with RMSE

        Args:
            census_data (Dataset): census dataset
            output_dir (str): output dir (Default: None)
        """
        assert (isinstance(census_data, Dataset)), "Input census_data must be a Dataset obj"
        assert (census_data.type == Dataset.TRAIN_TYPE), "Input census_data must have ground truth"

        pred_outputs = self.predict(census_data).to_numpy()[:, -3:]
        ground_truth = census_data.census_table[['CROPLAND_PER', 'PASTURE_PER', 'OTHER_PER']].to_numpy()
        plot_agland_pred_vs_ground_truth(ground_truth, pred_outputs, output_dir=output_dir)

    def save(self, output_dir):
        """
        Save GBT model

        Args:
            output_dir (str): output dir to save model parameters
        """
        model_path = h2o.save_model(model=self.model,
                                    path=output_dir, force=True)
        print('{} is saved'.format(model_path))

    def load(self, model_path):
        """
        Load GBT model parameters

        Args:
            model_path (str): model path to be loaded
        """
        self.model = h2o.load_model(model_path)
        print('Model parameters successfully loaded from {}'.format(model_path))
