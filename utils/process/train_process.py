from utils.dataset import *
from utils.tools.census_core import load_census_table_pkl
from models import gbt
import h2o


def pipeline(training_cfg, land_cover_cfg):
    """
    Training pipeline:
    1. Load input dataset for training from census output
    2. If model parameters are specified in config, load model weights,
       otherwise train on input dataset from scratch
    3. Save (direct) evaluation on model performance (for tuning or other references)

    Args:
        training_cfg (dict): training settings from yaml
        land_cover_cfg (dict): land cover settings from yaml

    Returns: (MultinomialGradientBoostingTree)
    """
    # Load census_table input as Dataset obj
    input_dataset = Dataset(
        census_table=load_census_table_pkl(
            training_cfg['path_dir']['inputs']['census_table_input']),
        land_cover_code=land_cover_cfg['code']['MCD12Q1'],
        remove_land_cover_feature_index=training_cfg['feature_remove'],
        invalid_data=training_cfg['invalid_data_handle'])

    # Declare model structure
    prob_est = gbt.MultinomialGradientBoostingTree(
        ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
        max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
        nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'])

    # Load model weights if specified, otherwise train on input_dataset
    if training_cfg['path_dir']['inputs']['load_model'] is not None:
        try:
            prob_est.load(training_cfg['path_dir']['inputs']['load_model'])
            print('Model loaded from {}'.format(
                training_cfg['path_dir']['inputs']['load_model']))
        except h2o.exceptions.H2OResponseError:
            raise h2o.exceptions.H2OResponseError(
                'File {} is not valid model path.'.format(
                    training_cfg['path_dir']['inputs']['load_model']))

    else:
        print('No parameters to load. Start training ...')
        prob_est.train(input_dataset)
        prob_est.save(training_cfg['path_dir']['outputs']['save_model'])
        print('Model saved at {}'.format(
            training_cfg['path_dir']['outputs']['save_model']))

        # Evaluate the model on the whole input dataset
        # Note: This generated raw prediction on the state level census table, therefore
        #       it shall directly reflect the training performance
        # _, _ = prob_est.evaluate(input_dataset)
