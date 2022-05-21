from utils import io
from utils.tools.census_core import load_census_table_pkl
from utils.dataset import *
from models import gbt
import os
import h2o

TRAINING_CFG = io.load_yaml_config('configs/training_cfg.yaml')
LAND_COVER_CFG = io.load_yaml_config('configs/land_cover_cfg.yaml')

# Load census_table input as Dataset
input_dataset = Dataset(census_table=load_census_table_pkl(TRAINING_CFG['path_dir']['census_table_input']),
                        land_cover_code=LAND_COVER_CFG['code']['MCD12Q1'],
                        remove_land_cover_feature_index=TRAINING_CFG['feature_remove'])

# Declare model
prob_est = gbt.GradientBoostingTree(ntrees=TRAINING_CFG['model']['gradient_boosting_tree']['ntrees'],
                                    max_depth=TRAINING_CFG['model']['gradient_boosting_tree']['max_depth'],
                                    nfolds=TRAINING_CFG['model']['gradient_boosting_tree']['nfolds'],
                                    distribution=TRAINING_CFG['model']['gradient_boosting_tree']['distribution'])

# Load model weights if specified
if TRAINING_CFG['path_dir']['load_model'] is not None:
    try:
        prob_est.load(TRAINING_CFG['path_dir']['load_model'])

    except h2o.exceptions.H2OResponseError:
        raise h2o.exceptions.H2OResponseError('File {} is not valid model path.'.format(TRAINING_CFG['path_dir']['load_model']))

else:
    prob_est.train(input_dataset)
    prob_est.save(TRAINING_CFG['path_dir']['save_model'])

# Evaluate the model on the whole input dataset
if TRAINING_CFG['path_dir']['pred_vs_ground_truth_fig'] is not None:
    prob_est.evaluate(input_dataset, TRAINING_CFG['path_dir']['pred_vs_ground_truth_fig'])

# Deploy
