from models.gbt import *
from utils import io
from utils.process.train_process import *
from itertools import product


EXPERIMENT_CFG = io.load_yaml_config('experiments/all_correct_to_FAO_scale_itr3_fr_0/configs/experiment_cfg_hyper_param.yaml')

h2o.init(enable_assertions=False)  # to prevent over aggressive warnings
h2o.no_progress()

training_cfg = io.load_yaml_config(EXPERIMENT_CFG['training_cfg'])
land_cover_cfg = io.load_yaml_config(EXPERIMENT_CFG['land_cover_cfg'])

# Load census_table input as Dataset obj
input_dataset = Dataset(
    census_table=load_census_table_pkl(
        training_cfg['path_dir']['inputs']['census_table_input']),
    land_cover_code=land_cover_cfg['code']['MCD12Q1'],
    remove_land_cover_feature_index=training_cfg['feature_remove'],
    invalid_data=training_cfg['invalid_data_handle'])

# Split train / test set
test_set_indices = input_dataset.spatial_sampling(
    method='uniform', 
    num_samples=int(EXPERIMENT_CFG['test_ratio']*len(input_dataset)), 
    masked_indices=[])
train_subset = input_dataset.remove_by_indices(indices=test_set_indices)
test_subset = input_dataset.get_subset_by_indices(indices=test_set_indices)

# Set Hyper parameter space
hyper_params_space = list(product(
    [i for i in EXPERIMENT_CFG['n_trees']],
    [i for i in EXPERIMENT_CFG['max_depth']], 
    [i for i in EXPERIMENT_CFG['min_rows']], 
    [i for i in EXPERIMENT_CFG['learn_rate']], 
    [i for i in EXPERIMENT_CFG['sample_rate']], 
    [i for i in EXPERIMENT_CFG['col_sample_rate']]
    )
)

# Set base model architecture
base_model = MultinomialGradientBoostingTreeWithCustomCV()
best_model = grid_search(base_model, 
                         hyper_params_space, 
                         train_subset, 
                         num_folds=EXPERIMENT_CFG['num_folds'], 
                         output_file=EXPERIMENT_CFG['output_file'])
