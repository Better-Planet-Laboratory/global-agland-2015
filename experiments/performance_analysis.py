import numpy as np
import matplotlib.pyplot as plt
from utils.process.train_process import *
from models.gbt import *
from utils import io


# Find top N best performance candidate from cv
N = 10
cv_performance = CVPerformance()
cv_performance.load_from_file('./experiments/MultinomialGradientBoostingTree_cv_logs.txt')
top_N_items_by_RMSE = cv_performance.find_top_N_hyper_param(best_model_criteria_by_RMSE, lower_is_better=True, N=N)
top_N_items_by_R2 = cv_performance.find_top_N_hyper_param(best_model_criteria_by_RMSE, lower_is_better=False, N=N)
top_2N_hyper_param = top_N_items_by_RMSE + top_N_items_by_R2
top_2N_hyper_param = set([i[0] for i in top_2N_hyper_param])

# Evaluate on test set
train_ratio = 0.9
test_ratio = 0.1

training_cfg = io.load_yaml_config('configs/training_cfg.yaml')
land_cover_cfg = io.load_yaml_config('configs/land_cover_cfg.yaml')

# Load census_table input as Dataset obj
input_dataset = Dataset(
    census_table=load_census_table_pkl(
        training_cfg['path_dir']['inputs']['census_table_input']),
    land_cover_code=land_cover_cfg['code']['MCD12Q1'],
    remove_land_cover_feature_index=training_cfg['feature_remove'],
    invalid_data=training_cfg['invalid_data_handle'])

test_set_indices = input_dataset.spatial_sampling(
    method='uniform', 
    num_samples=int(test_ratio*len(input_dataset)), 
    masked_indices=[])
train_subset = input_dataset.remove_by_indices(indices=test_set_indices)
test_subset = input_dataset.get_subset_by_indices(indices=test_set_indices)

for hyper_param in top_2N_hyper_param:
    base_model = MultinomialGradientBoostingTreeWithCustomCV()
    base_model.reset_hyper_params(
        ntrees=hyper_param[0], 
        max_depth=hyper_param[1], 
        min_rows=hyper_param[2], 
        learn_rate=hyper_param[3], 
        sample_rate=hyper_param[4], 
        col_sample_rate=hyper_param[5]
    )

    base_model.train(train_subset, shuffle=True)
    pred_results, ground_truth = base_model.evaluate(test_subset)
    RMSE_score, R2_score = compute_performance(pred_results, ground_truth)
    with open('./experiments/performance_on_test_set.txt', 'a+') as file:
        log_msg = f"{hyper_param} | RMSE_score: {RMSE_score} | R2_score: {R2_score}\n"
        file.write(log_msg)