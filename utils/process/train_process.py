from utils.dataset import *
from utils.tools.census_core import load_census_table_pkl
from models import gbt
import h2o
from tqdm import tqdm
from utils.metrics import *
import ast


class CVPerformance:
    def __init__(self):
        self.stats = {}

    @classmethod
    def write_single_performance_to_file(cls, output_file, hyper_param, performance):
        if output_file is not None: 
            with open(output_file, 'a+') as file:
                log_msg = f"{hyper_param}"
                for metric, p in performance.items():
                    log_msg += f" | {metric}: {p}"
                log_msg += "\n"
                file.write(log_msg)

    def __len__(self):
        return len(self.stats)

    def _decode_hyper_param_str(self, hyper_param_str):
        return ast.literal_eval(hyper_param_str)        

    def append(self, hyper_param, performance):
        assert isinstance(hyper_param, tuple), "Input hyper_param must be tuple"
        assert isinstance(performance, dict), "Input performance must be dict"
        self.stats[hyper_param] = performance

    def write_to_file(self, output_file):
        for hyper_param, performance in self.stats.items():
            CVPerformance.write_single_performance_to_file(output_file, hyper_param, performance)
    
    def load_from_file(self, output_file):
        with open(output_file, 'r') as file:
            for row in file:
                row_list = row.split(' | ')
                hyper_params = self._decode_hyper_param_str(row_list[0])
                performance = {}
                for i in range(len(row_list)-1):
                    metric_name = row_list[i+1].split(':')[0]
                    if i == len(row_list)-2:
                        metric_value = [float(j) for j in row_list[i+1][len(metric_name+': ['):-len('\n')-1].split()]
                    else:
                        metric_value = [float(j) for j in row_list[i+1][len(metric_name+': ['):-1].split()]
                    performance[metric_name] = metric_value
                self.append(hyper_params, performance)
        return self

    def find_top_N_hyper_param(self, criteria, lower_is_better=True, N=1):
        score_dict = {}
        for hyper_param, performance in self.stats.items():
            current_score = criteria(performance)
            score_dict[hyper_param] = current_score
        sorted_items = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        if lower_is_better:
            top_N_items = sorted_items[-N:]
        else:
            top_N_items = sorted_items[:N]
        return top_N_items

    def query_hyper_param(self, hyper_params):
        assert(isinstance(hyper_params, tuple) or isinstance(hyper_params, str)), "Input hyper_params must be str or tuples"
        if isinstance(hyper_params, str):
            query_key = self._decode_hyper_param_str(hyper_params)
        else:
            query_key = hyper_params        
        if query_key in set(self.stats.keys()):
            return self.stats[query_key]
        return None


def best_model_criteria_by_R2(performance):
    return np.mean(performance['mean_R2_score'])


def best_model_criteria_by_RMSE(performance):
    return np.mean(performance['mean_RMSE_score'])


def get_spatial_cross_validation_sets(data, num_folds=5):
    """
    Apply blocked spatial cross validation to an input data. Each fold contains 
    a centroid which is uniformly distributed over data's geospatial field, and 
    the rest of the data points in each fold find the N-1 nearest data point to 
    the centroid, where N is the number of samples in each fold based on num_folds

    Args:
        data (Dataset): input dataset to be cross validated on
        num_folds (int, optional): number of folds. Defaults to 5.

    Returns: (list) validation set indicies list
    """
    # Apply spatial cross validation to prepare N fold sets
    fold_size, remainder = divmod(len(data), num_folds)
    validation_set_sizes = [fold_size + 1 if i < remainder else fold_size for i in range(num_folds)]

    # To get N-fold validation sets, first get N spatial-uniformly sampled data points 
    # as centroids for blocks in each fold 
    centroid_indices = data.spatial_sampling(
        method='uniform', 
        num_samples=num_folds, 
        masked_indices=[])
        
    validation_set_indices = []
    current_mask = []
    for i, validation_set_size in enumerate(validation_set_sizes):
        # mask out any centroid indices except the current centroid
        # current_fold_indices already covers the last centroid 
        current_fold_indices = data.spatial_sampling(
            method='blocked', 
            num_samples=validation_set_size, 
            center_index=centroid_indices[i], 
            masked_indices=current_mask + centroid_indices[i+1:])
        current_mask += current_fold_indices
        validation_set_indices.append(current_fold_indices)
    
    return validation_set_indices


def compute_performance(pred_results, ground_truth):
    """ Return RMSE and R2 for each label class """
    RMSE_score = rmse(pred_results, ground_truth, axis=0)
    R2_score = np.asarray([r2(pred_results[:, i], ground_truth[:, i]) for i in range(pred_results.shape[1])])
    return RMSE_score, R2_score


def run_cross_validation(model, data, validation_set_indices):
    """
    Run cross validation and get performance stats for input model 

    Args:
        model (model): model
        data (Dataset): training dataset
        validation_set_indices (list): validation set indices 

    Returns: (tuple) mean_RMSE_score, mean_R2_score
    """
    for i, current_validation_set_indices in enumerate(validation_set_indices):
        validation_subset = data.get_subset_by_indices(current_validation_set_indices)
        train_subset = data.remove_by_indices(current_validation_set_indices)
        model.train(train_subset, shuffle=True)
        pred_results, ground_truth = model.evaluate(validation_subset)
        RMSE_score, R2_score = compute_performance(pred_results, ground_truth)

        if i == 0:
            mean_RMSE_score, mean_R2_score = RMSE_score, R2_score
        else:
            mean_RMSE_score += RMSE_score
            mean_R2_score += R2_score
    
    mean_RMSE_score /= len(validation_set_indices)
    mean_R2_score /= len(validation_set_indices)
    return mean_RMSE_score, mean_R2_score


def grid_search(base_model, hyper_params_space, data, num_folds=5, output_file=None, rerun=False):
    """
    Apply grid search on spatial cross validation of input data

    Args:
        base_model (model): model
        hyper_params_space (list): hyper_params_space iterable
        data (Dataset): train set
        num_folds (int, optional): number of folds. Defaults to 5.
        output_file (str, optional): dir of cv results file. Defaults to None.
        rerun (bool, optional): whether to rerun experiments in output_file. Defaults to False.

    Returns: (dict): best model based on RMSE or R2
    """
    best_RMSE = None
    best_R2 = None
    best_model = {'by_RMSE': None, 'by_R2': None}

    validation_set_indices = get_spatial_cross_validation_sets(data, num_folds=num_folds)
    cv_performance = CVPerformance()
    if output_file is not None:
        cv_performance.load_from_file(output_file)
    if not rerun:
        # If no rerun, we need to find the historical best hyper params
        best_hyper_param_RMSE, best_RMSE = cv_performance.find_best_hyper_param(best_model_criteria_by_RMSE, lower_is_better=True)
        best_hyper_param_R2, best_R2 = cv_performance.find_best_hyper_param(best_model_criteria_by_R2, lower_is_better=False)
        best_model['by_RMSE'] = best_hyper_param_RMSE
        best_model['by_R2'] = best_hyper_param_R2
        print(f"Best Model by RMSE: {best_hyper_param_RMSE} | RMSE: {best_RMSE}")
        print(f"Best Model by R2: {best_hyper_param_R2} | R2: {best_R2}")

    for hyper_params in tqdm(hyper_params_space, desc="Grid Search Process:"):
        if not rerun: 
            if cv_performance.query_hyper_param(hyper_params) is not None:
                # print(f"{hyper_params} skipped")
                continue
        
        base_model.reset_hyper_params(*hyper_params)

        mean_RMSE_score, mean_R2_score = run_cross_validation(base_model, data, validation_set_indices)
        avg_mean_RMSE_score = np.mean(mean_RMSE_score)
        avg_mean_R2_score = np.mean(mean_R2_score)

        # Log to output_file
        performance = {
            'mean_RMSE_score': mean_RMSE_score, 
            'mean_R2_score': mean_R2_score
        }
        CVPerformance.write_single_performance_to_file(output_file, 
                                                       hyper_params, 
                                                       performance)

        if best_RMSE is None or avg_mean_RMSE_score < best_RMSE:
            best_model['by_RMSE'] = hyper_params
            best_RMSE = avg_mean_RMSE_score
            print(f"Best Model by RMSE: {hyper_params} | RMSE: {best_RMSE} | R2: {avg_mean_R2_score}")
        if best_R2 is None or avg_mean_R2_score > best_R2:
            best_model['by_R2'] = hyper_params
            best_R2 = avg_mean_R2_score
            print(f"Best Model by R2: {hyper_params} | RMSE: {avg_mean_RMSE_score} | R2: {best_R2}")

    return best_model


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

    Returns: (GradientBoostingTree)
    """
    # Load census_table input as Dataset obj
    input_dataset = Dataset(
        census_table=load_census_table_pkl(
            training_cfg['path_dir']['inputs']['census_table_input']),
        land_cover_code=land_cover_cfg['code']['MCD12Q1'],
        remove_land_cover_feature_index=training_cfg['feature_remove'],
        invalid_data=training_cfg['invalid_data_handle'])

    # Declare model structure
    # prob_est = gbt.OvRBernoulliGradientBoostingTree(
    #     ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
    #     max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
    #     nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'])
    prob_est = gbt.MultinomialGradientBoostingTree(
            ntrees=training_cfg['model']['gradient_boosting_tree']['ntrees'],
            max_depth=training_cfg['model']['gradient_boosting_tree']['max_depth'],
            nfolds=training_cfg['model']['gradient_boosting_tree']['nfolds'], 
            min_rows=training_cfg['model']['gradient_boosting_tree']['min_rows'], 
            learn_rate=training_cfg['model']['gradient_boosting_tree']['learn_rate'], 
            sample_rate=training_cfg['model']['gradient_boosting_tree']['sample_rate'], 
            col_sample_rate=training_cfg['model']['gradient_boosting_tree']['col_sample_rate']
            )

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
