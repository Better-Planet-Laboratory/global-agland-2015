## Experiment Setup

There are four main components in the pipeline that could be varied to deliver different products:
* FAO calibration
  * Definition
    * A calibration factor to be applied to each state level in input dataset prior training, during fusion between FAOSTAT data and subnational level census data
  * Options
    * All calibrate to FAOSTAT 
    * All calibrate to subnational level
* Bias correction 
  * Definition
    * Part of post processing that is done at the end of deployment to bias correct the output agland map to match input dataset on each state level. This bias correction method forces each pixel in the posterior agland map to follow a probability distribution 
  * Options
    * scale 
    * ~~softmax~~ (Do not use - bad results)
* Iteration
  * Definition
    * Number of iterations of bias correction process. More iterations will lead to a convergence to the input dataset (if set to be 0, no bias correction will be applied)
  * Options
    * $itr \in \Z^+$
* Land cover features to be removed
  * Definition
    * Land cover class(es) to be removed from feature set for training. This feature selection process is usually done when 1. model suffers from overfitting 2. features show low correlation with labels 3. dimensionality of data needs to be reduced 4. two or more features show high correlation and are therefore more likely to be linearly dependent, etc. 
  * Options
    * [...] land cover class indices

It is highly recommanded to use mlflow to setup experiments for this project. Some example code are provided: [threshold space](./mlflow_ovrGBT_th.py), [mask order space](./mlflow_ovrGBT_masks.py). 

## Training Quality
There can be two ways to evaluate the model performance. One is based on metrics over the cross validation on direct input-output data, the other one is based on metrics over the deployed map. Due to the nature of gradient boosting tree models, it is relatively easier to get good results from former approach. Here we present the results on the latter one for iteration 0 and 3. 

|                                          all_correct_to_FAO_scale_itr0_fr_0                                          |                                          all_correct_to_FAO_scale_itr3_fr_0                                          |
| :------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
| ![perf0](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/pred_vs_ground_truth_fig_0.png) | ![perf3](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/pred_vs_ground_truth_fig_3.png) |


## Deployment and Bias Correction 
Since we are doing grid level prediction during deployment (20-by-20 kernel) and evaluation of output agland map on state level, the linkage between the two levels is unknown and unpresented to the model. Bias correction is an important step in the post-process that builds the missing bridge. We can see as iteration number increases, the output agland map converges to the input data. 

### *all_correct_to_FAO_scale_itr3_fr_0*
#### Order (top-down): Cropland, Pasture, Other
| iter 0                                                                                                                    | iter 1                                                                                                                    | iter 2                                                                                                                    | iter 3                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| ![cropland_map_itr0_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_0_cropland.png) | ![cropland_map_itr1_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_1_cropland.png) | ![cropland_map_itr2_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_2_cropland.png) | ![cropland_map_itr3_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_3_cropland.png) |
| ![pasture_map_itr0_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_0_pasture.png)   | ![pasture_map_itr1_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_1_pasture.png)   | ![pasture_map_itr2_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_2_pasture.png)   | ![pasture_map_itr3_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_3_pasture.png)   |
| ![other_map_itr0_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_0_other.png)       | ![other_map_itr1_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_1_other.png)       | ![other_map_itr2_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_2_other.png)       | ![other_map_itr3_1](../docs/source/_static/img/model_outputs/all_correct_to_FAO_scale_itr3_fr_0/output_3_other.png)       |
