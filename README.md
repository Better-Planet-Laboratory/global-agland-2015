## Global Agricultural Lands in the year 2015
This project is a continuation and update in methodology of the work from [Ramankutty et al. (2008)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007GB002952). We combine subnational level census data and national level FAOSTAT data to develop a global spatial dataset of croplands and pastures on a graticule of 5 arcminutes (~10 $km^2$ at the equator). These maps support a huge variety of research topics, from land use and food security to climate change and biodiversity loss. This repo includes a full set of replicable code for reproduction, modification and testing.

### Data Sources and Processing
- [FAOSTAT](FAOSTAT_data/README.md)
- [Subnational Census](subnational_stats/README.md)
- [GDD](gdd/README.md)
- [Land Cover](land_cover/README.md)
- [Aridity Map]()

### Results and Analysis
- [Evaluation](evaluation/README.md)
- [Experiments](experiments/README.md)

### Folder Structure
- [configs](./configs/) (user-specified settings)
- [utils](./utils/) (helper functions and tools)
- [census_processor](./census_processor/) (country class files for which we have subnational data and helps data loading)
- [gdd](./gdd/) (scripts and data for gdd filter mask)
- [land_cover](./land_cover/) (scripts and data for land cover maps)
- [models](./models/) (scripts and pre-trained model weights)
- [FAOSTAT_data](./FAOSTAT_data/) ([FAOSTAT](https://www.fao.org/faostat/en/) dataset)
- [subnational_stats](./subnational_stats/) (subnational census dataset)
- [shapefile](./shapefile/) (shapefile data from [GADM](https://gadm.org/))
- [evaluation](./evaluation/) (code and evaluation results between map predictions and independent sources)
- [experiments](./experiments/) (a collection of mlflow experiment scripts)
- [outputs](./outputs/) (a collection of experiment results)
- [docs/source](./docs/source/) (results figures and visualization scripts)

### Requirements
- Option 1 - PIP
  - Ubuntu users can run the following requirements directly
    - ``` pip install -r requirements.txt ```
- Option 2 - Docker
  - [Dockerfile](Dockerfile)
    - if you encounter issues while importing "gdal_array", I have included a [fix](./docs/source/readmes/gdal_array_fix.md)

### Merged Census Input
We use subnational data wherever it is available and fill in with national level data from FAOSTAT elsewhere. Thus we merge census and FAOSTAT data to generate the input dataset for our machine learning model. During the merging process, 2 filters are applied, namely NaN filter and GDD filter.
* NaN filter
  * Remove samples with NaN in either CROPLAND or PASTURE attribute (e.g. this happens if we had data for cropland but not for pasture for this unit)
* GDD filter 
  * Remove samples that geographically lay in GDD mask

For the data sources we used, originally we have 950 samples after merging, then the NaN filter removes 181 samples, and the GDD filter removes 42 samples, resulting in a total of 727 samples at the end of the census pipeline (Note: this is not the final input dataset, outliers that have CROPLAND and PASTURE sum greater than 100% will also be removed prior training). Implementation details can be found [here](./utils/process/census_process.py). To run the census pipeline, adjust the yaml files in the ```/configs``` and do:
```
python census.py
```
A visualization of the census inputs is also provided below. 
![merged_census_input_cropland](./docs/source/_static/img/census/cropland_census_input.png)
![merged_census_input_pasture](./docs/source/_static/img/census/pasture_census_input.png)

### Train
All training related configs could be found under ```/configs/training_cfg.yaml```. Note that one could also enable feature selection by specifying features (i.e. land cover types) to be removed. Removing a feature in land cover type does not simply remove it, instead a factor of 1/(1-[removed_class_sum]) is applied to the remaining features to maintain the property of probability distribution. All implementation details can be found [here](./utils/process/train_process.py). We employ a gradient boosting tree model using cross-validation to tune hyperparameters. To start training, run:
```
python train.py
```

### Deployment
During deployment, we use 20 x 20 block matrices of 500m MODIS grid cells as inputs for our model (detailed process is explained under [Prediction Input and Aggregation](./land_cover/README.md#prediction-input-and-aggregation)). Deployment configs can be modified under ```/configs/deploy_setting_cfg.yaml```, but need to be consistent with the settings used for the trained model (e.g. ```[path_dir][model]``` and ```[feature_remove]```). Make sure correct model and correct path of the model are loaded. Post processing implementation can be found [here](./utils/process/post_process.py). To run deployment to get the final cropland and pasture maps, run:
```
python deploy.py
```

### Output Figures
All visualization scripts are placed under ```/docs/source/scripts/```. Make sure the project path ```${workspaceFolder}``` is added to the PYTHONPATH, then run:
```
cd docs/source/scripts/
python SCRIPT_TO_RUN [FLAG] [ARG]
```

### Citation
```
```

