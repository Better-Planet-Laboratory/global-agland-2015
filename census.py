import os
from utils import io
from utils.tools.census_core import *
from utils.tools.gdd_core import *
from gdd.gdd_criteria import gdd_crop_criteria
from census_processor import Argentina, Australia, Austria, Belgium, Brazil, \
    Bulgaria, Canada, China, Croatia, Cyprus, Czechia, \
    Denmark, Estonia, Ethiopia, Finland, France, Germany, \
    Greece, Hungary, India, Indonesia, Ireland, Italy, Kazakhstan, \
    Latvia, Lithuania, Luxembourg, Malta, Mexico, Mongolia, \
    Mozambique, Namibia, Netherlands, Pakistan, Poland, Portugal, \
    Romania, Russia, SaudiArabia, Slovakia, Slovenia, SouthAfrica, \
    Spain, Sweden, Tanzania, Turkey, Uganda, UK, Ukraine, USA, World

# Load configs and census objects
CENSUS_SETTING_CFG = io.load_yaml_config('configs/census_setting_cfg.yaml')
GDD_CFG = io.load_yaml_config('configs/gdd_cfg.yaml')
LAND_COVER_CFG = io.load_yaml_config('configs/land_cover_cfg.yaml')
SHAPEFILE_CFG = io.load_yaml_config('configs/shapefile_cfg.yaml')
SUBNATIONAL_STATS_CFG = io.load_yaml_config('configs/subnational_stats_cfg.yaml')

SUBNATIONAL_CENSUS = {
    'Argentina': Argentina(SHAPEFILE_CFG['path_dir']['Argentina'],
                           SUBNATIONAL_STATS_CFG['path_dir']['Argentina'],
                           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Australia': Australia(SHAPEFILE_CFG['path_dir']['Australia'],
                           SUBNATIONAL_STATS_CFG['path_dir']['Australia'],
                           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Austria': Austria(SHAPEFILE_CFG['path_dir']['Austria'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Austria'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Belgium': Belgium(SHAPEFILE_CFG['path_dir']['Belgium'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Belgium'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Brazil': Brazil(SHAPEFILE_CFG['path_dir']['Brazil'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Brazil'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Bulgaria': Bulgaria(SHAPEFILE_CFG['path_dir']['Bulgaria'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Bulgaria'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Canada': Canada(SHAPEFILE_CFG['path_dir']['Canada'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Canada'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'China': China(SHAPEFILE_CFG['path_dir']['China'],
                   SUBNATIONAL_STATS_CFG['path_dir']['China'],
                   CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Croatia': Croatia(SHAPEFILE_CFG['path_dir']['Croatia'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Croatia'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Cyprus': Cyprus(SHAPEFILE_CFG['path_dir']['Cyprus'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Cyprus'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Czechia': Czechia(SHAPEFILE_CFG['path_dir']['Czechia'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Czechia'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Denmark': Denmark(SHAPEFILE_CFG['path_dir']['Denmark'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Denmark'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Estonia': Estonia(SHAPEFILE_CFG['path_dir']['Estonia'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Estonia'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ethiopia': Ethiopia(SHAPEFILE_CFG['path_dir']['Ethiopia'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Ethiopia'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Finland': Finland(SHAPEFILE_CFG['path_dir']['Finland'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Finland'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'France': France(SHAPEFILE_CFG['path_dir']['France'],
                     SUBNATIONAL_STATS_CFG['path_dir']['France'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Germany': Germany(SHAPEFILE_CFG['path_dir']['Germany'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Germany'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Greece': Greece(SHAPEFILE_CFG['path_dir']['Greece'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Greece'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Hungary': Hungary(SHAPEFILE_CFG['path_dir']['Hungary'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Hungary'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'India': India(SHAPEFILE_CFG['path_dir']['India'],
                   SUBNATIONAL_STATS_CFG['path_dir']['India'],
                   CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Indonesia': Indonesia(SHAPEFILE_CFG['path_dir']['Indonesia'],
                           SUBNATIONAL_STATS_CFG['path_dir']['Indonesia'],
                           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ireland': Ireland(SHAPEFILE_CFG['path_dir']['Ireland'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Ireland'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Italy': Italy(SHAPEFILE_CFG['path_dir']['Italy'],
                   SUBNATIONAL_STATS_CFG['path_dir']['Italy'],
                   CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Kazakhstan': Kazakhstan(SHAPEFILE_CFG['path_dir']['Kazakhstan'],
                             SUBNATIONAL_STATS_CFG['path_dir']['Kazakhstan'],
                             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Latvia': Latvia(SHAPEFILE_CFG['path_dir']['Latvia'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Latvia'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Lithuania': Lithuania(SHAPEFILE_CFG['path_dir']['Lithuania'],
                           SUBNATIONAL_STATS_CFG['path_dir']['Lithuania'],
                           CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Luxembourg': Luxembourg(SHAPEFILE_CFG['path_dir']['Luxembourg'],
                             SUBNATIONAL_STATS_CFG['path_dir']['Luxembourg'],
                             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Malta': Malta(SHAPEFILE_CFG['path_dir']['Malta'],
                   SUBNATIONAL_STATS_CFG['path_dir']['Malta'],
                   CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mexico': Mexico(SHAPEFILE_CFG['path_dir']['Mexico'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Mexico'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mongolia': Mongolia(SHAPEFILE_CFG['path_dir']['Mongolia'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Mongolia'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Mozambique': Mozambique(SHAPEFILE_CFG['path_dir']['Mozambique'],
                             SUBNATIONAL_STATS_CFG['path_dir']['Mozambique'],
                             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Namibia': Namibia(SHAPEFILE_CFG['path_dir']['Namibia'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Namibia'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Netherlands': Netherlands(SHAPEFILE_CFG['path_dir']['Netherlands'],
                               SUBNATIONAL_STATS_CFG['path_dir']['Netherlands'],
                               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Pakistan': Pakistan(SHAPEFILE_CFG['path_dir']['Pakistan'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Pakistan'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Poland': Poland(SHAPEFILE_CFG['path_dir']['Poland'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Poland'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Portugal': Portugal(SHAPEFILE_CFG['path_dir']['Portugal'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Portugal'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Romania': Romania(SHAPEFILE_CFG['path_dir']['Romania'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Romania'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Russia': Russia(SHAPEFILE_CFG['path_dir']['Russia'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Russia'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'SaudiArabia': SaudiArabia(SHAPEFILE_CFG['path_dir']['SaudiArabia'],
                               SUBNATIONAL_STATS_CFG['path_dir']['SaudiArabia'],
                               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Slovakia': Slovakia(SHAPEFILE_CFG['path_dir']['Slovakia'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Slovakia'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Slovenia': Slovenia(SHAPEFILE_CFG['path_dir']['Slovenia'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Slovenia'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'SouthAfrica': SouthAfrica(SHAPEFILE_CFG['path_dir']['SouthAfrica'],
                               SUBNATIONAL_STATS_CFG['path_dir']['SouthAfrica'],
                               CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Spain': Spain(SHAPEFILE_CFG['path_dir']['Spain'],
                   SUBNATIONAL_STATS_CFG['path_dir']['Spain'],
                   CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Sweden': Sweden(SHAPEFILE_CFG['path_dir']['Sweden'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Sweden'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Tanzania': Tanzania(SHAPEFILE_CFG['path_dir']['Tanzania'],
                         SUBNATIONAL_STATS_CFG['path_dir']['Tanzania'],
                         CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Turkey': Turkey(SHAPEFILE_CFG['path_dir']['Turkey'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Turkey'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Uganda': Uganda(SHAPEFILE_CFG['path_dir']['Uganda'],
                     SUBNATIONAL_STATS_CFG['path_dir']['Uganda'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'UK': UK(SHAPEFILE_CFG['path_dir']['UK'],
             SUBNATIONAL_STATS_CFG['path_dir']['UK'],
             CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'Ukraine': Ukraine(SHAPEFILE_CFG['path_dir']['Ukraine'],
                       SUBNATIONAL_STATS_CFG['path_dir']['Ukraine'],
                       CENSUS_SETTING_CFG['path_dir']['FAOSTAT']),
    'USA': USA(SHAPEFILE_CFG['path_dir']['USA'],
               SUBNATIONAL_STATS_CFG['path_dir']['USA'],
               CENSUS_SETTING_CFG['path_dir']['FAOSTAT'])
}

WORLD_CENSUS = World(SHAPEFILE_CFG['path_dir']['World'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT'],
                     CENSUS_SETTING_CFG['path_dir']['FAOSTAT_profile'])

# Use args to set flag for generating intermediate results
# args are for user pref. settings, census settings.yaml is for config in the pipeline
# if None, then dont output intermediate results, if input a dir then output


# Save intermediate outputs
# bias_factors_table - bias correction factors for each census sample
# census_states_count_table - number of states count for each country in census
bias_factors_table_dir, \
census_states_count_table_dir = CENSUS_SETTING_CFG['path_dir']['outputs']['bias_factors_table'], \
                                CENSUS_SETTING_CFG['path_dir']['outputs']['census_states_count_table']
if bias_factors_table_dir is not None:
    write_bias_factors_table_to_csv(get_bias_factors_table(SUBNATIONAL_CENSUS),
                                    bias_factors_table_dir)
if census_states_count_table_dir is not None:
    write_census_states_count_table(count_census_states(SUBNATIONAL_CENSUS),
                                    census_states_count_table_dir)

# Merge WORLD with SUBNATIONAL
merged_census = merge_subnation_to_world(WORLD_CENSUS, SUBNATIONAL_CENSUS, CENSUS_SETTING_CFG['bias_correct'])
print(len(merged_census))

# Apply 2 filters
# 1. nan_filter: nan in either CROPLAND or PASTURE
# 2. GDD_filter: GDD exclude: above 50d north with < 1000
#                GDD include / (GDD exclude + GDD include) < accept_ratio
merged_census = apply_nan_filter(merged_census)
merged_census = apply_GDD_filter(merged_census, GDD_CFG, CENSUS_SETTING_CFG['GDD_filter']['accept_ratio'],
                                 gdd_crop_criteria)

print(len(merged_census))


# Add land_cover percentage features to census table
merged_census = add_land_cover_percentage(merged_census, LAND_COVER_CFG['path_dir']['land_cover_map'],
                                          LAND_COVER_CFG['code']['MCD12Q1'])

print(len(merged_census))

# Add total area in kHa for each sample
merged_census = add_state_area(merged_census, LAND_COVER_CFG['path_dir']['global_area_map'],
                                          LAND_COVER_CFG['area_unit'])

print(len(merged_census))
print(merged_census)

# Data Pre-processing
# 2. Use land_cover map to add LAND_COVER percentages to each sample
# 3. Compute total area (in KHa, need to do a conversion here) for each sample
# 4. Add Attributes CROPLAND_PERCENTAGE, PASTURE_PERCENTAGE, OTHER_PERCENTAGE
