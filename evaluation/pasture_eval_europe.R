###############################################
###############################################
####### PASTURE MAP COMPARISON - EUROPE #######
###############################################
###############################################

# Julie Fortin
# 2022.10.12

#####################
### LOAD PACKAGES ###
#####################

library(here)
library(terra)


#################
### LOAD MAPS ###
#################

# Set iteration for this run (change to run for iters 0, 1, 2, 3)
iter <- "3"

# Load our global predictions for pasture
exp1_global <- rast(here("outputs/all_correct_to_FAO_scale_itr3_fr_0",
                         paste0("agland_map_output_", iter,".tif")))[[2]] # layer 2 is pasture
exp2_global <- rast(here("outputs/all_correct_to_subnation_scale_itr3_fr_0",
                         paste0("agland_map_output_", iter,".tif")))[[2]] # layer 2 is pasture

# Load our masks
water_body_mask <- rast(here("land_cover/water_body_mask.tif"))

# No need for GDD masks for eu (doesn't go above 50ºN)
gdd_mask <- rast(here("gdd/gdd_filter_map_360x720.tif"))

# Load reference maps
file_eu <- here("evaluation/pasture_reference_maps/europe/lcv_landcover.231_lucas.corine.eml_p_30m_0..0cm_2015_eumap_epsg3035_v0.2.tif")
r_eu <- rast(file_eu)


##########################
### PREP REFERENCE MAP ###
##########################

# For Europe, the data consists of a probability layer for pasture
# We want to aggregate to 10km grid cells
# And get the proportion of pasture in those grid cells
# We can treat the probabilities as the proportion of pasture within the 30m cells
# Then just take the average

# Aggregate probabilities
# aggfactor_eu <- 10000/30
# r_eu_agg <- aggregate(r_eu, aggfactor_eu, fun=mean, na.rm=T)
# writeRaster(r_eu_agg, here("evaluation/pasture_reference_maps/europe/eu_agg.tif"))
r_eu_agg <- rast(here("evaluation/pasture_reference_maps/europe/eu_agg.tif"))

# Convert to prop
names(r_eu_agg) <- "Value"
prop_eu <- r_eu_agg/100


###########################
### PREP PREDICTION MAP ###
###########################

# Project our reference map to match the global prediction rasters
proj_eu <- project(prop_eu, exp1_global)

# Crop our rasters (bounding box approx around eu)
exp1_eu <- crop(exp1_global, ext(-30, 40, 32, 75))
exp2_eu <- crop(exp2_global, ext(-30, 40, 32, 75))
proj_eu <- crop(proj_eu, ext(-30, 40, 32, 75))

# Mask out water bodies
water_body_mask_eu <- crop(water_body_mask, exp1_eu)
water_body_mask_eu <- project(water_body_mask_eu, exp1_eu)
exp1_eu <- mask(exp1_eu, water_body_mask_eu, maskvalues=0)
exp2_eu <- mask(exp2_eu, water_body_mask_eu, maskvalues=0)
proj_eu <- mask(proj_eu, water_body_mask_eu, maskvalues=0)

# Mask out GDD
gdd_mask_eu <- crop(gdd_mask, exp1_eu)
gdd_mask_eu <- project(gdd_mask_eu, exp1_eu)
exp1_eu <- mask(exp1_eu, gdd_mask_eu, maskvalues=0)
exp2_eu <- mask(exp2_eu, gdd_mask_eu, maskvalues=0)
proj_eu <- mask(proj_eu, gdd_mask_eu, maskvalues=0)

# Mask out just EU (not neighbouring countries)
mask_eu <- project(prop_eu, exp1_global)
mask_eu <- crop(mask_eu, ext(-30, 40, 32, 75))
mask_eu[!is.na(mask_eu)] <- 1
exp1_eu <- mask(exp1_eu, mask_eu)
exp2_eu <- mask(exp2_eu, mask_eu)
proj_eu <- mask(proj_eu, mask_eu)


####################################
### COMPARE REFERENCE+PREDICTION ###
####################################

# Calculate differences
absdif_exp1 <- proj_eu-exp1_eu
absdif_exp2 <- proj_eu-exp2_eu


##################
### SAVE PLOTS ###
##################

# Save all_correct_to_FAO_scale
writeRaster(mask(project(r_eu_agg, exp1_eu),mask_eu), here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe/eu_agg.tif"), overwrite=T)
writeRaster(proj_eu, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe/eu_reference.tif"), overwrite=T)
writeRaster(exp1_eu, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp1, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp1)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_musigma.csv")))

# Save all_correct_to_subnation_scale
writeRaster(mask(project(r_eu_agg, exp1_eu),mask_eu), here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe/eu_agg.tif"), overwrite=T)
writeRaster(proj_eu, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe/eu_reference.tif"), overwrite=T)
writeRaster(exp2_eu, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp2, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp2)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/europe", paste0("agland_map_output_",iter,"_eu_diff_musigma.csv")))

