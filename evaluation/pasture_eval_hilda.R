############################################
############################################
####### PASTURE MAP COMPARISON - HYDE #######
############################################
############################################

# Julie Fortin
# 2022.11.17

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
# exp2_global <- rast(here("outputs/all_correct_to_subnation_scale_itr3_fr_0",
#                          paste0("agland_map_output_", iter,".tif")))[[2]] # layer 2 is pasture

# Load our masks
water_body_mask <- rast(here("land_cover/water_body_mask.tif"))
gdd_mask <- rast(here("gdd/gdd_filter_map_21600x43200.tif"))

# Load reference map
file_hilda <- here("evaluation/pasture_reference_maps/hilda/hilda_plus_2015_states_GLOB-v1-0_base-map_wgs84-nn.tif") # from https://doi.pangaea.de/10.1594/PANGAEA.921846
r_hilda <- rast(file_hilda)


##########################
### PREP REFERENCE MAP ###
##########################

# The data consists of land cover classes
# We want to aggregate to 5'
# And estimate the proportion of pasture at that resolution

# Aggregate
# aggfactor <- floor(res(exp1_global)[1]/res(r_hilda)[1])
# s_hilda <- segregate(r_hilda, classes=33) # 33 is pasture/rangeland as per https://download.pangaea.de/dataset/921846/files/HILDAplus_GLOBv-1.0_geotiff_documentation.pdf
# r_hilda_agg <- aggregate(s_hilda, aggfactor, sum, na.rm=T)
# writeRaster(r_hilda_agg, here("evaluation/pasture_reference_maps/hilda/hilda_agg.tif"))
r_hilda_agg <- rast(here("evaluation/pasture_reference_maps/hilda/hilda_agg.tif"))

# Convert to prop
names(r_hilda_agg) <- "Value"
prop_hilda <- r_hilda_agg/100


###########################
### PREP PREDICTION MAP ###
###########################

# Project reference map to match predictioin
proj_hilda <- project(prop_hilda, exp1_global)

# Mask out GDD
gdd_mask <- project(gdd_mask, exp1_global)
exp1_global <- mask(exp1_global, gdd_mask, maskvalues=0)
# exp2_global <- mask(exp2_global, gdd_mask, maskvalues=0)
ref_global <- mask(proj_hilda, gdd_mask, maskvalues=0)

# Mask out water bodies
water_body_mask <- project(water_body_mask, exp1_global)
exp1_global <- mask(exp1_global, water_body_mask, maskvalues=0)
# exp2_global <- mask(exp2_global, water_body_mask, maskvalues=0)
ref_global <- mask(ref_global, water_body_mask, maskvalues=0)


####################################
### COMPARE REFERENCE+PREDICTION ###
####################################

# Calculate differences
absdif_exp1 <- ref_global-exp1_global
# absdif_exp2 <- ref_global-exp2_global


##################
### SAVE PLOTS ###
##################

# Save all_correct_to_FAO_scale
writeRaster(r_hilda_agg, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda/hilda_agg.tif"), overwrite=T)
writeRaster(ref_global, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda/hilda_reference.tif"), overwrite=T)
writeRaster(exp1_global, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp1, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp1)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_musigma.csv")))

# Save all_correct_to_subnation_scale
# writeRaster(r_hilda_agg, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda/hilda_agg.tif"), overwrite=T)
# writeRaster(ref_global, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda/hilda_reference.tif"), overwrite=T)
# writeRaster(exp2_global, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_pred_map.tif")), overwrite=T)
# writeRaster(absdif_exp2, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_map.tif")), overwrite=T)
# hist <- hist(absdif_exp2)
# histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
# mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
# sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
# musigma <- data.frame(mu=mu, sigma=sigma)
# write.csv(histdf, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_hist.csv")))
# write.csv(musigma, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/hilda", paste0("agland_map_output_",iter,"_diff_musigma.csv")))

