##################################################
##################################################
####### PASTURE MAP COMPARISON - AUSTRALIA #######
##################################################
##################################################

# Julie Fortin
# 2022.09.30

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

# No need for GDD masks for Australia (doesn't go above 50ºN)

# Load reference map
file_aus <- here("evaluation/pasture_reference_maps/australia/NLUM_ALUMV8_250m_2015_16_alb/NLUM_ALUMV8_250m_2015_16_alb.tif")
r_aus <- rast(file_aus)
levels_aus <- cats(r_aus)[[1]]
names(r_aus) <- "Value"


##########################
### PREP REFERENCE MAP ###
##########################

# Identify raster values corresponding to the classes of interest
values_aus <- c(210, 320, 420) # correspond to grazing native veg, grazing mod past, irrig past

# Create a new raster of counts of classes of interest (including "grazing native vegetation")
r_count <- r_aus 
r_count[r_aus %in% values_aus] <- 1 
r_count[!(r_aus %in% values_aus)] <- 0 

# Original resolution is 250m; aggregate to 10km
aggfactor_aus <- floor(10000/250)
prop_aus <- aggregate(r_count, fact=aggfactor_aus, fun=sum)/(aggfactor_aus^2)

# Mask by water body for plotting
water_body_mask_aus <- project(water_body_mask, prop_aus)
prop_aus <- mask(prop_aus, water_body_mask_aus, maskvalues=0)

# Mask by shapefile (water body mask introduced islands that are not part of Australia)
shp_aus <- vect(here("shapefile/Australia/gadm36_AUS_1.shp"))
prop_aus <- mask(prop_aus, shp_aus)


###########################
### PREP PREDICTION MAP ###
###########################

# Project our reference map to match the global prediction rasters
proj_aus <- project(prop_aus, exp1_global)

# Crop our rasters (bounding box approx around Australia)
exp1_aus <- crop(exp1_global, ext(109, 160, -46, -10))
exp2_aus <- crop(exp2_global, ext(109, 160, -46, -10))
proj_aus <- crop(proj_aus, ext(109, 160, -46, -10))

# No need to mask out GDD for Australia, does not go above 50ºN latitude

# Mask out water bodies
water_body_mask_aus <- crop(water_body_mask, exp1_aus)
water_body_mask_aus <- project(water_body_mask_aus, exp1_aus)
exp1_aus <- mask(exp1_aus, water_body_mask_aus, maskvalues=0)
exp2_aus <- mask(exp2_aus, water_body_mask_aus, maskvalues=0)
proj_aus <- mask(proj_aus, water_body_mask_aus, maskvalues=0)

# Mask out just aus (not neighbouring countries)
shp_aus <- vect(here("shapefile/AUS/gadm36_AUS_1.shp"))
exp1_aus <- mask(exp1_aus, shp_aus)
exp2_aus <- mask(exp2_aus, shp_aus)
proj_aus <- mask(proj_aus, shp_aus)


####################################
### COMPARE REFERENCE+PREDICTION ###
####################################

# Calculate differences 
absdif_exp1 <- prop_aus-exp1_aus
absdif_exp2 <- prop_aus-exp2_aus


##################
### SAVE PLOTS ###
##################

# Save outputs to tif, will be converted to png in python to match format/colour scheme used for cropland comparisons

# Save all_correct_to_FAO_scale
writeRaster(r_aus, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia/australia_landuse.tif"), overwrite=T)
writeRaster(prop_aus, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia/australia_reference.tif"), overwrite=T)
writeRaster(exp1_aus, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_pred_map.tif")))
writeRaster(absdif_exp1, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_map.tif")))
hist <- hist(absdif_exp1)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_musigma.csv")))

# Save all_correct_to_subnation_scale
writeRaster(r_aus, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia/australia_landuse.tif"), overwrite=T)
writeRaster(prop_aus, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia/australia_reference.tif"), overwrite=T)
writeRaster(exp2_aus, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_pred_map.tif")))
writeRaster(absdif_exp2, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_map.tif")))
hist <- hist(absdif_exp2)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/australia", paste0("agland_map_output_",iter,"_australia_diff_musigma.csv")))

