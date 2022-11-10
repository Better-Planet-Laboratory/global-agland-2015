###############################################
###############################################
####### PASTURE MAP COMPARISON - BRAZIL #######
###############################################
###############################################

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

# No need for GDD masks for Brazil (doesn't go above 50ºN)

# Load reference map
file_bra <- here("evaluation/pasture_reference_maps/brazil/brasil_coverage_2015.tif")
r_bra <- rast(file_bra)


#################################
### PREP REFERENCE MAP - LONG ###
#################################

### Uncomment this section if running for the first time
### But it takes very long to run/need to be run on HPC, so for subsequent testing
### Can just skip this and run the short version

# # Identify raster values corresponding to the classes of interest
# values_bra <- c(15, 21)
# 
# # Split raster into smaller more manageable chunks
# tile_template <- r_bra # make a template raster with same extent
# res(tile_template) <- res(tile_template)*10000 # change the resolution to much coarser
# makeTiles(r_bra, tile_template, "evaluation/pasture_reference_maps/brazil/tiles/brazil_tile.tif", extend=T)
# tile_list <- list.files(here("evaluation/pasture_reference_maps/brazil/tiles"))
# r_list <- vector(mode="list", length=length(tile_list))
# 
# for (i in 1:length(tile_list)) {
# 
#   # Create a new raster of counts of classes of interest
#   r <- rast(here("evaluation/pasture_reference_maps/brazil/tiles", tile_list[i]))
#   r_count <- r
#   r_count[r %in% values_bra] <- 1
#   r_count[!(r %in% values_bra)] <- 0
# 
#   # Original resolution is 30m; aggregate to approx 10km
#   aggfactor <- 10000/30
#   r_agg <- aggregate(r_count, floor(aggfactor), fun=sum) # aggregate needs to be on integer, hence floor()
# 
#   # Create a list of all aggregated rasters (needed for mosaic below)
#   r_list[[i]] <- r_agg
# 
# }
# 
# # Mosaic all aggregated rasters back to full Brazil extent
# r_list <- r_list[1:length(tile_list)]
# r_collection <- sprc(r_list)
# r_bra_agg <- mosaic(r_collection)
# writeRaster(r_bra_agg, "evaluation/pasture_reference_maps/brazil/brazil_agg.tif")


##################################
### PREP REFERENCE MAP - SHORT ###
##################################

### Comment this section out if running for the first time
### The long version is needed to generate an aggregated tif
### But it takes very long to run
### The rest of the time, you can just run this short versioin

# Identify raster values corresponding to the classes of interest
values_bra <- c(15, 21)

# Load aggregated raster
r_bra_agg <- rast(here("evaluation/pasture_reference_maps/brazil/brazil_agg.tif"))

# Calculate proportion of pasture per 10km grid cell
aggfactor <- 10000/30
prop_bra <- r_bra_agg/(aggfactor^2)


###########################
### PREP PREDICTION MAP ###
###########################

# Project our reference map to match the global prediction rasters
proj_bra <- project(prop_bra, exp1_global)

# Crop our rasters (bounding box approx around bra)
exp1_bra <- crop(exp1_global, ext(-75, -32, -35, 6))
exp2_bra <- crop(exp2_global, ext(-75, -32, -35, 6))
proj_bra <- crop(proj_bra, ext(-75, -32, -35, 6))

# No need to mask out GDD for Brazil, does not go above 50ºN latitude

# Mask out water bodies
water_body_mask_bra <- crop(water_body_mask, exp1_bra)
water_body_mask_bra <- project(water_body_mask_bra, exp1_bra)
exp1_bra <- mask(exp1_bra, water_body_mask_bra, maskvalues=0)
exp2_bra <- mask(exp2_bra, water_body_mask_bra, maskvalues=0)
proj_bra <- mask(proj_bra, water_body_mask_bra, maskvalues=0)

# Mask out just bra (not neighbouring countries)
shp_bra <- vect(here("shapefile/Brazil/gadm36_BRA_1.shp"))
exp1_bra <- mask(exp1_bra, shp_bra)
exp2_bra <- mask(exp2_bra, shp_bra)
proj_bra <- mask(proj_bra, shp_bra)


####################################
### COMPARE REFERENCE+PREDICTION ###
####################################

# Calculate differences
absdif_exp1 <- proj_bra-exp1_bra
absdif_exp2 <- proj_bra-exp2_bra


##################
### SAVE PLOTS ###
##################

# Save all_correct_to_FAO_scale
writeRaster(r_bra_agg, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil/brazil_landuse.tif"), overwrite=T)
writeRaster(proj_bra, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil/brazil_reference.tif"), overwrite=T)
writeRaster(exp1_bra, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp1, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil",paste0("agland_map_output_",iter,"_brazil_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp1)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_diff_musigma.csv")))

# Save all_correct_to_subnation_scale
writeRaster(r_bra_agg, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil/brazil_landuse.tif"), overwrite=T)
writeRaster(proj_bra, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil/brazil_reference.tif"), overwrite=T)
writeRaster(exp2_bra, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil",paste0("agland_map_output_",iter,"_brazil_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp2, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp2)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/brazil", paste0("agland_map_output_",iter,"_brazil_diff_musigma.csv")))

