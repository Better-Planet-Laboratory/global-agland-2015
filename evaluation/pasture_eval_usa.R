############################################
############################################
####### PASTURE MAP COMPARISON - USA #######
############################################
############################################

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
# exp2_global <- rast(here("outputs/all_correct_to_subnation_scale_itr3_fr_0",
#                          paste0("agland_map_output_", iter,".tif")))[[2]] # layer 2 is pasture

# Load our masks
water_body_mask <- rast(here("land_cover/water_body_mask.tif"))

# No need for GDD masks for USA (doesn't go above 50ºN)

# Load reference maps
file_usa_nlcd <- here("evaluation/pasture_reference_maps/usa/nlcd_2011_land_cover_l48_20210604/nlcd_2011_land_cover_l48_20210604.img")
r_usa_nlcd <- rast(file_usa_nlcd)

file_usa_range <- here("evaluation/pasture_reference_maps/usa/Rangelands_v1/Rangelands_v1.tif")
r_usa_range <- rast(file_usa_range)
levels_usa <- cats(r_usa_range)[[1]]


#################################
### PREP REFERENCE MAP - LONG ###
#################################

### Uncomment this section if running for the first time
### But it takes very long to run and requires HPC, so for subsequent testing
### Can just skip this and run the short version

### NLCD

# # Identify raster values corresponding to the classes of interest
# values_usa_nlcd <- 81
# 
# # Split raster into smaller more manageable chunks
# tile_template <- r_usa_nlcd # make a template raster with same extent
# res(tile_template) <- res(tile_template)*10000  # change the resolution to much coarser
# makeTiles(r_usa_nlcd, tile_template, "evaluation/pasture_reference_maps/usa/tiles/usa_tile_nlcd_.tif", extend=T, NAflag=NA)
# tile_list <- list.files(here("evaluation/pasture_reference_maps/usa/tiles"),"_nlcd")
# tile_list <- tile_list[endsWith(tile_list,"tif")]
# tile_list <- tile_list[c(1,100,111,122,133,144,155,166,177,2,13,24,35,46,57,68,79,90,99,101:110,112:121,123:132,134:143,145:154,156:165,167:176,178:187,3:12,14:23,25:34,36:45,47:56,58:67,69:78,80:89,91:98)]
# r_list <- vector(mode="list", length=length(tile_list))
# 
# for (i in 1:length(tile_list)) { # need large amount of memory to run this (or split into smaller loops)
#   
#   # Create a new raster of counts of classes of interest
#   r <- rast(here("evaluation/pasture_reference_maps/usa/tiles", tile_list[i]))
#   r_count <- r
#   r_count[r %in% values_usa_nlcd] <- 1
#   r_count[!(r %in% values_usa_nlcd)] <- 0
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
# # Mosaic all aggregated rasters back to full usa extent
# r_list <- r_list[1:length(tile_list)]
# r_collection <- sprc(r_list)
# r_usa_nlcd_agg <- mosaic(r_collection)
# plot(r_usa_nlcd_agg)
# writeRaster(r_usa_nlcd_agg, "evaluation/pasture_reference_maps/usa/tiles/usa_agg_nlcd.tif")

### RANGELANDS

# # Identify raster values corresponding to the classes of interest
# values_usa_range <- levels_usa$VALUE[which(levels_usa$LABEL %in% c("Rangeland"))]
# 
# # Split raster into smaller more manageable chunks
# tile_template <- r_usa_range # make a template raster with same extent
# res(tile_template) <- res(tile_template)*10000 # change the resolution to much coarser
# makeTiles(r_usa_range, tile_template, "evaluation/pasture_reference_maps/usa/tiles/usa_tile_range_.tif", extend=T, NAflag=NA)
# tile_list <- list.files(here("evaluation/pasture_reference_maps/usa/tiles"),"tile_range")
# tile_list <- tile_list[endsWith(tile_list, "tif")]
# tile_list <- tile_list[c(1,73,84,95,106,117,128,139,150,2,13,24,35,46,57,68,70:72,74:83,85:94,96:105,107:116,118:127,129:138,140:149,151:160,3:12,14:23,25:34,36:45,47:56,58:67,69)]
# r_list <- vector(mode="list", length=length(tile_list))
# 
# for (i in 1:length(tile_list)) { # need large amount of memory to run this (or split into smaller loops)
#   
#   # Create a new raster of counts of classes of interest
#   # makeTiles & writeRaster result in many tiles being NA for some unknown reason
#   # But the extents are correct
#   # So let's use those extents, and crop the main raster on the fly,
#   # Without needing to use writeRaster 
#   e <- ext(rast(here("evaluation/pasture_reference_maps/usa/tiles", tile_list[i])))
#   r <- crop(r_usa_range, e)
#   r_count <- r
#   r_count[r %in% values_usa_range] <- 1
#   r_count[!(r %in% values_usa_range)] <- 0
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
# # Mosaic all aggregated rasters back to full usa extent
# r_list <- r_list[1:length(tile_list)]
# r_collection <- sprc(r_list)
# r_usa_range_agg <- mosaic(r_collection)
# plot(r_usa_range_agg)
# writeRaster(r_usa_range_agg, "evaluation/pasture_reference_maps/usa/tiles/usa_agg_range.tif")


##################################
### PREP REFERENCE MAP - SHORT ###
##################################

### Comment this section out if running for the first time
### The long version is needed to generate an aggregated tif
### But it takes very long to run
### The rest of the time, you can just run this short versioin

### NLCD

# Identify raster values corresponding to the classes of interest
values_usa_nlcd <- 81

# Load aggregated raster
r_usa_nlcd_agg <- rast(here("evaluation/pasture_reference_maps/usa/usa_agg_nlcd.tif"))

### RANGELANDS

# Identify raster values corresponding to the classes of interest
values_usa_range <- levels_usa$VALUE[which(levels_usa$LABEL %in% c("Rangeland"))]

# Load aggregated raster
r_usa_range_agg <- rast(here("evaluation/pasture_reference_maps/usa/usa_agg_range.tif"))


########################
### MERGE NLCD RANGE ###
########################

# Make CRS & extents match
crs(r_usa_nlcd_agg) <- crs(r_usa_nlcd)
crs(r_usa_range_agg) <- crs(r_usa_range)
r_usa_nlcd_agg <- project(r_usa_nlcd_agg, r_usa_range_agg)

# Calculate proportion of pasture per 10km grid cell
aggfactor <- 10000/30
r_usa_agg <- r_usa_nlcd_agg + r_usa_range_agg
prop_usa <- r_usa_agg/(aggfactor^2)
prop_usa[prop_usa > 1] <- 1 # some cells add to more than 1, clamp them
names(prop_usa) <- "Value"


###########################
### PREP PREDICTION MAP ###
###########################

# Project our reference map to match the global prediction rasters
proj_usa <- project(prop_usa, exp1_global)

# Crop our rasters (bounding box approx around USA)
exp1_usa <- crop(exp1_global, ext(-126, -65, 24, 49))
# exp2_usa <- crop(exp2_global, ext(-126, -65, 24, 49))
proj_usa <- crop(proj_usa, ext(-126, -65, 24, 49))

# No need to mask out GDD for conterminous USA, does not go above 50ºN latitude

# Mask out water bodies
water_body_mask_usa <- crop(water_body_mask, exp1_usa)
water_body_mask_usa <- project(water_body_mask_usa, exp1_usa)
exp1_usa <- mask(exp1_usa, water_body_mask_usa, maskvalues=0)
# exp2_usa <- mask(exp2_usa, water_body_mask_usa, maskvalues=0)
proj_usa <- mask(proj_usa, water_body_mask_usa, maskvalues=0)

# Mask out just USA (not neighbouring countries)
shp_usa <- vect(here("shapefile/USA/gadm36_USA_1.shp"))
exp1_usa <- mask(exp1_usa, shp_usa)
# exp2_usa <- mask(exp2_usa, shp_usa)
proj_usa <- mask(proj_usa, shp_usa)


####################################
### COMPARE REFERENCE+PREDICTION ###
####################################

# Calculate differences
absdif_exp1 <- proj_usa-exp1_usa
# absdif_exp2 <- proj_usa-exp2_usa


##################
### SAVE PLOTS ###
##################

# Save all_correct_to_FAO_scale
writeRaster(mask(project(r_usa_nlcd_agg, exp1_usa),shp_usa), here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa/usa_nlcd.tif"), overwrite=T)
writeRaster(mask(project(r_usa_range_agg, exp1_usa),shp_usa), here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa/usa_range.tif"), overwrite=T)
writeRaster(proj_usa, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa/usa_reference.tif"), overwrite=T)
writeRaster(exp1_usa, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_pred_map.tif")), overwrite=T)
writeRaster(absdif_exp1, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_map.tif")), overwrite=T)
hist <- hist(absdif_exp1)
histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
musigma <- data.frame(mu=mu, sigma=sigma)
write.csv(histdf, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_hist.csv")))
write.csv(musigma, here("evaluation/all_correct_to_FAO_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_musigma.csv")))

# Save all_correct_to_subnation_scale
# writeRaster(mask(project(r_usa_nlcd_agg, exp1_usa),shp_usa), here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa/usa_nlcd.tif"), overwrite=T)
# writeRaster(mask(project(r_usa_range_agg, exp1_usa),shp_usa), here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa/usa_range.tif"), overwrite=T)
# writeRaster(proj_usa, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa/usa_reference.tif"), overwrite=T)
# writeRaster(exp2_usa, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_pred_map.tif")), overwrite=T)
# writeRaster(absdif_exp2, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_map.tif")), overwrite=T)
# hist <- hist(absdif_exp2)
# histdf <- data.frame(breaks=hist$breaks[-length(hist$breaks)], counts=hist$counts, density=hist$density, mids=hist$mids)
# mu <- round(mean(values(absdif_exp1), na.rm=T), 4)
# sigma <- round(sd(values(absdif_exp1), na.rm=T), 4)
# musigma <- data.frame(mu=mu, sigma=sigma)
# write.csv(histdf, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_hist.csv")))
# write.csv(musigma, here("evaluation/all_correct_to_subnation_scale_itr3_fr_0/usa", paste0("agland_map_output_",iter,"_usa_diff_musigma.csv")))

