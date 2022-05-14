####################################
####################################
########## NAMIBIA AGLAND ##########
####################################
####################################

# 2020.12.14

###############
### SUMMARY ###
###############

# The Agland2015 project seeks to create an updated global cropland and pastureland map centered around 2015
# First step is acquiring agricultural census data for countries
# Data for Namibia was available through https://nsa.org.na/microdata1/index.php/catalog/33/get-microdata
# But is in .dta format
# This script is to parse the data and export a useful csv table


##############
### SET UP ###
##############

# Install necessary packages
# install.packages("readstata13")
library(readstata13)
library(dplyr)
library(here)

# Locate file
namfile <- "./S3_S9_land_use_area_measurement_anonym.dta"


#################
### LOAD DATA ###
#################

df <- read.dta13(namfile)
View(df)


######################################
### EXTRACT CROPLAND & PASTURELAND ###
######################################

crop <- df %>%
  select(region, idhh, weight, q0301_crop_name, average_size) %>%
  filter(!(q0301_crop_name %in% c(" ", "Natural Forest Trees", "Other Forestry Trees", "Grazing Land", "Homestead", "Other Land", "Kraal", "Total parcel"))) %>%
  group_by(region, idhh, weight) %>%
  summarize(SUMSIZE = sum(average_size),
            WGTSUM = weight*SUMSIZE) %>%
  distinct() %>%
  ungroup(idhh, weight) %>%
  summarize(CROPLAND = sum(WGTSUM))

past <- df %>%
  select(region, idhh, weight, q0301_crop_name, average_size) %>%
  filter(q0301_crop_name %in% c("Grazing Land")) %>%
  group_by(region, idhh, weight) %>%
  summarize(SUMSIZE = sum(average_size),
            WGTSUM = weight*SUMSIZE) %>%
  distinct() %>%
  ungroup(idhh, weight) %>%
  summarize(PASTURE = sum(WGTSUM))

out <- full_join(crop, past, by="region")


##################
### WRITE DATA ###
##################

write.csv(df, here("S3_S9_land_use_area_measurement_anonym.csv"))
write.csv(out, here("subnational_stats/Namibia", "Namibia.csv"))



