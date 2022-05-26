from pyhdf.SD import SD, SDC
import numpy as np
from utils.io import save_array_as_tif
from osgeo import gdal, osr

# We use MCD12C1 product class 0: 'WATER BODIES' to extract a
# water body mask for later use
file = SD('land_cover/MCD12C1.A2015001.006.2018053185652.hdf', SDC.READ)
datasets_dic = file.datasets()
sds_obj = file.select('Majority_Land_Cover_Type_1')  # select sds
land_cover = sds_obj.get()  # get sds data

# Mask
land_cover[np.where(land_cover != 0)] = 1

save_array_as_tif('./water_body_mask.tif', land_cover,
                  x_min=-180, y_max=90,
                  pixel_size=0.05, epsg=4326,
                  no_data_value=-1,
                  dtype=gdal.GDT_UInt16)
