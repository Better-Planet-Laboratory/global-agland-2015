import geopandas as gpd
from utils.io import gdal, save_shp_as_tif

# Antarctica mask that has Antarctica region filled with 0s, the rest with 1s
antarctica_df = gpd.read_file('../shapefile/Antarctica/gadm41_ATA_0.shp')
antarctica_df['MASK_VALUE'] = 0
antarctica_df.to_file('antarctica_mask.shp')

# Convert shp to tif
# Output tif has size 2160 x 4320
save_shp_as_tif('antarctica_mask.tif',
                './antarctica_mask.shp',
                'MASK_VALUE',
                x_min=-180,
                y_max=90,
                pixel_size=0.083333333333333333333,
                epsg=4326,
                no_data_value=1,
                dtype=gdal.GDT_Float32)