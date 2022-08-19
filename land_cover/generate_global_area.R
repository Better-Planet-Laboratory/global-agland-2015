# This script is used to generate global area approximation for a square grid
library(raster)
library(rgdal)

# r <- raster(ncol=86400, nrows=43200)
r <- raster(ncol=4320, nrows=2160)
e <- extent(c(-180, 180,
              -90, 90))
extent(r) <- e
a <- area(r)

# writeRaster(a, './global_area_43200x86400.tif')
writeRaster(a, './global_area_2160x4320.tif')

plot(a)

