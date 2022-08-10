"""
Modified Pycnophylactic Interpolation function from:
https://github.com/danlewis85/pycno/
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import numpy as np
from numpy.ma import masked_invalid, masked_where
from pandas import DataFrame
from rasterio.features import rasterize


def pycno(gdf,
          value_field,
          x_min,
          y_max,
          pixel_size,
          r=0.2,
          handle_null=True,
          seperable_filter=[0.5, 0.0, 0.5],
          converge=3,
          type_data='intensive',
          verbose=True):
    """
    Returns a smooth pycnophylactic interpolation raster for a given geodataframe

    Modified: 
    This pycno interpolation is used to smooth the bias correction factor matrices, 
    input gdf is a weight table shapefile on global regions appeared in FAOSTAT and subnational 
    census. Unspecified regions have weights 1 as defualt

    Args:
    gdf (geopandas.geodataframe.GeoDataFrame): Input GeoDataFrame.
    value_field (str): Field name of values to be used to produce pycnophylactic surface
    x_min (float): x min in geo transform
    y_max (float): y max in geo transform
    pixel_size (float):  pixel size in geo transform
    r (float, optional): Relaxation parameter, default of 0.2 is generally fine.
    handle_null (boolean, optional): Changes how nodata values are smoothed. Default True.
    seperable_filter (list, optional): seperable filter to be applied as mean filter. Default [0.5, 0.0, 0.5].
    converge (int, optional): Index for stopping value, default 3 is generally fine.
    type_data (str, optional): data type must be either 'intensive' or 'extensive'. Default 'intensive'.
    verbose (boolean, optional): Print out progress at each iteration.

    Returns:
    Numpy Array: Smooth pycnophylactic interpolation.
    Rasterio geotransform
    GeoPandas crs
    """

    # The basic numpy convolve function doesn't handle nulls.
    def smooth2D(data):

        # Create function that calls a 1 dimensionsal smoother.
        s1d = lambda s: np.convolve(s, seperable_filter, mode="same")
        # pad the data array with the mean value
        padding_size = int((len(seperable_filter) - 1) / 2)
        padarray = np.pad(data,
                          padding_size,
                          "constant",
                          constant_values=np.nanmean(data))
        # make nodata mask
        mask = masked_invalid(padarray).mask
        # set nodata as zero to avoid eroding the raster
        padarray[mask] = 0.0
        # Apply the convolution along each axis of the data and average
        padarray = (np.apply_along_axis(s1d, 1, padarray) +
                    np.apply_along_axis(s1d, 0, padarray)) / 2
        # Reinstate nodata
        padarray[mask] = np.nan
        return padarray[padding_size:-padding_size, padding_size:-padding_size]

    # The convolution function from astropy handles nulls.
    def astroSmooth2d(data):

        s1d = lambda s: astro_convolve(s, seperable_filter)
        # pad the data array with the mean value
        padding_size = int((len(seperable_filter) - 1) / 2)
        padarray = np.pad(data,
                          padding_size,
                          "constant",
                          constant_values=np.nanmean(data))
        # Apply the convolution along each axis of the data and average
        padarray = (np.apply_along_axis(s1d, 1, padarray) +
                    np.apply_along_axis(s1d, 0, padarray)) / 2
        return padarray[padding_size:-padding_size, padding_size:-padding_size]

    def correct2Da(data):

        for idx, val in gdf[value_field].iteritems():
            # Create zone mask from feature_array
            mask = masked_where(feature_array == idx, feature_array).mask
            # Work out the correction factor
            correct = (val - np.nansum(data[mask])) / mask.sum()
            # Apply correction
            data[mask] += correct

        return data

    def correct2Dm(data):

        for idx, val in gdf[value_field].iteritems():
            # Create zone mask from feature_array
            mask = masked_where(feature_array == idx, feature_array).mask
            # Work out the correction factor
            correct = val / np.nansum(data[mask])
            if correct != 0.0:
                # Apply correction
                data[mask] *= correct

        return data

    try:
        from astropy.convolution import convolve as astro_convolve
    except (ImportError, ModuleNotFoundError):
        raise ImportError(
            "Pycnophylactic interpolation requires the astropy package")

    assert (type_data in ['intensive', 'extensive'
                          ]), "data type must be either intensive or extensive"

    # set nodata value
    # This nodata does not get involved in convolution, ok to hard code
    nodata = -9999

    x_max = -x_min
    y_min = -y_max
    xres = int((x_max - x_min) / pixel_size)
    yres = int((y_max - y_min) / pixel_size)

    # Work out transform so that we rasterize the area where the data are!
    trans = rasterio.Affine.from_gdal(x_min, pixel_size, 0, y_max, 0,
                                      -pixel_size)

    # First make a zone array
    # NB using index values as ids can often be too large/alphanumeric. Limit is int32 polygon features.
    # create a generator of geom, index pairs to use in rasterizing
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.index))
    # burn the features into a raster array
    feature_array = rasterize(shapes=shapes,
                              fill=nodata,
                              out_shape=(yres, xres),
                              transform=trans)

    # Get cell counts per index value (feature)
    unique, count = np.unique(feature_array, return_counts=True)
    cellcounts = np.asarray((unique, count)).T
    # Lose the nodata counts
    cellcounts = cellcounts[cellcounts[:, 0] != nodata, :]
    # Adjust value totals by cells
    # Make cell counts dataframe
    celldf = DataFrame(cellcounts[:, 1],
                       index=cellcounts[:, 0],
                       columns=["cellcount"])
    # Merge cell counts
    gdf = gdf.merge(celldf, how="left", left_index=True, right_index=True)

    # Adjust the input data for intensive variables
    if type_data == 'intensive':
        gdf[value_field] *= gdf["cellcount"]

    # Calculate cell values
    # create a generator of geom, cellvalue pairs to use in rasterizing
    # density array
    gdf["cellvalues"] = gdf[value_field] / gdf["cellcount"]

    shapes = ((geom, value)
              for geom, value in zip(gdf.geometry, gdf.cellvalues))
    # Now burn the initial value raster
    value_array = rasterize(shapes=shapes,
                            fill=nodata,
                            out_shape=(yres, xres),
                            transform=trans)

    # Set no data and empty weights (0s) to 1
    value_array[value_array == nodata] = 1
    # value_array[value_array == 0] = 1

    # Set stopper value based on converge parameter
    stopper = np.nanmax(value_array) * np.power(10.0, -converge)

    while True:

        # Store the current iteration
        old = np.copy(value_array)

        # Smooth the value_array
        if handle_null:
            sm = astroSmooth2d(value_array)
        else:
            sm = smooth2D(value_array)

        # Relaxation to prevent overcompensation in the smoothing step
        value_array = value_array * r + (1.0 - r) * sm

        # Perform correction
        value_array = correct2Da(value_array)

        # Reset any negative values to zero.
        value_array[value_array < 0] = 0.0

        # Perform correction
        value_array = correct2Dm(value_array)

        if verbose:
            print("Maximum Change: " +
                  str(round(np.nanmax(np.absolute(old - value_array)), 4)) +
                  " - will stop at " + str(round(stopper, 4)))

        if np.nanmax(np.absolute(old - value_array)) < stopper:
            break

    return (value_array, trans, gdf.crs)