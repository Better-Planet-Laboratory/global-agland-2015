import yaml
import pickle
from osgeo import gdal, osr, ogr
import numpy as np


def load_yaml_config(file_path):
    """ Load .yaml config files for training """
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_pkl(obj, directory):
    """ Save obj as pkl file """
    with open(directory + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(directory):
    """ Load pkl file """
    with open(directory + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_shp_as_tif(dst_filename, shp_file, attribute_name, x_min, y_max, pixel_size, epsg=4326, 
                    no_data_value=255, dtype=gdal.GDT_Float32):
    """
    Convert shp file with geometry and attribute to TIF file with projection 
    x_min, -x_min, -y_max, y_max

    Args:
        dst_filename (str): output tif file dir
        shp_file (str): input shapefile dir
        attribute_name (str): attribute name in shapefile
        x_min (float): x min in geo transform
        y_max (float): y max in geo transform
        pixel_size (float):  pixel size in geo transform
        epsg (int): epsg in geo transform
        no_data_value (int or float): replacement for nan (Default: 255)
        dtype (gdal dtype): data type
    """
    shp_ds = ogr.Open(shp_file)
    shp_layer = shp_ds.GetLayer()

    cols = int((2*(-x_min)) / pixel_size)
    rows = int((2*y_max) / pixel_size)

    target_ds = gdal.GetDriverByName('GTiff').Create(dst_filename, cols, rows, 1, dtype) 
    target_ds.SetGeoTransform((x_min, pixel_size, 0, -y_max, 0, pixel_size))
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    gdal.RasterizeLayer(target_ds, [1], shp_layer, options = ["ATTRIBUTE={}".format(attribute_name)])  
    target_dsSRS = osr.SpatialReference()
    target_dsSRS.ImportFromEPSG(epsg)
    target_ds.SetProjection(target_dsSRS.ExportToWkt())
    target_ds = None
    print('{} saved'.format(dst_filename))


def save_array_as_tif(dst_filename, data_array, x_min, y_max, pixel_size, epsg=4326,
                      no_data_value=255, dtype=gdal.GDT_UInt16):
    """
    TIF file contains projection transform info along with the image data
    x_min (along horizontal) indicates smallest column geo value on the left
    y_max (along vertical) indicates greatest row geo value on the top
    (x_min, y_max) indicates the geo coordinates in the top left corner

    Code example:
    https://gis.stackexchange.com/questions/58517/python-gdal-save-array-as-raster-with-projection-from-other-file

    gdal SetGeoTransform explanations:
    https://stackoverflow.com/questions/27166739/description-of-parameters-of-gdal-setgeotransform

    Args:
        dst_filename (str): output tif file dir
        data_array (np.array): 2D array to be saved
        x_min (float): x min in geo transform
        y_max (float): y max in geo transform
        pixel_size (float):  pixel size in geo transform
        epsg (int): epsg in geo transform
        no_data_value (int or float): replacement for nan (Default: 255)
        dtype (gdal dtype): data type
    """
    x_pixels, y_pixels = data_array.shape

    driver = gdal.GetDriverByName('GTiff')
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)

    dataset = driver.Create(
        dst_filename,
        y_pixels, x_pixels, 1, dtype, )

    dataset.SetGeoTransform((
        x_min, pixel_size, 0,
        y_max, 0, -pixel_size))

    # Check if data contains NaN, we need to set a NoDataValue
    # Make sure this value is not in data
    assert (no_data_value not in data_array)
    data_array[np.isnan(data_array)] = no_data_value
    dataset.GetRasterBand(1).SetNoDataValue(no_data_value)

    dataset.GetRasterBand(1).WriteArray(data_array)
    dataset.SetProjection(srs.ExportToWkt())
    dataset.FlushCache()  # Write to disk
    print('{} saved'.format(dst_filename))
