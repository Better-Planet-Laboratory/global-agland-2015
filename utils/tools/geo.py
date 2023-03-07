import geopandas as gpd
from shapely.ops import unary_union
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import mapping
import warnings
import rasterio
from rasterio.mask import mask


def polygon_union(polygon_list):
    """
    Cascaded union polygons in the input polygon_list into a single polygon

    Args:
        polygon_list (list of shapely poly): input list of shapely polygons

    Returns: (shapely.poly)
    """
    # Ignore ShapelyDeprecationWarning with geoms, currently using shapely-1.8.0
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    if len(polygon_list) == 0:
        return polygon_list
    else:
        return unary_union(polygon_list)


def get_border(index, shapefile):
    """
    Based on index, return the multipolygon from shapefile
    shapefile must have attribute geometry

    Args:
        index (int): index
        shapefile (pd): Dataframe

    Returns: (list) list of independent geometry
    """
    return [mapping(shapefile.iloc[index].geometry)]


def crop_intermediate_state(array, affine, census_table, index, crop=False):
    """
    Crop index of census_table state on array with affine, return a
    cropped matrix with nodata to be -1

    Args:
        array (np.ndarray): 2D map
        affine (affine.Affine): transform
        census_table (pd.DataFrame): input census table to be matched to
        index (int): index in census table
        crop (bool): crop (Default: False)

    Returns: (np.ndarray)
    """
    # Load numpy array as rasterio.Dataset from memory directly
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(driver='GTiff',
                          height=array.shape[0],
                          width=array.shape[1],
                          count=1,
                          dtype=array.dtype,
                          transform=affine) as dataset:
            dataset.write(array, 1)

        with memfile.open() as dataset:
            out, _ = mask(dataset,
                          get_border(index, census_table),
                          crop=crop,
                          nodata=-1)

    return out[0]
