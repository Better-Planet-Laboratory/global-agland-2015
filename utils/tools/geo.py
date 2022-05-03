import geopandas as gpd
from shapely.ops import unary_union
from shapely.errors import ShapelyDeprecationWarning
import warnings


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
