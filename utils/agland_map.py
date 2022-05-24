import numpy as np
from scipy.special import softmax
from osgeo import gdal, osr
import rasterio


def load_tif_as_AglandMap(input_dir, force_load=False):
    """
    Load tif as AglandMap obj. Input tif must have 3 bands, in the order of cropland,
    pasture and other map

    Args:
        input_dir (str): directory of GeoTIFF file with 3 bands
        force_load (bool): force the agland map to follow probability distribution
                           by calling CORRECTION_SCALE method (Default: False)
    Returns: (AglandMap)
    """
    data = rasterio.open(input_dir)
    assert (data.count == 3), "Input GeoTIFF must have 3 bands"

    return AglandMap(cropland_array=data.read(1),
                     pasture_array=data.read(2),
                     other_array=data.read(3), force_load=force_load)


class AglandMap:
    # Index in data slice for each agland type
    CROPLAND_IDX = 0
    PASTURE_IDX = 1
    OTHER_IDX = 2

    # Probability correction method
    CORRECTION_SCALE = 'scale'
    CORRECTION_SOFTMAX = 'softmax'

    def __init__(self, cropland_array, pasture_array, other_array, nodata=-1, force_load=False):
        """
        Constructor that takes cropland, pasture and other probability map that makes
        an agland map

        Args:
            cropland_array (np.array): 2D matrix for cropland probability
            pasture_array (np.array): 2D matrix for pasture probability
            other_array (np.array): 2D matrix for other probability
            nodata (int): indicator for missing data
            force_load (bool): force the agland map to follow probability distribution
                               by calling CORRECTION_SCALE method (Default: False)
        """
        assert (cropland_array.ndim == pasture_array.ndim == other_array.ndim == 2), \
            "cropland, pasture and other map must be 2D"

        self.height, self.width = cropland_array.shape
        self.data = np.zeros((self.height, self.width, 3))
        self.data[:, :, AglandMap.CROPLAND_IDX] = cropland_array
        self.data[:, :, AglandMap.PASTURE_IDX] = pasture_array
        self.data[:, :, AglandMap.OTHER_IDX] = other_array
        self.data[np.where(self.data == nodata)] = np.nan

        if not force_load:
            assert (((cropland_array + pasture_array + other_array) == np.ones_like(cropland_array)).all()), \
                "Input arrays must sum up to 1 (probability distribution)"
        else:
            self._prob_correct()

    def _prob_correct(self, method='scale'):
        """
        Correct agland map data by forcing each sample to be in probability distribution. Method
        could be 'scale' (applying a scaling factor) or 'softmax'

        Args:
            method (str): correction method, 'scale' or 'softmax' (Default: 'scale')
        """
        assert (method in [AglandMap.CORRECTION_SCALE, AglandMap.CORRECTION_SOFTMAX]), \
            "Unknown correction method"

        # Correct the updated values to probability distribution
        if method == AglandMap.CORRECTION_SCALE:
            # Apply scaling factor to force sum of values to be 1
            scaling_factor_matrix = np.sum(self.data, axis=2)
            for c in range(3):
                self.data[:, :, c] = np.divide(self.data[:, :, c], scaling_factor_matrix)

        elif method == AglandMap.CORRECTION_SOFTMAX:
            # Apply softmax to each sample
            self.data = softmax(self.data, axis=2)

    def get_cropland(self):
        """
        Return cropland probability map slice

        Returns: (np.array) cropland map
        """
        return self.data[:, :, AglandMap.CROPLAND_IDX]

    def get_pasture(self):
        """
        Return pasture probability map slice

        Returns: (np.array) pasture map
        """
        return self.data[:, :, AglandMap.PASTURE_IDX]

    def get_other(self):
        """
        Return other probability map slice

        Returns: (np.array) other map
        """
        return self.data[:, :, AglandMap.OTHER_IDX]

    def apply_factor(self, mask_index_cropland, mask_index_pasture, mask_index_other,
                     factor_cropland, factor_pasture, factor_other, correction_method='scale'):
        """
        Apply scalar multiplication factors to each channel (cropland, pasture, other). Since
        the AglandMap obj data must always follow a probability distribution, a correction_method
        needs to be specified (Default: 'scale') to make the factored values back to probability
        distribution range (sum up to 1)

        Args:
            mask_index_cropland (tuple of np.array): indices pairs indicating which pixels to
                                                     be factored in cropland slice
            mask_index_pasture (tuple of np.array): indices pairs indicating which pixels to
                                                    be factored in pasture slice
            mask_index_other (tuple of np.array): indices pairs indicating which pixels to
                                                  be factored in other slice
            factor_cropland (float): factor to be applied on cropland
            factor_pasture (float): factor to be applied on pasture
            factor_other (float): factor to be applied on other
            correction_method (str): correction method, 'scale' or 'softmax' (Default: 'scale')
        """
        assert (correction_method in [AglandMap.CORRECTION_SCALE, AglandMap.CORRECTION_SOFTMAX]), \
            "Unknown correction method"

        # Apply factor to cropland, pasture and other channel
        cropland_map_corr = self.get_cropland()
        pasture_map_corr = self.get_pasture()
        other_map_corr = self.get_other()

        cropland_map_corr[mask_index_cropland] *= factor_cropland
        pasture_map_corr[mask_index_pasture] *= factor_pasture
        other_map_corr[mask_index_other] *= factor_other

        self.data[:, :, AglandMap.CROPLAND_IDX] = cropland_map_corr
        self.data[:, :, AglandMap.PASTURE_IDX] = pasture_map_corr
        self.data[:, :, AglandMap.OTHER_IDX] = other_map_corr

        # Correct the updated values to probability distribution
        self._prob_correct(correction_method)

    def save_as_tif(self, output_dir):
        """
        Save AglandMap obj as a GeoTIFF file, with RasterBand cropland, pasture and other. Projection
        is set to be EPSG:4326

        Args:
            output_dir (str): path directory to the output map
        """
        epsg = 4326
        x_min = -180
        y_max = 90
        pixel_size = abs(x_min) * 2 / self.width

        driver = gdal.GetDriverByName('GTiff')
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)

        dataset = driver.Create(
            output_dir,
            self.width, self.height, 3, gdal.GDT_Float64, )

        dataset.SetGeoTransform((
            x_min, pixel_size, 0,
            y_max, 0, -pixel_size))

        dataset.GetRasterBand(1).WriteArray(self.get_cropland())
        dataset.GetRasterBand(2).WriteArray(self.get_pasture())
        dataset.GetRasterBand(3).WriteArray(self.get_other())

        dataset.SetProjection(srs.ExportToWkt())
        dataset.FlushCache()  # Write to disk
        print('{} saved'.format(output_dir))
