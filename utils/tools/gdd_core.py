import cv2
import numpy as np
from ..io import save_array_as_tif
from .visualizer import plot_gdd_map


def generate_GDD_filter(xyz_file_dir, xyz_shape, grid_size, output_dir,
                        x_min, y_max, epsg, criteria, *args):
    """
    GDD filter is used to filter samples from census_table that meets
    GDD criteria, therefore the excluded samples are not considered for training
    Output boolean mask is saved as GeoTIFF

    Args:
        xyz_file_dir (str): path dir to gdd.xyz.txt
        xyz_shape (tuple): matrix size
        grid_size (float): grid size in geo transform
        output_dir (str): output dir for GeoTIFF
        x_min (float): x min in geo transform
        y_max (float): y max in geo transform
        epsg (int): epsg in geo transform
        criteria (func): criteria function
        *args: arguments for criteria function
    """
    gdd = GDD(xyz_file_dir, xyz_shape, grid_size)
    _, gdd_filter_map = gdd.get_mask(criteria, *args)
    save_array_as_tif(output_dir, gdd_filter_map,
                      x_min=x_min, y_max=y_max, pixel_size=grid_size,
                      epsg=epsg)


def generate_GDD_mask(xyz_file_dir, xyz_shape, input_grid_size,
                      output_dir, output_grid_size, criteria, *args):
    """
    GDD mask map is intended to be applied to the final prediction product. In
    order to save storage and make the process time shorter, mask indices are
    saved as csv instead of raw GeoTIFF for a high product resolution

    To reconstruct gdd_mask matrix, do:
        mask_index = MaskIndex('gdd/gdd_mask_index.csv')
        gdd_mask = mask_index.create_mask_map(width=4320, height=2160)

    Args:
        xyz_file_dir (str): path dir to gdd.xyz.txt
        xyz_shape (tuple): matrix size
        input_grid_size (float): grid size used in gdd xyz file
        output_dir (str): output csv dir
        output_grid_size (float): grid size used in the final product
        criteria (func): criteria function
        *args: arguments for criteria function
    """
    gdd = GDD(xyz_file_dir, xyz_shape=xyz_shape, grid_size=input_grid_size)
    gdd.rescale(output_grid_size)
    mask_index, _ = gdd.get_mask(criteria, *args)
    mask_index.save(output_dir)


class GDD:

    def __init__(self, xyz_file_dir, xyz_shape=(360, 720), grid_size=0.5):
        """
        GDD constructor that takes in GDD xyz file, GDD map size and grid_size
        Map projects -90, 90, -180, 180 with resolution grid_size

        Args:
            xyz_file_dir (str): path dir to gdd.xyz.txt
            xyz_shape (tuple): matrix size (Default: (360, 720))
            grid_size (float): grid size (Default: 0.5)
        """
        h, w = xyz_shape[0], xyz_shape[1]
        self.gdd_map = np.loadtxt(xyz_file_dir)[:, 2].reshape(h, w)
        self.lat = np.arange(-90, 90, grid_size)
        self.lon = np.arange(-180, 180, grid_size)

    def rescale(self, grid_size, interpolation=cv2.INTER_LINEAR):
        """
        Rescale gdd map to match new input grid_size

        Args:
            grid_size (float): grid size
            interpolation (int): cv2 interpolation method

        Returns: (GDD)
        """
        self.lat = np.arange(-90, 90, grid_size)
        self.lon = np.arange(-180, 180, grid_size)
        gdd_map_scaled = cv2.resize(self.gdd_map, dsize=(self.lon.size, self.lat.size),
                                    interpolation=interpolation)
        self.gdd_map = gdd_map_scaled

        return self

    def get_mask(self, criteria, *args):
        """
        Based on input masking criteria, get output mask index and
        boolean mask matrix

        Args:
            criteria (func): criteria function
            *args: arguments for criteria function

        Returns: (MaskIndex, np.ndarray)
        """
        mask_index = MaskIndex(criteria(self.gdd_map, self.lat, self.lon, *args))
        gdd_map_mask = np.ones((self.lat.size, self.lon.size), dtype=bool)
        gdd_map_mask[mask_index.index_tuple] = False

        return mask_index, gdd_map_mask

    def set_gdd_map(self, gdd_map, grid_size):
        """
        Replace current gdd_map by the new input, update lat and lon based on
        input grid_size

        Args:
            gdd_map (np.ndarray): 2D np array
            grid_size (float): grid size
        """
        assert (gdd_map.ndim == 2), "Input gdd array must be 2D"

        self.gdd_map = gdd_map
        self.lat = np.arange(-90, 90, grid_size)
        self.lon = np.arange(-180, 180, grid_size)

    def plot(self, output_dir=None, nodata=-32768, cmap='viridis'):
        """
        Plot GDD map and save as png if output_dir is specified

        Args:
            gdd_array (np.ndarray): 2D np array
            output_dir (str): output dir (Default: None)
            nodata (int): no data indicator (Default: -32768)
            cmap (str or dict): matplotlib cmap
        """
        plot_gdd_map(self.gdd_map, output_dir, nodata, cmap)


class MaskIndex:

    @staticmethod
    def load_from_csv(csv_file, delimiter=','):
        """ Load mask indices tuple array from csv file """
        try:
            index_array = np.loadtxt(csv_file, delimiter)
        except FileNotFoundError:
            raise FileNotFoundError("File {} not found".format(csv_file))

        assert (index_array.shape[1] == 2), "Only accept 2D mask index"
        mask_index = (index_array[:, 0].reshape(-1), index_array[:, 1].reshape(-1))
        return mask_index

    @staticmethod
    def save_as_csv(csv_file, mask_index, delimiter=','):
        """
        Save mask_index tuple array as a 2 column csv files, where
        the first column represents the row index, and second as column index
        """
        x_idx, y_idx = mask_index
        np.savetxt(csv_file, np.concatenate([x_idx.reshape(-1, 1),
                                             y_idx.reshape(-1, 1)], axis=1), delimiter=delimiter)
        print('{} saved'.format(csv_file))

    def __init__(self, arg):
        """
        MaskIndex constructor that takes either a path dir to index csv file or a tuple array

        Args:
            arg (str or tuple array): path dir or index tuple array
        """
        if isinstance(arg, tuple):
            assert (len(arg) == 2), "Only accept 2D mask index"
            assert (isinstance(arg[0], np.ndarray) and isinstance(arg[1], np.ndarray)), "Input must be tuple array"
            self.index_tuple = arg

        elif isinstance(arg, str):
            self.index_tuple = MaskIndex.load_from_csv(arg)
        else:
            raise ValueError("Input arg must be either tuple array or str")

    def create_mask_map(self, width=4320, height=2160):
        """
        Reconstruct mask map from mask_index, with indiced values to be False (excluded),
        otherwise True (included)

        Args:
            width (int): width of mask map
            height (int): height of mask map

        Returns: (np.ndarray) boolean mask matrix
        """
        assert (np.max(self.index_tuple[0]) < height), "height too small"
        assert (np.max(self.index_tuple[1]) < width), "width too small"

        mask_map = np.ones((height, width), dtype=bool)
        mask_map[self.index_tuple] = False
        return mask_map

    def save(self, file_dir, delimiter=','):
        """
        Save MaskIndex index tuple as csv

        Args:
            file_dir (str): output csv path dir
            delimiter (str): delimiter (Default: ',')
        """
        MaskIndex.save_as_csv(file_dir, self.index_tuple, delimiter=delimiter)
