from numba import cuda
import numpy as np
import math
from osgeo import gdal
from tqdm import tqdm
import argparse
from utils.io import save_pkl, load_pkl, load_yaml_config
from utils.dataset import Dataset
import pandas as pd

LAND_COVER_CFG = load_yaml_config('../configs/land_cover_cfg.yaml')
TRAINING_CFG = load_yaml_config('../configs/training_cfg.yaml')


def split_block_matrix(matrix, block_x, block_y):
    """
    Split 2D input matrix into (block_x, block_y) block matrices

    Args:
        matrix (np.array): 2D matrix
        block_x (int): block size (row-wise)
        block_y (int): block size (col-wise)

    Returns: (dict) block index(tuple) -> block matrix(np.array)
    """
    h, w = matrix.shape
    assert ((h % block_x) == 0), "Height of the matrix must be divisable by block_x"
    assert ((w % block_y) == 0), "Width of the matrix must be divisable by block_y"

    # Divide the matrix into (block_x, block_y) block matrices
    # Each block matrix has size (h//block_x, w//block_y)
    blocks = {}
    for i in range(0, h, h // block_x):
        for j in range(0, w, w // block_y):
            blocks[(int(i / (h // block_x)), int(j / (w // block_y)))] = matrix[i:i + h // block_x, j:j + w // block_y]

    return blocks


def merge_block_matrix(blocks):
    """
    Merge block matrices into single matrix

    Args:
        blocks (dict): block index(tuple) -> block matrix(np.array)

    Returns: (np.array) single matrix (2D or 3D)
    """
    dtype = list(blocks.values())[0].dtype
    block_x, block_y = list(blocks.keys())[-1]
    block_x, block_y = block_x + 1, block_y + 1

    if list(blocks.values())[0].ndim == 2:
        block_h, block_w = list(blocks.values())[0].shape
        block_z = 1
    elif list(blocks.values())[0].ndim == 3:
        block_h, block_w, block_z = list(blocks.values())[0].shape

    matrix = np.zeros((block_h * block_x, block_w * block_y, block_z), dtype=dtype)

    for block_index, block_matrix in blocks.items():
        matrix[block_index[0] * block_h:(block_index[0] + 1) * block_h,
        block_index[1] * block_w:(block_index[1] + 1) * block_w, :] = block_matrix

    return matrix


def get_blocks_land_cover_percentage_gpu(blocks, kernel_size, code_index):
    """
    Count land cover percentage for each category in a kernel_size-by-kernel_size grid
    throughout the input blocks. Output (block_matrix.shape, num_categories) for each
    block

    Args:
        blocks (dict): block index(tuple) -> block matrix(np.array)
        kernel_size (int): kernel size
        code_index (list): list of keys for land cover types

    Returns: (dict) block index(tuple) -> block matrix histogram percentage(np.array)

    """
    categories = np.asarray(code_index)
    num_categories = categories.shape[0]

    blocks_land_cover_percentage = {}
    for block_index, block_matrix in tqdm(blocks.items()):
        d_block_matrix = cuda.to_device(np.ascontiguousarray(block_matrix))
        block_output = np.zeros(
            (block_matrix.shape[0] // kernel_size, block_matrix.shape[1] // kernel_size, num_categories))
        d_block_output = cuda.to_device(block_output)

        threadsperblock = (16, 16)
        blockspergrid_x = int(math.ceil(block_matrix.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(block_matrix.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        get_land_cover_percentage[blockspergrid, threadsperblock](d_block_matrix, d_block_output, kernel_size,
                                                                  categories)
        cuda.synchronize()

        blocks_land_cover_percentage[block_index] = d_block_output.copy_to_host()

    return blocks_land_cover_percentage


def convert_land_cover_percentage_hist_to_Dataset(merged_land_cover_percentage, land_cover_code,
                                                  remove_land_cover_feature_index):
    """
    Convert land cover percentage map into Dataset obj for deployment

    Args:
        merged_land_cover_percentage (np.array): (x, y, c) shape array with c axis representing
                                                 histogram percentage for land cover
        land_cover_code (dict): class types(int) -> (str)
        remove_land_cover_feature_index (list): index of land cover type code to be
                                                removed from features

    Returns: (Dataset)
    """
    print('{} are removed from land cover feature'.format(remove_land_cover_feature_index))
    h, w, z = merged_land_cover_percentage.shape

    # ROW_IDX, COL_IDX, <land cover type>, GRID_SIZE
    row_idx = np.repeat(np.arange(h), w)
    col_idx = np.tile(np.arange(w), h)
    grid_size = np.ones_like(row_idx) * (180 * 2 / w)
    land_cover_percentage = merged_land_cover_percentage.reshape(row_idx.shape[0], z)

    return Dataset(pd.DataFrame(data=np.hstack((row_idx.reshape(-1, 1),
                                                col_idx.reshape(-1, 1),
                                                land_cover_percentage,
                                                grid_size.reshape(-1, 1))),
                                columns=['ROW_IDX', 'COL_IDX'] + list(land_cover_code.keys()) + ['GRID_SIZE']),
                   land_cover_code,
                   remove_land_cover_feature_index)


@cuda.jit
def get_land_cover_percentage(x, out, kernel_size, categories):
    """
    Numba cuda implementation of histogram percentage count for a given 2D matrix x. Counting
    is defined as (occurrences / total_valid_pixels) over a kernel_size-by-kernel_size grid
    (total_valid_pixels is the total pixels within grid that also appears in categories).
    If grid matrix does not have any values in categories, default results for each category
    is set to be 0

    Args:
        x (np.array): input 2D matrix with shape (m, n)
        out (np.array): output 3D matrix with shape (m//kernel_size, n//kernel_size, num_categories)
        kernel_size (int): kernel size
        categories (np.array): array of categories indices
    """
    i1, j1 = cuda.grid(2)
    num_categories = categories.shape[0]

    if i1 < x.shape[0] and j1 < x.shape[1]:
        if (i1 % kernel_size == 0) and (j1 % kernel_size == 0):

            total_valid_pixels = 0
            for k1 in range(kernel_size):
                for k2 in range(kernel_size):

                    for c in range(num_categories):
                        if x[i1 + k1][j1 + k2] == categories[c]:
                            out[int(i1 // kernel_size)][int(j1 // kernel_size)][c] += 1
                            total_valid_pixels += 1

            # Convert count to percentage
            if total_valid_pixels != 0:
                for i in range(num_categories):
                    out[int(i1 // kernel_size)][int(j1 // kernel_size)][i] /= total_valid_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--land_cover_dir", type=str, default='./MCD12Q1_merged.tif',
                        help="directory of merged global land cover product")
    parser.add_argument("--kernel_size", type=int, default=20,
                        help="kernel size (must be divisible by land cover shape)")
    parser.add_argument("--land_cover_product_name", type=str, default='MCD12Q1',
                        help="land cover product name")
    parser.add_argument("--output_dir", type=str, default='./pred_input_map',
                        help="directory of output pkl file")
    args = parser.parse_args()
    print(args)

    assert (args.land_cover_product_name in LAND_COVER_CFG['code']), "Unknown land cover product name"

    # Load land cover map
    land_cover_map = np.array(gdal.Open(args.land_cover_dir).ReadAsArray())

    # Divide the land cover map into (5,5) 8640x17280 block matrices
    # Note: if GPU vram allows, could reduce block_x and block_y to feed larger
    #       image into GPU at once
    block_land_cover = split_block_matrix(land_cover_map, block_x=5, block_y=5)

    # Ouptut histogram percentage for each block
    block_output_hist = get_blocks_land_cover_percentage_gpu(block_land_cover,
                                                             args.kernel_size,
                                                             list(LAND_COVER_CFG['code']
                                                                  [args.land_cover_product_name].keys()))

    # Merge blocks
    output_hist = merge_block_matrix(block_output_hist)

    # Convert merged output into Dataset obj
    output_hist_dataset = convert_land_cover_percentage_hist_to_Dataset(output_hist,
                                                                        LAND_COVER_CFG['code']
                                                                        [args.land_cover_product_name],
                                                                        TRAINING_CFG['feature_remove'])
    # Save file
    if args.output_dir is not None:
        save_pkl(output_hist_dataset, args.output_dir)
        print('{} is saved'.format(args.output_dir))


if __name__ == '__main__':
    main()
