from osgeo import gdal
import numpy as np
import argparse
import pickle


def save_pkl(obj, directory):
    with open(directory + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_count_table(matrix, total_categories=17, invalid_index=255):
    """
    Assume the class labels start from 1 to total_categories
    :param matrix:
    :param total_categories:
    :param invalid_index:
    :return:
    """
    result_array = np.zeros((1, total_categories))
    for _, item in enumerate(matrix.flatten()):
        if item != invalid_index:
            result_array[0, int(item) - 1] += 1

    return result_array


def get_global_box_land_cover_input(world_class_dir, N=10, total_categories=17, invalid_index=255):
    """
    Pack input for models to evaluate upon
    """
    ds = gdal.Open(world_class_dir)
    world_land_cover = np.array(ds.ReadAsArray())
    h, w = world_land_cover.shape
    total_pixels = N * N  # this is the same for all partitioned map

    # Pre-allocate the input array
    # Note: when h or w is not divisible by N, fill last row or colum by the remainder
    num_box = ((h // N) + 1) * ((w // N) + 1)
    counter = 0
    processed_input_collection = np.zeros((num_box, total_categories))

    for i in np.arange(0, h, N):
        for j in np.arange(0, w, N):
            print('{} | {}'.format(i, j))
            partitioned_map = world_land_cover[i:i + N, j:j + N]  # automatically handles edge cases

            # Including spatial info as features
            processed_input = (get_count_table(partitioned_map,
                                               total_categories,
                                               invalid_index)[0, :] / total_pixels).reshape(1, -1)
            processed_input_collection[counter, :] = processed_input
            counter += 1

    packed_input = {'global_land_cover': processed_input_collection,
                    'width': w,
                    'height': h,
                    'pixel_size': N}

    return packed_input


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--land_cover_dir", type=str, default='./MCD12Q1_merged.tif',
                        help="directory of merged global MCD12Q1 product")
    parser.add_argument("--output_dir", type=str, default='../outputs/global_land_cover_for_deploy_20',
                        help="directory of output pkl file")
    parser.add_argument("--side_length", type=int, default=20,
                        help="side length of grid to count land cover types")
    parser.add_argument("--invalid_index", type=int, default=-9999,
                        help="invalid index")
    args = parser.parse_args()
    print(args)

    results = get_global_box_land_cover_input(args.land_cover_dir,
                                              N=args.side_length,
                                              total_categories=17,
                                              invalid_index=args.invalid_index)

    save_pkl(results, args.output_dir)


if __name__ == '__main__':
    main()
