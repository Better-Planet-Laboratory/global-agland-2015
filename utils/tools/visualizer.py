import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2

font = {'family': 'monospace',
        'weight': 'normal',
        'size': 20}

matplotlib.rc('font', **font)


def plot_gdd_map(gdd_array, output_dir=None, nodata=-32768, cmap='viridis'):
    """
    Plot GDD map and save as png if output_dir is specified

    Args:
        gdd_array (np.array): 2D np array
        output_dir (str): output dir (Default: None)
        nodata (int): no data indicator (Default: -32768)
        cmap (str or dict): matplotlib cmap
    """
    assert (gdd_array.ndim == 2), "Input gdd array must be 2D"
    bool_flag = True if np.unique(gdd_array).size == 2 else False
    cmap = plt.get_cmap(cmap, 2) if bool_flag else cmap

    # Hide invalid GDD values
    gdd_array_copy = gdd_array.copy()
    gdd_array_copy[gdd_array_copy == nodata] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    im = plt.imshow(gdd_array_copy, cmap=cmap)
    plt.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=-1, pack_start=True)
    fig.add_axes(cax)

    if bool_flag:
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, 1])
        cbar.ax.set_xticklabels(['0: Exclude', '1: Include'])
    else:
        cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_global_area_map(area_array, output_dir=None, scale=10):
    """

    Args:
        area_array (np.array): 2D np array
        output_dir (str): output dir (Default: None)
        scale (int): scale factor to be applied on image size (1/scale)
    """
    assert(scale > 0), "scale cannot be non negative"
    h, w = area_array.shape

    # Reduce image size for visualization
    area_array = cv2.resize(area_array, dsize=(int(w) // scale, int(h) // scale),
                            interpolation=cv2.INTER_NEAREST)

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(area_array, cmap='PuBuGn')
    plt.axis('off')

    axins = inset_axes(ax,
                       width="50%",
                       height="10%",
                       bbox_to_anchor=(0.165, 0.01, 1, 0.3),
                       bbox_transform=ax.transAxes,
                       borderpad=8.5
                       )
    cbar = fig.colorbar(im, cax=axins, orientation='horizontal')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_land_cover_map(land_cover_array, class_lookup, output_dir=None, nodata=-9999, scale=10):
    """
    Plot land cover map and save as png if output_dir is specified

    Args:
        land_cover_array (np.array): 2D np array
        class_lookup (dict): class index (int) -> class name (str)
        output_dir (str): output dir (Default: None)
        nodata (int): no data indicator (Default: -9999)
        scale (int): scale factor to be applied on image size (1/scale)
    """
    assert(scale > 0), "scale cannot be non negative"
    h, w = land_cover_array.shape
    class_name_list = list(class_lookup.values())
    class_index_list = list(class_lookup.keys())
    assert (len(class_name_list) <= 17), "Number of classes exceeds 17, add color in default colormap"

    # Reduce image size for visualization
    land_cover_array = cv2.resize(land_cover_array, dsize=(int(w) // scale, int(h) // scale),
                                  interpolation=cv2.INTER_NEAREST)

    # Remove null values
    land_cover_array[land_cover_array == nodata] = np.nan

    # Default colormap
    cmap = colors.ListedColormap(['#193300', '#00994C', '#CCCC00', '#CC6600', '#CCFFE5',
                                  '#4C0099', '#0000CC', '#CC0000', '#FF9933', '#99FFFF',
                                  '#003333', '#660033', '#000033', '#6666FF', '#FFFFCC',
                                  '#E5FFCC', '#E5CCFF'])
    boundaries = [min(class_index_list) - 0.5] + [i + 0.5 for i in class_index_list]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(land_cover_array, cmap=cmap, norm=norm)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0
                       )
    cbar = fig.colorbar(im, cax=axins, orientation='vertical',
                        ticks=class_index_list)
    cbar.ax.set_yticklabels(class_name_list)
    cbar.ax.tick_params(labelsize=10.5)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()
