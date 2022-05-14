import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
