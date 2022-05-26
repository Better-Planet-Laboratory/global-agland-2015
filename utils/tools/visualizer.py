import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2
from geopandas import GeoDataFrame
from ..metrics import *

font = {'family': 'monospace',
        'weight': 'normal',
        'size': 20}

matplotlib.rc('font', **font)

CROPLAND_CMAP10 = colors.ListedColormap([
    '#FEDFC9', '#FFCBA9', '#FFAE78',
    '#FF9147', '#FF7416', '#EC5F00',
    '#BB4B00', '#923B00', '#7A3100',
    '#491D00'
])

PASTURE_CMAP10 = colors.ListedColormap([
    '#E2FEE2', '#B9FEB9', '#78FF78',
    '#06FF06', '#00D400', '#00A300',
    '#007200', '#005100', '#003900',
    '#001800'
])

OTHER_CMAP10 = colors.ListedColormap([
    '#c4c4c4', '#a0a0a0', '#8e8e8e',
    '#7c7c7c', '#6b6b6b', '#595959',
    '#474747', '#353535', '#232323',
    '#000000'
])

LAND_COVER_CMAP17 = colors.ListedColormap([
    '#193300', '#00994C', '#CCCC00', '#CC6600', '#CCFFE5',
    '#4C0099', '#0000CC', '#CC0000', '#FF9933', '#99FFFF',
    '#003333', '#660033', '#000033', '#6666FF', '#FFFFCC',
    '#E5FFCC', '#E5CCFF'
])


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
    assert (scale > 0), "scale cannot be non negative"
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
    assert (scale > 0), "scale cannot be non negative"
    h, w = land_cover_array.shape
    class_name_list = list(class_lookup.values())
    class_index_list = list(class_lookup.keys())
    assert (len(class_name_list) <= 17), "Number of classes exceeds 17, add color in default colormap"

    # Reduce image size for visualization
    land_cover_array = cv2.resize(land_cover_array, dsize=(int(w) // scale, int(h) // scale),
                                  interpolation=cv2.INTER_NEAREST)

    # Remove null values
    land_cover_array[land_cover_array == nodata] = np.nan

    # Use land cover default colormap
    boundaries = [min(class_index_list) - 0.5] + [i + 0.5 for i in class_index_list]
    norm = colors.BoundaryNorm(boundaries, LAND_COVER_CMAP17.N, clip=True)

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(land_cover_array, cmap=LAND_COVER_CMAP17, norm=norm)
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


def plot_agland_map_slice(array, type, output_dir=None):
    """ """

    assert (type in ['cropland', 'pasture', 'other']), "Unknown type input matrix"
    # Default settings for agland map (cropland, pasture, other)
    max_val = 1
    min_val = 0
    num_bins = 10
    bins = [i for i in np.arange(min_val, max_val, (max_val-min_val)/num_bins)] + [max_val]
    norm = colors.BoundaryNorm(bins, num_bins, clip=True)

    if type == 'cropland':
        cmap = CROPLAND_CMAP10
    elif type == 'pasture':
        cmap = PASTURE_CMAP10
    elif type == 'other':
        cmap = OTHER_CMAP10

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18))
    im = ax.imshow(array, cmap=cmap, norm=norm)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0
                       )
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=10.5)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_FAO_census(world_census_table, attribute, cmap, num_bins, label, output_dir=None):
    """
    Plot FAO census map from world census table. Input attribute indicates whether to
    plot CROPLAND or PASTURE. cmap and num_bins must match length for a discrete
    colorbar. Input label is for colorbar label

    Args:
        world_census_table (pd): pd world census table with geometry
        attribute (str): 'CROPLAND' or 'PASTURE'
        cmap (str): matplolib cmap
        num_bins (int): number of bins
        label (str): label for colorbar
        output_dir (str): output dir (Default: None)
    """
    assert (attribute in ['CROPLAND', 'PASTURE']), \
        "Input attribute must be either CROPLAND or PASTURE for FAO"

    # Convert pd DataFrame to GeoDataFrame
    geo_FAO_census = GeoDataFrame(world_census_table)

    # Get min and max of values
    max_val = max(geo_FAO_census[attribute].to_list())
    min_val = min(geo_FAO_census[attribute].to_list())
    bins = [i for i in np.arange(min_val, max_val, (max_val - min_val) / (num_bins - 1))] + [max_val]
    bins = [int(i) for i in bins]  # only in FAO (values are in kHa, not %)

    # Plot
    ax = geo_FAO_census.plot(column=attribute,
                             edgecolor='black',
                             legend=True,
                             figsize=(20, 20),
                             cmap=cmap,
                             norm=colors.BoundaryNorm(bins, num_bins),
                             legend_kwds={'label': label,
                                          'orientation': 'vertical',
                                          'pad': 0,
                                          'shrink': 0.3})
    ax.set_axis_off()

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_agland_pred_vs_ground_truth(ground_truth, pred, output_dir=None):
    """
    Plot pred (x) vs. ground_truth (y) with a 1:1 line as reference for CROPLAND,
    PASTURE and OTHER. Function assumes both inputs are nx3 np.array with above
    order in place

    Args:
        ground_truth (np.array): n-by-3 array with columns CROPLAND, PASTURE, OTHER
        pred (np.array): n-by-3 array with columns CROPLAND, PASTURE, OTHER
        output_dir (str): output dir (Default: None)
    """

    def plot_helper(gt, pred, ax, x, m, b, title):
        """ Helper plotter """
        ax.scatter(pred, gt, marker='x')
        ax.plot(x, m * x + b, c='r')
        ax.plot(x, x, 'k-')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        ax.set_title(title)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    assert (ground_truth.ndim == pred.ndim == 2), "Input arrays must be 2D"
    assert (ground_truth.shape[1] == pred.shape[1] == 3), \
        "Input arrays must follow CROPLAND, PASTURE, OTHER order in columns"

    # Load agland data
    gt_cropland, gt_pasture, gt_other = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
    pred_cropland, pred_pasture, pred_other = pred[:, 0], pred[:, 1], pred[:, 2]

    # Get RMSE results
    x_range = np.linspace(0, 1, 50)
    m_cropland, b_cropland = np.polyfit(pred_cropland, gt_cropland, 1)
    m_pasture, b_pasture = np.polyfit(pred_pasture, gt_pasture, 1)
    m_other, b_other = np.polyfit(pred_other, gt_other, 1)

    rmse_cropland = rmse((m_cropland * x_range + b_cropland), x_range)
    rmse_pasture = rmse((m_pasture * x_range + b_pasture), x_range)
    rmse_other = rmse((m_other * x_range + b_other), x_range)

    # Plot results in 3 subplots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
    plot_helper(gt_cropland, pred_cropland, ax[0], x_range, m_cropland, b_cropland,
                'Cropland | RMSE:{}'.format(round(rmse_cropland, 4)))
    plot_helper(gt_pasture, pred_pasture, ax[1], x_range, m_pasture, b_pasture,
                'Pasture | RMSE:{}'.format(round(rmse_pasture, 4)))
    plot_helper(gt_other, pred_other, ax[2], x_range, m_other, b_other,
                'Other | RMSE:{}'.format(round(rmse_other, 4)))

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()
