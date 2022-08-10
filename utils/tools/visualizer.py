from matplotlib import projections
import rasterio
import rasterio.plot as rasterio_plot
import numpy as np
from osgeo import gdal
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cv2
from geopandas import GeoDataFrame
from ..metrics import *

font = {'family': 'monospace', 'weight': 'normal', 'size': 20}

matplotlib.rc('font', **font)

TRANSPARENT = colors.colorConverter.to_rgba('white', alpha=0)

GDD_MASK_CMAP = colors.ListedColormap(['#D4D4D4', TRANSPARENT])

PASTURE_CMAP10 = colors.ListedColormap([
    '#FEDFC9', '#FFCBA9', '#FFAE78', '#FF9147', '#FF7416', '#EC5F00',
    '#BB4B00', '#923B00', '#7A3100', '#491D00'
])

PASTURE_CMAP10_OUTLIER2 = colors.ListedColormap([
    '#D4D4D4', '#5D5D5D', '#FEDFC9', '#FFCBA9', '#FFAE78', '#FF9147',
    '#FF7416', '#EC5F00', '#BB4B00', '#923B00', '#7A3100', '#491D00'
])

CROPLAND_CMAP10 = colors.ListedColormap([
    '#E2FEE2', '#B9FEB9', '#78FF78', '#06FF06', '#00D400', '#00A300',
    '#007200', '#005100', '#003900', '#001800'
])

CROPLAND_CMAP10_OUTLIER2 = colors.ListedColormap([
    '#D4D4D4', '#5D5D5D', '#E2FEE2', '#B9FEB9', '#78FF78', '#06FF06',
    '#00D400', '#00A300', '#007200', '#005100', '#003900', '#001800'
])

OTHER_CMAP10 = colors.ListedColormap([
    '#c4c4c4', '#a0a0a0', '#8e8e8e', '#7c7c7c', '#6b6b6b', '#595959',
    '#474747', '#353535', '#232323', '#000000'
])

LAND_COVER_CMAP17 = colors.ListedColormap([
    '#193300', '#00994C', '#CCCC00', '#CC6600', '#CCFFE5', '#4C0099',
    '#0000CC', '#CC0000', '#FF9933', '#99FFFF', '#003333', '#660033',
    '#000033', '#6666FF', '#FFFFCC', '#E5FFCC', '#E5CCFF'
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
        cbar = fig.colorbar(im,
                            cax=cax,
                            orientation='horizontal',
                            ticks=[0, 1])
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
    Plot global area map 

    Args:
        area_array (np.array): 2D np array
        output_dir (str): output dir (Default: None)
        scale (int): scale factor to be applied on image size (1/scale)
    """
    assert (scale > 0), "scale cannot be non negative"
    h, w = area_array.shape

    # Reduce image size for visualization
    area_array = cv2.resize(area_array,
                            dsize=(int(w) // scale, int(h) // scale),
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
                       borderpad=8.5)
    cbar = fig.colorbar(im, cax=axins, orientation='horizontal')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_land_cover_map(land_cover_array,
                        class_lookup,
                        output_dir=None,
                        nodata=-9999,
                        scale=10):
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
    assert (len(class_name_list) <=
            17), "Number of classes exceeds 17, add color in default colormap"

    # Reduce image size for visualization
    land_cover_array = cv2.resize(land_cover_array,
                                  dsize=(int(w) // scale, int(h) // scale),
                                  interpolation=cv2.INTER_NEAREST)

    # Remove null values
    land_cover_array[land_cover_array == nodata] = np.nan

    # Use land cover default colormap
    boundaries = [min(class_index_list) - 0.5
                  ] + [i + 0.5 for i in class_index_list]
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
                       borderpad=0)
    cbar = fig.colorbar(im,
                        cax=axins,
                        orientation='vertical',
                        ticks=class_index_list)
    cbar.ax.set_yticklabels(class_name_list)
    cbar.ax.tick_params(labelsize=10.5)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_agland_map_slice(array, type, output_dir=None):
    """
    Helper function that plots slices of agland map ('cropland', 'pasture', 'other')

    Args:
        array (np.array): input array
        type (str): 'cropland', 'pasture', or 'other'
        output_dir (str, optional): output directory path. Defaults to None.
    """

    assert (type in ['cropland', 'pasture',
                     'other']), "Unknown type input matrix"
    # Default settings for agland map (cropland, pasture, other)
    max_val = 1
    min_val = 0
    num_bins = 10
    bins = [
        i for i in np.arange(min_val, max_val, (max_val - min_val) / num_bins)
    ] + [max_val]
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
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=10.5)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_FAO_census(world_census_table,
                    attribute,
                    cmap,
                    num_bins,
                    label,
                    output_dir=None):
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
    bins = [
        i for i in np.arange(min_val, max_val, (max_val - min_val) /
                             (num_bins - 1))
    ] + [max_val]
    bins = [int(i) for i in bins]  # only in FAO (values are in kHa, not %)

    # Plot
    ax = geo_FAO_census.plot(column=attribute,
                             edgecolor='black',
                             legend=True,
                             figsize=(20, 20),
                             cmap=cmap,
                             norm=colors.BoundaryNorm(bins, num_bins),
                             legend_kwds={
                                 'label': label,
                                 'orientation': 'vertical',
                                 'pad': 0,
                                 'shrink': 0.3
                             })
    ax.set_axis_off()

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()


def plot_merged_census(census_table, marker, gdd_config, output_dir=None):
    """
    Plot merged census map generated from pipeline. Input census_table must have
    CROPLAND and PASTURE attributes, with 2 marker values indicating states filtered
    by the 2 filers, namely nan_filter and gdd_filter

    Args:
        census_table (pd): pd census table with geometry
        marker (dict): (str) -> (int) indicators for filtered samples
        gdd_config (dict): gdd configs
        output_dir (str): output dir (Default: None)
    """
    # Load GDD mask
    gdd_mask_raster = rasterio.open(gdd_config['path_dir']['GDD_filter_map'])

    # Flip marker
    marker_flipped = {v: k for k, v in marker.items()}

    # Convert pd DataFrame to GeoDataFrame
    geo_census = GeoDataFrame(census_table)

    # Get min and max of values
    cropland_max_val = max(geo_census['CROPLAND'].to_list())
    cropland_min_val = sorted(set(
        geo_census['CROPLAND'].to_list()))[len(marker)]
    num_bins = 10
    cropland_bins = [
        i for i in np.arange(cropland_min_val, cropland_max_val,
                             (cropland_max_val - cropland_min_val) /
                             (num_bins - 1))
    ] + [cropland_max_val]
    nan_cropland_bins = sorted(list(marker.values())) + cropland_bins
    nan_cropland_bins = [int(i) for i in nan_cropland_bins
                         ]  # only in FAO (values are in kHa, not %)

    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    geo_census.plot(ax=ax,
                    column='CROPLAND',
                    edgecolor='black',
                    legend=True,
                    cmap=CROPLAND_CMAP10_OUTLIER2,
                    norm=colors.BoundaryNorm(nan_cropland_bins,
                                             len(nan_cropland_bins)),
                    legend_kwds={
                        'label': 'CROPLAND (kHa)',
                        'orientation': 'vertical',
                        'pad': 0.01,
                        'shrink': 0.3,
                        'ticks': nan_cropland_bins
                    })
    ax.set_axis_off()

    # Trick from stackoverflow to edit legend tick labels
    colourbar = ax.get_figure().get_axes()[1]
    colourbar.set_yticklabels(
        [marker_flipped[i] for i in nan_cropland_bins[0:len(marker)]] +
        nan_cropland_bins[len(marker):])

    # Plot GDD mask on top
    rasterio_plot.show(gdd_mask_raster,
                       ax=ax,
                       cmap=GDD_MASK_CMAP,
                       zorder=5,
                       alpha=0.9)

    if output_dir is not None:
        plt.savefig(output_dir + '/cropland_census_input.png',
                    format='png',
                    bbox_inches='tight')
        print('File {} generated'.format(output_dir))

    plt.show()
    plt.close()

    # Get min and max of values
    pasture_max_val = max(geo_census['PASTURE'].to_list())
    pasture_min_val = sorted(set(geo_census['PASTURE'].to_list()))[len(marker)]
    num_bins = 10
    pasture_bins = [
        i for i in np.arange(pasture_min_val, pasture_max_val,
                             (pasture_max_val - pasture_min_val) /
                             (num_bins - 1))
    ] + [pasture_max_val]
    nan_pasture_bins = sorted(list(marker.values())) + pasture_bins
    nan_pasture_bins = [int(i) for i in nan_pasture_bins
                        ]  # only in FAO (values are in kHa, not %)

    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    geo_census.plot(ax=ax,
                    column='PASTURE',
                    edgecolor='black',
                    legend=True,
                    cmap=PASTURE_CMAP10_OUTLIER2,
                    norm=colors.BoundaryNorm(nan_pasture_bins,
                                             len(nan_pasture_bins)),
                    legend_kwds={
                        'label': 'PASTURE (kHa)',
                        'orientation': 'vertical',
                        'pad': 0.01,
                        'shrink': 0.3,
                        'ticks': nan_pasture_bins
                    })
    ax.set_axis_off()

    colourbar = ax.get_figure().get_axes()[1]
    colourbar.set_yticklabels(
        [marker_flipped[i] for i in nan_pasture_bins[0:len(marker)]] +
        nan_pasture_bins[len(marker):])

    # Plot GDD mask on top
    rasterio_plot.show(gdd_mask_raster,
                       ax=ax,
                       cmap=GDD_MASK_CMAP,
                       zorder=5,
                       alpha=0.9)

    if output_dir is not None:
        plt.savefig(output_dir + '/pasture_census_input.png',
                    format='png',
                    bbox_inches='tight')
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
    gt_cropland, gt_pasture, gt_other = ground_truth[:,
                                                     0], ground_truth[:,
                                                                      1], ground_truth[:,
                                                                                       2]
    pred_cropland, pred_pasture, pred_other = pred[:, 0], pred[:, 1], pred[:,
                                                                           2]

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
    plot_helper(gt_cropland, pred_cropland, ax[0], x_range, m_cropland,
                b_cropland,
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


def plot_geowiki_cropland(geowiki_cropland_by_index, output_dir=None):
    """
    Scatter plots of geowiki cropland data. Input geowiki_cropland_by_index
    is a 2D array with 3 columns, respectively index_x, index_y, cropland (100%)

    Args:
        geowiki_cropland_by_index (np.array): 2D array of geowiki data in index form
        output_dir (str): output dir (Default: None)
    """
    fig, ax = plt.subplots(figsize=(18, 8))
    max_val = 1
    min_val = 0
    num_bins = 10
    bins = [
        i for i in np.arange(min_val, max_val, (max_val - min_val) / num_bins)
    ] + [max_val]
    norm = colors.BoundaryNorm(bins, num_bins, clip=True)
    im = plt.scatter(geowiki_cropland_by_index[:, 1],
                     geowiki_cropland_by_index[:, 0],
                     c=geowiki_cropland_by_index[:, 2] / 100,
                     cmap=CROPLAND_CMAP10,
                     norm=norm,
                     s=1)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)
    ax.invert_yaxis()

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_diff_geowiki_pred_cropland(geowiki_cropland_by_index,
                                    pred_results,
                                    output_dir=None):
    """
    Scatter plot of (geowiki-geowiki_cropland_by_index) cropland map, with nan removed.
    pred_results is the prediction results (in [0, 1]) that corresponds to
    geowiki_cropland_by_index indices

    Args:
        geowiki_cropland_by_index (np.array): 2D array of geowiki data in index form
        pred_results (np.array): 1D array of prediction results correspond to
                                 geowiki_cropland_by_index indices
        output_dir (str): output dir (Default: None)
    """
    assert (pred_results.ndim == 1), "Prediction results array must be 1D"
    assert (pred_results.shape[0] == geowiki_cropland_by_index.shape[0]), \
        "Prediction results array must be same size as geowiki cropland data"
    assert ((np.nanmax(pred_results) <= 1) and (np.nanmin(pred_results) >= 0)), \
        "Prediction results array must be in [0, 1]"

    # Get nan indices in pred_results
    nan_index = np.isnan(pred_results)

    # Difference map between Geowiki and pred
    # Note: to make the plot consistent with ones without having any nan,
    #       we need to temporarily assign 0 to geowiki reference, instead of
    #       removing the nan samples
    geowiki_cropland_by_index[nan_index] = 0
    pred_results[nan_index] = 0
    diff = geowiki_cropland_by_index[:, 2] / 100 - pred_results

    fig, ax = plt.subplots(figsize=(18, 8))
    im = plt.scatter(geowiki_cropland_by_index[:, 1],
                     geowiki_cropland_by_index[:, 0],
                     c=diff,
                     cmap='bwr',
                     vmin=-1,
                     vmax=1,
                     s=1)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)
    ax.invert_yaxis()

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_histogram_diff_geowiki_pred_cropland(geowiki_cropland_by_index,
                                              pred_results,
                                              output_dir=None):
    """
    Histogram plot of (geowiki-pred_results), with nan removed in pred_results

    Args:
        geowiki_cropland_by_index (np.array): 2D array of geowiki data in index form
        pred_results (np.array): 1D array of prediction results correspond to
                                 geowiki_cropland_by_index indices
        output_dir (str): output dir (Default: None)
    """
    # Get nan indices in pred_results
    nan_index = np.isnan(pred_results)

    # Compute difference map between Geowiki and pred
    diff = geowiki_cropland_by_index[~nan_index,
                                     2] / 100 - pred_results[~nan_index]
    rmse_error = np.sqrt(np.mean(diff**2))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.text(-0.85, 15000, r'$\mu={}$'.format(np.round(np.mean(diff), 4)))
    plt.text(-0.85, 13000, r'$\sigma={}$'.format(np.round(np.std(diff), 4)))
    plt.text(-0.85, 11000, r'RMSE=${}$'.format(np.round(rmse_error, 4)))
    plt.xlim(-1, 1)
    plt.hist(diff)
    plt.xlabel('Geowiki - Prediction')
    plt.ylabel('Frequency (#)')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_diff_maryland_pred_cropland(maryland_map, pred_map, output_dir=None):
    """
    Plot of difference between maryland cropland and prediction cropland

    Args:
        maryland_map (np.array): 2D array map of maryland cropland (reprojected to match agland)
        pred_map (np.array): 2D array map of predicted cropland
        output_dir (str): output dir (Default: None)
    """
    assert (maryland_map.shape == pred_map.shape
            ), "Input maps must have same shape"

    # Compute difference
    diff = maryland_map / 100 - pred_map

    # Plot
    fig, ax = plt.subplots(figsize=(18, 8))
    im = plt.imshow(diff, cmap='bwr', vmin=-1, vmax=1)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_histogram_diff_maryland_pred_cropland(maryland_map,
                                               pred_map,
                                               output_dir=None):
    """
    Histogram plot of (maryland_map-pred_map)

    Args:
        maryland_map (np.array): 2D array map of maryland cropland (reprojected to match agland)
        pred_map (np.array): 2D array map of predicted cropland
        output_dir (str): output dir (Default: None)
    """
    assert (maryland_map.shape == pred_map.shape
            ), "Input maps must have same shape"

    # Compute difference
    diff = maryland_map / 100 - pred_map
    rmse_error = np.sqrt(np.nanmean(diff**2))

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.text(-0.85, 1800000, r'$\mu={}$'.format(np.round(np.nanmean(diff), 4)))
    plt.text(-0.85, 1550000,
             r'$\sigma={}$'.format(np.round(np.nanstd(diff), 4)))
    plt.text(-0.85, 1300000, r'RMSE={}'.format(np.round(rmse_error, 4)))

    plt.xlim(-1, 1)
    plt.hist(diff.flatten())
    plt.xlabel('Maryland - Prediction')
    plt.ylabel('Frequency (#)')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_weights_array(weights_array):
    """
    Visualize 2D bias correction matrix

    Args:
        weights_array (np.array): 2D bias correction weights matrix
    """
    fig, ax = plt.subplots(figsize=(30, 30))
    im = plt.imshow(weights_array, cmap='magma')
    plt.axis('off')

    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
    fig.add_axes(cax)

    cbar = fig.colorbar(im, cax=cax, orientation='horizontal')

