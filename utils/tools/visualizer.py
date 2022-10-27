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
import geopandas as gpd

font = {'family': 'Helvetica', 'weight': 'normal', 'size': 24}

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

PASTURE_CMAP_FULL = colors.LinearSegmentedColormap.from_list(
    name='PASTURE_CMAP_FULL',
    colors=[
        '#FFFFFF', '#FFF7F2', '#FFF2EA', '#FEEDE2', '#FEE9DA', '#FEE4D2',
        '#FEDFC9', '#FEDAC1', '#FED5B9', '#FED0B1', '#FFCBA9', '#FFC7A1',
        '#FFC299', '#FFBD90', '#FFB888', '#FFB380', '#FFAE78', '#FFA970',
        '#FFA468', '#FFA05F', '#FF9B57', '#FF964F', '#FF9147', '#FF8C3F',
        '#FF8737', '#FF822E', '#FF7D26', '#FF791E', '#FF7416', '#FF6F0E',
        '#FF6A06', '#FC6600', '#F46200', '#EC5F00', '#E45C00', '#DC5800',
        '#D45500', '#CC5200', '#C34F00', '#BB4B00', '#B34800', '#AB4500',
        '#A34100', '#9B3E00', '#923B00', '#8A3700', '#823400', '#7A3100',
        '#722E00', '#6A2A00', '#612700', '#592400', '#512000', '#491D00',
        '#411A00', '#391700', '#301300', '#281000', '#200D00', '#180900',
        '#100600'
    ],
    N=1001)

CROPLAND_CMAP10 = colors.ListedColormap([
    '#E2FEE2', '#B9FEB9', '#78FF78', '#06FF06', '#00D400', '#00A300',
    '#007200', '#005100', '#003900', '#001800'
])

CROPLAND_CMAP10_OUTLIER2 = colors.ListedColormap([
    '#D4D4D4', '#5D5D5D', '#E2FEE2', '#B9FEB9', '#78FF78', '#06FF06',
    '#00D400', '#00A300', '#007200', '#005100', '#003900', '#001800'
])

CROPLAND_CMAP_FULL = colors.LinearSegmentedColormap.from_list(
    name='CROPLAND_CMAP_FULL',
    colors=[
        '#FFFFFF', '#F2FFF2', '#EAFFEA', '#E2FEE2', '#DAFEDA', '#D2FED2',
        '#C9FEC9', '#C1FEC1', '#B9FEB9', '#B1FEB1', '#A9FFA9', '#A1FFA1',
        '#99FF99', '#90FF90', '#88FF88', '#80FF80', '#78FF78', '#70FF70',
        '#68FF68', '#5FFF5F', '#57FF57', '#4FFF4F', '#47FF47', '#3FFF3F',
        '#37FF37', '#2EFF2E', '#26FF26', '#1EFF1E', '#16FF16', '#0eff0e',
        '#06FF06', '#00FC00', '#00F400', '#00EC00', '#00E400', '#00DC00',
        '#00D400', '#00CC00', '#00C300', '#00BB00', '#00B300', '#00AB00',
        '#00A300', '#009B00', '#009200', '#008A00', '#008200', '#007A00',
        '#007200', '#006A00', '#006100', '#005900', '#005100', '#004900',
        '#004100', '#003900', '#003000', '#002800', '#002000', '#001800',
        '#001000'
    ],
    N=1001)

OTHER_CMAP10 = colors.ListedColormap([
    '#c4c4c4', '#a0a0a0', '#8e8e8e', '#7c7c7c', '#6b6b6b', '#595959',
    '#474747', '#353535', '#232323', '#000000'
])

OTHER_CMAP_FULL = colors.LinearSegmentedColormap.from_list(
    name='OTHER_CMAP_FULL',
    colors=[
        '#FFFFFF', '#F8F8F8', '#F4F4F4', '#F0F0F0', '#ECECEC', '#E8E8E8',
        '#E4E4E4', '#E0E0E0', '#DCDCDC', '#D8D8D8', '#D4D4D4', '#D0D0D0',
        '#CCC', '#C7C7C7', '#C3C3C3', '#BFBFBF', '#BBB', '#B7B7B7', '#B3B3B3',
        '#AFAFAF', '#ABABAB', '#A7A7A7', '#A3A3A3', '#9F9F9F', '#9B9B9B',
        '#969696', '#929292', '#8E8E8E', '#8A8A8A', '#868686', '#828282',
        '#7E7E7E', '#7A7A7A', '#767676', '#727272', '#6E6E6E', '#6A6A6A',
        '#666', '#616161', '#5D5D5D', '#595959', '#555', '#515151', '#4D4D4D',
        '#494949', '#454545', '#414141', '#3D3D3D', '#393939', '#353535',
        '#303030', '#2C2C2C', '#282828', '#242424', '#202020', '#1C1C1C',
        '#181818', '#141414', '#101010', '#0C0C0C', '#080808'
    ],
    N=1001)

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

    # Reduce image size for visualization ONLY
    # Note: if area_array is used for computation, interplation with
    #       nearest neighbors will give wrong results, instead
    #       need to re-generate or sum (for downsampling) and average
    #       (for upsampling)
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


def plot_agland_map_tif(agland_map_tif,
                        type,
                        global_boundary_shp=None,
                        output_dir=None):
    """
    Helper function that plots slices of agland map ('cropland', 'pasture', 'other') that is 
    in tif format. global_boundary_shp is the shapefile of global boundary that is plotted 
    on top of the raster

    Args:
        agland_map_tif (str): path dir to agland map tif
        type (str): 'cropland', 'pasture', or 'other'
        global_boundary_shp (str, optional): path dir to boundary shp. Defaults to None.
        output_dir (str, optional): output directory path. Defaults to None.
    """
    assert (type in ['cropland', 'pasture',
                     'other']), "Unknown type input matrix"

    # Initialize figure
    fig, ax = plt.subplots(figsize=(18, 18), dpi=600)

    # Plot global boundary
    if global_boundary_shp is not None:
        global_boundary_gpd = gpd.read_file(global_boundary_shp)
        global_boundary_gpd.plot(ax=ax,
                                 column='min_zoom',
                                 edgecolor='black',
                                 legend=False,
                                 cmap=colors.ListedColormap([TRANSPARENT]))
        ax.set_axis_off()

    # Plot agland map content
    if type == 'cropland':
        cbar_label = 'Cropland Area (%)'
        # cmap = CROPLAND_CMAP10
        cmap = CROPLAND_CMAP_FULL
    elif type == 'pasture':
        cbar_label = 'Pasture Area (%)'
        # cmap = PASTURE_CMAP10
        cmap = PASTURE_CMAP_FULL
    elif type == 'other':
        cbar_label = 'Other Area (%)'
        # cmap = OTHER_CMAP10
        cmap = OTHER_CMAP_FULL

    agland_map = rasterio.open(agland_map_tif)
    im = rasterio_plot.show(agland_map,
                            ax=ax,
                            cmap=cmap,
                            zorder=5,
                            alpha=1,
                            vmin=0,
                            vmax=1)
    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0.05, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im.get_images()[0],
                        cax=axins,
                        orientation='vertical',
                        ticks=[i * 0.2 for i in range(6)])
    cbar.ax.set_yticklabels([str(i * 20) for i in range(6)], fontsize=11.5)
    cbar.set_label(cbar_label, rotation=90, labelpad=-80, y=0.55)

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
    # norm = colors.BoundaryNorm(bins, num_bins, clip=True)

    if type == 'cropland':
        cbar_label = 'Cropland Area %'
        # cmap = CROPLAND_CMAP10
        cmap = CROPLAND_CMAP_FULL
    elif type == 'pasture':
        cbar_label = 'Pasture Area %'
        # cmap = PASTURE_CMAP10
        cmap = PASTURE_CMAP_FULL
    elif type == 'other':
        cbar_label = 'Other Area %'
        # cmap = OTHER_CMAP10
        cmap = OTHER_CMAP_FULL

    # Plot
    fig, ax = plt.subplots(figsize=(18, 18), dpi=600)
    # im = ax.imshow(array, cmap=cmap, norm=norm)
    im = ax.imshow(array * 100, cmap=cmap)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0.05, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im,
                        cax=axins,
                        orientation='vertical',
                        ticks=[i * 0.2 for i in range(6)])
    cbar.ax.set_yticklabels([str(i * 20) for i in range(6)], fontsize=11.5)
    cbar.set_label(cbar_label, rotation=90, labelpad=-80, y=0.55)

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
                        'label': 'Cropland (kHa)',
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
                        'label': 'Pasture (kHa)',
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


def plot_agland_pred_vs_ground_truth(
        mark_index,
        output_pred_vs_ground_truth_data_collection,
        output_dir=None):
    """
    Plot pred (x) vs. ground_truth (y) with a 1:1 line as reference for CROPLAND,
    PASTURE and OTHER. Function assumes output_pred_vs_ground_truth_data_collection is not an 
    empty dict with ground_truth_collection and pred_collection info

    Args:
        mark_index (int): index in pred list to be highlighted red
        output_pred_vs_ground_truth_data_collection (dict): pred and gt info with iteration int as keys
        output_dir (str): output dir (Default: None)
    """

    def base_plot_helper(gt, pred, ax, x, m, b, title):
        """ Helper plotter """
        ax.scatter(pred, gt, marker='x')
        ax.plot(x, m * x + b, c='r')
        ax.plot(x, x, 'k-')
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Ground Truth')
        ax.set_title(title)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        return ax

    def sub_plot_helper(mark_index, iter_list, rmse_list, ax, markersize,
                        linewidth):
        """ Helper plotter """
        ax.plot(np.asarray(iter_list),
                np.asarray(rmse_list),
                linewidth=linewidth,
                c='k',
                zorder=1)
        ax.scatter(np.asarray(iter_list),
                   np.asarray(rmse_list),
                   marker='X',
                   c='b',
                   s=markersize,
                   zorder=10)
        ax.scatter(np.asarray(iter_list[mark_index]),
                   np.asarray(rmse_list[mark_index]),
                   marker='X',
                   c='r',
                   s=markersize,
                   zorder=10)
        ax.set_xticks(iter_list, ['iter_{}'.format(str(i)) for i in iter_list])
        ax.set_ylabel('RMSE')

    def add_subplot_axes(ax, rect, axisbg='w'):
        """
        Helper plotter
        Reference: 
        https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
        """
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        subax = fig.add_axes([x, y, width, height], facecolor=axisbg)
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.5
        y_labelsize *= rect[3]**0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)

        return subax

    assert (len(output_pred_vs_ground_truth_data_collection) >
            0), "output_pred_vs_ground_truth_data_collection cannot be empty"

    # subplot settings
    markersize = 400
    linewidth = 5
    subplot_rect = [0.5, 0.1, 0.4, 0.3]

    # Unpack data info and find RMSE results
    plot_data_table = {}
    x_range = np.linspace(0, 1, 50)
    for itr, data_info in output_pred_vs_ground_truth_data_collection.items():

        ground_truth = data_info['ground_truth_collection']
        pred = data_info['pred_collection']

        assert (ground_truth.ndim == pred.ndim == 2), "Input arrays must be 2D"
        assert (ground_truth.shape[1] == pred.shape[1] == 3), \
            "Input arrays must follow CROPLAND, PASTURE, OTHER order in columns"

        gt_cropland, gt_pasture, gt_other = ground_truth[:,
                                                         0], ground_truth[:,
                                                                          1], ground_truth[:,
                                                                                           2]
        pred_cropland, pred_pasture, pred_other = pred[:, 0], pred[:,
                                                                   1], pred[:,
                                                                            2]

        # Compute RMSE
        m_cropland, b_cropland = np.polyfit(pred_cropland, gt_cropland, 1)
        m_pasture, b_pasture = np.polyfit(pred_pasture, gt_pasture, 1)
        m_other, b_other = np.polyfit(pred_other, gt_other, 1)

        rmse_cropland = rmse((m_cropland * x_range + b_cropland), x_range)
        rmse_pasture = rmse((m_pasture * x_range + b_pasture), x_range)
        rmse_other = rmse((m_other * x_range + b_other), x_range)

        plot_data_table[itr] = {
            'gt_cropland': gt_cropland,
            'gt_pasture': gt_pasture,
            'gt_other': gt_other,
            'pred_cropland': pred_cropland,
            'pred_pasture': pred_pasture,
            'pred_other': pred_other,
            'm_cropland': m_cropland,
            'b_cropland': b_cropland,
            'm_pasture': m_pasture,
            'b_pasture': b_pasture,
            'm_other': m_other,
            'b_other': b_other,
            'rmse_cropland': rmse_cropland,
            'rmse_pasture': rmse_pasture,
            'rmse_other': rmse_other
        }

    # Use the last iteration as base plot
    iter_list = sorted(list(plot_data_table.keys()))
    base_iter = mark_index

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10), dpi=600)
    ax1 = base_plot_helper(
        plot_data_table[base_iter]['gt_cropland'],
        plot_data_table[base_iter]['pred_cropland'], ax[0], x_range,
        plot_data_table[base_iter]['m_cropland'],
        plot_data_table[base_iter]['b_cropland'], 'Cropland | RMSE:{}'.format(
            round(plot_data_table[base_iter]['rmse_cropland'], 4)))

    ax2 = base_plot_helper(
        plot_data_table[base_iter]['gt_pasture'],
        plot_data_table[base_iter]['pred_pasture'], ax[1], x_range,
        plot_data_table[base_iter]['m_pasture'],
        plot_data_table[base_iter]['b_pasture'], 'Pasture | RMSE:{}'.format(
            round(plot_data_table[base_iter]['rmse_pasture'], 4)))

    ax3 = base_plot_helper(
        plot_data_table[base_iter]['gt_other'],
        plot_data_table[base_iter]['pred_other'], ax[2], x_range,
        plot_data_table[base_iter]['m_other'],
        plot_data_table[base_iter]['b_other'], 'Other | RMSE:{}'.format(
            round(plot_data_table[base_iter]['rmse_other'], 4)))

    ax1_sub = add_subplot_axes(ax1, subplot_rect, axisbg='w')
    ax2_sub = add_subplot_axes(ax2, subplot_rect, axisbg='w')
    ax3_sub = add_subplot_axes(ax3, subplot_rect, axisbg='w')

    sub_plot_helper(mark_index,
                    iter_list,
                    [plot_data_table[i]['rmse_cropland'] for i in iter_list],
                    ax1_sub,
                    markersize=markersize,
                    linewidth=linewidth)
    sub_plot_helper(mark_index,
                    iter_list,
                    [plot_data_table[i]['rmse_pasture'] for i in iter_list],
                    ax2_sub,
                    markersize=markersize,
                    linewidth=linewidth)
    sub_plot_helper(mark_index,
                    iter_list,
                    [plot_data_table[i]['rmse_other'] for i in iter_list],
                    ax3_sub,
                    markersize=markersize,
                    linewidth=linewidth)

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
    Scatter plot of (pred_results - geowiki) cropland map, with nan removed.
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
    diff = pred_results - geowiki_cropland_by_index[:, 2] / 100

    fig, ax = plt.subplots(figsize=(18, 8), dpi=600)
    im = plt.scatter(geowiki_cropland_by_index[:, 1],
                     geowiki_cropland_by_index[:, 0],
                     c=diff * 100,
                     cmap='bwr',
                     vmin=-100,
                     vmax=100,
                     s=1)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0.08, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)
    cbar.set_label('Cropland Area \nDifference (%)',
                   rotation=90,
                   labelpad=-120,
                   y=0.55)
    ax.invert_yaxis()

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_histogram_diff_geowiki_pred_cropland(geowiki_cropland_by_index,
                                              pred_results,
                                              output_dir=None):
    """
    Histogram plot of (pred_results - geowiki), with nan removed in pred_results

    Args:
        geowiki_cropland_by_index (np.array): 2D array of geowiki data in index form
        pred_results (np.array): 1D array of prediction results correspond to
                                 geowiki_cropland_by_index indices
        output_dir (str): output dir (Default: None)
    """
    # Get nan indices in pred_results
    nan_index = np.isnan(pred_results)

    # Compute difference map between Geowiki and pred
    diff = pred_results[~nan_index] - geowiki_cropland_by_index[~nan_index,
                                                                2] / 100
    diff *= 100  # use percentage
    rmse_error = np.sqrt(np.mean(diff**2))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    plt.text(-90, 15000, r'$\mu={}$'.format(np.round(np.mean(diff), 4)))
    plt.text(-90, 13000, r'$\sigma={}$'.format(np.round(np.std(diff), 4)))
    plt.text(-90, 11000, r'RMSE=${}$'.format(np.round(rmse_error, 4)))
    plt.xlim(-100, 100)
    plt.hist(diff)
    plt.xlabel('Prediction - Geowiki (%)')
    plt.ylabel('Frequency (#)')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_diff_maryland_pred_cropland(maryland_map, pred_map, output_dir=None):
    """
    Plot of difference between prediction cropland and maryland cropland

    Args:
        maryland_map (np.array): 2D array map of maryland cropland (reprojected to match agland)
        pred_map (np.array): 2D array map of predicted cropland
        output_dir (str): output dir (Default: None)
    """
    assert (maryland_map.shape == pred_map.shape
            ), "Input maps must have same shape"

    # Compute difference
    diff = pred_map - maryland_map / 100
    diff *= 100  # use percentage

    # Plot
    fig, ax = plt.subplots(figsize=(18, 8), dpi=600)
    im = plt.imshow(diff, cmap='bwr', vmin=-100, vmax=100)
    plt.axis('off')

    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(0.08, 0.15, 0.3, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)
    cbar.set_label('Cropland Area \nDifference (%)',
                   rotation=90,
                   labelpad=-120,
                   y=0.55)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_histogram_diff_maryland_pred_cropland(maryland_map,
                                               pred_map,
                                               output_dir=None):
    """
    Histogram plot of (pred_map - maryland_map)

    Args:
        maryland_map (np.array): 2D array map of maryland cropland (reprojected to match agland)
        pred_map (np.array): 2D array map of predicted cropland
        output_dir (str): output dir (Default: None)
    """
    assert (maryland_map.shape == pred_map.shape
            ), "Input maps must have same shape"

    # Compute difference
    diff = pred_map - maryland_map / 100
    diff *= 100  # use percentage
    rmse_error = np.sqrt(np.nanmean(diff**2))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    plt.text(-90, 1800000, r'$\mu={}$'.format(np.round(np.nanmean(diff), 4)))
    plt.text(-90, 1550000, r'$\sigma={}$'.format(np.round(np.nanstd(diff), 4)))
    plt.text(-90, 1300000, r'RMSE={}'.format(np.round(rmse_error, 4)))

    plt.xlim(-100, 100)
    plt.hist(diff.flatten())
    plt.xlabel('Prediction - Maryland (%)')
    plt.ylabel('Frequency (#)')

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_diff_pred_pasture(diff_map, output_dir=None):
    """
    Scatter plot of pasture evaluation map.

    Args:
        diff_map (np.array): 2D array of pasture difference map
        output_dir (str): output dir (Default: None)
    """
    # Use percentage
    diff = -diff_map * 100

    # Plot
    fig, ax = plt.subplots(figsize=(18, 8), dpi=600)
    im = plt.imshow(diff, cmap='bwr', vmin=-100, vmax=100)
    plt.axis('off')

    # Default: (-0.08, 0.20, 0.5, 1)
    # brazil: (-0.08, 0.20, 0.6, 0.92)
    # australia: (-0.08, 0.20, 0.5, 0.92)
    # europe: diff = diff[80:-1, 150:] for demo, (-0.05, 0.20, 0.4, 0.92)
    # usa: (-0.08, 0.20, 0.26, 0.99)
    axins = inset_axes(ax,
                       width="5%",
                       height="50%",
                       loc='lower left',
                       bbox_to_anchor=(-0.08, 0.20, 0.26, 0.99),
                       bbox_transform=ax.transAxes,
                       borderpad=0)
    cbar = fig.colorbar(im, cax=axins, orientation='vertical')
    cbar.ax.tick_params(labelsize=11.5)
    cbar.set_label('Pasture Area \nDifference (%)',
                   rotation=90,
                   labelpad=-120,
                   y=0.55)

    if output_dir is not None:
        plt.savefig(output_dir, format='png', bbox_inches='tight')

    plt.show()
    plt.close()


def plot_histogram_diff_pred_pasture(diff_map, output_dir=None):
    """
    Histogram plot of prediction map - reference for pasture

    Args:
        diff_map (np.array): 2D array of pasture difference map
        output_dir (str): output dir (Default: None)
    """
    # Use percentage
    diff = -diff_map * 100

    rmse_error = np.sqrt(np.nanmean(diff**2))

    # all correct to FAO
    # australia
    # iter-0,1,2,3: (-80, -80, -80), (18200, 15600, 13000)
    # brazil
    # iter-0,1: (-80, -80, -80), (35000, 30000, 25000)
    # iter-2,3: (-90, -90, -90), (26000, 22000, 18000)
    # europe
    # iter-0: (-90, -90, -90), (23000, 19000, 15000)
    # iter-1,2,3: (-90, -90, -90), (30000, 25000, 20000)
    # usa
    # iter-0: (-95, -95, -95), (29000, 24500, 20000)
    # iter-1: (-95, -95, -95), (38000, 32000, 26000)
    # iter-2,3: (-90, -90, -90), (32000, 26000, 20000)

    # all correct to subnation
    # australia
    # iter-0: (-80, -80, -80), (22000, 18500, 15000)
    # iter-1: (-80, -80, -80), (21200, 18100, 15000)
    # iter-2,3: (-80, -80, -80), (18800, 15900, 13000)
    # brazil
    # iter-0,1,2,3: (-85, -85, -85), (33000, 27500, 22000)
    # europe
    # iter-0: (-85, -85, -85), (33000, 27500, 22000)
    # iter-1,2,3: (-85, -85, -85), (43000, 36500, 30000)
    # usa
    # iter-0: (-95, -95, -95), (30000, 25000, 20000)
    # iter-1,2,3: (-95, -95, -95), (31000, 25500, 20000)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
    plt.text(-95, 32000, r'$\mu={:0.4f}$'.format(np.round(np.nanmean(diff),
                                                          4)))
    plt.text(-95, 26000,
             r'$\sigma={:0.4f}$'.format(np.round(np.nanstd(diff), 4)))
    plt.text(-95, 20000, r'RMSE={:0.4f}'.format(np.round(rmse_error, 4)))

    plt.xlim(-100, 100)
    plt.hist(diff.flatten())
    plt.xlabel('Prediction - Reference (%)')
    plt.ylabel('Frequency (#)')

    if output_dir is not None:
        plt.savefig(output_dir,
                    format='png',
                    bbox_inches='tight',
                    transparent=True)

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
