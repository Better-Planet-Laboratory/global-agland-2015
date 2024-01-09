from utils.io import gdal, save_array_as_tif
import rasterio
import numpy as np
import cv2


def load_aridity_map(tif_path, tfw_path=None, scaling_factor=0.0001):
    """
    Helper function that loads Global-AI_ET0_v3_annual map as numpy array. Based on 
    the documentation provided, there is a default scaling factor of 0.0001 that 
    needs to be applied on the raw map to convert from int to actual float value

    Args:
        tif_path (str): path to the tif file
        tfw_path (str, optional): path to the tfw file. Defaults to None.
        scaling_factor (float, optional): scaling factor. Defaults to 0.0001.

    Returns: (np.ndarray) Aridity map
    """

    if tfw_path is None:
        aridity_map = rasterio.open(tif_path).read()[0, :] * scaling_factor
    else:
        raw_tif = gdal.Open(tif_path)
        with open(tfw_path, 'r') as f:
            tfw_lines = f.readlines()
            transformation = [float(line.strip()) for line in tfw_lines]
        raw_tif.SetGeoTransform(transformation)
        aridity_map = raw_tif.ReadAsArray() * scaling_factor

    return aridity_map


def load_AEI_map(asc_path):
    """
    Load AEI map from .asc file as numpy array. Based on the documentation 
    found here: https://zenodo.org/record/6886564#.ZAdxOezML0r, and according 
    to the authors, the unit in the dataset is Ha

    Args:
        asc_path (str): path to .asc file

    Returns: (np.ndarray) AEI map
    """
    aei_map = np.loadtxt(asc_path, skiprows=6)
    return aei_map


def make_aridity_mask(threshold_AEI, threshold_AI, aridity_map, aei_map, size=(2160, 4320)):
    """
    To make an aridity mask, a threshold_AEI is first applied on aei_map to get a 
    binary map, any region that has AEI value < threshold_AEI (unirrigated region) is 
    considered in aridity_map. Within these regions in aridity_map, any pixels with AI value 
    < threshold_AI is marked as excluded (0) in the final mask, otherwise included (1)

    Args:
        threshold_AEI (float): threshold_AEI
        threshold_AI (float): threshold_AI
        aridity_map (np.ndarray): aridity_map
        aei_map (np.ndarray): aei_map
        size (tuple, optional): size. Defaults to (2160, 4320).

    Returns:
        (np.ndarray): aridity_mask
    """
    # Process AEI mask
    aei_map = cv2.resize(aei_map, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    aei_mask = np.zeros_like(aei_map)

    aei_mask[np.where(aei_map >= threshold_AEI)] = 1
    aei_mask = 1 - aei_mask
    aei_mask[np.where(aei_mask == 0)] = np.nan

    # Process AI mask
    aridity_map = cv2.resize(aridity_map, dsize=(size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    aridity_mask = np.zeros_like(aridity_map)
    aridity_map *= aei_mask

    aridity_mask[np.where(aridity_map >= threshold_AI)] = 1
    aridity_mask[np.where(np.isnan(aridity_map))] = 1
    aridity_mask.astype(np.uint16)

    return aridity_mask

if __name__ == '__main__':

    for threshold_AEI in [0.01]:
        for threshold_AI in [0.05]:
            aridity_mask = make_aridity_mask(threshold_AEI=threshold_AEI, 
                                            threshold_AI=threshold_AI, 
                                            aridity_map=load_aridity_map(tif_path='./Global-AI_ET0_v3_annual/ai_v3_yr.tif', 
                                                                        tfw_path='./Global-AI_ET0_v3_annual/ai_v3_yr.tfw'),
                                            aei_map=load_AEI_map('./G_AEI_2015.asc'), 
                                            size=(2160, 4320))

            save_array_as_tif(f"./aridity_masks/aridity_mask_thAEI_{str(threshold_AEI)}_thAI_{str(threshold_AI).replace('.', '')}.tif", aridity_mask, 
                              x_min=-180, y_max=90,
                              pixel_size=0.083333333333333333333, epsg=4326,
                              no_data_value=-1,
                              dtype=gdal.GDT_UInt16)
