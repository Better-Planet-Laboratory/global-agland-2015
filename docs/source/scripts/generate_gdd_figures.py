import argparse
from utils.tools.gdd_core import *
from gdd.gdd_criteria import gdd_crop_criteria


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gdd_xyz_dir", type=str, default='../../../gdd/gdd.xyz.txt',
                        help="path dir to gdd xyz file")
    parser.add_argument("--gdd_raw_output_dir", type=str, default='../_static/img/gdd/gdd_raw.png',
                        help="path dir to gdd raw output")
    parser.add_argument("--gdd_mask_output_dir", type=str, default='../_static/img/gdd/gdd_mask.png',
                        help="path dir to gdd mask output")
    parser.add_argument("--grid_size", type=float, default=0.5/60,
                        help="grid size of gdd input")
    # parser.add_argument("--land_cover_dir", type=str, default="../../../land_cover/MCD12Q1_merged.tif",
    parser.add_argument("--land_cover_dir", type=str, default="../../../land_cover/land_cover_map_cropland_mask.tif",
                        help="land cover map file dir")

    args = parser.parse_args()
    print(args)


    # To upsample gdd mask, we need to first create a binary mask to filter out non-gdd areas
    gdd_raw = GDD(args.gdd_xyz_dir)
    gdd_regions =  gdd_raw.gdd_map.copy()
    gdd_regions[gdd_regions < 0] = 0
    gdd_regions[gdd_regions > 0] = 1

    gdd_map = gdd_raw.gdd_map * gdd_regions

    # Dump upsampled gdd and its mask locally to save memory
    tmp_gdd_map_filepath= "./tmp_gdd_map.dat"
    tmp_gdd_regions_filepath= "./tmp_gdd_regions.dat"
    
    tmp_gdd_map = np.memmap(tmp_gdd_map_filepath, 
                        dtype=np.float32, mode='w+',
                        shape=(43200, 21600))
    tmp_gdd_regions = np.memmap(tmp_gdd_regions_filepath, 
                        dtype=np.uint8, mode='w+',
                        shape=(43200, 21600))
    tmp_gdd_map = cv2.resize(gdd_map,
                          (43200, 21600), 
                          interpolation=cv2.INTER_LANCZOS4)
    tmp_gdd_regions = cv2.resize(gdd_regions,
                          (43200, 21600), 
                          interpolation=cv2.INTER_NEAREST)
    gdd_raw.set_gdd_map(tmp_gdd_map, args.grid_size)
    _, gdd_mask = gdd_raw.get_mask(gdd_crop_criteria, args.land_cover_dir, tmp_gdd_regions)
    gdd_raw.set_gdd_map(gdd_mask, args.grid_size)
    gdd_raw.plot(output_dir=args.gdd_mask_output_dir, cmap='Greys_r')

    # Save gdd_mask as tif 
    save_array_as_tif(args.gdd_mask_output_dir[:-len("png")] + "tif", gdd_mask,
                    x_min=-180, y_max=90, pixel_size=args.grid_size,
                    epsg=4326)


if __name__ == '__main__':
    main()
