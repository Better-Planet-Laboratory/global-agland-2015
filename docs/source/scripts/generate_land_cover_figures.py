import argparse
import rasterio
import numpy as np
from utils.io import load_yaml_config
from utils.tools.visualizer import plot_land_cover_map

LAND_COVER_CFG = load_yaml_config('../../../configs/land_cover_cfg.yaml')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--land_cover_dir", type=str, default='../../../land_cover/MCD12Q1_merged.tif',
                        help="path dir to land cover tif file")
    parser.add_argument("--output_dir", type=str, default='../_static/img/land_cover/land_cover.png',
                        help="path dir to land cover figure output")
    parser.add_argument("--scale", type=int, default=1,
                        help="1/scale factor to image size")

    args = parser.parse_args()
    print(args)
    print('Loading MCD12Q1 table by default')

    plot_land_cover_map(rasterio.open(args.land_cover_dir).read()[0, :].astype(np.uint8),
                        LAND_COVER_CFG['code']['MCD12Q1'],
                        args.output_dir,
                        LAND_COVER_CFG['null_value'],
                        args.scale)


if __name__ == '__main__':
    main()
