import argparse
import rasterio
from utils.tools.visualizer import plot_global_area_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--global_area_dir",
        type=str,
        default='../../../land_cover/global_area_43200x86400.tif',
        help="path dir to global area map tif file")
    parser.add_argument("--output_dir",
                        type=str,
                        default='../_static/img/land_cover/global_area.png',
                        help="path dir to global area figure output")
    parser.add_argument("--scale",
                        type=int,
                        default=10,
                        help="1/scale factor to image size")

    args = parser.parse_args()
    print(args)

    plot_global_area_map(
        rasterio.open(args.global_area_dir).read()[0, :], args.output_dir,
        args.scale)


if __name__ == '__main__':
    main()
