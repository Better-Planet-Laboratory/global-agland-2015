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
    parser.add_argument("--grid_size", type=float, default=0.5,
                        help="grid size of gdd input")

    args = parser.parse_args()
    print(args)

    gdd_raw = GDD(args.gdd_xyz_dir)
    gdd_raw.plot(output_dir=args.gdd_raw_output_dir, cmap='viridis')
    _, gdd_mask = gdd_raw.get_mask(gdd_crop_criteria)
    gdd_raw.set_gdd_map(gdd_mask, args.grid_size)
    gdd_raw.plot(output_dir=args.gdd_mask_output_dir, cmap='Greys_r')


if __name__ == '__main__':
    main()
