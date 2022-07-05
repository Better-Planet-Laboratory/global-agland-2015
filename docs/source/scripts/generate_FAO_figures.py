import argparse
from census_processor.world import World
from utils import io

ROOT = './' # run from root

CENSUS_SETTING_CFG = io.load_yaml_config(ROOT + 'configs/census_setting_cfg.yaml')
SHAPEFILE_CFG = io.load_yaml_config(ROOT + 'configs/shapefile_cfg.yaml')

WORLD_CENSUS = World(ROOT + SHAPEFILE_CFG['path_dir']['World'],
                     ROOT + CENSUS_SETTING_CFG['path_dir']['FAOSTAT'],
                     ROOT + CENSUS_SETTING_CFG['path_dir']['FAOSTAT_profile'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cropland_output_dir", type=str, default='docs/source/_static/img/FAOSTAT/cropland_FAO.png',
                        help="path dir output cropland from FAO census")
    parser.add_argument("--pasture_output_dir", type=str, default='docs/source/_static/img/FAOSTAT/pasture_FAO.png',
                        help="path dir output pasture from FAO census")

    args = parser.parse_args()
    print(args)

    WORLD_CENSUS.plot_cropland(args.cropland_output_dir)
    WORLD_CENSUS.plot_pasture(args.pasture_output_dir)


if __name__ == '__main__':
    main()
