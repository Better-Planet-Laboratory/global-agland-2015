from utils import io
from utils.process.train_process import pipeline

TRAINING_CFG = io.load_yaml_config('configs/training_cfg.yaml')
LAND_COVER_CFG = io.load_yaml_config('configs/land_cover_cfg.yaml')

if __name__ == '__main__':
    pipeline(TRAINING_CFG, LAND_COVER_CFG)
