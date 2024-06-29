from utils import io
from utils.process.post_process import pipeline

DEPLOY_SETTING_CFG = io.load_yaml_config('configs/deploy_setting_cfg.yaml')
TRAINING_CFG = io.load_yaml_config('configs/training_cfg.yaml')
LAND_COVER_CFG = io.load_yaml_config('configs/land_cover_cfg.yaml')

if __name__ == '__main__':
    pipeline(DEPLOY_SETTING_CFG, LAND_COVER_CFG, TRAINING_CFG)