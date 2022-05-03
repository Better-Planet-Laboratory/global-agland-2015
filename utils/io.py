import yaml
import pickle


def load_yaml_config(file_path):
    """ Load .yaml config files for training """
    with open(file_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def save_pkl(obj, directory):
    """ Save obj as pkl file """
    with open(directory + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(directory):
    """ Load pkl file """
    with open(directory + '.pkl', 'rb') as f:
        return pickle.load(f)
