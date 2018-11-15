import yaml


def load_model_config():
    with open("model_config.yml") as config_file:
        model_config = yaml.load(config_file)
        # Todo: check_model_config(model_config)

    return model_config
