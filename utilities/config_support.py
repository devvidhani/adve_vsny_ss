import yaml

class ConfigReader:
    def __init__(self, config_file):
        self.config = self.read_configs(config_file)

    def read_configs(self, config_file):
        with open(config_file) as f:
            # config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.safe_load(f)
        return config

    def get_model_inferencing_configs(self, processingtype):
        model_inferencing_config = None
        model_inference_types = self.config.get("ModelInferencingInputs", {}).get("InferenceTypes", {})
        if model_inference_types is not None:
            model_inferencing_config = model_inference_types.get(processingtype)
        return processingtype, model_inferencing_config

    def get_models_configs(self, model_to_load=None):
        if model_to_load is not None:
            return self.config.get("ModelInferencingInputs", {}).get("Models", {}).get(model_to_load)
        else:
            return self.config.get("ModelInferencingInputs", {}).get("Models", {})

    def get_data_configs(self):
        data_configs = self.config["data"]
        return data_configs

    def get_input_configs(self):
        input_configs = self.config["inputs"]
        return input_configs

    def get_output_configs(self):
        output_configs = self.config["outputs"]
        return output_configs