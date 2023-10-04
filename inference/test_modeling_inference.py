from model_inferencing_base import ModelInferencingInputBase

class TestModelingInference(ModelInferencingInputBase):
    def __init__(self, config):
        super().__init__(config)
        self._field1 = config["field1"]

    def isTest(self):
        return True

    def prompt_from_transcripts_csv_fields(self):
        # yield super().prompt_from_transcripts_csv_fields_basic()
        for item in super().prompt_from_transcripts_csv_fields_basic():
            yield item
