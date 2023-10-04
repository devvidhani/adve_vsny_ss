from model_inferencing_base import ModelInferencingInputBase
import csv

class TranscriptsTextTypegen(ModelInferencingInputBase):
    def __init__(self, config):
        super().__init__(config)
        # self.inputs_extraction_method = config.get("inputs_extraction_method", "")
        self._output_file = config["output_file"]
        self._output_file_format = config.get("output_file_format", "csv")
        self._field1 = config["field1"] # "text" field
        self._field2 = config["field2"] # "dirname" field
        self._field3 = config["field3"] # "starttime_sec" field

    def preinference_processing(self):
        # Assume format is CSV for now
        self._csv_writer = None
        if self._output_file is not None:
            self.csv_file = open(self._output_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file, delimiter='|')
            # Write header row
            self.csv_writer.writerow([self._field2, self._field3, "text_type"])

    def prompt_from_transcripts_csv_fields(self):
        for item in super().prompt_from_transcripts_csv_fields_basic():
            yield item

    def model_results_nextsteps(self, **kwargs):
        # Best place to review error handling (OOM, etc.)
        text_type = "Q" if "question" in kwargs["gentext"][15:23] else "A"
        # if debug:
        print(self.row_dict[self._field2], self.row_dict[self._field3], kwargs["gentext"][:50])
        # Assume format is CSV for now
        if self.csv_writer:
            self.csv_writer.writerow([
                self.row_dict[self._field2],
                self.row_dict[self._field3],
                text_type
            ])

    def postinference_processing(self, **kwargs):
        if self.csv_writer:
            self.csv_file.close()