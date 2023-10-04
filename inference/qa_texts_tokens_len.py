from model_inferencing_base import ModelInferencingInputBase
import csv

class CreateTextTokensLen(ModelInferencingInputBase):
    def __init__(self, config):
        super().__init__(config)
        self._output_file = config["output_file"]
        self._output_file_format = config.get("output_file_format", "csv")
        self._field1 = config["field1"] # "text" field
        self._field2 = config["field2"] # "dirname" field
        self._field3 = config["field3"] # "starttime_sec" field
        self._field4 = config["field4"] # "aggregated_answer_text" field
        # Do we need it?
        self._output_headers = config.get("output_headers", [self._field2, self._field3, "text_type"])
        self._output_headers += [action_type for action_type in self.output_type if action_type != "input_text"]

    def preinference_processing(self):
        # Assume format is CSV for now
        self._csv_writer = None
        if self._output_file is not None:
            self.csv_file = open(self._output_file, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file, delimiter='|')
            # Write header row
            self.csv_writer.writerow(self._output_headers)

    def prompt_from_csv_get_answers(self):
        with open(self.input_file) as f:
            reader = csv.reader(f, delimiter='|')
            fieldnames = next(reader)
            for row in reader:
                self.row_dict = dict(zip(fieldnames, row))
                self.texttype = ""
                item = "[INST] {} {} [/INST]".format(self.prompt_instr if self.prompt_instr is not None else '', self.row_dict[self._field1])
                # item = "[INST] {} {} [/INST]".format(self.prompt_instr, self.row_dict[self._field1])
                yield item

    def prompt_from_csv_fields_QA_dataset(self):
        with open(self.input_file) as f:
            reader = csv.reader(f, delimiter='|')
            fieldnames = next(reader)
            for row in reader:
                self.row_dict = dict(zip(fieldnames, row))
                self.texttype = "Q"
                item = "[INST] {} {} [/INST]".format(self.prompt_instr if self.prompt_instr is not None else '', self.row_dict[self._field1])
                # item = "[INST] {} {} [/INST]".format(self.prompt_instr, self.row_dict[self._field1])
                yield item
                self.texttype = "A"
                item = "[INST] {} [/INST]".format(self.row_dict[self._field4])
                yield item
                self.texttype = "QA"
                item = "[INST] {} {} {} [/INST]".format(self.prompt_instr if self.prompt_instr is not None else '', self.row_dict[self._field1], self.row_dict[self._field4])
                yield item

    def model_results_nextsteps(self, **kwargs):
        # Best place to review error handling (OOM, etc.)
        if self.csv_writer:
            row_values = [
                self.row_dict[self._field2],
                self.row_dict[self._field3],
                self.texttype,
            ]

            for action_type in self.output_type:
                if action_type != "input_text":
                    row_values.append(kwargs.get(action_type, ""))
                else: # Handle input_text separately if needed
                    pass
                    # row_values.append(kwargs.get("input_text", ""))

            # if debug, print the row to the command line
            print("CSV Row:", row_values)

            # Write the row to the CSV
            self.csv_writer.writerow(row_values)

        # Assume format is CSV for now
        # if self.csv_writer:
        #     self.csv_writer.writerow([
        #         self.row_dict[self._field2],
        #         self.row_dict[self._field3],
        #         self.lentype,
        #         self.lenval,
        #         kwargs[self.output_type]
        #     ])

    def postinference_processing(self, **kwargs):
        if self.csv_writer:
            self.csv_file.close()