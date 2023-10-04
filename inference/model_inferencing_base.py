import csv

class ModelInferencingInputBase:
    def __init__(self, config):
        self.config = config
        self.prompt_instr = config.get("prompt_instr", "")
        self.input_file = config.get("input_file", "")
        self.max_length = config.get("max_length", 100)
        # self.output_type = config.get("output_type", "origtext")
        self.output_type = config.get("output_type", ["input_text"])
        self.inputs_extraction_method = config.get("inputs_extraction_method", "pass_method")

    # Defined in derived classes if Test is True
    def isTest(self):
        return False

    def get_output_type(self):
        return self.output_type
    
    def get_maxlen(self):
        return self.max_length

    # Redefined in derived classes if Test is True
    def preinference_processing(self):
        pass

    # Redefined in derived classes if Test is True
    def prompt_from_inputs(self):
        method_name = self.inputs_extraction_method
        method_to_call = getattr(self, method_name, None)
        
        if callable(method_to_call):
            return method_to_call()
        else:
            # Handle the case when the method is not found or not callable
            return None  # Or some other appropriate value
        # yield self.inputs_extraction_method()()

    def prompt_from_transcripts_csv_fields_basic(self):
        with open(self.input_file) as f:
            reader = csv.reader(f)
            fieldnames = next(reader)
            for row in reader:
                self.row_dict = dict(zip(fieldnames, row))
                yield "[INST] {} {} [/INST]".format(self.prompt_instr, row[fieldnames.index(self._field1)])
                # yield returnval

    # Custom method
    def pass_method(self):
        yield None

    def model_results_nextsteps(self, **kwargs):
        print(kwargs["input_text"])

    def postinference_processing(self, **kwargs):
        pass