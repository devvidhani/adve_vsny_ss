from transformers import AutoModelForCausalLM, AutoTokenizer
from utilities.config_support import ConfigReader
import argparse
import torch
from model_inferencing_base import ModelInferencingInputBase
from transcript_text_type_gen import TranscriptsTextTypegen
from test_modeling_inference import TestModelingInference
from qa_texts_tokens_len import CreateTextTokensLen
from qa_responses import CreateTextResponses
import gc

class ModelWrapper:
    def __init__(self, model_configs, tokn):
        self.ft_models_dir = model_configs.get("ft_models_dir", "")
        self.model_name = model_configs.get("model_name", "")
        self.tokenizer_name = model_configs.get("tokenizer_name", self.model_name)
        self.model_path = self.ft_models_dir + "/" + self.model_name
        self.tokn = tokn
        self.model = None
        self.tokenizer = None
        return

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, device_map="auto", load_in_4bit=True, token=self.tokn)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, use_fast=True, token=self.tokn)
        return

    def generate(self, prompt, action=["input_text"], max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda:0")
        outputs = {}

        torch.cuda.empty_cache()
        gc.collect()

        # if action == "inptext_token_count" or action == "output_token_count":
        if "input_text_token_len" in action:
            outputs["input_text_token_len"] = torch.numel(inputs["input_ids"])

        if "input_text_len" in action:
            outputs["input_text_len"] = len(prompt.split()) # self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
        
        # if action == "origtext":
        if "input_text" in action:
            outputs["input_text"] = prompt # Just return the orig text as is
        
        if bool(set(action) & set(["output_text_token_len", "output_text_len", "gentext"])):
            try:
                self.tokenizer.add_special_tokens({"pad_token":"<PAD>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

                gen_config = {"temperature": 0.7,
                              "top_p": 0.1,
                            #   "repetition_penalty": 1.18,
                              "repetition_penalty": 1.19,
                              "top_k": 40,
                              "do_sample": True,
                            #   "max_new_tokens": 100,  
                              "pad_token_id": 18610
                             }

                output = self.model.generate(max_length=max_length, **inputs, **gen_config)
                output_only = output[0][inputs['input_ids'].shape[-1]:]

                if "output_text_token_len" in action:
                    outputs["output_text_token_len"] = len(output_only)

                if "gentext" in action:
                    outputs["gentext"] = self.tokenizer.decode(output_only, skip_special_tokens=True)
                    # outputs["gentext"] = self.tokenizer.decode(output[0], skip_special_tokens=True)

                if "output_text_len" in action:
                    outputs["output_text_len"] = len(outputs["gentext"].split())

            except RuntimeError as e:
                outputs["output_text_token_len"] = 0 # 0 to indicate error
                outputs["output_text_len"] = 0 # 0 to indicate error
                if "CUDA out of memory" in str(e):
                    outputs["gentext"] = "CUDA out of memory error."
                else: 
                    outputs["gentext"] = "Error: {}".format(e)

        # if action not in ["input_text_token_len", "input_text_len", "input_text", "output_text_token_len", "output_text_len", "gentext"]:
        if not(bool(set(action) & set(["input_text_token_len", "input_text_len", "input_text", "output_text_token_len", "output_text_len", "gentext"]))):
            outputs["error"] = "Invalid action: {}".format(action)

        return outputs

class PromptGenerator:
    def __init__(self, inferencing_input_class_inst):
        self._inferencing_input_class_inst = inferencing_input_class_inst

    # In future, model's prompt limitations may be passed from here - like inserting PAD, BOS, EOS, etc.
    def get_prompt(self):
        for item in self._inferencing_input_class_inst.prompt_from_inputs():
            yield item


import sys
# Main method
def main(args):
    config_reader = ConfigReader(args.config_file)
    model_inferencing_class, model_inferencing_config = config_reader.get_model_inferencing_configs(args.modelinferencingclass)
    tokn = ConfigReader(args.token_file).config["hf_token"]

    # Load model
    if (args.model_to_load is not None):
        model_wrapper = ModelWrapper(config_reader.get_models_configs(args.model_to_load), tokn)
        model_wrapper.load_model()

    # Model Execution
    # Instantiate the appropriate class based on the class name based on inferencing type
    if model_inferencing_class in globals():
        inferencing_input_class_inst = globals()[model_inferencing_class](model_inferencing_config)
    else:
        print("Invalid model inferencing class: {}. Reverting to Base Inferencing".format(model_inferencing_class))
        inferencing_input_class_inst = ModelInferencingInputBase(model_inferencing_config)

    inferencing_input_class_inst.preinference_processing()
    prompt_generator = PromptGenerator(inferencing_input_class_inst)
    model_gen_results = {}
    for prompt in prompt_generator.get_prompt():
        model_gen_results = model_wrapper.generate(prompt,
                                                   inferencing_input_class_inst.get_output_type(),
                                                   inferencing_input_class_inst.get_maxlen())
        inferencing_input_class_inst.model_results_nextsteps(**model_gen_results)
    inferencing_input_class_inst.postinference_processing()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    parser.add_argument("--token_file", type=str, default="configs/env.yaml", help="Path to the token yaml file")
    parser.add_argument("--model_to_load", type=str, default="hf_llama2_7b_chat", help="Name of the model to load")
    parser.add_argument("--modelinferencingclass", type=str, default="TestModelingInference", help="Model inferencing inputs")
    args = parser.parse_args()

    main(args)
