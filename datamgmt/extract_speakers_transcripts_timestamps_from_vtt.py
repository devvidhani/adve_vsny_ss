import csv
import os
import re
import yaml
from utilities.utils import timestamp_to_seconds_or_string_or_datetime
from utilities.config_support import ConfigReader
import argparse

def parse_transcripts(filename):
    data = []
    with open(filename) as f:
        starttime = endtime = starttime_sec = endtime_sec = ""
        for line in f:
            if '-->' in line:
                starttime, endtime = line.split('-->')
                starttime, endtime = starttime.strip(), endtime.strip()
                starttime_sec, endtime_sec = (timestamp_to_seconds_or_string_or_datetime(starttime), 
                                              timestamp_to_seconds_or_string_or_datetime(endtime))
            else:
                match = re.match(r'\[SPEAKER_(\d+)\]: (.*)', line)
                if match:
                    speaker = match.group(1)
                    text = match.group(2)
                    text_len = len(text)
                    
                    row = {
                        'starttime': starttime,
                        'endtime': endtime,
                        'starttime_sec': starttime_sec,
                        'endtime_sec': endtime_sec,
                        'speaker': speaker,
                        'text': text,
                        'text_length': text_len,
                        'text_type': ''
                    }
                    data.append(row)
    return data

def write_csv(data, output_csv_path):
    fieldnames = ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def write_transcripts_with_timestamps(filename, outputdir, output_file_name):
    data = parse_transcripts(filename)
    output_csv_path = os.path.join(outputdir, f"{output_file_name}.csv")
    write_csv(data, output_csv_path)

def process_transcripts_in_subdirectories(input_path, output_file_name):
    all_trans_vtt_files = []

    if os.path.isdir(input_path):  # Directory processing
        for root, _, files in os.walk(input_path):
            all_trans_vtt_files.extend([os.path.join(root, file) for file in files if file == "transcripts_inputs_by_speakers.vtt"])
    
    elif os.path.isfile(input_path):  # File processing
        if os.path.basename(input_path) == "transcripts_inputs_by_speakers.vtt":
            all_trans_vtt_files.append(input_path)
        
        else:  # File containing directory names
            with open(input_path, 'r') as f:
                directories = [line.strip() for line in f.readlines()]
                for directory in directories:
                    audio_file = os.path.join(directory, "transcripts_inputs_by_speakers.vtt")
                    if os.path.exists(audio_file):
                        all_trans_vtt_files.append(audio_file)

    for input_file in all_trans_vtt_files:
        write_transcripts_with_timestamps(input_file, os.path.dirname(input_file), output_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions from description.txt files and dump them to a file with timestamps.")
    # mutually exclusive group: --input and --config_file
    config_input_group = parser.add_mutually_exclusive_group()
    config_input_group.add_argument("--input", type=str, help="Input vtt file (for testing).")
    config_input_group.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    parser.add_argument("--output_file", type=str, default="transcripts_from_vtt_with_timestamps", help="Name of the output file.")
    args = parser.parse_args()

    if not args.input:
        config_reader = ConfigReader(args.config_file)
        data_configs = config_reader.get_data_configs()
        data_dir = data_configs["youtube_downloaded_videos_dir"]
        # output_configs = config_reader.get_output_configs()
        # outputs_dir = output_configs["output_dir"]
        process_transcripts_in_subdirectories(data_dir, args.output_file)
    else:
        process_transcripts_in_subdirectories(args.input, args.output_file)