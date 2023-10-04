import sys
import re
import os
import argparse
from utilities.config_support import ConfigReader

def merge_continuous_speakers(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Remove unwanted lines
    lines = [line.strip() for line in lines if line.strip() and "WEBVTT" not in line]

    merged_lines = []
    current_speaker = None
    current_text = ""
    start_time = None
    end_time = None

    i = 0
    while i < len(lines):
        tsline, speakerline = lines[i], lines[i+1].strip()
        start, _, end = tsline.split(' ')
        match = re.search(r"^\[SPEAKER_(\d+)\]: (.*)", speakerline)
        # pattern = r"^\[SPEAKER_(\d+)\]: (.*)"
        if match:
            new_speaker = match.group(1)
        else:
            new_speaker = '99'

        if current_speaker == new_speaker:
            end_time = end
            if match and match.group(2):
                current_text += ". " + match.group(2)
            else:
                current_text += ". " + speakerline
        else:
            if (start_time and end_time):
                merged_lines.append(f"{start_time} --> {end_time}")
                if (current_speaker == '99'):
                    current_text = "[SPEAKER_99]: " + current_text
                # merged_lines.append(current_speaker)
                merged_lines.append(current_text)
            start_time = start
            end_time = end
            current_text = speakerline
            current_speaker = new_speaker
        i += 2

    if current_speaker:
        merged_lines.append(f"{start_time} --> {end_time}")
        merged_lines.append(current_speaker)
        merged_lines.append(current_text)

    with open(output_file, 'w') as f:
        for line in merged_lines:
            f.write(line + "\n")

def process_subdirectories(input_path):
    all_vtt_files = []

    if os.path.isdir(input_path):  # Directory processing
        for root, _, files in os.walk(input_path):
            all_vtt_files.extend([os.path.join(root, file) for file in files if file.lower().endswith("audio.vtt")])
    
    elif os.path.isfile(input_path):  # File processing
        if input_path.lower().endswith(".vtt"):  # .vtt file
            all_vtt_files.append(input_path)
        
        else:  # File containing directory names
            with open(input_path, 'r') as f:
                directories = [line.strip() for line in f.readlines()]
                for directory in directories:
                    audio_file = os.path.join(directory, "audio.vtt")
                    if os.path.exists(audio_file):
                        all_vtt_files.append(audio_file)

    for input_file in all_vtt_files:
        output_file = input_file.replace("audio.vtt", "transcripts_inputs_by_speakers.vtt")
        merge_continuous_speakers(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="consolidate audio.vtt generated from whisperx. Put continous speakers entries with different timestamps durations as one single speaker entry with longer timestamp duration entry.")
    # mutually exclusive group: --input and --config_file
    config_input_group = parser.add_mutually_exclusive_group()
    config_input_group.add_argument("--input", type=str, help="Input audio.vtt file (for testing).")
    config_input_group.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    args = parser.parse_args()

    if not args.input:
        config_reader = ConfigReader(args.config_file)
        data_configs = config_reader.get_data_configs()
        data_dir = data_configs["youtube_downloaded_videos_dir"]
        process_subdirectories(data_dir)
    else:
        process_subdirectories(args.input)
