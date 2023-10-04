import pandas as pd
import csv
import os
import re
import yaml
from utilities.utils import timestamp_to_seconds_or_string_or_datetime
from utilities.config_support import ConfigReader
import argparse

def find_main_speaker_pandas(transcripts_filepath):
    transcripts_df = pd.read_csv(transcripts_filepath, delimiter='|')
    transcripts_df['duration'] = transcripts_df['endtime_sec'] - transcripts_df['starttime_sec']
    speaker_duration = transcripts_df.groupby('speaker')['duration'].sum()
    main_speaker = speaker_duration.idxmax()
    return main_speaker

def merge_sort_append_answers(questions_filepath, transcripts_filepath, main_speaker):
    questions_df = pd.read_csv(questions_filepath, delimiter='|')
    transcripts_df = pd.read_csv(transcripts_filepath, delimiter='|')
    
    merged_df = pd.concat([questions_df, transcripts_df]).sort_values(by='starttime_sec')
    
    results = []
    prev_question = None
    
    for i, row in merged_df.iterrows():
        is_question = row['text_type'] == 0
        
        if is_question and prev_question is not None:
            start_time = prev_question['starttime_sec']
            end_time = row['starttime_sec']
            answers = transcripts_df[
                (transcripts_df['speaker'] == main_speaker) & 
                (transcripts_df['starttime_sec'] >= start_time) & 
                (transcripts_df['starttime_sec'] <= end_time)
            ]
            answer_text = " ".join(answers['text'])
            answer_starttime_sec = answers['starttime_sec'].min() if not answers.empty else ''
            answer_endtime_sec = answers['endtime_sec'].max() if not answers.empty else ''
            
            results.append({
                'question': prev_question['text'],
                'question_starttime_sec': prev_question['starttime_sec'],
                'answers_from_main_speaker': answer_text,
                'answers_starttime_sec': answer_starttime_sec,
                'answers_endtime_sec': answer_endtime_sec
            })
        
        if is_question:
            prev_question = row
    
    return pd.DataFrame(results)

def write_results_pandas(results_df, outputdir, output_file_name):
    output_filepath = os.path.join(outputdir, f"{output_file_name}.csv")
    results_df.to_csv(output_filepath, sep='|', index=False)
    return output_filepath

def create_individual_video_dataset(outputdir, trans_csv_file, quest_csv_file, output_file_name):
    main_speaker = find_main_speaker_pandas(trans_csv_file)
    results_df = merge_sort_append_answers(quest_csv_file, trans_csv_file, main_speaker)
    return write_results_pandas(results_df, outputdir, output_file_name)

def process_subdirectories_for_datacreation(input_path, outputs_dir, output_file_name, final_dataset_file):
    all_trans_csv_files = []
    all_quest_csv_files = []

    if os.path.isdir(input_path):  # Directory processing
        for root, _, files in os.walk(input_path):
            all_trans_csv_files.extend([os.path.join(root, file) for file in files if file == "transcripts_from_vtt_with_timestamps.csv"])
            all_quest_csv_files.extend([os.path.join(root, file) for file in files if file == "questions_from_description_with_timestamps.csv"])
    
    elif os.path.isfile(input_path):  # File processing
        # File containing directory names
        with open(input_path, 'r') as f:
            directories = [line.strip() for line in f.readlines()]
            for directory in directories:
                trans_csv_file = os.path.join(directory, "transcripts_from_vtt_with_timestamps.csv")
                quest_csv_file = os.path.join(directory, "questions_from_description_with_timestamps.csv")
                if os.path.exists(trans_csv_file) and os.path.exists(quest_csv_file):
                    all_trans_csv_files.append(trans_csv_file)
                    all_quest_csv_files.append(quest_csv_file)

    all_dataset_files = []
    for i,_ in enumerate(all_trans_csv_files):
        all_dataset_files.append(create_individual_video_dataset(os.path.dirname(all_quest_csv_files[i]), all_trans_csv_files[i],
                                                                 all_quest_csv_files[i], output_file_name))

    # Combine all the individual dataset files into one - but only question and answers_from_main_speaker columns
    all_df = []
    for file in all_dataset_files:
        df = pd.read_csv(file, delimiter='|')
        all_df.append(df[['question', 'answers_from_main_speaker']])
    # Write the combined dataset to a file
    final_df = pd.concat(all_df)
    final_df.to_csv(os.path.join(outputs_dir, final_dataset_file), sep='|', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine questions from description.txt and speakers info from transcripts to generate dataset.")
    # mutually exclusive group: --input and --config_file
    config_input_group = parser.add_mutually_exclusive_group()
    config_input_group.add_argument("--input", type=str, help="Input directory (for testing).")
    config_input_group.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    parser.add_argument("--output_file", type=str, default="adve_vsny_dataset.csv", help="Name of the output file.")
    parser.add_argument("--final_dataset_file", type=str, default="adve_vsny_qa_dataset.csv", help="Name of the output file.")
    args = parser.parse_args()

    if not args.input:
        config_reader = ConfigReader(args.config_file)
        data_configs = config_reader.get_data_configs()
        data_dir = data_configs["youtube_downloaded_videos_dir"]
        output_configs = config_reader.get_output_configs()
        outputs_dir = output_configs["output_dir"]
        process_subdirectories_for_datacreation(data_dir, outputs_dir, args.output_file, args.final_dataset_file)
    else:
        process_subdirectories_for_datacreation(args.input, os.getcwd(), args.output_file, args.final_dataset_file)