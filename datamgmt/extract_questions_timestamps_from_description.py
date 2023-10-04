import os
import re
# from datetime import datetime
import argparse
from utilities.config_support import ConfigReader
from utilities.utils import timestamp_to_seconds_or_string_or_datetime
import csv

# Example usage
# Note: Provide a valid path to a directory containing subdirectories with description.txt files for real usage
# example_datadir = "/path/to/directory"
# extracted_lines = extract_description_lines(example_datadir)

# Define the search pattern (case-insensitive)
pattern = re.compile(r'with time-stamps|with timestamps', re.IGNORECASE)

def extract_description_lines(input_file):
    """
    Traverse through datadir, find description.txt in each subdirectory, 
    extract relevant lines based on the pattern, and return them as a list of strings.
    """
    questions_section_from_description = []
    
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    found_pattern = False
    for line in lines:
        if not found_pattern:
            found_pattern = pattern.search(line)
        elif found_pattern:
            if (line.strip() == "") or (re.search(r'new york', line, re.IGNORECASE)):
                break
            questions_section_from_description.append(line.strip())
                        
    return questions_section_from_description


# Example usage
# Note: Provide a valid questions_section_from_description list for real usage
# example_questions_section = ["1:00 Question 1", "Some additional info", "2:00 Question 2"]
# extracted_questions = extract_questions(example_questions_section)
def extract_questions(questions_section_from_description):
    """
    Take a list of lines (questions_section_from_description), process them to aggregate lines 
    that belong together, and return them as a new list of strings.
    """
    questions_only_from_description_question_section = []
    current_line = ""
    
    # Function to check if a line contains the pattern \d:\d\d
    def contains_pattern(line):
        return re.search(r'\d:\d\d', line)
    
    for line in questions_section_from_description:
        line = line.strip()
        
        # Check if the line contains the pattern
        if contains_pattern(line):
            # If a new line is starting, add the current_line to the output_lines
            if current_line:
                questions_only_from_description_question_section.append(current_line)
            current_line = line
        else:
            # If the line doesn't contain the pattern, append it to the current_line
            current_line += ' ' + line

    # Add the last line to the output
    if current_line:
        questions_only_from_description_question_section.append(current_line)
    
    return questions_only_from_description_question_section


# Example usage
# Note: Provide a valid questions_only_from_description_question_section list for real usage
# example_questions_only = ["1:00 Question 1 a) part 1 b) part 2", "2:00 Question 2 (a) part A (b) part B"]
# split_questions_list = split_questions(example_questions_only)
def split_questions(questions_only_from_description_question_section):
    """
    Take a list of lines, split lines at specified patterns, and return them as 
    a new list of strings.
    """
    simple_questions_first_attempt = []
    split_patterns = [r'\(.\)', r' .\)']

    def split_lines_at_patterns(input_lines, patterns):
        output_lines = []
        current_line = ""

        for line in input_lines:
            line_copy = line  # Create a copy to preserve the original line
            match_count = 0  # Initialize match count
            for pattern in patterns:
                matches = re.findall(pattern, line_copy)
                for match in matches:
                    match_count += 1
                    if match_count > 1:  # Add '\n' after the first match
                        line_copy = line_copy.replace(match, f'\n{match}', 1)
            output_lines.extend(line_copy.split('\n'))  # Split the line at '\n' and add to output
        return output_lines

    simple_questions_first_attempt = split_lines_at_patterns(
        questions_only_from_description_question_section, split_patterns)
    
    return simple_questions_first_attempt

# Example usage
# Note: Provide a valid simple_questions_first_attempt list and output directory for real usage
# example_simple_questions = ["1:00 Question 1", "Question 1 continued", "2:00 Question 2"]
# example_output_dir = "/path/to/output/directory"
# created_file_path = write_questions_with_timestamps(example_simple_questions, example_output_dir)
def write_questions_with_timestamps(simple_questions_first_attempt, outputdir, output_file_name):
    """
    Take a list of lines (simple_questions_first_attempt), process them, 
    and write them to an output file named questions_from_description_with_timestamps.txt
    and questions_from_description_with_timestamps.csv in the initial input directory.
    """
    # Regular expression pattern to match timestamps
    timestamp_pattern = r"(\d{1,2}:\d{2}:\d{2}|\d{1,2}:\d{2})"

    # Initialize the last read timestamp
    last_timestamp = None

    # Function to extract and update the last timestamp
    def extract_timestamp(line):
        nonlocal last_timestamp  # Use the nonlocal last_timestamp variable
        match = re.search(timestamp_pattern, line)
        if match:
            last_timestamp = match.group(0)  # Update the last timestamp
        return last_timestamp

    def formatted_lines_to_csv_data(qu_timestamps, qu_lines):
        data = []
        for i, line in enumerate(qu_timestamps):
            timestamp_str = qu_timestamps[i].strip()
            text = qu_lines[i].strip()
            starttime_sec = timestamp_to_seconds_or_string_or_datetime(timestamp_str)
            
            if i < len(qu_timestamps) - 1:
                endtime_str = qu_timestamps[i+1].strip()
                endtime_sec = timestamp_to_seconds_or_string_or_datetime(endtime_str)
            else:
                endtime_str = "99:99:99"
                endtime_sec = timestamp_to_seconds_or_string_or_datetime(endtime_str)
            
            row = {
                'starttime': timestamp_str,
                'endtime': endtime_str,
                'starttime_sec': starttime_sec,
                'endtime_sec': endtime_sec,
                'speaker': 99,
                'text': text,
                'text_length': len(text),
                'text_type': 0
            }
            data.append(row)
        return data

    def write_csv(data, filepath):
        fieldnames = ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    # Process the lines and extract timestamps
    formatted_lines = []
    qu_timestamps = []
    qu_lines = []
    for line in simple_questions_first_attempt:
        timestamp = extract_timestamp(line)
        if timestamp:
            line = re.sub(timestamp_pattern, '', line)  # Remove timestamp from the line
            match = re.match(r"Q\.\d*", line)
            if match:
                line = line[match.end():]  # Remove the matched part
            line = line[line.find(next(filter(str.isalpha, line))):]  # Extract from first alpha character onwards
            if line.lower() == "intro\n" or line.lower() == "intro":
                continue  # Skip lines that only contain "Intro" only
            qu_timestamps.append(timestamp)
            qu_lines.append(line)
    
    # Create the output file name with the current timestamp to avoid overwriting previous files
    # output_file_name = f"questions_from_description_with_timestamps_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # output_file_name = f"questions_from_description_with_timestamps.txt"
    
    # Write the formatted lines to the output file
    output_file_path = os.path.join(outputdir, f"{output_file_name}.txt")
    for i, line in enumerate(qu_timestamps):
        timestamp = qu_timestamps[i].ljust(8) + ": "  # Format timestamp
        formatted_lines.append(f"{timestamp}{qu_lines[i]}")
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n".join(formatted_lines))
    
    # Write the output file with current timestamps to 
    csv_data = formatted_lines_to_csv_data(qu_timestamps, qu_lines)
    output_csv_path = os.path.join(outputdir, f"{output_file_name}.csv")
    write_csv(csv_data, output_csv_path)

    return output_file_path  # Return the path of the created file for reference

def process_desc_in_subdirectories(input_path, output_file_name):
    all_desc_files = []

    if os.path.isdir(input_path):  # Directory processing
        for root, _, files in os.walk(input_path):
            all_desc_files.extend([os.path.join(root, file) for file in files if file == "description.txt"])
    
    elif os.path.isfile(input_path):  # File processing
        if os.path.basename(input_path) == "description.txt":
            all_desc_files.append(input_path)
        
        else:  # File containing directory names
            with open(input_path, 'r') as f:
                directories = [line.strip() for line in f.readlines()]
                for directory in directories:
                    audio_file = os.path.join(directory, "description.txt")
                    if os.path.exists(audio_file):
                        all_desc_files.append(audio_file)

    for input_file in all_desc_files:
        # Function 1: Extract lines from description.txt files
        questions_section_from_description = extract_description_lines(input_file)

        # Function 2: Extract and aggregate questions
        questions_only_from_description_question_section = extract_questions(questions_section_from_description)

        # Function 3: Split questions into separate lines
        simple_questions_first_attempt = split_questions(questions_only_from_description_question_section)

        # Function 4: Write questions with timestamps to a file
        output_file_path = write_questions_with_timestamps(simple_questions_first_attempt, os.path.dirname(input_file), output_file_name)
        print("Created output file: ", output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract questions from description.txt files and write them to a file with timestamps.")
    # mutually exclusive group: --input and --config_file
    config_input_group = parser.add_mutually_exclusive_group()
    config_input_group.add_argument("--input", type=str, help="Input description.txt file (for testing).")
    config_input_group.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    parser.add_argument("--output_file", type=str, default="questions_from_description_with_timestamps", help="Name of the output file.")
    args = parser.parse_args()

    if not args.input:
        config_reader = ConfigReader(args.config_file)
        data_configs = config_reader.get_data_configs()
        data_dir = data_configs["youtube_downloaded_videos_dir"]
        # output_configs = config_reader.get_output_configs()
        # outputs_dir = output_configs["output_dir"]
        process_desc_in_subdirectories(data_dir, args.output_file)
        # process_desc_in_subdirectories(data_dir, args.output_file, outputs_dir)
    else:
        process_desc_in_subdirectories(args.input, args.output_file)
        # process_desc_in_subdirectories(args.input, args.output_file)