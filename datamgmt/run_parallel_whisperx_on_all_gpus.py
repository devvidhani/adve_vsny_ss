import os
import subprocess
import argparse
from pytube import YouTube
from utilities.config_support import ConfigReader
from concurrent.futures import ProcessPoolExecutor

def get_n_gpus():
    try:
        result = subprocess.check_output("nvidia-smi -L | wc -l", shell=True)
        return int(result.strip())
    except Exception as e:
        print(f"Error getting GPU count: {str(e)}")
        return 0

N_GPUS = get_n_gpus()

def convert_mp4_to_mp3_if_needed(mp4_file):
    """Convert MP4 to MP3 if the MP3 file doesn't exist."""
    mp3_file = os.path.join(os.path.dirname(mp4_file), "audio.mp3")
    if not os.path.exists(mp3_file):
        try:
            subprocess.run(["ffmpeg", "-i", mp4_file, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", mp3_file], check=True)
            print(f"Converted {mp4_file} to {mp3_file}")
        except subprocess.CalledProcessError:
            print(f"ffmpeg failed for {mp4_file}")
        except Exception as e:
            print(f"Error during conversion for {mp4_file}: {str(e)}")

def run_whisperx(mp3_file, output_dir, gpu_id, hf_token, conda_env_name, conda_activate_path):
    try:
        # script_path = os.path.abspath("./datamgmt/run_whisperx.sh")
        command = [
            "./datamgmt/run_whisperx.sh",
            "--env", conda_env_name,
            "--path", conda_activate_path,
            "whisperx",
            "--model", "large-v2",
            "--diarize",
            "--batch_size", "1",
            "--min_speakers", "1",
            "--max_speakers", "10",
            "--output_format", "vtt",
            "--hf_token", hf_token,
            mp3_file,
            "--output_dir", output_dir
        ]

        working_directory = "."

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print("Command being called:", command)
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=working_directory, check=True)
        # result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, shell=True, cwd=working_directory, check=True)

        print(result.stdout)
        print(result.stderr)
        print("Done")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}:")
        print(e.output)
    except Exception as e:
        print(f"Error running whisperx: {str(e)}")

def process_mp4(mp4_file, mp3_conversion_only, hf_token, conda_env_name, conda_activate_path):
    """Convert MP4 to MP3 and optionally run whisperx."""
    convert_mp4_to_mp3_if_needed(mp4_file)
    if not mp3_conversion_only:
        mp3_file = os.path.join(os.path.dirname(mp4_file), "audio.mp3")
        gpu_id = (process_mp4.call_count % N_GPUS)
        run_whisperx(mp3_file, os.path.dirname(mp3_file), gpu_id, hf_token, conda_env_name, conda_activate_path)
        process_mp4.call_count += 1

process_mp4.call_count = 0

def process_input(input_path, mp3_conversion_only, hf_token, conda_env_name, conda_activate_path):
    """Process a directory or individual file based on provided input path."""
    all_mp4_files = []
    all_mp3_files = []

    # Collecting all the .mp4 and .mp3 files
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            all_mp4_files.extend([os.path.join(root, file) for file in files if file.lower().endswith(".mp4")])
            all_mp3_files.extend([os.path.join(root, file) for file in files if file.lower().endswith(".mp3")])
    else:
        if input_path.lower().endswith(".mp4"):
            all_mp4_files.append(input_path)
        elif input_path.lower().endswith(".mp3"):
            all_mp3_files.append(input_path)
        else:
            print("Invalid input. Please provide a valid file or directory.")
            return

    # Convert all the .mp4 files to .mp3 files in batches
    for i in range(0, len(all_mp4_files), 40):
        batch_mp4_files = all_mp4_files[i:i+40]
        with ProcessPoolExecutor(max_workers=40) as executor:
            futures = {executor.submit(process_mp4, mp4_file, mp3_conversion_only, hf_token, conda_env_name, conda_activate_path): mp4_file for mp4_file in batch_mp4_files}
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {futures[future]}: {str(e)}")

    # If we don't want to process with whisperx, we can return now
    if mp3_conversion_only:
        return

    # Process .mp3 files with whisperx in batches
    for i in range(0, len(all_mp3_files), 5 * N_GPUS):
        batch_mp3_files = all_mp3_files[i:i+5*N_GPUS]
        with ProcessPoolExecutor(max_workers=5 * N_GPUS) as executor:
            futures = {executor.submit(process_mp4, mp4_file, mp3_conversion_only, hf_token, conda_env_name, conda_activate_path): mp4_file for mp4_file in batch_mp4_files}
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing {futures[future]}: {str(e)}")

process_input.call_count = 0

parser = argparse.ArgumentParser(description="Process mp4 or mp3 files with whisperx.")

# mutually exclusive group: --input and --config_file
config_input_group = parser.add_mutually_exclusive_group()
config_input_group.add_argument("--input", type=str, help="Input directory or mp4/mp3 file.")
config_input_group.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")

parser.add_argument("--env_file", type=str, default="configs/env.yaml", help="Path to the token yaml file")
parser.add_argument("--mp3_conversion_only", action="store_true", help="Perform MP3 conversion only, without running WhisperX.")
args = parser.parse_args()

env_configs = ConfigReader(args.env_file).config
hf_token = env_configs["hf_token"]
conda_env_name = env_configs["conda_env"]
conda_activate_path = env_configs["conda_activate_path"]
if not args.input:
    config_reader = ConfigReader(args.config_file)
    data_configs = config_reader.get_data_configs()
    data_dir = data_configs["youtube_downloaded_videos_dir"]
    process_input(data_dir, args.mp3_conversion_only, hf_token, conda_env_name, conda_activate_path)
else:
    process_input(args.input, args.mp3_conversion_only, hf_token, conda_env_name, conda_activate_path)
