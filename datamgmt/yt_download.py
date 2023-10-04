from pytube import YouTube, exceptions, Channel
import os
import hashlib
import argparse
from utilities.config_support import ConfigReader

# Define a function to read URLs from a file
def read_urls(urlfilepath):
    # Open the file in read mode
    with open(urlfilepath, "r") as f:
        # Read the file line by line and store the URLs in a list
        urls = f.read().splitlines()
    # Return the list of URLs
    return urls

def save_audio(url, path):
    # mp4URL = "https://www.youtube.com/watch?v=21X5lGlDOfg"
    # try:
    #     yt = YouTube(url)
    #     yt.streams.filter(only_audio=True).first().download(path)
    try:
        yt = YouTube(url)
        stream = yt.streams.filter(only_audio=True).first()
        if stream is None:
            raise Exception(f"No audio stream found for {url}")
        stream.download(path)
        title = yt.title
        with open(os.path.join(path, "title.txt"), "w") as f:
            f.write(title)
        description = yt.description
        with open(os.path.join(path, "description.txt"), "w") as f:
            f.write(description)
        # Get the video length in seconds
        length = yt.length
        with open(os.path.join(path, "length.txt"), "w") as f:
            f.write(str(length))
        # Save the URL in the path directory
        with open(os.path.join(path, "url.txt"), "w") as f:
            f.write(url)

    except exceptions.VideoUnavailable:
        print(f'Video {url} is unavailable, skipping.')
    except exceptions.LiveStreamError:
        print(f'Video {url} is a live stream, skipping.')
    except exceptions.VideoPrivate:
        print(f'Video {url} is a private video, skipping.')
    except exceptions.PytubeError as e:
        print(f'An unknown error occurred: {e}')
    except Exception as e:
        print(f'An custom error occurred: {e}')

def main(args):
    config_reader = ConfigReader(args.config_file)
    # config download-ytdata-path
    data_configs = config_reader.get_data_configs()
    input_configs = config_reader.get_input_configs()
    youtube_downloaded_videos_dir = data_configs['youtube_downloaded_videos_dir']
    # config youtube_video_urls
    youtube_video_urls_dir = input_configs['youtube_video_urls_dir']

    args = parser.parse_args()
    input_files_with_urls = args.files_with_urls
    urls_files_path_name = os.path.join(youtube_video_urls_dir, input_files_with_urls)

    # Call the read_urls function and get the list of URLs

    if input_files_with_urls.startswith("https://www.youtube.com"):
        # If it is a channel URL, create a channel object
        c = Channel(input_files_with_urls)
        # Print the number of videos and the first video URL as an example
        print(f"There are {len(c.video_urls)} videos in the channel.")
        print(f"The first video URL is: {c.video_urls[0]}")
        
        # Loop through the video URLs and call the save_audio function for each one
        # An issue with the pytube library channels to video_urls enumeration
        # https://github.com/pytube/pytube/issues/1408 suggests using the following
        # command pip install git+https://github.com/pishiko/pytube.git@42a7d8322dd7749a9e950baf6860d115bbeaedfc
        # another site https://stackoverflow.com/questions/74334535/pytube-channel-video-urls-is-does-not-working
        # suggests following:- pip install git+https://github.com/24makee/pytube.git@c709202d4f2c0d36d9484314d44fd26744225b7d
        # (The above command worked for me)
        # Or even consider from their site (but unsure if all bug fixes are merged in)
        # python -m pip install git+https://github.com/pytube/pytube

        for url in c.video_urls:
            print(f"Downloading URL : {url}")
            # Generate a hash from the URL
            hash = hashlib.md5(url.encode()).hexdigest()
            # Create a subdirectory for each URL using its hash
            sub_path = os.path.join(youtube_downloaded_videos_dir, hash)
            if os.path.isdir(sub_path):
                print('  skipping...')
                # Skip the current iteration if the directory exists
                continue
            os.makedirs(sub_path, exist_ok=True)
            save_audio(url, sub_path)
            
    # Check if the input is a file or a channel URL
    elif os.path.isfile(urls_files_path_name):
        urls = read_urls(urls_files_path_name)
        # Print the number of URLs and the first URL as an example
        print(f"There are {len(urls)} URLs in the file.")
        print(f"The first URL is: {urls[0]}")
        
        # Loop through the URLs and call the save_audio function for each one
        for url in urls:
            print(f"Downloading URL : {url}")
            # Generate a hash from the URL
            hash = hashlib.md5(url.encode()).hexdigest()
            # Create a subdirectory for each URL using its hash
            sub_path = os.path.join(youtube_downloaded_videos_dir, hash)
            if os.path.isdir(sub_path):
                print('  skipping...')
                # Skip the current iteration if the directory exists
                continue
            os.makedirs(sub_path, exist_ok=True)
            save_audio(url, sub_path)

    else:
        # If it is neither a file nor a channel URL, print an error message
        print(f"Invalid input: {input_files_with_urls}")
        print(f"Please provide either a file path with youtube URS in it or a channel URL.")


# Call the main function from __main__
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/config.yaml", help="Path to the config yaml file")
    parser.add_argument('--files_with_urls', required=True, type=str, help='URL to test, or, file containing URLs')
    args = parser.parse_args()

    main(args)
