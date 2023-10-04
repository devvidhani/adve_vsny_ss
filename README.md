# adve_vsny_ss: Advaita Vedanta : Vedanta Society of New York : Swami Sarvapriyananda Models

This is an attempt to build a workflow to build a finetuned Llama2 model for answering questions on the topic of Advaita Vedanta in the style of Swami Sarvapriyananda. [Swami Sarvapriyananda](https://en.wikipedia.org/wiki/Swami_Sarvapriyananda) is a Hindu monk belonging to the Ramakrishna Order. Since January 2017, he is the current resident Swami and head of the [Vedanta Society of New York](https://www.vedantany.org). He is a well-known speaker on Vedanta and his talks are extremely popular worldwide. Many of his talks delivered at various forums are available on YouTube. The first version of model creation code is based on [AskSwami Q&A | Swami Sarvapriyananda](https://www.youtube.com/playlist?list=PLDqahtm2vA70VohJ__IobJSOGFJ2SdaRO) Question and Answer sessions led by Swami Sarvapriyananda at the Vedanta Society of New York.

The main goal for building this workflow (and models as a result) is to ask a spiritual question to the model, that can then respond a well founded answer in the tone and style of Swami Sarvapriyananda (as is the case when a question is posed to Swami).

## Disclaimer
adve_vsny_ss models are not released publicly. One the code and workflow is released to help promote/build similar models for similar use cases. Demo outputs are being shared to inspire the potential of fine tuning for these use cases.

## Index
* [Sample Responses](#-Samples)
* [Updates](#-updates)
* [Features](#-features)
* [Installation](#-installation)
* [Usage](#-usage)
* [Workflow: Build your model](#-Workflow)
* [TODO](#-todo)

## Sample Responses from some finetuned models of different sizes

## Updates
**2023.10.04**
- First version quick notes
  - Based off AskSwami QA videos format only
  - Scalable download and transcription of YouTube videos (and metadata) list, creation of question-answer training dataset for LLMs, creation and testing of finetuned models
  

## Features
- Automatic downloads a set of YouTube AskSwami QA videos using their urls (and corresponding video descriptions) using [PyTube](https://github.com/pytube/pytube) - a lightweight, dependency-free Python library (and command-line utility) for downloading YouTube videos. There were some newer versions that needed to be installed in conda enviroment to help work with newer YouTube API.
  - Parallel transcription of videos with [WhisperX](https://github.com/m-bain/whisperX) - a fast automatic speech recognition with word-level timestamps and multi-speaker diarization.
  - Creation of a Question-Answer dataset through a series of video description and video transcriptions transformations
  - Create finetuned [Llama-2-7b-chat-hf](https://huggingface.co/blog/llama2#demo) models. Note that to download and use Meta's Llama models, both HuggingFace and Meta's forms need to be filled out. Users are provided access to the repository once both forms are filled after few hours. The model size (7B, 13B or 70B) that can be finetuned depends upon the GPU power, quantization techniques, etc. With permissions from YouTube video owners, one can relesae the dataset on public forums (HuggingFace, etc.)
  - Deploy and query finetuned models on/using one or multiple platforms/frameworks (E.g., [HuggingFace](https://huggingface.co/models), [Oobabooga](https://github.com/oobabooga/text-generation-webui))

## Installation

These installation and usage instructions have been tested on a dual-4090 AMD-workstation (powered by 5975wx CPU)

#### Setup conda environment
```bash
conda create -n your_env_name python=3.10
conda activate your_env_name
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Download and setup adve_vsny_ss repository
```bash
git clone git@github.com:devvidhani/adve_vsny_ss.git
```
#### Install requirements.txt. Note that it is superflous, but complete for entire workflow (pytube+patches, whisperx, ffmpeg, trl, etc.)
```bash
pip install -r requirements.txt
```
#### Setup input, data, output and test directories in adve_vsny_ss directories
```bash
cd adve_vsny_ss
mkdir -p ./inputdir/ ./datadir/download_videos/ ./datadir/intermediate_data/ ./outputdir/outputs/
```

#### Setup configs
  - setup configs/config.yaml file
```bash
inputs:
  youtube_video_urls_dir: ./inputdir/

data:
  youtube_downloaded_videos_dir: ./datadir/download_videos/
  inprocess_data_dir: ./datadir/intermediate_data/

outputs:
  output_dir: ./outputdir/outputs/
```
  - setup configs/env.yaml file
```bash
hf_token: hf_........
conda_env: your_env_name
conda_activate_path: <path_to_miniconda3>/bin/activate
```

#### Models use setup
- Fill Meta Llama2 downlaod and usage forms on HuggingFace and Meta's website to download HuggingFace models
- Set up HuggingFace [UAT (User Access Token)](https://huggingface.co/settings/tokens)

## Usage (Python)
The finetuned model can be passed the list of questions that would be answered sequentially by the model uploaded. The python script usage is described below in workflow section's last step.
## Workflow: Build your model

The adve_vsny_ss model(s) are finetuned Meta's Llama2 chat model based off HuggingFace's versions of Llama2.

1. Follow installation instructions to setup the environment

2. Set environment var
```bash
export PYTHONPATH=.:$PYTHONPATH
```

4. Download Youtube Videos data and metadata
  - in configs/config.yaml inputdir, create a file which contains URLs of Youtube to be downloaded (yturls.txt)
```bash
python ./datamgmt/yt_download.py --files_with_urls yturls.txt
```

5. Convert Youtube mp4 files to mp3 and extract audio using whisperx (genreates audio.vtt) in parallel on all GPUs
```bash
python ./datamgmt/run_parallel_whisperx_on_all_gpus.py
```

6. Read each audio.vtt above and consolidate continuous same speaker's timestamps into a single timestamp for that speaker
```bash
python ./datamgmt/consolidate_audiovtt_speakers.vtt
```

7. Extract questions from description.txt downloaded along with youtube videos. Luckily those description files have questions with timestamps. despite a complex flow, one (of 39) video's question extraction didnt work. There are some heuristics - e.g., QA begins in sentence after the work timestamp (or time-stamp); multi-line questions are collated, questions with parts with numbering like (a), (b), etc. are separated with same timestamp
   These are the fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
```bash
python ./datamgmt/extract_questions_timestamps_from_description.py
```

8. Generate CSV of transcripts from consolidate_audiovtt_speakers with these fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
```bash
python ./datamgmt/extract_speakers_transcripts_timestamps_from_vtt.py
```

9. For now, merge questions/answers around main speakers' around the timestamp from description question file
```bash
python ./datamgmt/create_question_answer_dataset.py
```

10. Create finetuned model
- Sample model creation for 100 steps using special tokenizer padding "<PAD>"
```bash
python ./train/SFT_wrapper.py --output_dir ./outputdir/avde_fintun_llama2/100_steps_PAD --hf_token <your_hf_token> --max_steps 100
```

11. Execute finetuned model
- Sample execution
```bash
python ./inference/model_inference.py --model_to_load avde_hf_llama2_13b_chat_100steps_PAD --modelinferencingclass CreateTextResponses
```
- Different inferencing procedures can be creating using same framework for model_inference.py. The framework can be extended by defining input data format, input data processing, exact input inferencing action, output data processing, etc.

## TODO
- [x] Initial README.md writeup
- [ ] Fix bug with ./inference/model_inference.py. As per current testing, the private version of code is producing results fine, but final finetuned version needs further testing.
- [ ] Cleanup configs for SFT_wrapper.py and model_inference.py
- [ ] Refine questions/answers without manual intervention
  - Based on improved alignment from QA dataset
  - Fix (quite rare) speaker diarization issue
  - Drop irrelevant text (E.g., "next question please", "come forward and speak your name", etc.)
- [ ] Manage individual row length
  - Long answers refinement when multiple questions answered in a single long response
- [ ] Voice and tone cloning to hear responses in original voice
  - Current attempts using [Bark](https://github.com/serp-ai/bark-with-voice-clone) are limited to short audio generation with reasonable but not complete success.
- [ ] Summarize responses using existing models to answer in generic styles
- [ ] Finetune on more generic YouTube videos (lectures, QA at end of lectures)
  - Knowledge base SFT