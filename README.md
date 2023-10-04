# A simple extensible workflow for creating LLM training dataset and finetuning LLMs from YouTube QA style videos

This is an attempt to build a workflow to build a finetuned LLM ([Meta's Llama2](https://ai.meta.com/llama/) currently) model from answering questions on a specific topic based on YouTube question/answers videos on that topic. There are four parts to generating/testing model

1. Bulk download and bulk transcription using Audio-To-Text transformation
2. Transcription transformation to generate training LLM training dataset
3. Finetune on specific base LLM
4. Deploy finetuned model

The current release's workflow's data management (tranformations) is built for a specific use case. It would be simplified for generic plugins in next releases

#### Testing the workflow

_To validate the workflow, it is currently tested for answering questions on the topic of Advaita Vedanta in the style of [Swami Sarvapriyananda](https://en.wikipedia.org/wiki/Swami_Sarvapriyananda), a Hindu monk belonging to the Ramakrishna Order, and a well-known speaker on Vedanta and his talks are extremely popular worldwide. Many of his talks delivered at various forums are available on YouTube. The first version of model creation code is based on [AskSwami Q&A | Swami Sarvapriyananda](https://www.youtube.com/playlist?list=PLDqahtm2vA70VohJ__IobJSOGFJ2SdaRO) Question and Answer sessions led by Swami Sarvapriyananda at the Vedanta Society of New York._

_The main goal for this testing/building this workflow (and models as a result) is to ask a spiritual question to the model, that can then respond a well founded answer in the tone and style of Swami Sarvapriyananda (as is the case when a question is posed to Swami)._

## Disclaimer
adve_vsny_ss models are not released publicly. Only the code and workflow is released to help promote/build similar models for similar use cases. Demo outputs are being shared to inspire the potential of fine tuning for these use cases.

## Index
* [Sample Responses](#sample-responses-from-some-finetuned-models-of-different-sizes)
* [Updates](#updates)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage-python)
* [Workflow: Build your model](#workflow-build-your-model)
* [TODO](#todo)

## Sample Responses from some finetuned models of different sizes

<details>
  <summary>
    Sample Question 1 and Model's Answer
  </summary>
Question: What are four yogas for self-realization? 
Swami: so, what is the path of spirituality?. the path of spirituality.. it's not just one way.. there are many ways.. but they all lead to that one goal which we have been talking about.. that's why i said earlier, there may be different paths but only one destination.. and those paths are called yogas.. four yogas.. they are called yogas because they bring you closer to god or brahman.. now, these four yogas are very interesting and beautifully described in the gita.. first one is bhakti yoga.. devotion.. you love god.. you want to know more about god.. you feel an attraction towards god.. you pray to god.. you worship god.. this is bhakti yoga.. second one is jnana yoga.. knowledge.. you want to know who am i?. who is this universe?. how did it come into being?. what is my relationship with the universe?. these questions will take me onwards to knowing myself as brahman.. third one is raja yoga.. meditation.. when you sit down quietly and try to focus your mind upon something like om or aum or any other mantra or even silence itself.. then you can go deeper and deeper within yourself.. into the depths of your own consciousness.. and when you do so, you begin to experience higher states of consciousness.. and then finally, the fourth one is karma yoga.. action.. service.. selfless service.. not doing anything for yourself.. doing things for others.. helping people around you.. serving them.. all of these are yogas.. they are paths leading us to our true nature.. to realize ourselves as satchidananda.. existence, consciousness, bliss.. we are that already.. but we don't know it.. we think we are limited.. we are not aware of our full potential.. so we need to follow some methodology to become aware of our full potential.. that's why we call them yogas.. they help us to become aware of our full potential.. thank you."
</details>

<details>
  <summary>
    Sample Question 2 and Model's Answer
  </summary>
Question: what is benefit of karma yoga?
Swami: so, what's the advantage of doing karma yoga? the first thing that comes to mind is you get good karma and bad karma gets wiped out. that's not true. you don't have to do anything for your own good or bad karma. it will come automatically. whatever you are going to experience in this life, whatever you deserve, you will get it. and if you think i am doing something good so i must be getting some reward. no, no, no. if you are doing something good then you are fulfilling your duty. you are supposed to do it. but if you are thinking i am doing something good so i must be getting some reward. then you are actually doing harm because now you are trying to get a reward instead of just doing your duty. now you are trying to earn merits. earning merits means you want more than what you already deserve. so you are trying to earn more than what you deserve. this desire for merit is called punya bhanga. punya means merit. bhanga means breakage. so breaking through the merit. merit itself is fine but wanting more than what you deserve is like greed. greed for merit. desire for merit. so when you start doing things with an eye on the reward, you are actually causing yourself harm. because you are creating a new desire which was not there before. when you were doing your duty earlier, you didn't feel any need for a reward. now you are feeling a need for a reward. so you are creating a new desire. new desires create new problems. old desires continue to cause problems. so you are creating new problems for yourself. not only that, you are also making your spiritual progress slower. why? because you are still stuck in the same desire. you wanted to do good deeds to get a reward. instead of being free from all desires, you are still attached to one particular desire. so you are still stuck in samsara. samsara means the cycle of birth and death. in hinduism we call it samsara chakra. cycle of birth and death. we keep coming back into this world again and again until we become enlightened. until we realize our real nature as brahman. until we realize that i am not this body-mind personality. i am pure consciousness. until we realize that, we are trapped in samsara. trapped in the cycle of birth and death. so you are still trapped in samsara. stuck in samsara. how can you escape from samsara? by becoming free from all desires. all desires mean all desires. good desires, bad desires, neutral desires. all desires. becoming free from all desires. freedom from all desires is called nirvana. nirvana means freedom from all desires. so how does one achieve freedom from all desires? one way is by practicing non-dual meditation. dhyanam. turiyatita. non-dual meditation. another way is by performing selfless service. karma yoga. selfless action. service done without expectation of reward. without attachment to result. without attachment to outcome. just doing it for its own sake. for the joy of serving others. serving others without expecting anything in return. even if you expect nothing in return, still you are benefiting. you are helping others. helping others benefits you too. benefitting others helps you too. so you are benefited twice over. once by helping others and once by experiencing the joy of helping others. joy of helping others. there is another level beyond that. beyond the joy of helping others. there is the bliss of helping others. ananda of helping others. bliss of helping others. that's the highest level of karma yoga. where you are not even interested in the joy of helping others. you are not even interested in the satisfaction of having helped others. you are not even interested in the happiness of helping others. you are simply interested in helping others. period. end of story. nothing else matters. only helping others. that's the ultimate goal of karma yoga. ultimately, why should you help others? because they are suffering. they are unhappy. they are struggling. they are crying. they are hungry. they are thirsty. they are sick. they are dying. somebody has to help them. you are here. you are capable of helping them. you are responsible for helping them. you owe it to them. you cannot say i am not interested in helping others. i am not interested in their sufferings. i am not interested in their pain.
</details>
>Answer manually truncated at the end as the model phrases repititions started

<details>
  <summary>
    Sample Question 3 and Model's Answer
  </summary>
Question - what is purpose of my life.
Swami: the first thing that comes up in your mind when you ask this question, what's the purpose of my life? i want to be happy and fulfilled. that's why we are here. we all have come into existence for one reason only, to experience happiness. and how do we experience happiness? by realizing our true nature as pure consciousness or brahman. so that's the ultimate goal. but then immediately another question arises. how does one realize one's true nature? you need some methodology. some technique. there are many techniques available. one can meditate. one can pray. one can study spiritual texts like the bhagavad gita, the upanishads, the talmud, whatever. all these things will help us on our journey towards self-realization. now, let me tell you something about this journey. it's not just one way. it's two ways. firstly, there is an inner journey which takes place within yourself where you try to understand who am i really? am i this body? no. then what am i? mind? no. then what am i? thoughts? no. then what am i? emotions? no. then what am i? consciousness itself? yes! that's the final destination. when you say i am consciousness, i am awareness, i am aware of everything around me. this is called self-enquiry. self-enquiry means asking oneself questions until you reach the truth. until you find out who you truly are. and once you know that, you become enlightened. once you know that, you become god realized. once you know that, you become liberated from suffering. once you know that, you become free forever. but remember, this journey has to be done with love. love for yourself. love for others. love for god. love for everybody. if you don't have love, you won't go far along this path. secondly, there is also an outer journey. an outer journey. what is that? service to humanity. serving people. helping them. loving them. caring for them. feeding them. clothing them. educating them. healing them. protecting them. these are all examples of service to humanity. not because you expect anything in return but simply because they exist. they deserve it. and if you serve them, you will feel good. you will feel happy. you will feel fulfilled. and ultimately, you will see god everywhere. in every person, in every creature,```
</details>
>Answer manually truncated at the end as the model phrases repititions started

## Updates
**2023.10.04**
- First release quick notes
  - Scalable download and transcription of youtube videos (and metadata) list, creation of question-answer training dataset for llms, creation and testing of finetuned models
    - Testing based off askswami qa videos format only
  

## Features
- Automatic downloads a set of youtube askswami qa videos using their urls (and corresponding video descriptions) using [pytube](https://github.com/pytube/pytube) - a lightweight, dependency-free python library (and command-line utility) for downloading youtube videos. there were some newer versions that needed to be installed in conda enviroment to help work with newer youtube api.
  - Parallel transcription of videos with [whisperx](https://github.com/m-bain/whisperx) - a fast automatic speech recognition with word-level timestamps and multi-speaker diarization.
  - Creation of a question-answer dataset through a series of video description and video transcriptions transformations
  - Create finetuned [llama-2-7b-chat-hf](https://huggingface.co/blog/llama2#demo) models. note that to download and use meta's llama models, both huggingface and meta's forms need to be filled out. users are provided access to the repository once both forms are filled after few hours. the model size (7b, 13b or 70b) that can be finetuned depends upon the gpu power, quantization techniques, etc. with permissions from youtube video owners, one can relesae the dataset on public forums (huggingface, etc.)
  - Deploy and query finetuned models on/using one or multiple platforms/frameworks (e.g., [huggingface](https://huggingface.co/models), [oobabooga](https://github.com/oobabooga/text-generation-webui))

## Installation

These installation and usage instructions have been tested on a dual-4090 amd-workstation (powered by 5975wx cpu)

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
#### Install requirements.txt. note that it is superflous, but complete for entire workflow (pytube+patches, whisperx, ffmpeg, trl, etc.)
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
- Fill meta llama2 downlaod and usage forms on huggingface and meta's website to download huggingface models
- Set up huggingface [uat (user access token)](https://huggingface.co/settings/tokens)

## Usage (python)
The finetuned model can be passed the list of questions that would be answered sequentially by the model uploaded. the python script usage is described below in workflow section's last step.
## Workflow: build your model

The adve_vsny_ss model(s) are finetuned meta's llama2 chat model based off huggingface's versions of llama2.

1. Follow installation instructions to setup the environment

2. Set environment var
```bash
export pythonpath=.:$pythonpath
```

3. Download youtube videos data and metadata
  - in configs/config.yaml inputdir, create a file which contains urls of youtube to be downloaded (yturls.txt)
```bash
python ./datamgmt/yt_download.py --files_with_urls yturls.txt
```

4. Convert youtube mp4 files to mp3 and extract audio using whisperx (genreates audio.vtt) in parallel on all gpus
```bash
python ./datamgmt/run_parallel_whisperx_on_all_gpus.py
```

5. Read each audio.vtt above and consolidate continuous same speaker's timestamps into a single timestamp for that speaker
```bash
python ./datamgmt/consolidate_audiovtt_speakers.vtt
```

6. Extract questions from description.txt downloaded along with youtube videos. luckily those description files have questions with timestamps. despite a complex flow, one (of 39) video's question extraction didnt work. there are some heuristics - e.g., qa begins in sentence after the work timestamp (or time-stamp); multi-line questions are collated, questions with parts with numbering like (a), (b), etc. are separated with same timestamp
   these are the fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
```bash
python ./datamgmt/extract_questions_timestamps_from_description.py
```

7. Generate csv of transcripts from consolidate_audiovtt_speakers with these fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
```bash
python ./datamgmt/extract_speakers_transcripts_timestamps_from_vtt.py
```

8. For now, merge questions/answers around main speakers' around the timestamp from description question file
```bash
python ./datamgmt/create_question_answer_dataset.py
```

9. Create finetuned model
- Sample model creation for 100 steps using special tokenizer padding "<pad>"
```bash
python ./train/sft_wrapper.py --output_dir ./outputdir/avde_fintun_llama2/100_steps_pad --hf_token <your_hf_token> --max_steps 100
```

10. Execute finetuned model
- Sample execution
```bash
python ./inference/model_inference.py --model_to_load avde_hf_llama2_13b_chat_100steps_pad --modelinferencingclass createtextresponses
```
- Different inferencing procedures can be creating using same framework for model_inference.py. the framework can be extended by defining input data format, input data processing, exact input inferencing action, output data processing, etc.

## Todo
- [x] Initial readme.md writeup
- [ ] Pluggable data transformations
- [ ] Fix repietition of model's response towards the end
- [ ] Fix bug with ./inference/model_inference.py. as per current testing, the private version of code is producing results fine, but final finetuned version needs further testing.
- [ ] Cleanup configs for sft_wrapper.py and model_inference.py
- [ ] Refine questions/answers without manual intervention
  - Based on improved alignment from qa dataset
  - Fix (quite rare) speaker diarization issue
  - Drop irrelevant text (e.g., "next question please", "come forward and speak your name", etc.)
- [ ] Manage individual row length
  - Long answers refinement when multiple questions answered in a single long response
- [ ] Voice and tone cloning to hear responses in original voice
  - Current attempts using [bark](https://github.com/serp-ai/bark-with-voice-clone) are limited to short audio generation with reasonable but not complete success.
- [ ] Summarize responses using existing models to answer in generic styles
- [ ] Finetune on more generic youtube videos (lectures, qa at end of lectures)
  - Knowledge base SFT (supervised fine tuning)