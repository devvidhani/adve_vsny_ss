# adve_vsny_ss: Advaita Vedanta : Vedanta Society of New York : Swami Sarvapriyananda Models

This is an attempt to build a workflow to build a finetuned Llama2 model for answering questions on the topic of Advaita Vedanta in the style of Swami Sarvapriyananda. [Swami Sarvapriyananda](https://en.wikipedia.org/wiki/Swami_Sarvapriyananda) is a Hindu monk belonging to the Ramakrishna Order. Since January 2017, he is the current resident Swami and head of the [Vedanta Society of New York](https://www.vedantany.org). He is a well-known speaker on Vedanta and his talks are extremely popular worldwide. Many of his talks delivered at various forums are available on YouTube. The first version of model creation code is based on [AskSwami Q&A | Swami Sarvapriyananda](https://www.youtube.com/playlist?list=PLDqahtm2vA70VohJ__IobJSOGFJ2SdaRO) Question and Answer sessions led by Swami Sarvapriyananda at the Vedanta Society of New York.

The main goal for building this workflow (and models as a result) is to ask a spiritual question to the model, that can then respond a well founded answer in the tone and style of Swami Sarvapriyananda (as is the case when a question is posed to Swami).

## Disclaimer
adve_vsny_ss models are not released publicly. One the code and workflow is released to help promote/build similar models for similar use cases. Demo outputs are being shared to inspire the potential of fine tuning for these use cases.

## Index
* [Sample Responses](#sample-responses-from-some-finetuned-models-of-different-sizes)
* [Updates](#updates)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage-python)
* [Workflow: Build your model](#workflow-build-your-model)
* [TODO](#todo)

## Sample Responses from some finetuned models of different sizes
#### Question: What are four yogas for self-realization?

> Swami: So, what is the path of spirituality?. The path of spirituality.. It's not just one way.. There are many ways.. But they all lead to that one goal which we have been talking about.. That's why I said earlier, there may be different paths but only one destination.. And those paths are called Yogas.. Four Yogas.. They are called Yogas because they bring you closer to God or Brahman.. Now, these four Yogas are very interesting and beautifully described in the Gita.. First one is Bhakti Yoga.. Devotion.. You love God.. You want to know more about God.. You feel an attraction towards God.. You pray to God.. You worship God.. This is Bhakti Yoga.. Second one is Jnana Yoga.. Knowledge.. You want to know who am I?. Who is this universe?. How did it come into being?. What is my relationship with the universe?. These questions will take me onwards to knowing myself as Brahman.. Third one is Raja Yoga.. Meditation.. When you sit down quietly and try to focus your mind upon something like OM or AUM or any other mantra or even silence itself.. Then you can go deeper and deeper within yourself.. Into the depths of your own consciousness.. And when you do so, you begin to experience higher states of consciousness.. And then finally, the fourth one is Karma Yoga.. Action.. Service.. Selfless service.. Not doing anything for yourself.. Doing things for others.. Helping people around you.. Serving them.. All of these are Yogas.. They are paths leading us to our true nature.. To realize ourselves as Satchidananda.. Existence, Consciousness, Bliss.. We are that already.. But we don't know it.. We think we are limited.. We are not aware of our full potential.. So we need to follow some methodology to become aware of our full potential.. That's why we call them Yogas.. They help us to become aware of our full potential.. Thank you."]

#### Question: What is benefit of Karma Yoga?
<details>
  <summary>
    Swami: So, what's the advantage of doing Karma Yoga? The first thing that comes t
  </summary>
  details
Swami: So, what's the advantage of doing Karma Yoga? The first thing that comes to mind is you get good karma and bad karma gets wiped out. That's not true. You don't have to do anything for your own good or bad karma. It will come automatically. Whatever you are going to experience in this life, whatever you deserve, you will get it. And if you think I am doing something good so I must be getting some reward. No, no, no. If you are doing something good then you are fulfilling your duty. You are supposed to do it. But if you are thinking I am doing something good so I must be getting some reward. Then you are actually doing harm because now you are trying to get a reward instead of just doing your duty. Now you are trying to earn merits. Earning merits means you want more than what you already deserve. So you are trying to earn more than what you deserve. This desire for merit is called punya bhanga. Punya means merit. Bhanga means breakage. So breaking through the merit. Merit itself is fine but wanting more than what you deserve is like greed. Greed for merit. Desire for merit. So when you start doing things with an eye on the reward, you are actually causing yourself harm. Because you are creating a new desire which was not there before. When you were doing your duty earlier, you didn't feel any need for a reward. Now you are feeling a need for a reward. So you are creating a new desire. New desires create new problems. Old desires continue to cause problems. So you are creating new problems for yourself. Not only that, you are also making your spiritual progress slower. Why? Because you are still stuck in the same desire. You wanted to do good deeds to get a reward. Instead of being free from all desires, you are still attached to one particular desire. So you are still stuck in samsara. Samsara means the cycle of birth and death. In Hinduism we call it samsara chakra. Cycle of birth and death. We keep coming back into this world again and again until we become enlightened. Until we realize our real nature as Brahman. Until we realize that I am not this body-mind personality. I am pure consciousness. Until we realize that, we are trapped in samsara. Trapped in the cycle of birth and death. So you are still trapped in samsara. Stuck in samsara. How can you escape from samsara? By becoming free from all desires. All desires mean all desires. Good desires, bad desires, neutral desires. All desires. Becoming free from all desires. Freedom from all desires is called nirvana. Nirvana means freedom from all desires. So how does one achieve freedom from all desires? One way is by practicing non-dual meditation. Dhyanam. Turiyatita. Non-dual meditation. Another way is by performing selfless service. Karma Yoga. Selfless action. Service done without expectation of reward. Without attachment to result. Without attachment to outcome. Just doing it for its own sake. For the joy of serving others. Serving others without expecting anything in return. Even if you expect nothing in return, still you are benefiting. You are helping others. Helping others benefits you too. Benefitting others helps you too. So you are benefited twice over. Once by helping others and once by experiencing the joy of helping others. Joy of helping others. There is another level beyond that. Beyond the joy of helping others. There is the bliss of helping others. Ananda of helping others. Bliss of helping others. That's the highest level of Karma Yoga. Where you are not even interested in the joy of helping others. You are not even interested in the satisfaction of having helped others. You are not even interested in the happiness of helping others. You are simply interested in helping others. Period. End of story. Nothing else matters. Only helping others. That's the ultimate goal of Karma Yoga. Ultimately, why should you help others? Because they are suffering. They are unhappy. They are struggling. They are crying. They are hungry. They are thirsty. They are sick. They are dying. Somebody has to help them. You are here. You are capable of helping them. You are responsible for helping them. You owe it to them. You cannot say I am not interested in helping others. I am not interested in their sufferings. I am not interested in their pain.
</details>
-  Truncated the response as then repititions started...

#### Question - what is purpose of my life.
```Swami: The first thing that comes up in your mind when you ask this question, what's the purpose of my life? I want to be happy and fulfilled. That's why we are here. We all have come into existence for one reason only, to experience happiness. And how do we experience happiness? By realizing our true nature as pure consciousness or Brahman. So that's the ultimate goal. But then immediately another question arises. How does one realize one's true nature? You need some methodology. Some technique. There are many techniques available. One can meditate. One can pray. One can study spiritual texts like the Bhagavad Gita, the Upanishads, the Talmud, whatever. All these things will help us on our journey towards self-realization. Now, let me tell you something about this journey. It's not just one way. It's two ways. Firstly, there is an inner journey which takes place within yourself where you try to understand who am I really? Am I this body? No. Then what am I? Mind? No. Then what am I? Thoughts? No. Then what am I? Emotions? No. Then what am I? Consciousness itself? Yes! That's the final destination. When you say I am consciousness, I am awareness, I am aware of everything around me. This is called self-enquiry. Self-enquiry means asking oneself questions until you reach the truth. Until you find out who you truly are. And once you know that, you become enlightened. Once you know that, you become God realized. Once you know that, you become liberated from suffering. Once you know that, you become free forever. But remember, this journey has to be done with love. Love for yourself. Love for others. Love for God. Love for everybody. If you don't have love, you won't go far along this path. Secondly, there is also an outer journey. An outer journey. What is that? Service to humanity. Serving people. Helping them. Loving them. Caring for them. Feeding them. Clothing them. Educating them. Healing them. Protecting them. These are all examples of service to humanity. Not because you expect anything in return but simply because they exist. They deserve it. And if you serve them, you will feel good. You will feel happy. You will feel fulfilled. And ultimately, you will see God everywhere. In every person, in every creature,```

-  Truncated the response as then repititions started...

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
- [ ] Repition of text towards the end
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