# adve_vsny_ss

1. Create directories called inputdir, outputdir, datadir and testdir

2. populate configs/config.yaml and configs/env.yaml

3. Set environment var
  - export PYTHONPATH=.:$PYTHONPATH

4. Download Youtube Videos data and metadata
  - in inputdir, create a file which contains URLs of Youtube to be downloaded (yturls.txt)
  - python ./datamgmt/yt_download.py --files_with_urls yturls.txt

5. Convert Youtube mp4 files to mp3 and extract audio using whisperx (genreates audio.vtt)
  - python ./datamgmt/run_parallel_whisperx_on_all_gpus.py

6. Read audio.vtt above and consolidate continuous same speaker's timestamps into a single timestamp for that speaker
  - python ./datamgmt/consolidate_audiovtt_speakers.vtt

7. Extract questions from description.txt downloaded along with youtube videos. Luckily those description files have questions with timestamps. despite a complex flow, one (of 39) video's question extraction didnt work. There are some heuristics - e.g., QA begins in sentence after the work timestamp (or time-stamp); multi-line questions are collated, questions with parts with numbering like (a), (b), etc. are separated with same timestamp
   These are the fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
  - python ./datamgmt/extract_questions_timestamps_from_description.py

8. Generate CSV of transcripts from consolidate_audiovtt_speakers with these fields ['starttime', 'endtime', 'starttime_sec', 'endtime_sec', 'speaker', 'text', 'text_length', 'text_type']
  - python ./datamgmt/extract_speakers_transcripts_timestamps_from_vtt.py

9. For now, merge questions/answers around main speakers' around the timestamp from description question file
  - python ./datamgmt/create_question_answer_dataset.py

10. Create finetuned model:

11. Execute finetuned model:
