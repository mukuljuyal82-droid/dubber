# Video Dubber (test version)

Chinese video -> Hindi or English dubbed video. First 10 minutes by default.

## Setup
1. Install ffmpeg and Python 3.10+
2. `pip install -r requirements.txt`
3. Free API key from console.groq.com, then `export GROQ_API_KEY=...`

## Easiest: deploy online (Render / Railway)
1. Put this folder in a GitHub repo.
2. On Render or Railway, create a new Web Service from the repo (it detects the Dockerfile).
3. Add environment variable `GROQ_API_KEY` (free key from console.groq.com).
4. Open the link you get, upload a video, wait, download.
Free plans are slow and may restart; fine for the 10-minute test.

## Voice samples first
`python voice_sample.py` makes 4 sample clips (Hindi Swara/Madhur, English Aria/Guy).

## Run on one video
`python pipeline.py video.mp4 --lang hi --minutes 10`
Output: `video_work/dubbed_hi.mp4`. Steps are cached in the work folder, so a re-run
with another language only redoes translation and voice.

## Web UI
`uvicorn app:app --port 8000` then open http://localhost:8000

## How sync works
Each translated line gets a character budget based on how long the Chinese line is.
Voice clips may use the silence before the next line, are sped up at most 1.3x if
still too long, and are pushed later instead of overlapping.

## Not done yet
Demucs background separation (`--separate`, needs GPU for long videos), job queue,
cloud storage, credits and payments, Azure voices.
