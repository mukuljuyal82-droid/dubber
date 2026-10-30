"""Chinese video -> Hindi / English dubbed video.

Steps: extract audio -> Whisper (Groq) -> translate with length budget (Groq LLM)
-> TTS (edge-tts, free; swap for Azure later) -> fit clips to timeline -> mix -> mux.

Usage:
    export GROQ_API_KEY=...
    python pipeline.py input.mp4 --lang hi --minutes 10
"""
import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GROQ = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3"
LLM_MODEL = "llama-3.3-70b-versatile"

VOICES = {"hi": "hi-IN-SwaraNeural", "en": "en-US-AriaNeural"}
LANG_NAME = {"hi": "Hindi (Devanagari script)", "en": "English"}
# Rough spoken characters per second at natural speed; used as a translation length budget.
CPS = {"hi": 13.0, "en": 15.0}

CHUNK_SECONDS = 600
MAX_SPEEDUP = 1.3
GAP = 0.05


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr[-800:]}")
    return r.stdout


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- sync planning (pure logic, unit-testable) ----------
def plan_placements(segments, durations, max_speed=MAX_SPEEDUP, gap=GAP):
    """segments: list of {start,end}; durations: natural TTS length per segment (s).

    Each clip may use the silence up to the next segment's start. If it is still too
    long it is sped up (capped at max_speed); anything left over pushes the clip later
    instead of overlapping the previous one.
    """
    plan = []
    cursor = -gap
    for i, seg in enumerate(segments):
        dur = durations[i]
        nxt = segments[i + 1]["start"] if i + 1 < len(segments) else seg["end"] + 30
        allowed = max(nxt - seg["start"] - gap, seg["end"] - seg["start"], 0.2)
        speed = min(max_speed, max(1.0, dur / allowed))
        out_dur = dur / speed
        place = max(seg["start"], cursor + gap)
        cursor = place + out_dur
        plan.append({"start": place, "speed": speed, "out_dur": out_dur})
    return plan


# ---------- steps ----------
def extract_chunks(video, workdir, minutes):
    chunks_dir = workdir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    if not list(chunks_dir.glob("chunk_*.mp3")):
        cmd = ["ffmpeg", "-y", "-i", str(video)]
        if minutes:
            cmd += ["-t", str(minutes * 60)]
        cmd += ["-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
                "-f", "segment", "-segment_time", str(CHUNK_SECONDS),
                "-reset_timestamps", "1", str(chunks_dir / "chunk_%03d.mp3")]
        run(cmd)
    return sorted(chunks_dir.glob("chunk_*.mp3"))


def transcribe(chunks, workdir):
    import requests
    out = workdir / "transcript.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    key = os.environ["GROQ_API_KEY"]
    segs = []
    for idx, chunk in enumerate(chunks):
        log(f"transcribing chunk {idx + 1}/{len(chunks)}")
        for attempt in range(4):
            with open(chunk, "rb") as f:
                r = requests.post(
                    f"{GROQ}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {key}"},
                    files={"file": (chunk.name, f, "audio/mpeg")},
                    data={"model": WHISPER_MODEL, "language": "zh",
                          "response_format": "verbose_json"},
                    timeout=300,
                )
            if r.status_code == 200:
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"transcription failed: {r.status_code} {r.text[:300]}")
        for s in r.json().get("segments", []):
            text = s["text"].strip()
            if text:
                segs.append({"start": s["start"] + idx * CHUNK_SECONDS,
                             "end": s["end"] + idx * CHUNK_SECONDS, "text": text})
    out.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    return segs


def translate(segs, lang, workdir):
    import requests
    out = workdir / f"translation_{lang}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    key = os.environ["GROQ_API_KEY"]
    result = []
    batch_size = 20
    for b in range(0, len(segs), batch_size):
        batch = segs[b:b + batch_size]
        items = [{"id": i, "zh": s["text"],
                  "max_chars": max(8, int((s["end"] - s["start"]) * CPS[lang] * 1.25))}
                 for i, s in enumerate(batch)]
        prompt = (
            f"Translate these Chinese subtitle lines into natural, spoken {LANG_NAME[lang]} "
            "for a video voiceover. Keep names consistent, keep the tone of the speaker. "
            "Each translation MUST be at most max_chars characters, so shorten by rephrasing "
            "rather than cutting meaning. No notes, no brackets.\n"
            'Return JSON only: {"items":[{"id":0,"text":"..."}]}\n\n'
            + json.dumps(items, ensure_ascii=False)
        )
        translated = {}
        for attempt in range(4):
            r = requests.post(
                f"{GROQ}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": LLM_MODEL, "temperature": 0.3,
                      "response_format": {"type": "json_object"},
                      "messages": [{"role": "user", "content": prompt}]},
                timeout=120,
            )
            if r.status_code == 200:
                try:
                    data = json.loads(r.json()["choices"][0]["message"]["content"])
                    translated = {int(x["id"]): x["text"].strip() for x in data["items"]}
                    break
                except (KeyError, ValueError, TypeError):
                    pass
            time.sleep(4 * (attempt + 1))
        else:
            raise RuntimeError(f"translation failed: {r.status_code} {r.text[:300]}")
        for i, s in enumerate(batch):
            result.append({**s, "dub": translated.get(i, "")})
        log(f"translated {min(b + batch_size, len(segs))}/{len(segs)} lines")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


async def _tts_all(items, voice, clips_dir):
    import edge_tts
    sem = asyncio.Semaphore(6)

    async def one(i, text):
        path = clips_dir / f"{i:05d}.mp3"
        if path.exists() and path.stat().st_size > 0:
            return
        async with sem:
            for attempt in range(3):
                try:
                    await edge_tts.Communicate(text, voice).save(str(path))
                    return
                except Exception:
                    await asyncio.sleep(2 * (attempt + 1))
            raise RuntimeError(f"TTS failed for line {i}")

    await asyncio.gather(*(one(i, t) for i, t in items))


def synthesize(lines, lang, workdir):
    clips_dir = workdir / f"clips_{lang}"
    clips_dir.mkdir(exist_ok=True)
    items = [(i, l["dub"]) for i, l in enumerate(lines) if l["dub"]]
    log(f"generating {len(items)} voice clips")
    asyncio.run(_tts_all(items, VOICES[lang], clips_dir))
    return clips_dir


def separate_background(video, workdir, minutes):
    """Optional: remove original voices with Demucs, keep music/effects."""
    wav = workdir / "full.wav"
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    if minutes:
        cmd += ["-t", str(minutes * 60)]
    run(cmd + ["-vn", "-ac", "2", "-ar", "44100", str(wav)])
    run([sys.executable, "-m", "demucs", "--two-stems=vocals", "-o", str(workdir / "demucs"), str(wav)])
    found = list((workdir / "demucs").glob("*/full/no_vocals.wav"))
    if not found:
        raise RuntimeError("demucs output not found")
    return found[0]


def mix_and_mux(video, lines, clips_dir, lang, workdir, minutes, background):
    from pydub import AudioSegment
    import tempfile

    spoken = [(i, l) for i, l in enumerate(lines) if l["dub"] and (clips_dir / f"{i:05d}.mp3").exists()]
    clips = {i: AudioSegment.from_file(clips_dir / f"{i:05d}.mp3") for i, _ in spoken}
    durations = [len(clips[i]) / 1000.0 for i, _ in spoken]
    plan = plan_placements([l for _, l in spoken], durations)

    total_ms = int((max(l["end"] for l in lines) + 5) * 1000)
    track = AudioSegment.silent(duration=total_ms, frame_rate=24000)
    for (i, _), p in zip(spoken, plan):
        clip = clips[i]
        if p["speed"] > 1.01:
            with tempfile.TemporaryDirectory() as td:
                src, dst = Path(td) / "a.wav", Path(td) / "b.wav"
                clip.export(src, format="wav")
                run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={p['speed']:.3f}", str(dst)])
                clip = AudioSegment.from_file(dst)
        track = track.overlay(clip, position=int(p["start"] * 1000))
    track = track.set_channels(2).set_frame_rate(44100)

    if background:
        bg = AudioSegment.from_file(background)
    else:
        src_wav = workdir / "orig.wav"
        cmd = ["ffmpeg", "-y", "-i", str(video)]
        if minutes:
            cmd += ["-t", str(minutes * 60)]
        run(cmd + ["-vn", "-ac", "2", "-ar", "44100", str(src_wav)])
        bg = AudioSegment.from_file(src_wav) - 22  # faint original under the dub
    mixed = bg[:len(track)].overlay(track) if len(bg) >= len(track) else track.overlay(bg)
    mix_path = workdir / f"mix_{lang}.wav"
    mixed.export(mix_path, format="wav")

    out = workdir / f"dubbed_{lang}.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(video), "-i", str(mix_path)]
    if minutes:
        cmd += ["-t", str(minutes * 60)]
    cmd += ["-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)]
    run(cmd)
    return out


def dub(video, lang="hi", workdir=None, minutes=None, separate=False, progress=log):
    video = Path(video)
    workdir = Path(workdir or video.with_suffix("").name + "_work")
    workdir.mkdir(parents=True, exist_ok=True)
    progress("extracting audio")
    chunks = extract_chunks(video, workdir, minutes)
    segs = transcribe(chunks, workdir)
    if not segs:
        raise RuntimeError("no speech found")
    lines = translate(segs, lang, workdir)
    clips_dir = synthesize(lines, lang, workdir)
    background = separate_background(video, workdir, minutes) if separate else None
    progress("mixing and exporting")
    return mix_and_mux(video, lines, clips_dir, lang, workdir, minutes, background)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--lang", choices=["hi", "en"], default="hi")
    ap.add_argument("--minutes", type=int, default=10, help="only process first N minutes (0 = all)")
    ap.add_argument("--workdir")
    ap.add_argument("--separate", action="store_true", help="remove original voices with Demucs (slow without GPU)")
    a = ap.parse_args()
    print(dub(a.video, a.lang, a.workdir, a.minutes or None, a.separate))
