"""Chinese video -> Hindi / English dubbed video.

Steps: extract audio -> Whisper (Groq) -> translate with length budget (Groq LLM)
-> TTS (Cartesia if CARTESIA_API_KEY is set, else edge-tts) -> fit clips to timeline -> mix -> mux.

The original Chinese voice is REMOVED from the output. Only the dubbed voice is kept
(plus music/effects if you use --separate, which needs Demucs).

Usage:
    export GROQ_API_KEY=...
    export CARTESIA_API_KEY=...
    export GEMINI_API_KEY=...      # optional, better Hindi translation
    python pipeline.py input.mp4 --lang hi --minutes 10
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GROQ = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3"
# Groq retired llama-3.3-70b-versatile (Aug 2026). Tried in order; only ones the account can see are used.
LLM_MODELS = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

# ---- Gemini (optional). Set GEMINI_API_KEY for better translation; Groq is the automatic backup. ----
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
# Model names change often: the code asks the API which ones the key can use. GEMINI_MODEL forces one.
GEMINI_FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
GEMINI_SKIP = ("image", "tts", "live", "audio", "embed", "vision", "robotics", "computer", "learnlm",
               "gemma", "nano", "banana", "veo", "imagen", "native", "thinking")

# ---- edge-tts (free fallback, used only when CARTESIA_API_KEY is missing) ----
VOICES = {"hi": "hi-IN-SwaraNeural", "en": "en-US-AriaNeural"}

# ---- Cartesia ----
CARTESIA_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_MODEL = os.environ.get("CARTESIA_MODEL", "sonic-3")
# Tried in order; the first version the API accepts is used.
CARTESIA_VERSIONS = [v for v in (os.environ.get("CARTESIA_VERSION"), "2026-03-01", "2025-04-16") if v]
# Voice IDs. Change them freely: browse play.cartesia.ai/voices, copy the ID, set env var
# CARTESIA_VOICE_HI / CARTESIA_VOICE_EN (or edit here). Hindi voice MUST be a Hindi-native voice.
CARTESIA_VOICES = {
    "hi": os.environ.get("CARTESIA_VOICE_HI", "ac7ee4fa-25db-420d-bfff-f590d740aeb2"),
    "en": os.environ.get("CARTESIA_VOICE_EN", "f786b574-daa5-4673-aa0c-cbe3e8534c02"),
}
CARTESIA_WORKERS = int(os.environ.get("CARTESIA_WORKERS", "2"))  # free plan allows few parallel requests

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
    chunks = sorted(chunks_dir.glob("chunk_*.mp3"))
    # A leftover sliver at the end can be rejected by the API; drop near-empty chunks (~<0.7s).
    return [c for c in chunks if c.stat().st_size > 4000]


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
            if r.status_code == 400 and "too short" in r.text:
                r = None  # tiny leftover chunk, nothing to transcribe
                break
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"transcription failed: {r.status_code} {r.text[:300]}")
        if r is None:
            continue
        for s in r.json().get("segments", []):
            text = s["text"].strip()
            if text:
                segs.append({"start": s["start"] + idx * CHUNK_SECONDS,
                             "end": s["end"] + idx * CHUNK_SECONDS, "text": text})
    out.write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    return segs


def pick_models(key):
    import requests
    try:
        r = requests.get(f"{GROQ}/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        have = {m["id"] for m in r.json().get("data", [])}
        picked = [m for m in LLM_MODELS if m in have]
        if picked:
            return picked
    except Exception:
        pass
    return list(LLM_MODELS)


def llm_json(prompt, key, models):
    """Ask the LLM for JSON; falls back across models and request styles.

    Reasoning models (gpt-oss, qwen3) can spend the whole token budget thinking and return an
    empty answer, so we ask for low reasoning effort and a large completion budget first.
    """
    import requests
    last = ""
    for attempt in range(3):
        for model in models:
            extras = {"max_completion_tokens": 8000}
            if model.startswith("openai/gpt-oss"):
                extras["reasoning_effort"] = "low"
            variants = [(True, extras), (False, extras), (False, {})]
            for json_mode, extra in variants:
                body = {"model": model, "temperature": 0.3,
                        "messages": [{"role": "user", "content": prompt}], **extra}
                if json_mode:
                    body["response_format"] = {"type": "json_object"}
                r = requests.post(f"{GROQ}/chat/completions",
                                  headers={"Authorization": f"Bearer {key}"},
                                  json=body, timeout=180)
                if r.status_code == 200:
                    choice = r.json()["choices"][0]
                    text = choice["message"].get("content") or ""
                    try:
                        return json.loads(text[text.index("{"): text.rindex("}") + 1])
                    except ValueError:
                        last = f"{model}: bad JSON (finish={choice.get('finish_reason')}) {text[:150]!r}"
                        continue
                last = f"{model}: {r.status_code} {r.text[:250]}"
                if r.status_code == 429:
                    time.sleep(10)
                if r.status_code == 404:
                    break  # model not available, try the next one
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"translation failed: {last}")


# ---------- Gemini (optional: better Hindi; Groq stays as backup) ----------
def _gemini_rank(name):
    """Sort key for text models: stable Flash first, then Flash-Lite, then Pro; newest version first."""
    m = re.match(r"gemini-(\d+(?:\.\d+)*)-(flash-lite|flash|pro)(.*)$", name)
    if not m or any(w in name for w in GEMINI_SKIP):
        return None
    ver = tuple(int(x) for x in m.group(1).split("."))
    ver = (ver + (0, 0, 0))[:3]
    tier = {"flash": 0, "flash-lite": 1, "pro": 2}[m.group(2)]
    unstable = 1 if ("preview" in m.group(3) or "exp" in m.group(3)) else 0
    return (tier, unstable, tuple(-x for x in ver), len(m.group(3)))


def gemini_models(key):
    """Models this key can use, best first. Names change often, so ask the API instead of hard-coding."""
    import requests
    forced = os.environ.get("GEMINI_MODEL")
    if forced:
        return [forced]
    names, token = [], None
    try:
        for _ in range(5):
            params = {"pageSize": 200}
            if token:
                params["pageToken"] = token
            r = requests.get(f"{GEMINI_BASE}/models", headers={"x-goog-api-key": key},
                             params=params, timeout=30)
            if r.status_code != 200:
                break
            data = r.json()
            for m in data.get("models", []):
                if "generateContent" in m.get("supportedGenerationMethods", []):
                    names.append(m["name"].split("/", 1)[-1])
            token = data.get("nextPageToken")
            if not token:
                break
    except Exception:
        pass
    ranked = sorted((n for n in set(names) if _gemini_rank(n)), key=_gemini_rank)
    return ranked or list(GEMINI_FALLBACK_MODELS)


def gemini_json(prompt, key, models):
    """Ask Gemini for JSON. Drops models that are unavailable or out of free quota (edits `models`)."""
    import requests
    last = ""
    for attempt in range(3):
        for model in list(models):
            body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}}
            r = requests.post(f"{GEMINI_BASE}/models/{model}:generateContent",
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=180)
            if r.status_code == 200:
                try:
                    parts = r.json()["candidates"][0]["content"]["parts"]
                    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                    return json.loads(text[text.index("{"): text.rindex("}") + 1])
                except (KeyError, IndexError, ValueError, TypeError):
                    last = f"{model}: unreadable answer {r.text[:150]!r}"
                    continue
            last = f"{model}: {r.status_code} {r.text[:200]}"
            low = r.text.lower()
            if r.status_code in (400, 401, 403) and "api key" in low:
                raise RuntimeError(f"Gemini key rejected: {last}")
            if r.status_code in (403, 404):
                models.remove(model)  # this key cannot use this model
            elif r.status_code == 429:
                squashed = low.replace("_", "").replace(" ", "")
                if "limit:0" in squashed or "perday" in squashed:
                    models.remove(model)  # free quota for this model is used up
                else:
                    time.sleep(15)  # per-minute limit, wait a bit
            elif r.status_code >= 500:
                time.sleep(3 * (attempt + 1))
        if not models:
            break
        time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"Gemini failed: {last}")


def make_llm():
    """Returns (call, label). call(prompt) -> dict. Gemini first when GEMINI_API_KEY is set; Groq is the backup."""
    groq_key = os.environ["GROQ_API_KEY"]
    groq_models = pick_models(groq_key)
    gkey = os.environ.get("GEMINI_API_KEY")
    gem = {"key": gkey, "models": gemini_models(gkey), "fails": 0} if gkey else None

    def call(prompt):
        if gem and gem["models"]:
            try:
                data = gemini_json(prompt, gem["key"], gem["models"])
                gem["fails"] = 0
                return data
            except RuntimeError as e:
                gem["fails"] += 1
                log(f"Gemini failed ({str(e)[:150]}); using Groq for this step")
                if gem["fails"] >= 2:
                    gem["models"] = []
                    log("Gemini switched off for the rest of this video")
        return llm_json(prompt, groq_key, groq_models)

    label = f"Gemini ({gem['models'][0]})" if gem and gem["models"] else f"Groq ({groq_models[0]})"
    return call, label


# ---------- translation ----------
def _sample_text(texts, limit):
    """Up to `limit` characters, taken evenly from the whole video (not just the start)."""
    full = "\n".join(texts)
    if len(full) <= limit:
        return full
    k = 8
    size = limit // k
    step = (len(full) - size) // (k - 1)
    return "\n...\n".join(full[i * step: i * step + size] for i in range(k))


def build_glossary(segs, lang, workdir, call, limit):
    """One pass over the whole video: summary + one fixed rendering for every name, so names never change."""
    out = workdir / f"glossary_{lang}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    prompt = (
        "Below are Chinese subtitle lines sampled from one video.\n"
        "1) Write a 1-2 sentence English summary: topic, story/genre, tone, who is speaking.\n"
        "2) List the important names (people, places, groups, titles, special terms) that appear, "
        f"each with ONE consistent rendering in {LANG_NAME[lang]} for the voiceover, written the way "
        "a native speaker would say it aloud.\n"
        'Return JSON only: {"summary":"...","terms":[{"zh":"...","to":"..."}]}\n\n'
        + _sample_text([s["text"] for s in segs], limit)
    )
    try:
        data = call(prompt)
    except RuntimeError as e:
        log(f"glossary skipped: {str(e)[:150]}")
        return {"summary": "", "terms": []}
    terms = []
    for t in data.get("terms", []):
        if isinstance(t, dict) and t.get("zh") and t.get("to"):
            terms.append({"zh": str(t["zh"]).strip(), "to": str(t["to"]).strip()})
    glossary = {"summary": str(data.get("summary", ""))[:600], "terms": terms[:80]}
    out.write_text(json.dumps(glossary, ensure_ascii=False, indent=1), encoding="utf-8")
    return glossary


def _translate_prompt(lang, summary, terms, before, after, items):
    parts = [
        "You are a professional dubbing translator. Translate the Chinese subtitle lines below into "
        f"natural, spoken {LANG_NAME[lang]} for a voiceover that a narrator reads aloud. Keep the speaker's tone."
    ]
    if summary:
        parts.append(f"About the video: {summary}")
    if terms:
        parts.append("Always render these names/terms exactly like this: "
                     + json.dumps(terms, ensure_ascii=False))
    if before:
        parts.append("Lines just before (already translated, context only, do NOT output them): "
                     + json.dumps([{"zh": x["text"], "done": x["dub"]} for x in before], ensure_ascii=False))
    if after:
        parts.append("Lines just after (context only, do NOT translate): "
                     + json.dumps([s["text"] for s in after], ensure_ascii=False))
    rules = (
        "Rules: translate the meaning, not word for word, and do not add or drop information. "
        "A sentence can continue across lines, so make the lines flow one after another without "
        "repeating words from the previous line. EVERY line must get a translation of at most max_chars "
        "characters: shorten by rephrasing, never by cutting meaning. No notes, brackets or emoji."
    )
    if lang == "hi":
        rules += " Write Hindi in Devanagari, in simple spoken language, the way a storyteller would say it."
    parts.append(rules)
    parts.append('Return JSON only: {"items":[{"id":0,"text":"..."}]}')
    parts.append("Lines to translate:\n" + json.dumps(items, ensure_ascii=False))
    return "\n\n".join(parts)


def _parse_items(data):
    translated = {}
    for x in data.get("items", []):
        try:
            text = str(x["text"]).strip()
            if text:
                translated[int(x["id"])] = text
        except (KeyError, ValueError, TypeError):
            continue
    return translated


def translate(segs, lang, workdir):
    out = workdir / f"translation_{lang}_v2.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    call, label = make_llm()
    log(f"translating with {label}")
    glossary = build_glossary(segs, lang, workdir, call, 14000 if label.startswith("Gemini") else 5000)
    log(f"glossary ready: {len(glossary['terms'])} names/terms")
    result = []
    batch_size = 15
    for b in range(0, len(segs), batch_size):
        batch = segs[b:b + batch_size]
        before = [x for x in result[-4:] if x["dub"]]
        after = segs[b + batch_size:b + batch_size + 3]
        items = [{"id": i, "zh": s["text"],
                  "max_chars": max(8, int((s["end"] - s["start"]) * CPS[lang] * 1.25))}
                 for i, s in enumerate(batch)]
        seen = " ".join(s["text"] for s in batch + before + after)
        terms = [t for t in glossary["terms"] if t["zh"] in seen][:40]
        translated = {}
        try:
            translated = _parse_items(call(_translate_prompt(
                lang, glossary["summary"], terms, before, after, items)))
        except RuntimeError as e:
            log(f"batch failed, will retry line by line: {str(e)[:150]}")
        missing = [it for it in items if it["id"] not in translated]
        if missing:  # the model skipped some lines: ask again for just those
            try:
                translated.update(_parse_items(call(_translate_prompt(
                    lang, glossary["summary"], terms, [], [], missing))))
            except RuntimeError as e:
                log(f"retry failed: {str(e)[:150]}")
        gaps = sum(1 for i in range(len(batch)) if i not in translated)
        if gaps:
            log(f"warning: {gaps} line(s) left without translation (silent gap)")
        for i, s in enumerate(batch):
            result.append({**s, "dub": translated.get(i, "")})
        log(f"translated {min(b + batch_size, len(segs))}/{len(segs)} lines")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


# ---------- text to speech ----------
def _cartesia_one(text, lang, path, key):
    """Generate one clip with Cartesia and save it as a WAV file."""
    import requests
    body = {
        "model_id": CARTESIA_MODEL,
        "transcript": text,
        "voice": {"mode": "id", "id": CARTESIA_VOICES[lang]},
        "language": lang,
        "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 24000},
    }
    last = ""
    for attempt in range(5):
        for version in CARTESIA_VERSIONS:
            r = requests.post(
                CARTESIA_URL,
                headers={"X-API-Key": key, "Authorization": f"Bearer {key}",
                         "Cartesia-Version": version, "Content-Type": "application/json"},
                json=body, timeout=120)
            if r.status_code == 200 and r.content:
                path.write_bytes(r.content)
                return
            last = f"{r.status_code} {r.text[:300]}"
            if r.status_code in (400, 422) and "version" in r.text.lower():
                continue  # this API version is not accepted, try the next one
            break
        if "401" in last[:3] or "403" in last[:3]:
            raise RuntimeError(f"Cartesia key rejected: {last}")
        time.sleep(3 * (attempt + 1))  # rate limit (429) or temporary error
    raise RuntimeError(f"Cartesia TTS failed: {last}")


def _tts_cartesia(items, lang, clips_dir):
    key = os.environ["CARTESIA_API_KEY"]

    def one(item):
        i, text = item
        path = clips_dir / f"{i:05d}.wav"
        if path.exists() and path.stat().st_size > 0:
            return
        _cartesia_one(text, lang, path, key)

    done = 0
    with ThreadPoolExecutor(max_workers=CARTESIA_WORKERS) as pool:
        for _ in pool.map(one, items):
            done += 1
            if done % 25 == 0:
                log(f"voice clips {done}/{len(items)}")


async def _tts_edge(items, voice, clips_dir):
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
    use_cartesia = bool(os.environ.get("CARTESIA_API_KEY"))
    # the voice id is part of the folder name, so changing the voice never re-uses old clips
    tag = f"cartesia_{CARTESIA_VOICES[lang][:8]}" if use_cartesia else "edge"
    clips_dir = workdir / f"clips_{lang}_{tag}"
    clips_dir.mkdir(exist_ok=True)
    items = [(i, l["dub"]) for i, l in enumerate(lines) if l["dub"]]
    log(f"generating {len(items)} voice clips with {'Cartesia' if use_cartesia else 'edge-tts'}")
    if use_cartesia:
        _tts_cartesia(items, lang, clips_dir)
    else:
        asyncio.run(_tts_edge(items, VOICES[lang], clips_dir))
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


RATE = 24000


def _pcm(args):
    r = subprocess.run(["ffmpeg", "-v", "error", *args], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr.decode("utf-8", "ignore")[-500:])
    return r.stdout


def _clip_file(clips_dir, i):
    for ext in (".wav", ".mp3"):
        p = clips_dir / f"{i:05d}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def mix_and_mux(video, lines, clips_dir, lang, workdir, minutes, background):
    """Low-memory mixing: voice clips are written into one raw PCM file on disk (no overlap,
    guaranteed by plan_placements), then a single ffmpeg call builds the final audio.

    The original audio track is NOT used. With a Demucs background (music/effects only) the
    dub is mixed over it; without one, the output audio is the dubbed voice only.
    """
    spoken = [(i, l) for i, l in enumerate(lines) if l["dub"] and _clip_file(clips_dir, i)]
    raws, durations = {}, []
    for i, _ in spoken:
        raw = clips_dir / f"{i:05d}.raw"
        if not raw.exists():
            raw.write_bytes(_pcm(["-i", str(_clip_file(clips_dir, i)),
                                  "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"]))
        raws[i] = raw
        durations.append(raw.stat().st_size / 2 / RATE)
    plan = plan_placements([l for _, l in spoken], durations)

    dub_raw = workdir / f"dub_{lang}.raw"
    with open(dub_raw, "wb") as f:
        for (i, _), p in zip(spoken, plan):
            if p["speed"] > 1.01:
                data = _pcm(["-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", str(raws[i]),
                             "-filter:a", f"atempo={p['speed']:.3f}",
                             "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-"])
            else:
                data = raws[i].read_bytes()
            f.seek(int(p["start"] * RATE) * 2)
            f.write(data)

    out = workdir / f"dubbed_{lang}.mp4"
    if background:
        # music/effects (voices already removed) + dubbed voice
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(background),
               "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", str(dub_raw),
               "-filter_complex",
               "[1:a:0]aresample=44100[bg];[2:a]aresample=44100[fg];"
               "[bg][fg]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[out]"]
    else:
        # dubbed voice only: the original audio track is dropped completely
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video),
               "-f", "s16le", "-ar", str(RATE), "-ac", "1", "-i", str(dub_raw),
               "-filter_complex", "[1:a]aresample=44100,alimiter=limit=0.95[out]"]
    cmd += ["-map", "0:v:0", "-map", "[out]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k"]
    if minutes:
        cmd += ["-t", str(minutes * 60)]
    run(cmd + [str(out)])
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
    ap.add_argument("--separate", action="store_true",
                    help="keep music/effects by removing only the original voices with Demucs (slow without GPU)")
    a = ap.parse_args()
    print(dub(a.video, a.lang, a.workdir, a.minutes or None, a.separate))
