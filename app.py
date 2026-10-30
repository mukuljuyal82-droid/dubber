"""Minimal web UI: upload a video, pick a language, get a dubbed video.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import pipeline

DATA = Path("jobs")
DATA.mkdir(exist_ok=True)
JOBS = {}
app = FastAPI()

PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Video Dubber</title>
<body style="font-family:sans-serif;max-width:520px;margin:2rem auto;padding:0 1rem">
<h2>Video Dubber (test)</h2>
<form id=f><input type=file name=file accept="video/*" required><br><br>
<select name=lang><option value=hi>Hindi</option><option value=en>English</option></select>
<select name=minutes><option value=10>First 10 min (test)</option><option value=0>Full video</option></select><br><br>
<button>Start</button></form><pre id=s></pre>
<script>
f.onsubmit=async e=>{e.preventDefault();s.textContent='Uploading...';
const r=await fetch('/upload',{method:'POST',body:new FormData(f)});const j=await r.json();
if(!r.ok){s.textContent=j.detail;return}
const t=setInterval(async()=>{const x=await (await fetch('/status/'+j.id)).json();
s.textContent=x.status+(x.message?': '+x.message:'');
if(x.status=='done'){clearInterval(t);s.innerHTML='Done. <a href="/download/'+j.id+'">Download</a>'}
if(x.status=='error')clearInterval(t)},3000)}
</script>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


def _work(job_id, video, lang, minutes):
    try:
        JOBS[job_id] = {"status": "processing", "message": "starting"}
        out = pipeline.dub(video, lang, DATA / job_id / "work", minutes or None,
                           progress=lambda m: JOBS[job_id].update(message=m))
        JOBS[job_id] = {"status": "done", "file": str(out)}
    except Exception as e:  # surface the error to the UI
        JOBS[job_id] = {"status": "error", "message": str(e)[:500]}


@app.post("/upload")
async def upload(file: UploadFile = File(...), lang: str = Form("hi"), minutes: int = Form(10)):
    if lang not in pipeline.VOICES:
        raise HTTPException(400, "unsupported language")
    job_id = uuid.uuid4().hex[:12]
    folder = DATA / job_id
    folder.mkdir()
    video = folder / ("input" + Path(file.filename or "v.mp4").suffix)
    with open(video, "wb") as f:
        while chunk := await file.read(1 << 20):
            f.write(chunk)
    JOBS[job_id] = {"status": "queued"}
    threading.Thread(target=_work, args=(job_id, video, lang, minutes), daemon=True).start()
    return {"id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    return JSONResponse(JOBS.get(job_id, {"status": "unknown"}))


@app.get("/download/{job_id}")
def download(job_id: str):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404, "not ready")
    return FileResponse(job["file"], filename="dubbed.mp4")
