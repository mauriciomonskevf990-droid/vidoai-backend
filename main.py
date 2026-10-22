import os
import uuid
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional

import fal_client
import requests
import imageio_ffmpeg

from fastapi import (
    FastAPI, UploadFile, File, Form,
    HTTPException, BackgroundTasks
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


APP_NAME = "VidoAI Backend"

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WORK_DIR = DATA_DIR / "work"

for folder in (UPLOAD_DIR, OUTPUT_DIR, WORK_DIR):
    folder.mkdir(parents=True, exist_ok=True)

FAL_KEY = os.getenv("FAL_KEY", "").strip()

TEXT_MODEL = os.getenv(
    "FAL_TEXT_VIDEO_MODEL",
    "fal-ai/wan/v2.2-a14b/text-to-video"
)

IMAGE_MODEL = os.getenv(
    "FAL_IMAGE_VIDEO_MODEL",
    "fal-ai/wan/v2.2-a14b/image-to-video"
)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "12"))
MAX_DURATION_MINUTES = int(os.getenv("MAX_DURATION_MINUTES", "60"))
MAX_SCENES_PER_JOB = int(os.getenv("MAX_SCENES_PER_JOB", "720"))
SCENE_SECONDS = max(5, int(os.getenv("SCENE_SECONDS", "5")))

ALLOWED_LANGUAGES = {
    "auto", "tg", "ru", "en", "fa", "ar",
    "tr", "es", "fr", "de", "zh", "hi", "other"
}

app = FastAPI(
    title=APP_NAME,
    description="VidoAI video generation backend using fal.ai",
    version="0.2.0"
)

origins = [
    x.strip()
    for x in os.getenv("CORS_ORIGINS", "*").split(",")
    if x.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount(
    "/outputs",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="outputs"
)

# Prototype storage: jobs are lost when the server restarts.
JOBS = {}
JOBS_LOCK = threading.Lock()


def set_job(job_id, **values):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(values)
            JOBS[job_id]["updated_at"] = time.time()


def public_job(job):
    keys = [
        "job_id", "status", "progress", "message",
        "duration_minutes", "scene_count",
        "voice_language", "subtitle_language",
        "video_url", "subtitle_url", "error", "note"
    ]
    return {key: job[key] for key in keys if key in job}


def make_scene_prompts(prompt, scene_count, style):
    beats = [
        "Establish the location and atmosphere",
        "Introduce the main subject clearly",
        "Show the beginning of the action",
        "Develop the story with natural movement",
        "Show a close-up of the main subject",
        "Reveal more of the environment",
        "Build tension through cinematic movement",
        "Show the central action",
        "Use atmospheric transition shots",
        "Show the strongest dramatic moment",
        "Resolve the action",
        "Finish with a cinematic closing shot"
    ]

    prompts = []

    for index in range(scene_count):
        beat = beats[index % len(beats)]

        scene_prompt = (
            f"{prompt}. "
            f"Scene {index + 1} of {scene_count}. "
            f"Action: {beat}. "
            f"Visual style: {style}. "
            "Keep characters, clothing and location consistent. "
            "            TEXT_MODEL,
      Natural motion, cinematic composition, no text or logos."
        )

        prompts.append(scene_prompt)

    return prompts


def download_file(url, destination):
    with requests.get(
        url,
        stream=True,
        timeout=(20, 180)
    ) as response:
        response.raise_for_status()

        with open(destination, "wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)


def run_ffmpeg(arguments, timeout=1800):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    result = subprocess.run(
        [ffmpeg, "-y", *arguments],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FFmpeg error: " +
            (result.stderr[-1500:] if result.stderr else "unknown error")
        )

    return result


def generate_scene(scene_prompt, image_url=None):
    if not FAL_KEY:
        raise RuntimeError("FAL_KEY is missing in Render Environment.")

    os.environ["FAL_KEY"] = FAL_KEY

    if image_url:
        result = fal_client.subscribe(
            IMAGE_MODEL,
            arguments={
                "image_url": image_url,
                "prompt": scene_prompt,
                "num_frames": 81,
                "frames_per_second": 16,
                "resolution": "480p",
                "aspect_ratio": "16:9"
            },
            with_logs=False
        )
    else:
        result = fal_client.subscribe(
      arguments={
                "prompt": scene_prompt,
                "num_frames": 81,
                "frames_per_second": 16,
                "resolution": "480p",
                "aspect_ratio": "16:9"
            },
            with_logs=False
        )

    video = result.get("video", {}) if isinstance(result, dict) else {}
    url = video.get("url") if isinstance(video, dict) else None

    if not url:
        raise RuntimeError(
            "fal.ai returned no video URL. Check model access and API key."
        )

    return url

@app.get("/")
def root():
    return {
        "service": APP_NAME,
        "status": "ok",
        "version": "0.2.0",
        "message": "VidoAI Backend работает"
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "fal_key_configured": bool(FAL_KEY)
    }


@app.get("/api/config")
def api_config():
    return {
        "ok": True,
        "text_to_video": True,
        "image_to_video": True,
        "max_duration_minutes": MAX_DURATION_MINUTES,
        "supported_durations": [3, 5, 10, 15, 30, 45, 60],
        "voice_and_subtitles": "language selection only; generation not implemented yet"
    }


@app.post("/api/generate")
async def generate_video(
    background_tasks: BackgroundTasks,
    prompt: str = Form(""),
    duration_minutes: int = Form(3),
    style: str = Form("cinematic"),
    voice_language: str = Form("tg"),
    subtitle_language: str = Form("ru"),
    photo: Optional[UploadFile] = File(None)
):
    prompt = prompt.strip()

    if not FAL_KEY:
        raise HTTPException(
            status_code=503,
            detail="FAL_KEY не настроен в Render Environment."
        )

    if not prompt and photo is None:
        raise HTTPException(
            status_code=400,
            detail="Введите описание видео или загрузите фотографию."
        )

    if duration_minutes not in [3, 5, 10, 15, 30, 45, 60]:
        raise HTTPException(
            status_code=400,
            detail="Выберите длительность от 3 до 60 минут из списка."
        )

    if voice_language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Неизвестный язык озвучки.")

    if subtitle_language not in ALLOWED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Неизвестный язык субтитров.")

    if photo is not None:
        content_type = (photo.content_type or "").lower()

        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=415,
                detail="Загрузите изображение JPG, PNG или WEBP."
            )

        raw = await photo.read()

        if not raw:
            raise HTTPException(status_code=400, detail="Фотография пустая.")

        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail=f"Фотография должна быть не больше {MAX_UPLOAD_MB} МБ."
            )

        photo_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.jpg"
        photo_path.write_bytes(raw)
    else:
        photo_path = None

    job_id = uuid.uuid4().hex[:12]

    with JOBS_LOCK:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "message": "Задание принято",
            "duration_minutes": duration_minutes,
            "scene_count": duration_minutes * 60 // SCENE_SECONDS,
            "voice_language": voice_language,
            "subtitle_language": subtitle_language,
            "created_at": time.time(),
            "updated_at": time.time(),
            "note": "Прототип: озвучка и настоящие субтитры пока не генерируются."
        }

    return public_job(JOBS[job_id])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)

        if job is None:
            raise HTTPException(status_code=404, detail="Задание не найдено.")

        return public_job(job)


@app.get("/api/jobs")
def list_jobs():
    with JOBS_LOCK:
        return {
            "jobs": [
                public_job(job)
                for job in JOBS.values()
            ]
        }
