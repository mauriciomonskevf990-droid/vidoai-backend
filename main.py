import os
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vidoai")

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
VIDEO_PROVIDER = os.getenv("VIDEO_PROVIDER", "fal-ai").strip()
TEXT_VIDEO_MODEL = os.getenv("TEXT_VIDEO_MODEL", "Wan-AI/Wan2.2-TI2V-5B").strip()
IMAGE_VIDEO_MODEL = os.getenv("IMAGE_VIDEO_MODEL", "Wan-AI/Wan2.2-I2V-A14B").strip()
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_PROMPT_CHARS = 1500

app = FastAPI(title="VidoAI Backend", version="0.1.0")

# For local testing. In production, set CORS_ORIGINS to your real frontend domain.
origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://127.0.0.1:5500,http://localhost:5500,http://127.0.0.1:8000,http://localhost:8000").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


def get_client() -> InferenceClient:
    if not HF_TOKEN:
        raise HTTPException(
            status_code=503,
            detail="Не настроен HF_TOKEN. Создайте токен Hugging Face с правами Inference Providers и добавьте его в .env."
        )
    return InferenceClient(provider=VIDEO_PROVIDER, api_key=HF_TOKEN, timeout=300)


@app.get("/")
def root():
    return {
        "service": "VidoAI Backend",
        "status": "ok",
        "message": "API работает. Откройте /docs для проверки эндпоинтов.",
        "generation_note": "Один запрос создаёт короткий клип, а не готовый фильм на 3–60 минут."
    }


@app.get("/health")
def health():
    return {"ok": True, "token_configured": bool(HF_TOKEN)}


@app.post("/api/generate")
async def generate_video(
    prompt: str = Form(""),
    photo: Optional[UploadFile] = File(None),
    style: str = Form("cinematic"),
):
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"Описание слишком длинное. Максимум {MAX_PROMPT_CHARS} символов.")
    if not prompt and not photo:
        raise HTTPException(status_code=400, detail="Введите описание или загрузите фотографию.")

    style_map = {
        "cinematic": "cinematic lighting, smooth camera movement, coherent composition, high detail",
        "realistic": "photorealistic, natural movement, realistic lighting",
        "animation": "high quality 3D animation, expressive motion, polished visuals",
        "documentary": "documentary film look, natural colors, handheld camera realism",
        "fantasy": "epic fantasy film look, atmospheric light, rich visual details",
        "advert": "premium commercial film look, polished camera movement, product-quality lighting",
    }
    style_hint = style_map.get(style.lower(), style_map["cinematic"])
    full_prompt = f"{prompt}. Visual style: {style_hint}."

    try:
        client = get_client()
        job_id = uuid.uuid4().hex[:12]
        output_path = OUTPUT_DIR / f"{job_id}.mp4"

        if photo is not None:
            content_type = (photo.content_type or "").lower()
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=415, detail="Загрузите файл изображения JPG, PNG или WEBP.")
            raw = await photo.read()
            if not raw:
                raise HTTPException(status_code=400, detail="Файл фотографии пуст.")
            if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
                raise HTTPException(status_code=413, detail=f"Фото должно быть меньше {MAX_UPLOAD_MB} МБ.")
            log.info("Starting image-to-video job %s", job_id)
            video_bytes = client.image_to_video(
                raw,
                model=IMAGE_VIDEO_MODEL,
                prompt=full_prompt,
                num_frames=49,
            )
        else:
            log.info("Starting text-to-video job %s", job_id)
            video_bytes = client.text_to_video(
                full_prompt,
                model=TEXT_VIDEO_MODEL,
                num_frames=49,
            )

        if not video_bytes:
            raise HTTPException(status_code=502, detail="AI-сервис вернул пустой результат.")
        output_path.write_bytes(video_bytes)
        return {
            "ok": True,
            "job_id": job_id,
            "video_url": f"/outputs/{job_id}.mp4",
            "filename": f"{job_id}.mp4",
            "message": "Короткий клип готов. Длительность зависит от выбранной модели."
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Video generation failed")
        # Don't leak API keys or provider internals to the browser.
        raise HTTPException(
            status_code=502,
            detail="Генерация не удалась. Проверьте HF_TOKEN, доступность модели и баланс/лимиты провайдера. Подробности смотрите в терминале сервера."
        ) from exc
