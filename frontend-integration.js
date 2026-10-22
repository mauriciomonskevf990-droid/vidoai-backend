/*
Подключение к существующему VidoAI frontend.
Добавьте этот код в app.js вместо текущего обработчика generateBtn.
Ожидаются элементы: prompt, photoInput, style, generateBtn, resultBox.
Локальный backend: http://127.0.0.1:8000
*/
const API_BASE = "http://127.0.0.1:8000";

document.getElementById("generateBtn").addEventListener("click", async () => {
  const prompt = document.getElementById("prompt").value.trim();
  const photo = document.getElementById("photoInput").files[0];
  const style = document.getElementById("style").value;

  if (!prompt && !photo) {
    alert("Введите описание или загрузите фотографию.");
    return;
  }

  const button = document.getElementById("generateBtn");
  const result = document.getElementById("resultBox");
  const form = new FormData();
  form.append("prompt", prompt);
  form.append("style", style);
  if (photo) form.append("photo", photo);

  button.disabled = true;
  button.textContent = "Генерация клипа… подождите";
  result.hidden = false;
  result.textContent = "Отправляем задачу AI-сервису. Это может занять несколько минут.";

  try {
    const response = await fetch(`${API_BASE}/api/generate`, {
      method: "POST",
      body: form
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Не удалось создать видео.");

    const videoUrl = `${API_BASE}${data.video_url}`;
    result.innerHTML = "";
    const title = document.createElement("strong");
    title.textContent = "Клип готов";
    const video = document.createElement("video");
    video.src = videoUrl;
    video.controls = true;
    video.playsInline = true;
    video.style.width = "100%";
    video.style.display = "block";
    video.style.marginTop = "12px";
    const link = document.createElement("a");
    link.href = videoUrl;
    link.download = data.filename;
    link.textContent = "Скачать MP4";
    link.style.display = "inline-block";
    link.style.marginTop = "12px";
    link.style.color = "#c4b5fd";
    result.append(title, video, link);
  } catch (error) {
    result.textContent = `Ошибка: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "✦ Подготовить проект";
  }
});
