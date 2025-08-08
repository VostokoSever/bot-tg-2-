# Smart Bot 2.0 — Webhook (Render Free)

Эта версия запускается как **Web Service** на Render (бесплатный план).

## Быстрый деплой
1. Создай бота у @BotFather и скопируй токен.
2. Залей файлы репозитория в GitHub.
3. На https://render.com → **New → Web Service** → выбери репозиторий.
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python smart_bot.py`
   - Добавь переменную окружения: `TELEGRAM_BOT_TOKEN`
   - Остальные — по желанию: `OPENAI_API_KEY`, `BROTHER_USERNAME`
4. Нажми **Create Web Service**. После статуса **Live** Render задаст `RENDER_EXTERNAL_URL`.
5. Перезапусти сервис (**Manual Deploy → Clear cache & deploy**) — бот выставит webhook сам.
6. Напиши боту `/start` в Telegram.

Если при первом старте увидишь ошибку про `WEBHOOK_BASE/RENDER_EXTERNAL_URL` — это нормально:
после первого запуска у Render появляется `RENDER_EXTERNAL_URL`. Просто **ещё раз задеплой** (Manual Deploy).
