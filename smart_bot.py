import os
import re
import random
import logging
from datetime import datetime, timezone, timedelta

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters, PicklePersistence
)

# ---------- Config ----------
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "PASTE_TOKEN_HERE")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # optional
BROTHER_USERNAME = os.environ.get("BROTHER_USERNAME")  # like "@vasya"
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

KYIV_TZ = timezone(timedelta(hours=3))
RATE_LIMIT_PER_MIN = 8

# Webhook base URL (no trailing slash). Prefer RENDER_EXTERNAL_URL; fallback to WEBHOOK_BASE.
WEBHOOK_BASE = (os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("WEBHOOK_BASE") or "").rstrip("/")
PORT = int(os.environ.get("PORT", "10000"))

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("smart-bot")

# ---------- Optional OpenAI ----------
_openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("OpenAI client initialized.")
    except Exception as e:
        log.warning("OpenAI init failed: %s", e)
        _openai_client = None

async def llm_answer(prompt: str, sys: str = "You are a concise, witty assistant."):
    if _openai_client:
        try:
            resp = _openai_client.chat.completions.create(
                model=MODEL_NAME,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": sys},
                    {"role": "user", "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.warning("LLM error: %s", e)
    canned = [
        "Коротко: сформулируй цель, определи входные данные, проверь крайние случаи.",
        "Думай в модели «вход→преобразование→выход».",
        "Объясни 10‑летнему: если получилось — понял тему."
    ]
    return random.choice(canned)

# ---------- Data ----------
FACTS = {
    "наука": [
        "Квантовая запутанность не передаёт информацию быстрее света.",
        "Чёрные дыры теоретически излучают (Хокинг).",
        "CRISPR — «ножницы» для ДНК."
    ],
    "история": [
        "Великая стена — сеть укреплений, а не ровная линия.",
        "Письменность шумеров — клинопись на табличках.",
        "Римские дороги многослойны, поэтому живучи."
    ],
    "математика": [
        "Мощность континуума больше, чем у натуральных чисел.",
        "Числа Фибоначчи всплывают в алгоритмах и природе.",
        "NP-полные задачи легко проверить, трудно найти решение."
    ],
}

QUIZ = [
    {"q": "Что больше: 2^100 или 10^30?", "opts": ["2^100", "10^30"], "a": 0, "ex": "2^10≈10^3 ⇒ 2^100≈10^30×1.024"},
    {"q": "Термин «чёрная дыра» популяризовал…", "opts": ["Дж. Уилер", "С. Хокинг", "А. Эйнштейн"], "a": 0, "ex": "Джон Уилер, 1960-е"},
    {"q": "JVM в основе написана на…", "opts": ["Java", "C/C++", "Rust"], "a": 1, "ex": "На C/C++"},
]

# ---------- Helpers ----------
def now_kyiv():
    return datetime.now(tz=KYIV_TZ).strftime("%Y-%m-%d %H:%M")

def is_brother(update: Update) -> bool:
    if not BROTHER_USERNAME:
        return False
    u = update.effective_user
    if not u or not u.username:
        return False
    return ("@" + u.username) == BROTHER_USERNAME

class RateLimiter:
    def __init__(self, per_min=RATE_LIMIT_PER_MIN):
        self.per_min = per_min
        self.tokens = {}  # user_id -> (tokens, ts)

    def check(self, uid: int) -> bool:
        import time
        now = time.time()
        t, last = self.tokens.get(uid, (self.per_min, now))
        elapsed = now - last
        t = min(self.per_min, t + elapsed * (self.per_min / 60.0))
        if t < 1:
            self.tokens[uid] = (t, now)
            return False
        self.tokens[uid] = (t - 1, now)
        return True

rate = RateLimiter()

# ---------- Handlers ----------
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    kb = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎯 Викторина", callback_data="quiz:start"),
         InlineKeyboardButton("🧠 Факт", callback_data="fact:menu")],
        [InlineKeyboardButton("🤖 Спросить ИИ", callback_data="ask:hint"),
         InlineKeyboardButton("🪄 Объясни", callback_data="explain:hint")],
    ])
    text = (
        f"Привет, {update.effective_user.first_name or 'друг'}! Я Умный Бот 2.0 🤓\n"
        f"Сейчас в Киеве {now_kyiv()}.\n\n"
        "Команды:\n"
        "/fact — умный факт по теме\n"
        "/quiz — интеллектуальная викторина\n"
        "/ask вопрос — ответ «как ChatGPT»\n"
        "/explain тема — объясню просто\n"
        "/roast — лёгкая подколка\n"
        "/help — помощь"
    )
    await update.message.reply_text(text, reply_markup=kb)

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "• /fact [наука|история|математика]\n"
        "• /quiz — квиз\n"
        "• /ask Как работает квантовый компьютер?\n"
        "• /explain Индексы в БД — просто\n"
        "• /roast — шутка (для брата — особый режим)"
    )

async def send_fact(update_or_cb, topic: str, edit=False):
    msg = f"🧠 {topic.capitalize()}: {random.choice(FACTS[topic])}"
    if hasattr(update_or_cb, "message") and update_or_cb.message:
        await update_or_cb.message.reply_text(msg)
    else:
        if edit:
            await update_or_cb.edit_message_text(msg)
        else:
            await update_or_cb.message.reply_text(msg)

async def cmd_fact(update: Update, context: ContextTypes.DEFAULT_TYPE):
    args = context.args or []
    if args and args[0].lower() in FACTS:
        await send_fact(update, args[0].lower())
        return
    kb = InlineKeyboardMarkup([[
        InlineKeyboardButton("Наука", callback_data="fact:наука"),
        InlineKeyboardButton("История", callback_data="fact:история"),
        InlineKeyboardButton("Математика", callback_data="fact:математика"),
    ]])
    await update.message.reply_text("Выбери раздел:", reply_markup=kb)

async def cmd_roast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target = "брат" if is_brother(update) else "собеседник"
    lines = [
        f"Кажется, мой {target} опять пытается меня переплюнуть… смело 😌",
        "Я не говорю, что я умнее всех — просто статистика упорно кивает.",
        "Могу объяснить проще — включу режим «для тех, кто смотрит туториалы на 2×».",
    ]
    await update.message.reply_text("😏 " + random.choice(lines))

async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Напиши так: /ask твой вопрос.")
        return
    if not rate.check(update.effective_user.id):
        await update.message.reply_text("Погоди секунду, не спамим. Попробуй чуть позже.")
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    q = " ".join(context.args)
    ans = await llm_answer(q, sys="You are a sharp, helpful assistant. Answer concisely with examples when useful.")
    await update.message.reply_text(ans)

async def cmd_explain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Напиши так: /explain тема или текст.")
        return
    if not rate.check(update.effective_user.id):
        await update.message.reply_text("Чуть-чуть подождём. Ещё мгновение.")
        return
    await update.message.chat.send_action(ChatAction.TYPING)
    topic = " ".join(context.args)
    prompt = "Объясни простым языком с 1 аналогией и 3 шагами для разборки. Тема: " + topic
    ans = await llm_answer(prompt)
    await update.message.reply_text(ans)

async def cmd_quiz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.setdefault("quiz", {"i": 0, "score": 0})
    context.user_data["quiz"]["order"] = random.sample(range(len(QUIZ)), len(QUIZ))
    await send_quiz_q(update, context)

async def send_quiz_q(update_or_cb, context: ContextTypes.DEFAULT_TYPE, edit=False):
    st = context.user_data.get("quiz", {"i": 0, "score": 0, "order": list(range(len(QUIZ)))})
    if st["i"] >= len(QUIZ):
        msg = f"Итог: {st['score']}/{len(QUIZ)}. {'Неплохо!' if st['score'] >= len(QUIZ)*0.6 else 'Потренируемся ещё?'}"
        if hasattr(update_or_cb, "message") and update_or_cb.message:
            await update_or_cb.message.reply_text(msg)
        else:
            await update_or_cb.edit_message_text(msg)
        context.user_data["quiz"] = {"i": 0, "score": 0, "order": random.sample(range(len(QUIZ)), len(QUIZ))}
        return
    q_idx = st["order"][st["i"]]
    item = QUIZ[q_idx]
    kb = InlineKeyboardMarkup([[InlineKeyboardButton(opt, callback_data=f"quiz:ans:{n}") for n, opt in enumerate(item["opts"])]])
    text = f"🎯 Вопрос {st['i']+1}/{len(QUIZ)}:\n{item['q']}"
    if hasattr(update_or_cb, "message") and update_or_cb.message:
        await update_or_cb.message.reply_text(text, reply_markup=kb)
    else:
        if edit:
            await update_or_cb.edit_message_text(text, reply_markup=kb)
        else:
            await update_or_cb.message.reply_text(text, reply_markup=kb)

async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    data = q.data
    if data.startswith("fact:"):
        _, t = data.split(":", 1)
        if t == "menu":
            kb = InlineKeyboardMarkup([[
                InlineKeyboardButton("Наука", callback_data="fact:наука"),
                InlineKeyboardButton("История", callback_data="fact:история"),
                InlineKeyboardButton("Математика", callback_data="fact:математика"),
            ]])
            await q.edit_message_text("Выбери раздел:", reply_markup=kb)
        else:
            await send_fact(q, t, edit=True)
    elif data.startswith("quiz:"):
        parts = data.split(":")
        if len(parts) == 2 and parts[1] == "start":
            context.user_data["quiz"] = {"i": 0, "score": 0, "order": random.sample(range(len(QUIZ)), len(QUIZ))}
            await send_quiz_q(q, context, edit=True)
        elif len(parts) == 3 and parts[1] == "ans":
            idx = int(parts[2])
            st = context.user_data["quiz"]
            q_idx = st["order"][st["i"]]
            item = QUIZ[q_idx]
            correct = (idx == item["a"])
            if correct:
                st["score"] += 1
            st["i"] += 1
            feedback = ("✅ Верно! " if correct else "❌ Не совсем. ") + item["ex"]
            await q.edit_message_text(feedback)
            await send_quiz_q(q, context, edit=False)
    elif data.startswith("ask:"):
        await q.edit_message_text("Напиши: /ask твой вопрос — отвечу умно и по делу.")
    elif data.startswith("explain:"):
        await q.edit_message_text("Напиши: /explain тема — объясню простыми словами.")

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = update.message.text.strip().lower()
    if re.search(r"\b(кто\s+умн|умнее)\b", txt):
        await update.message.reply_text("Очевидно, я 🤖. Но и ты хорош, просто дай мне блеснуть.")
        return
    if txt.endswith("?") and len(txt) > 5:
        await cmd_ask(update, context)
        return
    topic = random.choice(list(FACTS.keys()))
    await update.message.reply_text(f"Интересно 🤔. А пока вот факт из «{topic}»: {random.choice(FACTS[topic])}")

def build_app():
    persistence = PicklePersistence(filepath="smart_state.pkl")
    app = ApplicationBuilder().token(TOKEN).persistence(persistence).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("fact", cmd_fact))
    app.add_handler(CommandHandler("quiz", cmd_quiz))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("explain", cmd_explain))
    app.add_handler(CommandHandler("roast", cmd_roast))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    return app

def main():
    if not TOKEN or TOKEN == "PASTE_TOKEN_HERE":
        raise SystemExit("Укажи TELEGRAM_BOT_TOKEN в переменных окружения.")
    app = build_app()

    # Webhook mode for Render Web Service
    if not WEBHOOK_BASE:
        raise SystemExit("Не задан WEBHOOK_BASE/RENDER_EXTERNAL_URL. На Render он появляется автоматически после деплоя. Перезапусти после первого старта.")
    webhook_path = f"/webhook/{TOKEN}"
    webhook_url = f"{WEBHOOK_BASE}{webhook_path}"
    log.info("Starting webhook server on port %s, path %s", PORT, webhook_path)
    log.info("Setting webhook to %s", webhook_url)
    app.run_webhook(
        listen="0.0.0.0",
        port=PORT,
        url_path=webhook_path,
        webhook_url=webhook_url,
        drop_pending_updates=True,
    )

if __name__ == "__main__":
    main()
