import asyncio
import logging
import sys
from os import getenv
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import re
import html
import uuid
from urllib.parse import urlparse
import json
from pathlib import Path
from typing import List
from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, FSInputFile

# Загрузка переменных окружения
load_dotenv()

# Путь к файлу с ID пользователей
USERS_FILE = Path("user_ids.json")

def load_user_ids():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, 'r') as f:
                return set(int(x) for x in json.load(f))
        except (json.JSONDecodeError, ValueError):
            pass
    return set()

def save_user_ids(user_ids_set):
    with open(USERS_FILE, 'w') as f:
        json.dump(sorted(list(user_ids_set)), f)

# Глобальные переменные
user_ids = load_user_ids()
user_context = {}
user_model = {}
user_last_message = {}
FLOOD_COOLDOWN = 1.5
MAX_CONTEXT_MESSAGES = 10

# Укажите ваш Telegram ID!
ADMIN_ID = 1680340118  # ⚠️ ЗАМЕНИТЕ НА СВОЙ ID!

# Цветной логгер
class ColoredFormatter(logging.Formatter):
    COLORS = {
        "INFO": "\x1b[36m", "WARNING": "\x1b[33m", "ERROR": "\x1b[31m",
        "CRITICAL": "\x1b[35m", "DEBUG": "\x1b[37m", "RESET": "\x1b[0m"
    }
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{color}⚡️[{timestamp}] {record.levelname:>8}{reset} [{record.name}] {record.getMessage()}"

logger = logging.getLogger("OlvexAI_Bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)
logging.getLogger("aiogram").handlers.clear()
logging.getLogger("aiogram").addHandler(handler)
logging.getLogger("aiogram").setLevel(logging.INFO)

# Проверка токенов
TOKEN = getenv("BOT_TOKEN")
AI_TOKEN = getenv("AI_TOKEN")
if not TOKEN or not AI_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN или AI_TOKEN не найден. Проверьте .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_TOKEN
)

# Настройка бота
dp = Dispatcher()
router = Router()
openai_semaphore = asyncio.Semaphore(3)

# ---------------------------
# Markdown → HTML
# ---------------------------

_markdown_code_pattern = re.compile(r"```(\w*)\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)
_link_pattern = re.compile(r'<a\s+href\s*=\s*["\']([^"\']*)["\']\s*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_b_pattern = re.compile(r"<b>(.*?)</b>", re.IGNORECASE | re.DOTALL)
_i_pattern = re.compile(r"<i>(.*?)</i>", re.IGNORECASE | re.DOTALL)
_u_pattern = re.compile(r"<u>(.*?)</u>", re.IGNORECASE | re.DOTALL)
_s_pattern = re.compile(r"<s>(.*?)</s>", re.IGNORECASE | re.DOTALL)

ALLOWED_SCHEMES = ("http", "https")

def _is_safe_href(href: str) -> bool:
    try:
        parsed = urlparse(href)
        return parsed.scheme.lower() in ALLOWED_SCHEMES and bool(parsed.netloc)
    except Exception:
        return False

import markdown2
from urllib.parse import urlparse

def _is_safe_href(href: str) -> bool:
    try:
        parsed = urlparse(href)
        return parsed.scheme.lower() in ALLOWED_SCHEMES and bool(parsed.netloc)
    except Exception:
        return False

# вставьте это вместо вашей markdown_to_html_safe + отправки по кускам

TG_LIMIT = 4096

def markdown_to_html_safe(text: str) -> str:
    """
    Преобразует Markdown (включая ```блоки``` и `inline`) в безопасный HTML,
    экранирует угловые скобки вне тегов, сохраняет <pre><code>...</code></pre> и <code>...</code>.
    """
    text = (text or "").strip()

    # --- Извлекаем и экранируем блоки кода и inline-код, заменяя на плейсхолдеры
    code_blocks = []

    def repl_backticks(m):
        inner = m.group(1)
        escaped = html.escape(inner)
        code_blocks.append(f"<pre><code>{escaped}</code></pre>")
        return f"@@CODE_{len(code_blocks)-1}@@"

    text = re.sub(r"```(?:\w*)\n(.*?)```", repl_backticks, text, flags=re.DOTALL)

    # --- Заменяем заголовки ## и ### на ▎ ---
    text = re.sub(r"^###\s*(.*)", r"▎ \1", text, flags=re.MULTILINE)
    text = re.sub(r"^##\s*(.*)", r"▎ \1", text, flags=re.MULTILINE)

    def repl_inline(m):
        inner = m.group(1)
        escaped = html.escape(inner)
        code_blocks.append(f"<code>{escaped}</code>")
        return f"@@CODE_{len(code_blocks)-1}@@"

    text = re.sub(r"`([^`]+?)`", repl_inline, text, flags=re.DOTALL)

    # --- Преобразования Markdown для НЕ-кодовой части ---
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text, flags=re.DOTALL)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text, flags=re.DOTALL)
    text = re.sub(r"^\s*[-*]\s+(.*)", r"• \1", text, flags=re.MULTILINE)

    def sanitize_links(match):
        inner = match.group(1)
        href = match.group(2)
        if _is_safe_href(href):
            return f'<a href="{html.escape(href, quote=True)}">{html.escape(inner)}</a>'
        else:
            return html.escape(inner)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", sanitize_links, text, flags=re.DOTALL)

    text = re.sub(r"</?(p|div|span|br|hr|h\d)[^>]*>", "", text, flags=re.IGNORECASE)

    # --- Экранируем угловые скобки только вне тегов и плейсхолдеров
    def escape_outside(m):
        tag, txt = m.group(1), m.group(2)
        if tag:
            return tag
        else:
            return txt.replace("<", "&lt;").replace(">", "&gt;")
    # Разделяем на теги и текст; теги (включая те, что мы хотим оставить) пропускаем через group(1)
    text = re.sub(r"(<[^>]+>)|([^<>]+)", lambda m: escape_outside(m), text)

    # --- Вернём кодовые блоки
    for i, block in enumerate(code_blocks):
        text = text.replace(f"@@CODE_{i}@@", block)

    text = text.replace("\r\n", "\n")  # нормализуем переводы строк
    text = re.sub(r"\n{2,}", "\n\n", text)  # минимум две строки для абзаца
    return text.strip()


def split_message_preserve_code(html_text: str, limit: int = TG_LIMIT) -> List[str]:
    """
    Разбивает html_text на части длиной <= limit.
    Не разрывает внутри <pre><code>...</code></pre>.
    Если отдельный кодовый блок > limit, разрезает его на несколько корректных <pre><code>..</code></pre>.
    """
    parts: List[str] = []

    # Найдём все <pre><code>...</code></pre>
    code_re = re.compile(r"<pre><code>.*?</code></pre>", flags=re.DOTALL | re.IGNORECASE)
    segments = []
    last = 0
    for m in code_re.finditer(html_text):
        if m.start() > last:
            segments.append(("text", html_text[last:m.start()]))
        segments.append(("code", m.group(0)))
        last = m.end()
    if last < len(html_text):
        segments.append(("text", html_text[last:]))

    cur = ""

    def flush_cur():
        nonlocal cur
        if cur:
            parts.append(cur)
            cur = ""

    for kind, seg in segments:
        if kind == "text":
            # пытаемся добавить сегмент целиком, иначе разбиваем аккуратно
            if len(cur) + len(seg) <= limit:
                cur += seg
            else:
                remaining = seg
                while remaining:
                    space_left = limit - len(cur)
                    if space_left <= 0:
                        flush_cur()
                        space_left = limit
                    if len(remaining) <= space_left:
                        cur += remaining
                        remaining = ""
                    else:
                        # стараемся разрезать по последнему переводу строки внутри лимита
                        cut = remaining.rfind("\n", 0, space_left)
                        if cut <= 0:
                            # fallback: найдем безопасную позицию (чтобы не разорвать тег)
                            cut = space_left
                            safe_cut = None
                            # откатываемся до 500 символов в поиске безопасного края
                            start_check = max(0, cut - 500)
                            prefix = cur
                            for idx in range(cut, start_check - 1, -1):
                                seg_prefix = prefix + remaining[:idx]
                                if seg_prefix.count("<") == seg_prefix.count(">"):
                                    safe_cut = idx
                                    break
                            if safe_cut:
                                cut = safe_cut
                        cur += remaining[:cut]
                        remaining = remaining[cut:]
                        flush_cur()
        else:  # code
            # если код помещается в текущую часть — добавляем
            if len(cur) + len(seg) <= limit:
                cur += seg
            else:
                # сначала сбрасываем накопленное
                flush_cur()
                # извлечём содержимое кода (без тегов)
                inner = re.sub(r"(?i)^<pre><code>|</code></pre>$", "", seg)
                # теперь разбиваем содержимое на куски, чтобы каждый фрагмент влезал
                overhead = len("<pre><code></code></pre>")
                max_chunk = max(1, limit - overhead)
                start = 0
                while start < len(inner):
                    chunk = inner[start:start + max_chunk]
                    parts.append(f"<pre><code>{chunk}</code></pre>")
                    start += max_chunk

    flush_cur()
    return parts


def clean_ai_response(text: str) -> str:
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"\[\/?OUT\]", "", text)
    text = re.sub(r"<<SYS>>.*?<<\/SYS>>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|im_start\|>.*?<\|im_end\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"^<s>\s*", "", text)
    text = re.sub(r"\s*</s>$", "", text)
    text = re.sub(r"<[^>]+>", "", text)  # Remove any HTML tags to prevent parsing errors
    return text.strip()

# ---------------------------
# Модели
# ---------------------------

AVAILABLE_MODELS = {
    "deepseek": "deepseek/deepseek-chat-v3.1",
    "r1": "deepseek/deepseek-r1-0528",
    "qwen8b": "deepseek/deepseek-r1-0528-qwen3-8b",
    "mistral": "mistralai/mistral-small-3.2-24b-instruct",
    "qwen": "qwen/qwen3-coder",
    "gemma": "google/gemma-3n-e4b-it",
    "gpt20b": "openai/gpt-oss-20b",
}


MODEL_NAMES = {
    "deepseek": "🧠 DeepSeek Chat v3.1",
    "r1": "🚀 DeepSeek R1 (0528)",
    "qwen8b": "🧩 DeepSeek + Qwen3 8B",
    "mistral": "🔥 Mistral Small 24B",
    "qwen": "💻 Qwen3 Coder",
    "gemma": "🟢 Google Gemma 3n E4B",
    "gpt20b": "🔵 GPT-OSS 20B",
}

# ---------------------------
# Команды
# ---------------------------

@router.message(CommandStart())
async def command_start_handler(message: Message):
    user = message.from_user
    if user.id not in user_ids:
        user_ids.add(user.id)
        save_user_ids(user_ids)
        logger.info(f"📥 Новый пользователь: {user.full_name} (ID: {user.id})")
    safe_name = html.escape(user.full_name or "Пользователь")
    text = (
        f"Привет, <b>{safe_name}</b>! 👋\n\n"
        "🧠 Я — ваш ИИ-ассистент.\n"
        "Задавайте вопросы, просите написать код, объяснить тему.\n\n"
        "📌 Используйте /help — чтобы узнать все возможности."
    )
    await message.answer(text, parse_mode=ParseMode.HTML)

@router.message(F.text.startswith("/help"))
async def help_command(message: Message):
    models_list = "\n".join([f"🔸 <code>/{key}</code> — {value}" for key, value in MODEL_NAMES.items()])
    text = (
        "📚 <b>OlvexAI | DeepSeek</b>\n"
        "Умный бот, готовый ответить на любые вопросы.\n\n"
        "🔹 <b>Команды:</b>\n"
        "🔸 /start — начать\n"
        "🔸 /help — это меню\n"
        "🔸 /model — текущая модель\n"
        "🔸 /clear — очистить историю диалога\n"
        "🔸 /retry — перегенерировать последний ответ\n"
        "🔸 /stats — статистика (только для админа)\n"
        "🔸 /broadcast — рассылка (только для админа)\n\n"
        f"🔹 <b>Модели:</b>\n{models_list}\n\n"
        "📎 Прикрепите файл (.txt, .py, .js и др.) — я прочитаю и объясню.\n\n"
        "💡 Бот запоминает контекст — можно вести диалог.\n"
        "⚠️ Сервера могут временно отключаться.\n"
        "👤 Разработка: @vazor_code"
    )
    await message.answer(text, parse_mode=ParseMode.HTML)

@router.message(F.text.startswith("/model"))
async def show_model(message: Message):
    user_id = message.from_user.id
    model = user_model.get(user_id, "deepseek/deepseek-chat-v3.1")
    key = next((k for k, v in AVAILABLE_MODELS.items() if v == model), "deepseek")
    await message.answer(f"📦 Текущая модель: <b>{MODEL_NAMES[key]}</b>", parse_mode=ParseMode.HTML)

# --- Переключение моделей ---
@router.message(F.text.regexp(r"^/(deepseek|r1|qwen8b|mistral|qwen|qwen4b|gemma|gpt20b)$"))
async def quick_switch(message: Message):
    key = message.text.lstrip("/")
    user_model[message.from_user.id] = AVAILABLE_MODELS[key]
    await message.answer(f"✅ Модель переключена на <b>{MODEL_NAMES[key]}</b>", parse_mode=ParseMode.HTML)

# --- Очистка контекста ---
@router.message(F.text == "/clear")
async def clear_context(message: Message):
    user_context.pop(message.from_user.id, None)
    await message.answer("🧹 Контекст очищен!")


# --- Админ-команды ---
@router.message(F.text.startswith("/broadcast"))
async def broadcast_message(message: Message):
    global bot
    if message.from_user.id != ADMIN_ID:
        await message.answer("❌ У вас нет прав на рассылку.")
        return
    if bot is None:
        await message.answer("⚠️ Бот ещё не инициализирован.")
        return

    text = message.text[len("/broadcast"):].strip()
    if not text:
        await message.answer("📝 Введите текст после команды.")
        return

    await message.answer("🚀 Рассылка начата...")
    success = blocked = errors = 0
    for uid in user_ids:
        try:
            await bot.send_message(uid, text, parse_mode=ParseMode.HTML)
            success += 1
            await asyncio.sleep(0.05)
        except Exception as e:
            if "BotBlocked" in str(e) or "ChatNotFound" in str(e):
                blocked += 1
            else:
                errors += 1
    await message.answer(f"📬 Рассылка завершена!\n✅ {success} | 🚫 {blocked} | ⚠️ {errors}")


@router.message(F.text == "/stats")
async def show_stats(message: Message):
    global bot
    if message.from_user.id != ADMIN_ID:
        return
    if bot is None:
        await message.answer("⚠️ Бот ещё не инициализирован.")
        return

    total_users = len(user_ids)
    active_chats = len(user_context)
    model_usage = {}
    for uid, model in user_model.items():
        key = next((k for k, v in AVAILABLE_MODELS.items() if v == model), "unknown")
        model_usage[key] = model_usage.get(key, 0) + 1
    usage_list = "\n".join([f"🔸 {MODEL_NAMES.get(k, k)}: {v}" for k, v in sorted(model_usage.items(), key=lambda x: -x[1])])
    await message.answer(
        f"📊 <b>Статистика</b>\n\n👥 Пользователей: {total_users}\n💬 Диалогов: {active_chats}\n\n<b>Модели:</b>\n{usage_list}",
        parse_mode=ParseMode.HTML
    )

# --- Повтор ---
@router.message(F.text == "/retry")
async def retry_last(message: Message):
    user_id = message.from_user.id
    if user_id not in user_context or len(user_context[user_id]) < 2:
        await message.answer("⚠️ Нет предыдущего сообщения для повтора.")
        return
    last_user_message = next((m["content"] for m in reversed(user_context[user_id]) if m["role"] == "user"), None)
    if not last_user_message:
        await message.answer("⚠️ Не найдено последнее сообщение пользователя.")
        return
    await echo_handler(Message(message_id=message.message_id, from_user=message.from_user, text=last_user_message, chat=message.chat))

# --- Отправка файлов ---
@router.message(F.document)
async def handle_document(message: Message):
    doc = message.document
    if not doc.file_name.lower().endswith((".txt", ".py", ".js", ".md", ".json")):
        await message.answer("⚠️ Поддерживаются только текстовые файлы.")
        return
    file = await message.bot.get_file(doc.file_id)
    path = await message.bot.download_file(file.file_path)
    content = path.read().decode("utf-8", errors="ignore")
    user_context.setdefault(message.from_user.id, []).append({"role": "user", "content": f"Файл {doc.file_name}:\n{content}"})
    await message.answer("📄 Файл получен! Я учту его в контексте.")

# --- Основной обработчик ---
@router.message(F.text)
async def echo_handler(message: Message):
    user_id = message.from_user.id
    user = message.from_user
    if user.id not in user_ids:
        await message.answer(
            "⚠️ <b>Привет!</b>\n\n"
            "Чтобы бот начал сохранять ваш <b>контекст</b> и <b>статистику</b>, "
            "пожалуйста, напишите команду <code>/start</code>.",
            parse_mode=ParseMode.HTML
        )
        user_ids.add(user.id)
        save_user_ids(user_ids)
        return  # дальше не обрабатываем старое сообщение
    now = asyncio.get_event_loop().time()
    if user_id in user_last_message and now - user_last_message[user_id] < FLOOD_COOLDOWN:
        return
    user_last_message[user_id] = now

    user_text = message.text.strip()
    logger.info(f"📨 От {message.from_user.full_name}: {user_text}")

    thinking_msg = None
    try:
        model = user_model.get(user_id, "deepseek/deepseek-chat-v3.1")
        user_context.setdefault(user_id, []).append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": "Ты — умный ассистент, отвечай по-русски, красиво форматируй код."}] + user_context[user_id]
        thinking_msg = await message.answer("💭 Думаю...")

        async with openai_semaphore:
            completion = await asyncio.to_thread(lambda: client.chat.completions.create(model=model, messages=messages, temperature=0.7))
        raw = completion.choices[0].message.content.strip()
        cleaned = clean_ai_response(raw)
        user_context[user_id].append({"role": "assistant", "content": cleaned})
        if len(user_context[user_id]) > MAX_CONTEXT_MESSAGES * 2:
            user_context[user_id] = user_context[user_id][:1] + user_context[user_id][-MAX_CONTEXT_MESSAGES*2:]
        html_reply = markdown_to_html_safe(cleaned)
        await thinking_msg.delete()

        safe_parts = split_message_preserve_code(html_reply, TG_LIMIT)
        for part in safe_parts:
            await message.answer(part, parse_mode=ParseMode.HTML)
        logger.info("✅ Ответ отправлен")
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        if thinking_msg:
            try:
                await thinking_msg.delete()
            except Exception:
                pass
        await message.answer("⚠️ Ошибка генерации. Попробуйте позже.")


# --- Запуск ---
async def main():
    global bot
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp.include_router(router)
    logger.info(f"🚀 OlvexAI Bot запущен! Админ: {ADMIN_ID}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("💤 Бот остановлен.")
    except Exception as e:
        logger.critical(f"🛑 Ошибка: {e}")
