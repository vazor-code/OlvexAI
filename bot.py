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

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

# Загрузка переменных окружения
load_dotenv()

# Цветной логгер
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\x1b[36m', 'WARNING': '\x1b[33m', 'ERROR': '\x1b[31m',
        'CRITICAL': '\x1b[35m', 'DEBUG': '\x1b[37m', 'RESET': '\x1b[0m'
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
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

# ✅ Исправлено: убраны пробелы в URL
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=AI_TOKEN)

# Настройка бота
dp = Dispatcher()
router = Router()
openai_semaphore = asyncio.Semaphore(3)

# ---------------------------
# Парсинг кода и HTML
# ---------------------------

_code_block_pattern = re.compile(r'```(\w*)\s*\n(.*?)\n```', re.DOTALL | re.IGNORECASE)
_link_pattern = re.compile(r'<a\s+href\s*=\s*["\']([^"\']*)["\']\s*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_b_pattern = re.compile(r'<b>(.*?)</b>', re.IGNORECASE | re.DOTALL)
_i_pattern = re.compile(r'<i>(.*?)</i>', re.IGNORECASE | re.DOTALL)
_u_pattern = re.compile(r'<u>(.*?)</u>', re.IGNORECASE | re.DOTALL)
_s_pattern = re.compile(r'<s>(.*?)</s>', re.IGNORECASE | re.DOTALL)

ALLOWED_SCHEMES = ("http", "https")

def _is_safe_href(href: str) -> bool:
    try:
        parsed = urlparse(href)
        return parsed.scheme.lower() in ALLOWED_SCHEMES and bool(parsed.netloc)
    except Exception:
        return False

def extract_code_blocks(text: str):
    """Извлекает блоки кода, возвращает (остальной_текст, [(язык, код)])"""
    codes = []
    parts = []
    last_end = 0

    for match in _code_block_pattern.finditer(text):
        if match.start() > last_end:
            parts.append(('text', text[last_end:match.start()]))
        lang = match.group(1).strip() or "txt"
        code = match.group(2).strip('\n')
        codes.append((lang, code))
        last_end = match.end()

    if last_end < len(text):
        parts.append(('text', text[last_end:]))

    return parts, codes

def sanitize_ai_html(text: str) -> str:
    """Оставляет только безопасные HTML-теги."""
    placeholders = {}

    def make_token():
        return f"__PH_{uuid.uuid4().hex}__"

    temp = text
    temp = _link_pattern.sub(lambda m: (t := make_token(), placeholders.update({t: ("a", m.group(1), m.group(2))}) or t)[-1], temp)
    temp = _b_pattern.sub(lambda m: (t := make_token(), placeholders.update({t: ("b", m.group(1))}) or t)[-1], temp)
    temp = _i_pattern.sub(lambda m: (t := make_token(), placeholders.update({t: ("i", m.group(1))}) or t)[-1], temp)
    temp = _u_pattern.sub(lambda m: (t := make_token(), placeholders.update({t: ("u", m.group(1))}) or t)[-1], temp)
    temp = _s_pattern.sub(lambda m: (t := make_token(), placeholders.update({t: ("s", m.group(1))}) or t)[-1], temp)

    escaped = html.escape(temp)

    for token, data in placeholders.items():
        tag = data[0]
        if tag == "a":
            href, inner = data[1], data[2]
            inner_esc = html.escape(inner)
            if _is_safe_href(href):
                href_esc = html.escape(href, quote=True)
                replacement = f'<a href="{href_esc}">{inner_esc}</a>'
            else:
                replacement = inner_esc
        elif tag == "b":
            replacement = f"<b>{html.escape(data[1])}</b>"
        elif tag == "i":
            replacement = f"<i>{html.escape(data[1])}</i>"
        elif tag == "u":
            replacement = f"<u>{html.escape(data[1])}</u>"
        elif tag == "s":
            replacement = f"<s>{html.escape(data[1])}</s>"
        else:
            replacement = html.escape(str(data))
        escaped = escaped.replace(token, replacement)

    return escaped


# ---------------------------
# Хэндлеры
# ---------------------------

@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    user = message.from_user
    safe_name = html.escape(user.full_name or "Пользователь")
    welcome_text = (
        f"Привет, <b>{safe_name}</b>! 👋\n\n"
        "🧠 Я — ваш ИИ-ассистент.\n"
        "Задавайте вопросы, просите написать код, объяснить тему.\n\n"
        "<i>Код будет отображаться красиво — его можно копировать одним кликом.</i>"
    )
    logger.info(f"👋 /start от {user.full_name}")
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


# Защита от флуда
FLOOD_COOLDOWN = 1.5
user_last_message = {}

@router.message(F.text)
async def echo_handler(message: Message) -> None:
    user_id = message.from_user.id
    now = asyncio.get_event_loop().time()

    if user_id in user_last_message and now - user_last_message[user_id] < FLOOD_COOLDOWN:
        return
    user_last_message[user_id] = now

    user_text = message.text.strip()
    logger.info(f"📨 От {message.from_user.full_name}: {user_text}")

    thinking_msg = await message.answer("💭 Думаю...")

    try:
        async with openai_semaphore:
            completion = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Ты — ИИ-ассистент. Всегда отвечай на русском языке.\n"
                                "• Для кода используй формат: ```py\\nкод\\n```\n"
                                "• Для ссылок: <a href='https://example.com'>текст</a>\n"
                                "• Жирный: <b>текст</b>, курсив: <i>текст</i>\n"
                                "• Никогда не используй <pre> или <code> в ответе."
                            ),
                        },
                        {"role": "user", "content": user_text},
                    ],
                    temperature=0.7,
                )
            )

        ai_reply = completion.choices[0].message.content.strip()
        non_code_parts, code_blocks = extract_code_blocks(ai_reply)

        await thinking_msg.delete()

        # Сначала отправляем обычный текст (с HTML-разметкой)
        if non_code_parts:
            full_text = ''.join(part for _, part in non_code_parts)
            clean_text = sanitize_ai_html(full_text)
            MAX_LEN = 4096
            for i in range(0, len(clean_text), MAX_LEN):
                chunk = clean_text[i:i + MAX_LEN]
                await message.answer(chunk, parse_mode=ParseMode.HTML)

                # Отправка блоков кода с разбивкой на части
        MAX_MESSAGE_LENGTH = 4096 - 100  # запас на теги

        for lang, code in code_blocks:
            code_lines = code.strip().splitlines()
            chunk = ""
            current_length = 0

            def make_code_block(text):
                return f'<pre><code class="language-{html.escape(lang)}">{html.escape(text)}</code></pre>'

            for line in code_lines:
                line_escaped = html.escape(line) + "\n"
                if current_length + len(line_escaped) > MAX_MESSAGE_LENGTH:
                    # Отправляем текущий кусок
                    if chunk:
                        await message.answer(make_code_block(chunk), parse_mode=ParseMode.HTML)
                    # Начинаем новый
                    chunk = line
                    current_length = len(line_escaped)
                else:
                    chunk += line + "\n"
                    current_length += len(line_escaped)

            # Отправляем остаток
            if chunk.strip():
                await message.answer(make_code_block(chunk), parse_mode=ParseMode.HTML)

        logger.info("✅ Ответ отправлен (код как <pre><code>)")

    except Exception as e:
        logger.error(f"Ошибка: {e}")
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await message.answer("⚠️ Ошибка генерации. Попробуйте позже.")


# ---------------------------
# Запуск бота
# ---------------------------

async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp.include_router(router)
    logger.info("🚀 Бот запущен!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("💤 Бот остановлен.")
    except Exception as e:
        logger.critical(f"🛑 Ошибка: {e}")