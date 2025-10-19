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

from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

load_dotenv()

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\x1b[36m',
        'WARNING': '\x1b[33m',
        'ERROR': '\x1b[31m',
        'CRITICAL': '\x1b[35m',
        'DEBUG': '\x1b[37m',
        'RESET': '\x1b[0m'
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{log_color}⚡️[{timestamp}] {record.levelname:>8}{reset} [{record.name}] {record.getMessage()}"

logger = logging.getLogger("OlvexAI_Bot")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

logging.getLogger("aiogram").handlers.clear()
logging.getLogger("aiogram").addHandler(handler)
logging.getLogger("aiogram").setLevel(logging.INFO)

TOKEN = getenv("BOT_TOKEN")
AI_TOKEN = getenv("AI_TOKEN")
if not TOKEN or not AI_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN или AI_TOKEN не найден. Проверьте .env")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=AI_TOKEN)

dp = Dispatcher()
router = Router()
openai_semaphore = asyncio.Semaphore(3)

# ---------------------------
# Sanitizer: разрешаем только <b>, <i>, <u>, <s>, <a href="...">...</a>
# ---------------------------

# Паттерны (нежадные)
_link_pattern = re.compile(r'<a\s+href\s*=\s*["\']([^"\']*)["\']\s*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
_b_pattern = re.compile(r'<b>(.*?)</b>', re.IGNORECASE | re.DOTALL)
_i_pattern = re.compile(r'<i>(.*?)</i>', re.IGNORECASE | re.DOTALL)
_u_pattern = re.compile(r'<u>(.*?)</u>', re.IGNORECASE | re.DOTALL)
_s_pattern = re.compile(r'<s>(.*?)</s>', re.IGNORECASE | re.DOTALL)

ALLOWED_SCHEMES = ("http", "https")

def _is_safe_href(href: str) -> bool:
    try:
        parsed = urlparse(href)
        if parsed.scheme.lower() in ALLOWED_SCHEMES and parsed.netloc:
            return True
    except Exception:
        return False
    return False

def sanitize_ai_html(text: str) -> str:
    """
    Принимает HTML-строку от ИИ и возвращает безопасный HTML,
    содержащий только разрешённые теги (<b>, <i>, <u>, <s>, <a href="...">).
    Всё остальное — эскейпится.
    """

    placeholders = {}
    # Уникальный токен генератор
    def _make_token() -> str:
        return f"__PH_{uuid.uuid4().hex}__"

    # 1) Вырежем разрешённые теги, сохраним их в плейсхолдерах.
    def _save_link(m):
        href = m.group(1)
        inner = m.group(2) or ""
        token = _make_token()
        placeholders[token] = ("a", href, inner)
        return token

    def _save_b(m):
        inner = m.group(1) or ""
        token = _make_token()
        placeholders[token] = ("b", inner)
        return token

    def _save_i(m):
        inner = m.group(1) or ""
        token = _make_token()
        placeholders[token] = ("i", inner)
        return token

    def _save_u(m):
        inner = m.group(1) or ""
        token = _make_token()
        placeholders[token] = ("u", inner)
        return token

    def _save_s(m):
        inner = m.group(1) or ""
        token = _make_token()
        placeholders[token] = ("s", inner)
        return token

    # Применяем по порядку (ссылки сначала)
    temp = _link_pattern.sub(_save_link, text)
    temp = _b_pattern.sub(_save_b, temp)
    temp = _i_pattern.sub(_save_i, temp)
    temp = _u_pattern.sub(_save_u, temp)
    temp = _s_pattern.sub(_save_s, temp)

    # 2) Эскейпим всё, что осталось (это удалит любые незапрошенные теги)
    escaped = html.escape(temp)

    # 3) Вставляем назад безопасные теги (с эскейпом их содержимого и валидацией href)
    for token, data in placeholders.items():
        if data[0] == "a":
            _, href, inner = data
            inner_esc = html.escape(inner)
            if _is_safe_href(href):
                href_esc = html.escape(href, quote=True)
                replacement = f'<a href="{href_esc}">{inner_esc}</a>'
            else:
                # Небезопасный href — вставляем только текст ссылки (без тега)
                replacement = inner_esc
        elif data[0] == "b":
            replacement = f"<b>{html.escape(data[1])}</b>"
        elif data[0] == "i":
            replacement = f"<i>{html.escape(data[1])}</i>"
        elif data[0] == "u":
            replacement = f"<u>{html.escape(data[1])}</u>"
        elif data[0] == "s":
            replacement = f"<s>{html.escape(data[1])}</s>"
        else:
            replacement = html.escape(str(data))

        # token в escaped не будет преобразован html.escape'ом, потому что
        # token состоит из безопасных ASCII-символов (буквы/цифры/подчёрки).
        escaped = escaped.replace(token, replacement)

    return escaped

# ---------------------------
# Хэндлеры
# ---------------------------

@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    user = message.from_user
    # Используем HTML-парсинг, поэтому экранируем имя
    safe_name = html.escape(user.full_name or "пользователь")
    welcome_text = (
        f"Привет, <b>{safe_name}</b>! 👋\n\n"
        "🧠 Добро пожаловать в <b>OlvexAI</b> — ваш персональный ИИ-ассистент.\n"
        "Я могу генерировать текст, писать код, объяснять сложное и многое другое.\n\n"
        "Просто напиши своё первое сообщение — и поехали."
    )

    logger.info(f"👋 /start от {user.full_name} (id={user.id})")
    # Явно передаём parse_mode=HTML — это безопаснее и понятнее
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)

@router.message()
async def echo_handler(message: Message) -> None:
    user = message.from_user
    logger.info(f"📨 Сообщение от {user.full_name}: {message.text}")

    thinking_msg = await message.answer("💭 Думаю над ответом...")

    try:
        async with openai_semaphore:
            completion = await asyncio.to_thread(
                lambda: client.chat.completions.create(
                    model="deepseek/deepseek-chat-v3.1",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Ты ассистент. Используй только HTML-теги: "
                                "<b>, <i>, <u>, <s>, <a>. "
                                "Не используйте <code> или <pre> — их Telegram не принимает."
                            ),
                        },
                        {"role": "user", "content": message.text},
                    ],
                    temperature=0.7,
                )
            )

        ai_reply = completion.choices[0].message.content.strip()
        # Санитизируем HTML от ИИ, разрешая только нужные теги
        formatted_reply = sanitize_ai_html(ai_reply)
        await thinking_msg.delete()

        MAX_LEN = 4096
        # Отправляем, нарезая по лимиту Telegram
        if len(formatted_reply) <= MAX_LEN:
            await message.answer(formatted_reply, parse_mode=ParseMode.HTML)
        else:
            # Разбиваем аккуратно — при желании можно разбивать по предложениям,
            # но базовая нарезка по символам тоже работает.
            for i in range(0, len(formatted_reply), MAX_LEN):
                chunk = formatted_reply[i:i + MAX_LEN]
                await message.answer(chunk, parse_mode=ParseMode.HTML)

        logger.info(f"🤖 Ответ пользователю {user.full_name}: {ai_reply[:80]}...")

    except Exception as e:
        logger.error(f"Ошибка при запросе к DeepSeek: {e}")
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        err = "⚠️ Ошибка при генерации ответа. Попробуй позже."
        await message.answer(html.escape(err), parse_mode=ParseMode.HTML)

async def main() -> None:
    # Устанавливаем HTML как дефолтный режим парсинга
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp.include_router(router)

    logger.info("🚀 Бот OlvexAI запущен и готов к работе!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("💤 Бот остановлен пользователем.")
    except Exception as e:
        logger.critical(f"🛑 Критическая ошибка: {e}")
