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

# Загрузка переменных окружения из .env
load_dotenv()

# Настройка цветного логгера
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

# Проверка токенов
TOKEN = getenv("BOT_TOKEN")
AI_TOKEN = getenv("AI_TOKEN")
if not TOKEN or not AI_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN или AI_TOKEN не найден. Проверьте .env")

# Инициализация OpenAI клиента (OpenRouter)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=AI_TOKEN)

# Настройка диспетчера
dp = Dispatcher()
router = Router()
openai_semaphore = asyncio.Semaphore(3)

# ---------------------------
# Регулярные выражения для разбора форматированного текста
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
    """Извлекает блоки кода и возвращает остаток текста и список кодов."""
    codes = []
    non_code_parts = []

    last_end = 0
    for match in _code_block_pattern.finditer(text):
        if match.start() > last_end:
            non_code_parts.append(('text', text[last_end:match.start()]))
        lang = match.group(1).strip() or "txt"
        code_content = match.group(2).rstrip('\n')
        codes.append((lang, code_content))
        last_end = match.end()

    if last_end < len(text):
        non_code_parts.append(('text', text[last_end:]))

    return non_code_parts, codes

def sanitize_ai_html(text: str) -> str:
    """Оставляет только безопасные HTML-теги: <b>, <i>, <u>, <s>, <a>."""
    placeholders = {}

    def make_token():
        return f"__PH_{uuid.uuid4().hex}__"

    # Сохраняем разрешённые теги через плейсхолдеры
    temp = _link_pattern.sub(lambda m: (token := make_token(), placeholders.update({token: ("a", m.group(1), m.group(2))}) or token)[-1], text)
    temp = _b_pattern.sub(lambda m: (token := make_token(), placeholders.update({token: ("b", m.group(1))}) or token)[-1], temp)
    temp = _i_pattern.sub(lambda m: (token := make_token(), placeholders.update({token: ("i", m.group(1))}) or token)[-1], temp)
    temp = _u_pattern.sub(lambda m: (token := make_token(), placeholders.update({token: ("u", m.group(1))}) or token)[-1], temp)
    temp = _s_pattern.sub(lambda m: (token := make_token(), placeholders.update({token: ("s", m.group(1))}) or token)[-1], temp)

    # Экранируем весь остальной HTML
    escaped = html.escape(temp)

    # Восстанавливаем разрешённые теги
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
        "🧠 Добро пожаловать в <b>OlvexAI</b> — ваш персональный ИИ-ассистент.\n"
        "Я могу:\n"
        "• Писать и объяснять код\n"
        "• Отвечать на вопросы\n"
        "• Помогать с учёбой и работой\n\n"
        "<i>Просто напишите, что вам нужно!</i>"
    )
    logger.info(f"👋 /start от {user.full_name} (id={user.id})")
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)


# Глобальная защита от флуда
FLOOD_COOLDOWN = 2
user_last_message = {}

@router.message(F.text)
async def echo_handler(message: Message) -> None:
    user_id = message.from_user.id
    now = asyncio.get_event_loop().time()

    if user_id in user_last_message and now - user_last_message[user_id] < FLOOD_COOLDOWN:
        return

    user_last_message[user_id] = now

    user = message.from_user
    user_text = message.text.strip()
    logger.info(f"📨 Сообщение от {user.full_name}: {user_text}")

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
                                "Ты — полезный ассистент. Форматируй ответы так:\n"
                                "• Для кода используй: ```py\\nкод\\n```\n"
                                "• Для ссылок: <a href='https://example.com'>сайт</a>\n"
                                "• Жирный: <b>текст</b>, курсив: <i>текст</i>\n"
                                "• Никогда не используй <pre>, <code>, <br>."
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

        # Отправляем обычный текст (HTML)
        if non_code_parts:
            full_text = ''.join(part for _, part in non_code_parts)
            cleaned_text = sanitize_ai_html(full_text)
            MAX_LEN = 4096
            for i in range(0, len(cleaned_text), MAX_LEN):
                chunk = cleaned_text[i:i + MAX_LEN]
                await message.answer(chunk, parse_mode=ParseMode.HTML)

        # Отправляем блоки кода с кнопкой "Скопировать"
        for lang, code in code_blocks:
            # Убираем лишние пробелы в начале/конце
            code_clean = code.rstrip('\n').lstrip('\n')
            # Отправляем код как сообщение
            code_msg = f"```{lang}\n{code_clean}\n```"
            sent_msg = await message.answer(code_msg, parse_mode=None)

            # Создаём кнопку "Скопировать код"
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="📋 Скопировать код", callback_data=f"copy_code_{lang}_{sent_msg.message_id}")]
            ])
            # Редактируем сообщение, добавляя кнопку
            await sent_msg.edit_text(code_msg, reply_markup=keyboard, parse_mode=None)

        logger.info(f"✅ Ответ отправлен пользователю {user.full_name}")

    except Exception as e:
        logger.error(f"Ошибка при запросе к ИИ: {e}")
        try:
            await thinking_msg.delete()
        except Exception:
            pass
        await message.answer("⚠️ Ошибка генерации. Попробуйте позже.", parse_mode=ParseMode.HTML)


# ---------------------------
# Запуск бота
# ---------------------------

from aiogram.filters import CallbackData

class CopyCodeCallback(CallbackData, prefix="copy_code"):
    lang: str
    msg_id: int

@router.callback_query(CopyCodeCallback.filter())
async def copy_code_callback(callback: types.CallbackQuery, callback_data: CopyCodeCallback):
    # Получаем ID сообщения, где был код
    msg_id = callback_data.msg_id
    lang = callback_data.lang

    # Находим сообщение с кодом
    try:
        msg = await callback.message.bot.get_message(chat_id=callback.message.chat.id, message_id=msg_id)
        # Извлекаем текст кода (убираем ```py и ``` из начала и конца)
        code_text = msg.text
        if code_text.startswith(f"```{lang}") and code_text.endswith("```"):
            code_content = code_text[len(f"```{lang}"): -3].rstrip('\n')
            # Отправляем чистый код как сообщение
            await callback.message.answer(f"```{lang}\n{code_content}\n```", parse_mode=None)
            # Уведомляем пользователя
            await callback.answer("✅ Код скопирован в буфер обмена!", show_alert=True)
        else:
            await callback.answer("❌ Не удалось извлечь код.", show_alert=True)
    except Exception as e:
        await callback.answer(f"❌ Ошибка: {str(e)}", show_alert=True)

async def main() -> None:
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