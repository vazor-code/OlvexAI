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
from io import StringIO

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery

# ---------------------------
# Настройка логгера
# ---------------------------
load_dotenv()

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

# ---------------------------
# Проверка токенов
# ---------------------------
TOKEN = getenv("BOT_TOKEN")
AI_TOKEN = getenv("AI_TOKEN")
if not TOKEN or not AI_TOKEN:
    raise RuntimeError("❌ BOT_TOKEN или AI_TOKEN не найден. Проверьте .env")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=AI_TOKEN
)

dp = Dispatcher()
router = Router()
openai_semaphore = asyncio.Semaphore(3)

# ---------------------------
# Модели и пользователи
# ---------------------------
user_context = {}
user_model = {}
user_last_message = {}
FLOOD_COOLDOWN = 1.5

AVAILABLE_MODELS = {
    "deepseek": "deepseek/deepseek-chat-v3-0324:free",
    "r1": "deepseek/deepseek-r1-0528:free",
    "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
    "qwen": "qwen/qwen3-coder:free",
    "gptoss": "openai/gpt-oss-20b:free",
}

MODEL_NAMES = {
    "deepseek": "🧠 DeepSeek Chat v3.0324",
    "r1": "🚀 DeepSeek R1 (0528)",
    "mistral": "🔥 Mistral Small 24B",
    "qwen": "💻 Qwen3 Coder",
    "gptoss": "🔵 GPT-OSS 20B",
}

# Порядок резервных моделей при 429
MODEL_FALLBACK_ORDER = ["deepseek", "r1", "mistral", "qwen", "gptoss"]

# ---------------------------
# Вспомогательные функции
# ---------------------------
def clean_ai_response(text: str) -> str:
    text = re.sub(r'<\|.*?\|>', '', text)
    text = re.sub(r'\[\/?OUT\]', '', text)
    text = re.sub(r'<<SYS>>.*?<<\/SYS>>', '', text, flags=re.DOTALL)
    text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    text = re.sub(r'^<s>\s*', '', text)
    text = re.sub(r'\s*</s>$', '', text)
    return text.strip()

# ---------------------------
# 🆕 Safe OpenAI Call
# ---------------------------
async def safe_openai_call(user_id: int, user_text: str, message: Message, temperature: float = 0.7):
    """
    Безопасный вызов модели: проверка лимитов и автопереключение моделей.
    Уведомления отправляются в чат.
    """
    models_cycle = list(AVAILABLE_MODELS.values())
    start_index = models_cycle.index(user_model.get(user_id, models_cycle[0]))
    
    for i in range(len(models_cycle)):
        model = models_cycle[(start_index + i) % len(models_cycle)]
        user_model[user_id] = model
        display_name = next((name for key, name in MODEL_NAMES.items() if AVAILABLE_MODELS[key] == model), model)
        
        try:
            async with openai_semaphore:
                # Передаем temperature внутри
                completion = await asyncio.to_thread(
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "Ты — ИИ-ассистент, отвечай на русском."},
                            {"role": "user", "content": user_text}
                        ],
                        temperature=temperature,
                    )
                )
            return completion.choices[0].message.content.strip()
        
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate-limited" in error_str.lower():
                logger.warning(f"⚠️ Модель {display_name} исчерпала лимит. Пробуем следующую...")
                await message.answer(f"⚠️ Модель <b>{display_name}</b> исчерпала лимит. Пробуем следующую...", parse_mode=ParseMode.HTML)
                continue
            else:
                logger.error(f"Ошибка модели {display_name}: {error_str}")
                raise e

    raise RuntimeError("❌ Все модели исчерпали лимит, попробуйте позже.")

# ---------------------------
# /help
# ---------------------------
@router.message(F.text.startswith("/help"))
async def help_command(message: Message):
    help_text = (
        "📚 <b>OlvexAI</b>\n"
        "Умный бот, готовый ответить на любые вопросы.\n\n"
        "🔹 <b>Команды:</b>\n"
        "🔸 <code>/start</code> — начать\n"
        "🔸 <code>/help</code> — это меню\n"
        "🔸 <code>/model</code> — выбрать или увидеть текущую модель\n\n"
        "📎 Прикрепите файл (.txt, .py, .js и др.) — я прочитаю и объясню.\n\n"
        "💡 Бот запоминает контекст — можно вести диалог.\n"
        "⚠️ Сервера могут временно отключаться.\n"
        "👤 Разработка: @vazor_code"
    )
    await message.answer(help_text, parse_mode=ParseMode.HTML)

# ---------------------------
# /model
# ---------------------------
@router.message(F.text.startswith("/model"))
async def change_model(message: Message):
    user_id = message.from_user.id
    args = message.text.split(maxsplit=1)

    # Если указана модель вручную
    if len(args) > 1:
        model_key = args[1].strip().lower()
        if model_key in AVAILABLE_MODELS:
            user_model[user_id] = AVAILABLE_MODELS[model_key]
            display_name = MODEL_NAMES.get(model_key, model_key)
            await message.answer(
                f"✅ Модель изменена на <b>{display_name}</b> (<code>{model_key}</code>)",
                parse_mode=ParseMode.HTML
            )
        else:
            await message.answer("⚠️ Такой модели нет. Нажмите /model, чтобы выбрать из списка.")
        return

    # Текущая модель
    current_model = user_model.get(user_id, "deepseek/deepseek-chat-v3-0324:free")
    model_key = next((k for k, v in AVAILABLE_MODELS.items() if v == current_model), "deepseek")
    display_name = MODEL_NAMES.get(model_key, "Неизвестная модель")

    text = (
        f"🤖 <b>Выбор модели</b>\n\n"
        f"🔹 <b>Текущая:</b> {display_name} (<code>{model_key}</code>)\n\n"
        "Выберите одну из доступных моделей ниже 👇"
    )

    grouped_models = {
        "🧠 DeepSeek": ["deepseek", "r1"],
        "🔥 Mistral": ["mistral"],
        "💻 Qwen": ["qwen"],
        "🔵 OpenAI": ["gptoss"]
    }

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[[InlineKeyboardButton(text="🔄 Обновить список", callback_data="refresh_models")]]
    )

    rows = []
    for group_name, keys in grouped_models.items():
        rows.append([InlineKeyboardButton(text=group_name, callback_data="none")])
        for key in keys:
            model_text = MODEL_NAMES.get(key, key)
            rows.append([InlineKeyboardButton(text=model_text, callback_data=f"set_model:{key}")])
        rows.append([InlineKeyboardButton(text="—", callback_data="none")])
    keyboard.inline_keyboard = rows

    await message.answer(text, parse_mode=ParseMode.HTML, reply_markup=keyboard)


@router.callback_query(F.data.startswith("set_model:"))
async def set_model_callback(callback: CallbackQuery):
    user_id = callback.from_user.id
    model_key = callback.data.split(":", 1)[1]

    if model_key in AVAILABLE_MODELS:
        user_model[user_id] = AVAILABLE_MODELS[model_key]
        display_name = MODEL_NAMES.get(model_key, model_key)
        await callback.message.edit_text(
            f"✅ Модель изменена на <b>{display_name}</b> (<code>{model_key}</code>)\n\n"
            "💬 Теперь можно писать запрос 👇",
            parse_mode=ParseMode.HTML
        )
        await callback.answer("Модель успешно переключена ✅")
    else:
        await callback.answer("❌ Такой модели нет", show_alert=True)

# ---------------------------
# Хэндлеры команд
# ---------------------------
@router.message(CommandStart())
async def command_start_handler(message: Message):
    user = message.from_user
    safe_name = html.escape(user.full_name or "Пользователь")
    welcome_text = (
        f"Привет, <b>{safe_name}</b>! 👋\n\n"
        "🧠 Я — ваш ИИ-ассистент.\n"
        "Задавайте вопросы, просите написать код, объяснить тему.\n\n"
        "<i>Код будет отображаться красиво — его можно копировать одним кликом.</i>\n\n"
        "📌 Используйте /help — чтобы узнать все возможности.\n"
        "🔹 Чтобы выбрать модель — /model"
    )
    logger.info(f"👋 /start от {user.full_name}")
    await message.answer(welcome_text, parse_mode=ParseMode.HTML)

# ---------------------------
# Обработка текста от пользователя
# ---------------------------
@router.message(F.text)
async def echo_handler(message: Message):
    user_id = message.from_user.id
    now = asyncio.get_event_loop().time()

    if user_id in user_last_message and now - user_last_message[user_id] < FLOOD_COOLDOWN:
        return
    user_last_message[user_id] = now

    user_text = message.text.strip()
    logger.info(f"📨 От {message.from_user.full_name}: {user_text}")

    thinking_msg = await message.answer("💭 Думаю...")

    if user_id not in user_context:
        user_context[user_id] = []

    # Добавляем пользовательский ввод в контекст
    user_context[user_id].append({"role": "user", "content": user_text})

    messages = [
        {"role": "system", "content": (
            "Ты — ИИ-ассистент. Всегда отвечай на русском языке.\n"
            "• Для кода используй формат: ```py\\nкод\\n```\n"
            "• Для ссылок: <a href='https://example.com'>текст</a>\n"
            "• Жирный: <b>текст</b>, курсив: <i>текст</i>\n"
            "• Никогда не используй <pre> или <code> в ответе."
        )}
    ] + user_context[user_id]

    try:
        # Вызов модели через безопасную функцию
        raw_reply = await safe_openai_call(user_id, messages, message)
        cleaned_reply = clean_ai_response(raw_reply)
        user_context[user_id].append({"role": "assistant", "content": cleaned_reply})

        await message.answer(cleaned_reply, parse_mode=ParseMode.HTML)
        logger.info("✅ Ответ отправлен")

    except Exception as e:
        logger.error(f"Ошибка генерации: {e}")
        await message.answer("⚠️ Ошибка генерации. Попробуйте позже.")

    finally:
        # Безопасное удаление thinking_msg
        if thinking_msg:
            try:
                await thinking_msg.delete()
            except Exception:
                pass

# ---------------------------
# Запуск бота
# ---------------------------
async def main():
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp.include_router(router)
    logger.info("🚀 OlvexAI Bot запущен!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("💤 Бот остановлен.")
    except Exception as e:
        logger.critical(f"🛑 Ошибка: {e}")