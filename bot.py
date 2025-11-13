import logging

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)
from ai import (
    analyze_agent_performance,
    analyze_emotional_dynamics,
    analyze_sales_phrases
)
from config import TELEGRAM_API_TOKEN

BOT_TOKEN = TELEGRAM_API_TOKEN
logging.basicConfig(level=logging.INFO)

user_context = {}

def main_menu():
    keyboard = [
        [InlineKeyboardButton("📈 Оценка эффективности агентов", callback_data="feature_1")],
        [InlineKeyboardButton("💬 Эмоциональная динамика клиентов", callback_data="feature_2")],
        [InlineKeyboardButton("🔑 Фразы успешных продаж", callback_data="feature_3")]
    ]
    return InlineKeyboardMarkup(keyboard)

def run_and_back_button(feature_code: int):
    keyboard = [
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")],
        [InlineKeyboardButton("▶️ Запустить анализ", callback_data="run_analysis_feature_f{feature_code}")]
    ]
    return InlineKeyboardMarkup(keyboard)

def back_button():
    keyboard = [
        [InlineKeyboardButton("⬅️ Назад", callback_data="back_to_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я бот для анализа эффективности сотрудников колл-центра.\n\n"
        "Выберите, что хотите проанализировать:",
        reply_markup=main_menu()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "ℹ️ Я помогаю анализировать звонки и результаты работы операторов.\n\n"
        "Доступные команды:\n"
        "/start — главное меню\n"
        "/help — справка"
    )


async def handle_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data

    if data == "feature_1":
        await query.edit_message_text(
            "📊 Фича 1 — Комплексная оценка эффективности агента.\n\n",
            reply_markup=run_and_back_button(1)
        )

    elif data == "feature_2":
        await query.edit_message_text(
            "💬 Фича 2 — Анализ эмоциональной динамики клиента.\n\n",
            reply_markup=run_and_back_button(2)
        )

    elif data == "feature_3":
        await query.edit_message_text(
            "🔑 Фича 3 — Определение фраз, связанных с успешными продажами.\n\n",
            reply_markup=run_and_back_button(3)
        )

    elif data == "back_to_menu":
        await query.edit_message_text(
            "Выберите действие:",
            reply_markup=main_menu()
        )

    elif data.startswith("run_analysis"):
        await run_analysis(update, data[-1])


async def run_analysis(update: Update, feature: int):
    query = update.callback_query

    await query.edit_message_text("🧠 Выполняется анализ... Пожалуйста, подождите несколько минут.")

    try:
        if feature == 1:
            result_text = analyze_agent_performance()
        elif feature == 2:
            result_text = analyze_emotional_dynamics()
        else:
            result_text = analyze_sales_phrases()

        await query.message.reply_text(
            f"✅ Анализ завершён!\n\n{result_text}",
            reply_markup=back_button()
        )

    except Exception as e:
        logging.exception(e)
        await query.message.reply_text(
            "⚠️ Произошла ошибка при анализе данных. Проверьте корректность файлов и попробуйте снова.",
            reply_markup=back_button()
        )


def main():
    # print(analyze_agent_performance())
    # analyze_emotional_dynamics()
    # analyze_sales_phrases()
    
    # application = ApplicationBuilder().token(BOT_TOKEN).build()

    # application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("help", help_command))
    # application.add_handler(CallbackQueryHandler(handle_menu))

    # logging.info("🤖 Бот запущен...")
    # application.run_polling()


if __name__ == "__main__":
    main()
