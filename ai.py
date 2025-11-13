from config import OPENAI_API_TOKEN
import pandas as pd
import json
from openai import OpenAI

# def analyze_agent_performance() -> str:

def analyze_agent_performance():
    """
    Отправляет данные в OpenAI и получает полный аналитический отчёт,
    рассчитанный и сформулированный моделью.
    """
    # ----------------------------
    # 1. Загрузка и подготовка данных (только для формирования входа)
    # ----------------------------
    model = "gpt-4o"
    call_center = pd.read_csv('data/Call_Center_Data.csv', sep=';', encoding='cp1251')
    transcripts = pd.read_csv('data/final_transcripts_enriched_v2.csv', sep=';', encoding='cp1251')
    sales = pd.read_csv('data/sales_data_sample.csv', sep=';', encoding='cp1251')

    # Приведение типов
    call_center['Id_agent'] = call_center['Id_agent'].astype(str)
    transcripts['id'] = transcripts['id'].astype(str)
    transcripts['Name'] = transcripts['Name'].astype(str)
    sales['CUSTOMERNAME'] = sales['CUSTOMERNAME'].astype(str)

    # Определяем успешные продажи
    completed_customers = set(sales[sales['STATUS'] == 'Completed']['CUSTOMERNAME'])

    # Обогащаем транскрипты статусом продажи
    # transcripts['Has_Sale'] = transcripts['Customer_ID'].isin( completed_customers)

    # Объединяем с длительностью звонков
    call_center.rename(columns={'Id_agent': 'id'}, inplace=True)
    call_center['id'] = call_center['id'].astype(str)

    # Объединяем данные по Agent_ID
    merged = transcripts.merge(
        call_center[['id', 'Talk Duration (AVG)']],
        on='id',
        how='left'
    )

    # Выбираем только нужные поля для отправки
    data_for_ai = merged[[
        'id',
        'Name',
        'Sentiment',
        # 'Call_Outcome',
        # 'Has_Sale',
        'Talk Duration (AVG)'
    ]].copy()

    # Преобразуем в список словарей (JSON-совместимый)
    records = data_for_ai.to_dict(orient='records')

    # Ограничиваем объём данных (на случай большого файла)
    if len(records) > 2000:
        records = records[:2000]  # OpenAI имеет лимит токенов

    # ----------------------------
    # 2. Формируем промпт для OpenAI
    # ----------------------------
    prompt = f"""
    Ты — эксперт по аналитике контактных центров. Тебе предоставлены данные о звонках клиентов.

    Данные представлены в виде списка записей. Каждая запись содержит:
    - Agent_ID: идентификатор агента
    - Customer_ID: идентификатор клиента
    - Sentiment_Score: тональность реплики клиента (-1.0 до +1.0)
    - Call_Outcome: результат звонка ("Success" или другой)
    - Has_Sale: совершил ли клиент покупку после звонка (true/false)
    - Talk Duration (AVG): средняя длительность разговора агента в формате "XmYs" (например, "3m12s")

    Твоя задача:
    1. Для каждого агента рассчитай:
    - Среднюю тональность (Avg_Sentiment)
    - Конверсию продаж (доля клиентов с Has_Sale=true)
    - Среднюю длительность разговора в минутах (преобразуй "XmYs" → число минут с дробной частью)
    2. Рассчитай интегральный Performance_Score по формуле:
    Performance_Score = 0.4 * нормализованный(Avg_Sentiment) + 
                        0.4 * нормализованный(Conversion_Rate) + 
                        0.2 * (1 - нормализованный(Avg_Duration_in_minutes))
    Нормализацию проведи по всем агентам (Min-Max scaling).
    3. Построй рейтинг агентов по Performance_Score.
    4. Выдели ТОП-5 и BOTTOM-5.
    5. Дай краткий профессиональный вывод для руководителя:
    - Какова общая эффективность команды?
    - Что объединяет лучших агентов?
    - Что можно улучшить у слабых?
    - Предложи 2–3 конкретные рекомендации.

    Ответ должен быть на русском языке, структурированным, без кода, не более 300 слов.

    Данные ({len(records)} записей):
    {json.dumps(records, ensure_ascii=False, indent=2)}
    """

    # ----------------------------
    # 3. Запрос к OpenAI
    # ----------------------------
    client = OpenAI(api_key=OPENAI_API_TOKEN)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — аналитик. Ты даёшь точные, краткие и полезные выводы на основе данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=800
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Ошибка при обращении к OpenAI: {str(e)}"

def analyze_emotional_dynamics():
    """
    Анализ динамики эмоционального состояния клиентов.
    Вся логика анализа выполняется в OpenAI.
    """
    # ----------------------------
    # 1. Загрузка и фильтрация данных
    # ----------------------------
    model = "gpt-4o"
    transcripts = pd.read_csv('data/final_transcripts_enriched_v2.csv')

    # Оставляем только клиентов и необходимые поля
    customer_data = transcripts[
        (transcripts['Speaker'] == 'Customer') &
        (transcripts['Sentiment_Score'].notna())
    ][['Call_ID', 'Customer_ID', 'Agent_ID', 'Sentiment_Score']].copy()

    # Убедимся, что Sentiment_Score — число
    customer_data['Sentiment_Score'] = pd.to_numeric(customer_data['Sentiment_Score'], errors='coerce')
    customer_data = customer_data.dropna(subset=['Sentiment_Score'])

    # Ограничиваем объём для укладки в токены
    if len(customer_data) > 3000:
        customer_data = customer_data.head(3000)

    # Преобразуем в список словарей
    records = customer_data.to_dict(orient='records')

    # ----------------------------
    # 2. Промпт для OpenAI
    # ----------------------------
    prompt = f"""
    Ты — аналитик контактного центра. Тебе предоставлены реплики клиентов с оценкой тональности.

    Каждая запись содержит:
    - Call_ID: уникальный идентификатор звонка
    - Customer_ID: идентификатор клиента
    - Agent_ID: идентификатор оператора
    - Sentiment_Score: тональность реплики клиента (от -1.0 до +1.0)

    Твоя задача:
    1. Для каждого звонка (Call_ID) определи первую и последнюю реплику клиента.
    2. Рассчитай:
    - Start_Sentiment = тональность первой реплики
    - End_Sentiment = тональность последней реплики
    - ΔSentiment = End_Sentiment - Start_Sentiment
    3. Классифицируй каждый звонок:
    - "Улучшение", если ΔSentiment > 0.1
    - "Ухудшение", если ΔSentiment < -0.1
    - "Без изменений" в остальных случаях
    4. Рассчитай статистику:
    - Общее число звонков
    - Распределение по трендам (%)
    - Среднее значение ΔSentiment
    - Примеры звонков с наибольшим улучшением и ухудшением (укажи Customer_ID и Δ)
    5. Проведи анализ по агентам:
    - Какие агенты чаще ведут к "Улучшению"?
    - Есть ли агенты, после общения с которыми настроение клиентов часто ухудшается?
    6. Дай рекомендации руководителю:
    - Что помогает улучшать эмоциональное состояние?
    - Какие практики стоит внедрить или избегать?

    Ответ должен быть на русском языке, структурированным, без кода и таблиц, не более 250 слов.

    Данные ({len(records)} реплик клиента):
    {json.dumps(records, ensure_ascii=False, indent=2)}
    """

    # ----------------------------
    # 3. Запрос к OpenAI
    # ----------------------------
    client = OpenAI(api_key=OPENAI_API_TOKEN)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — эксперт по анализу клиентского опыта. Ты даёшь краткие, точные и actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Ошибка при обращении к OpenAI: {str(e)}"# def analyze_emotional_dynamics() -> str:
#     return 'Я проанализировал и получил вот это: ТУТ ОТВЕТ'

# def analyze_sales_phrases() -> str:
#     return 'Я проанализировал и получил вот это: ТУТ ОТВЕТ'

# Убедимся, что punkt и stopwords загружены (если нет — раскомментируйте)
# nltk.download('punkt')
# nltk.download('stopwords')

def analyze_sales_phrases():
    """
    Анализ ключевых фраз, влияющих на конверсию.
    Вся аналитика (парсинг, частоты, Impact Score, выводы) выполняется в OpenAI.
    """
    # ----------------------------
    # 1. Загрузка и объединение данных
    # ----------------------------
    model = "gpt-4o"
    transcripts = pd.read_csv('data/final_transcripts_enriched_v2.csv')
    sales = pd.read_csv('data/sales_data_sample.csv')

    # Приведение типов
    sales['Customer_ID'] = sales['Customer_ID'].astype(str)
    transcripts['Customer_ID'] = transcripts['Customer_ID'].astype(str)

    # Определяем успешные продажи
    completed_statuses = {'completed', 'success', 'won', 'paid'}
    success_customers = set(
        sales[sales['STATUS'].str.lower().isin(completed_statuses)]['Customer_ID']
    )

    # Оставляем только нужные столбцы
    data_for_ai = transcripts[['Customer_ID', 'Keywords']].copy()
    data_for_ai['Is_Success'] = data_for_ai['Customer_ID'].isin(success_customers)

    # Удаляем строки без Keywords
    data_for_ai = data_for_ai.dropna(subset=['Keywords'])

    # Ограничиваем объём (чтобы не превысить лимит токенов)
    if len(data_for_ai) > 1000:
        data_for_ai = data_for_ai.head(1000)

    # Преобразуем в список словарей
    records = data_for_ai.to_dict(orient='records')

    # ----------------------------
    # 2. Промпт для OpenAI
    # ----------------------------
    prompt = f"""
    Ты — NLP-аналитик в контактном центре. Тебе предоставлены ключевые фразы из звонков и информация о том, завершился ли звонок продажей.

    Каждая запись содержит:
    - Customer_ID: идентификатор клиента
    - Keywords: строка с ключевыми фразами (разделены запятыми, точкой с запятой или переносом строки)
    - Is_Success: true — если клиент совершил покупку, false — если нет

    Твоя задача:
    1. Раздели Keywords на отдельные фразы (игнорируй пустые и короткие <3 символов).
    2. Приведи все фразы к нижнему регистру, удали лишние пробелы и спецсимволы.
    3. Для каждой уникальной фразы посчитай:
    - freq_success = доля упоминаний в успешных звонках (от общего числа упоминаний фразы)
    - freq_failure = доля упоминаний в неуспешных звонках
    - Impact_Score = freq_success - freq_failure
    4. Выдели:
    - ТОП-5 фраз с наибольшим Impact_Score (>0)
    - ТОП-5 фраз с наименьшим Impact_Score (<0)
    5. Сформулируй вывод:
    - Какие фразы явно способствуют продажам?
    - Какие фразы сигнализируют о риске отказа?
    - Какие рекомендации можно дать операторам?

    Ответ должен быть на русском языке, кратким (не более 250 слов), без таблиц и кода.

    Данные ({len(records)} записей):
    {json.dumps(records, ensure_ascii=False, indent=2)}
    """

    # ----------------------------
    # 3. Запрос к OpenAI
    # ----------------------------
    client = OpenAI(api_key=OPENAI_API_TOKEN)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Ты — эксперт по конверсии в продажах. Ты выявляешь паттерны и даёшь actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=700
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Ошибка при обращении к OpenAI: {str(e)}"