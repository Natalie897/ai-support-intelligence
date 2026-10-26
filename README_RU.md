# AI Support Intelligence — инструкция на русском

Это учебный **ML + Python Backend** проект для портфолио: классификация англоязычных обращений в банковскую поддержку. Классификатор обучается на текстах, а **FastAPI** позволяет отправлять запросы и получать предполагаемую категорию.

## Что внутри

- `ml/train.py` — сравнение **TF-IDF + Logistic Regression** и **TF-IDF + Multinomial Naive Bayes** с оценкой `accuracy` и `macro-F1`;
- `app/api.py` — REST API с методами `GET /health`, `GET /categories`, `POST /predict`;
- `app/storage.py` — опциональное сохранение **только метаданных** предсказаний в PostgreSQL;
- `tests/` — тесты API, данных и обучения;
- `Dockerfile`, `compose.yaml` — контейнеризация и PostgreSQL;
- `.github/workflows/ci.yml` — запуск тестов при загрузке на GitHub.

## Быстрая проверка на Windows (PowerShell)

Запусти PowerShell **в корне репозитория**. Требуется Python 3.11 или новее.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ml.train --demo
.\.venv\Scripts\python.exe -m uvicorn app.api:app --reload
```

После этого в браузере открой `http://127.0.0.1:8000/docs` и выбери `POST /predict` → **Try it out**. Пример тела запроса:

```json
{"text":"My card has not arrived yet", "top_k":3}
```

**Важно:** `--demo` обучает модель на небольших вымышленных примерах (6 категорий), чтобы проверить программу без интернета. Это **не** показатель качества на реальном BANKING77.

## Обучение на реальном наборе BANKING77

```powershell
.\.venv\Scripts\python.exe -m ml.train
```

При первом запуске команда скачает оригинальные CSV (при наличии интернета), проверит контрольные суммы SHA-256, сравнит две модели на валидационной части обучающих данных и посчитает качество победителя на отдельной тестовой части. Итог — в `artifacts/metrics.json`, модель — в `artifacts/model.joblib`. Эти файлы **не загружаются в Git**. После обучения запусти API заново.

Источник: https://github.com/PolyAI-LDN/task-specific-datasets (лицензия набора данных CC BY 4.0). Данные англоязычные; ввод на русском не поддерживается надёжно. Оценки уверенности модели не откалиброваны, поэтому сервис учебный и не подходит для принятия решений без участия человека.

## Запуск с Docker

```powershell
docker compose --profile training run --build --rm train
docker compose up --build -d api
```

API: `http://127.0.0.1:8000/docs`. PostgreSQL запускается локально в Compose. Для остановки: `docker compose down`. В примере используются демонстрационные учётные данные для БД; публично публиковать API без авторизации нельзя.

## Как использовать в резюме

*AI Support Intelligence — ML-классификатор обращений: TF-IDF, Logistic Regression/Naive Bayes, FastAPI REST API, pytest, Docker, PostgreSQL (опционально).*

Если полный датасет ещё не обучен и не оценён, **не указывай в резюме выдуманные метрики или модель на 77 категориях как уже обученную**. Для исходного README и полной технической документации см. [README.md](README.md).
