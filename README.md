# GLA - Gated Linear Attention Transformer

## Описание проекта

Этот проект представляет собой реализацию модели Gated Linear Attention (GLA), основанной на статье "Gated Linear Attention Transformers with Hardware-Efficient Training". Цель проекта — создать и обучить трансформерную модель, способную восстанавливать входную последовательность символов, решая задачу автоэнкодера.

## Структура репозитория

- `gla.py` — реализация слоя Gated Linear Attention.
- `gla_model.py` — архитектура модели GLATransformer.
- `train.py` — код для обучения модели.
- `test.py` — скрипт для тестирования модели.
- `models/` — папка для сохранения обученных моделей (содержит предобученную `gla_learned_model`).
- `README.md` — этот файл с инструкцией по запуску.

## Установка и зависимости

Перед запуском установите необходимые библиотеки:

```bash
pip install torch datasets einops
```

## Использование

### Обучение модели

Запустите скрипт `train.py` для обучения модели:

```bash
python train.py
```

После успешного обучения модель сохранится в `models/gla_model_best_accuracy.pth`.

### Тестирование модели

Запустите `test.py`, чтобы протестировать модель:

```bash
python test.py
```

Вы также можете протестировать уже предобученную модель `gla_learned_model`, находящуюся в папке `models/`.

## Работа модели с датасетом

Модель обучается на текстовых данных из датасета Wikitext-103, содержащего статьи из Википедии. Основная цель обучения — восстановление входной последовательности символов без искажений, что соответствует задаче автоэнкодера.

Во время обучения текст преобразуется в числовые представления, где каждому символу соответствует индекс из словаря. Затем данные подаются в слои внимания модели GLA, которые позволяют эффективно обрабатывать последовательности, минимизируя вычислительные затраты. Итоговая модель способна кодировать и декодировать последовательности символов, сохраняя их исходную структуру.

## Дополнительная информация

Проект разработан в рамках выполнения задания на ИМШ и ориентирован на проведение исследований в области улучшения архитектур нейронных сетей для обработки последовательностей.

## Лицензия

Данный проект распространяется по лицензии MIT, так как использует материалы статьи [Gated Linear Attention Transformers with Hardware-Efficient Training](https://icml.cc/virtual/2024/poster/33349).



