import torch
from gla_model import GLATransformer
from datasets import load_dataset
from torch.utils.data import DataLoader
import os
import string
import random

# Создаем словарь символов
chars = string.printable  # Все печатные символы
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


def load_model(model, model_name="gla_model_best.pth"):
    """
    Загружает модель из папки models.
    """
    model_path = os.path.join("models", model_name)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Модель загружена из {model_path}")
    else:
        print(f"Файл модели {model_path} не найден.")
    return model


def test_model(model, dataloader, num_examples=5):
    """
    Тестирует модель на нескольких примерах.
    """
    model.eval()  # Переводим модель в режим оценки
    """
    # Свой текст
    custom_text = "hello world"

    # Преобразуем в индексы
    input_ids = torch.tensor([[char_to_idx.get(c, 0) for c in custom_text]], dtype=torch.long)

    # Передаём в модель
    with torch.no_grad():
        logits = model(input_ids)

    # Получаем предсказания
    predictions = torch.argmax(logits, dim=-1)

    # Декодируем текст
    predicted_text = "".join([idx_to_char.get(idx.item(), "") for idx in predictions[0]])

    print("Исходный текст:", custom_text)
    print("Предсказание модели:", predicted_text)
    """

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_examples:
                break

            # Проверка на пустые данные
            if batch.numel() == 0:
                print(f"Пример {i + 1}: Пустой батч, пропускаем.")
                continue
            # batch = [batch_size, seq_len]
            # Получаем предсказания модели
            logits = model(batch)  # [batch_size, seq_len, vocab_size]

            # Получаем предсказания (индексы с максимальными значениями)
            predictions = torch.argmax(logits, dim=-1)

            # Преобразуем индексы в символы
            input_text = "".join([idx_to_char.get(idx.item(), "") for idx in batch[0]])
            predicted_text = "".join([idx_to_char.get(idx.item(), "") for idx in predictions[0]])

            # Выводим пример
            print(f"Пример {i + 1}:")
            print(f"Вход: {input_text}")
            print(f"Предсказание: {predicted_text}")
            print("-" * 50)


def collate_fn(batch):
    """
    Функция для подготовки батча данных.
    """
    # Фильтруем пустые строки
    batch = [sample for sample in batch if sample["text"].strip()]

    if not batch:
        return torch.tensor([], dtype=torch.long)  # Возвращаем пустой тензор, если все строки пусты

    # Преобразуем текст в индексы с использованием словаря
    sequences = [torch.tensor([char_to_idx.get(c, 0) for c in sample["text"]], dtype=torch.long) for sample in batch]

    # Применяем padding
    return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)


# Загрузка данных с использованием wikitext
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

# Выбираем случайные примеры из датасета
num_examples = 5  # Количество примеров для тестирования
random_indices = random.sample(range(len(dataset)), num_examples)  # Случайные индексы
random_samples = [dataset[i] for i in random_indices]  # Случайные примеры

# Создаем DataLoader для случайных примеров
dataloader = DataLoader(random_samples, batch_size=1, collate_fn=collate_fn)

# Инициализация модели
model = GLATransformer(d_model=512, n_heads=4, d_k=64, d_v=64, num_layers=6, vocab_size=vocab_size)

# Загрузка модели из папки models
model = load_model(model, "gla_learned_model.pth")

# Тестирование модели
test_model(model, dataloader, num_examples=num_examples)