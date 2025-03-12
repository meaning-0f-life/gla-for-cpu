from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from gla_model import GLATransformer
import torch.nn.functional as F
import torch
import os
import string

# Создаем словарь символов
chars = string.printable  # Все печатные символы
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}


def save_model(model, model_name="gla_model.pth"):
    """
    Сохраняет модель в папку models.

    Args:
        model (nn.Module): Обученная модель.
        model_name (str): Имя файла для сохранения модели.
    """
    # Создаем папку models, если её нет
    if not os.path.exists("models"):
        os.makedirs("models")

    # Сохраняем модель
    model_path = os.path.join("models", model_name)
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в {model_path}")


def calculate_accuracy(logits, targets):
    """
    Вычисляет accuracy (процент правильных предсказаний).

    Args:
        logits (torch.Tensor): Выход модели (логиты) с размерностью [batch_size, seq_len, vocab_size].
        targets (torch.Tensor): Реальные значения с размерностью [batch_size, seq_len].

    Returns:
        float: Процент правильных предсказаний.
    """
    # Получаем предсказания (индексы с максимальными значениями)
    predictions = torch.argmax(logits, dim=-1)

    # Сравниваем предсказания с реальными значениями
    correct = (predictions == targets).float()

    # Вычисляем accuracy
    accuracy = correct.sum() / targets.numel()

    return accuracy.item()


# Загрузка данных с использованием wikitext
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")


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


dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn)

# Инициализация модели
model = GLATransformer(d_model=512, n_heads=4, d_k=64, d_v=64, num_layers=6, vocab_size=vocab_size)
optimizer = AdamW(model.parameters(), lr=1e-4)

# Цикл обучения
best_accuracy = 0.0  # Лучшая accuracy

for epoch in range(3):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()

        # Получаем предсказания модели
        logits = model(batch)

        # Вычисляем loss
        loss = F.cross_entropy(logits.view(-1, vocab_size), batch.view(-1))

        # Вычисляем accuracy
        accuracy = calculate_accuracy(logits, batch)

        # Обратное распространение и обновление весов
        loss.backward()
        optimizer.step()

        # Сохраняем модель, если accuracy улучшилась
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_model(model, f"gla_model_best_accuracy.pth")
            print(f"Новая лучшая accuracy: {accuracy * 100:.2f}%. Модель сохранена.")

        # Выводим loss и accuracy
        print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {accuracy * 100:.2f}%")

# Сохранение модели
save_model(model, "gla_model.pth")