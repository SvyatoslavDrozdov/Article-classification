# Article Classification

В данном учебном проекте реализован трансформер-энкодер для задачи классификации научных статей по трём темам:

- **physics**
- **mathematics**
- **computer_science**

Модель работает только с текстами на английском языке.

## Данные

Модель обучалась на датасете [TimSchopf/arxiv_categories](https://huggingface.co/datasets/TimSchopf/arxiv_categories).

Количество объектов обучающей выборки по классам:

```text
physics             92506
mathematics         33816
computer_science    32994
```
## Архитектура модели

### Основные компоненты модели

- Токенизация текста выполняется с помощью токенизатора Hugging Face.
- Каждый токен преобразуется в векторное представление с помощью слоя `Embedding`.
- Дополнительно используются позиционные эмбеддинги, чтобы модель учитывала порядок токенов в последовательности.
- Далее вход проходит через несколько encoder block.

### Encoder block

Каждый encoder block состоит из двух основных частей:

1. Multi-Head Self-Attention  
2. Feed-Forward Network

Также в каждом блоке используются:
- residual connections
- LayerNorm
- Dropout

В конце используется линейный слой для получения логитов.

## Обучение модели
Обучение модели представлено в model_train.ipynb. Чекпоинт с данными модели находится по адресу https://huggingface.co/Svyat-dr/article_classifier/blob/main/transformer_checkpoint.pt.

## Фронтенд

Для удобства использования модели был реализован простой веб-интерфейс на фреймворке Streamlit.

Приложение позволяет ввести текст статьи и получить предсказание наиболее вероятной темы из следующих:

- **physics**
- **mathematics**
- **computer_science**

На данный момент приложение развернуто с помощью Streamlit Cloud и доступно по адресу https://article-classification-a2qnp9i6a8t27kr4rntvaa.streamlit.app/.
