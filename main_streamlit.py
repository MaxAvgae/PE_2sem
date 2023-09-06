from transformers import pipeline
import streamlit as st
from langdetect import detect


# создаем пайплайн для определения тональности текста
def get_pipeline():
    pipe = pipeline("sentiment-analysis")
    return pipe


def sentiment_and_score(inputs):
    """Выполняет определение тональности текста и возвращает метку и оценку.

    Args:
        inputs (str): Текст для определения тональности.

    Returns:
        tuple: Кортеж, в котором 1й элемент - метка тональности, а 2й - оценка.

    """
    pipe = get_pipeline()
    result = pipe(inputs)[0]
    return result["label"], result["score"]


def language_test(inputs):
    lang = detect(inputs)
    return lang == "en"


# Определение тональности текста.
st.title("ОПРЕДЕЛЕНИЕ ТОНАЛЬНОСТИ ТЕКСТА.")
# Поле ввода текста. value - значение по умолчанию.
context = st.text_input("CONTEXT:", value="Life is awesome!")

# Кнопка, нажатие на которую запускает процесс определения тональности.
result = st.button("ОПРЕДЕЛИТЬ ТОНАЛЬНОСТЬ")
if context:
    if language_test(context):
        label, score = sentiment_and_score(context)
        st.text(f"LABEL={label}\nSCORE={score}")
    else:
        st.error("Язык не является английским")
else:
    st.error("Пустое поле ввода")
