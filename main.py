from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


# опрелеляем класс, который наследуется от BaseModel
class Item(BaseModel):
    text: str


app = FastAPI()
# пайплайн для анализа тональности текста
classifier = pipeline("sentiment-analysis")


@app.get("/")
def root():
    """Выводит сообщение в формате JSON

    Returns:
        dict: JSON messege
    """
    return {"messege": "Ohh. It's working"}


@app.post("/predict/")
def predict(item: Item):
    """Реализацияя предсказаний модели

    Returns:
        tup: где на [0] стоит оценка "POSITIVE" или "NEGATIVE"
    """
    return classifier(item.text)[0]
