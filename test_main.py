from fastapi.testclient import TestClient
from main import app  # копируем наше "приложение" из main


client = TestClient(app)


def test_root_main():
    response = client.get("/")
    # проверяем код ответа
    assert response.status_code == 200
    # и сам ответ
    assert response.json() == {"messege": "Ohh. It's working"}


def test_predict_positive():
    response_post = client.post("/predict/", json={"text": "I like PE"})
    json_data = response_post.json()
    assert response_post.status_code == 200
    assert json_data["label"] == "POSITIVE"


def test_predict_negative():
    response_post = client.post("/predict/", json={"text": "I hate PE"})
    json_data = response_post.json()
    assert response_post.status_code == 200
    assert json_data["label"] == "NEGATIVE"


def test_predict_neutral():
    response_post = client.post("/predict/", json={"text": "It's PE"})
    json_data = response_post.json()
    assert response_post.status_code == 200
    assert json_data["label"] == "NEUTRAL"
