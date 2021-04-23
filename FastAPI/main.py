from fastapi import FastAPI
import ktrain

app = FastAPI()
predictor = ktrain.load_predictor('suomi24_5_topic_model_FinBERT_finnish')

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/hello_name/{name}")
def read_root(name: str):
    string_to_return = str("Hello {}".format(name))
    return {"string_to_return": string_to_return}

@app.get("/predict/{text}")
def read_item(text: str):
    prediction = predictor.predict(text)
    return {"prediction": prediction}