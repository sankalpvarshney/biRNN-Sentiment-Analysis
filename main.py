from fastapi import FastAPI, File
import numpy as np
from src.utils.common import read_yaml
from src.components.inference import SentimentAnalysis


params = read_yaml("config/config.yaml")
sentiment = SentimentAnalysis(params)

app = FastAPI(title="Sentiment Analysis on IMDB comments")


@app.post("/sentimentPrediction")
async def enter_any_comment_for_sentiment_prediction(comment : str):

    score = sentiment.predict(comment)

    output = {
        "score" : float(score)
    }

    return output


###################### uvicorn main:app --host '0.0.0.0' --port 3354 --reload  ########################################