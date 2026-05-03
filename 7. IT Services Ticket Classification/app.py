import joblib
from pydantic import BaseModel
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from backend_program import Backend_Program

model_path = "logistic_regression_model.joblib"
label_en_path = "label_encoder_model.joblib"

backend =  Backend_Program(model_path, label_en_path)

app = FastAPI()

class TicketRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TicketRequest):
    result = backend.predict_ticket_category(request.text)
    return {"Prediction" : result}
