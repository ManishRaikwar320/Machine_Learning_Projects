import joblib
from sentence_transformers import SentenceTransformer

class Backend_Program:
    def __init__(self, model_path, label_en_path):
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_en_path)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Initial Requirment Loaded")
        print()

    def predict_ticket_category(self, text):
        text_emb = self.embed_model.encode([text])
        prd = self.model.predict(text_emb)
        label = self.label_encoder.inverse_transform(prd)[0]
        return f"Predicted Category: {label}"

    def sample_data(self, data, num_samples=5):
        for i in range(num_samples):
            sample_test = data['Document'][i]
            labels = data['Topic_group'][i]
            print(f"{sample_test} :-- {labels}")