import torch
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from model import StarClassifier

num_classes = 6
in_features = 6
weights_path = "checkpoints/"
modelname = "StarClassifier_90.pth"
model_path = weights_path + modelname

def load_model() -> StarClassifier:
    model = StarClassifier(in_features, 64, num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    return model

class Star(BaseModel):
    temperature: float
    luminosity: float
    radius: float
    abs_magn: float
    color: int
    spectral_class: int

async def predict_star(model: StarClassifier, star: Star) -> list[float]:
    star_np = np.array([star.temperature, star.luminosity, star.radius, star.abs_magn, star.color, star.spectral_class], dtype=np.float32)
    logits = model(torch.tensor(star_np).unsqueeze(dim=0))

    probs = logits.softmax(dim=1)

    return probs.tolist()
    
app = FastAPI()
model = load_model()

@app.post("/predict")
async def get_prediction(star: Star):
    return await predict_star(model, star)


