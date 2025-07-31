from datetime import datetime

from .utils import save_torch_model


def save_model_step(model):
    now = datetime.now()
    formatted = now.strftime("%d.%m_%H:%M")
    save_torch_model(model, f'/home/marinda/Документы/ml_projects/ProductsMatching/models/checkpoints/bert_{formatted}.pth')