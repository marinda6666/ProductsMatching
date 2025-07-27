from fastapi import FastAPI, File, UploadFile, Query
from PIL import Image
import base64
import io
import asyncio
from fastapi import HTTPException
from triton_client import InferenceModule  # Импортируем ваш модуль инференса


THRESHOLD = 0.7

app = FastAPI()

inference_module = InferenceModule()  # Создаем экземпляр модуля инференса

@app.post("/predict/", description="Выполняет классификацию изображения с использованием указанной модели.")
async def predict(
    text1: str,
    text2: str
):
    """
    Выполнить классификацию изображения.

    Args:
        file (UploadFile): Загружаемое изображение.
        model_name (str): Имя модели для использования в инференсе.

    Returns:
        dict: Название класса и значение логита.
    """
    try:

        result = await inference_module.infer_text(text1, text2)

        class_id = result["all_probabilities"]

        return result["all_probabilities"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")