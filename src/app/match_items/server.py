from fastapi import FastAPI, File, UploadFile, Query
import asyncio
from fastapi import HTTPException
from triton_inference import InferenceModule

THRESHOLD = 0.7

app = FastAPI()

inference_module = InferenceModule()


@app.get("/predict/")
async def predict(text1: str, text2: str):
    try:

        result = await inference_module.infer_text(text1, text2)

        class_id = result["all_probabilities"]

        return result["all_probabilities"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
