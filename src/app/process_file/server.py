from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pandas as pd
from io import BytesIO

from utils import process_table

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
@app.get("/upload-csv", response_class=HTMLResponse)
async def get_upload_form(request: Request):
    return templates.TemplateResponse("load_file.html", {"request": request, "table": None, "columns": None, "error": None})

@app.post("/", response_class=HTMLResponse)
@app.post("/upload-csv", response_class=HTMLResponse)
async def upload_csv(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        return templates.TemplateResponse(
            "load_file.html",
            {"request": request, "table": None, "columns": None, "error": "Можно загружать только CSV файлы!"}
        )
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        table = process_table(df)
        table = table.values.tolist()
        columns = list(df.columns)
        return templates.TemplateResponse(
            "load_file.html",
            {"request": request, "table": table, "columns": columns, "error": None}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "load_file.html",
            {"request": request, "table": None, "columns": None, "error": f"Ошибка обработки файла: {e}"}
        )
