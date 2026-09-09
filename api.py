from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import rag

# PRUEBA TEMPORAL:
# Cargamos la librería sentence-transformers,
# pero NO cargamos todavía el modelo SBERT.
from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="tIAnqui API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProductoRequest(BaseModel):
    producto: str


@app.get("/")
def inicio():
    return {
        "ok": True,
        "mensaje": "API funcionando con sentence-transformers importado"
    }


@app.post("/analizar")
def analizar_producto(request: ProductoRequest):
    return {
        "ok": True,
        "mensaje": "SentenceTransformer importado, modelo todavía no cargado",
        "producto": request.producto
    }