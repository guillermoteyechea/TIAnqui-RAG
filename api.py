from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# PRUEBA TEMPORAL:
# Importamos rag.py, pero NO ejecutamos todavía responder()
# ni inicializamos el modelo SBERT.
import rag

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
        "mensaje": "API de tIAnqui funcionando con módulo RAG importado"
    }


@app.post("/analizar")
def analizar_producto(request: ProductoRequest):
    return {
        "ok": True,
        "mensaje": "API funcionando con RAG importado, pero sin ejecutar el modelo",
        "producto": request.producto
    }