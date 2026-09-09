from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# TEMPORAL: desactivamos el RAG para comprobar si Render puede arrancar
# from rag import responder

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
        "mensaje": "API de tIAnqui funcionando"
    }

@app.post("/analizar")
def analizar_producto(request: ProductoRequest):
    return {
        "ok": True,
        "mensaje": "API funcionando sin cargar RAG",
        "producto": request.producto
    }