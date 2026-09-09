from pathlib import Path
import os
import pickle
import re

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai import OpenAI

# =========================
# Rutas base
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
CACHE_DIR = BASE_DIR / "cache"

TAX_PATH = DATA_DIR / "Tax.csv"
EXPORT_PATH = DATA_DIR / "exportaciones.csv"
APIKEY_PATH = BASE_DIR / "apikey.txt"
EMB_PATH = CACHE_DIR / "embeddings_tax.pkl"

# =========================
# Estado global del sistema
# =========================
RAG_INICIALIZADO = False
MODEL_REF = None
TAX_REF = None
EXPORTACIONES_REF = None
EMBEDDINGS_REF = None

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# =========================
# API KEY
# =========================
def cargar_api_key(path=APIKEY_PATH):
    with open(path, "r", encoding="utf-8") as f:
        key = f.read().strip()

    os.environ["OPENAI_API_KEY"] = key
    return key


# =========================
# LLM
# =========================
def llamar_llm(prompt: str, model: str = "gpt-5.4-mini") -> str:
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("No se encontró la variable OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text.strip()


def reescribir_consulta_con_llm(consulta_usuario: str) -> str:
    prompt = f"""
Eres un normalizador de consultas para comercio internacional.

Tarea:
- Corrige ortografía.
- Reduce diminutivos y variantes a su producto base.
- Conserva el significado.
- Devuelve SOLO una línea.
- Devuelve SOLO el nombre del producto principal, que debe ser una fruta, verdura, hortaliza o en general tambien frutos secos, como nueces. Tambien considera las bebidas.
- Sin comillas.
- Sin explicación.

Ejemplos:
tequilita -> tequila
tekila -> tequila
manguitos -> mango
limoncito -> limon
nuez -> nueces

Consulta: {consulta_usuario}
""".strip()

    respuesta = llamar_llm(prompt)
    return clean_text(respuesta)


def es_producto_valido(consulta_usuario: str) -> bool:
    prompt = f"""
Responde únicamente con SI o NO.

La siguiente consulta se refiere claramente a una fruta, verdura, hortaliza, grano, fruto seco o bebida?

Consulta: {consulta_usuario}
""".strip()

    try:
        respuesta = llamar_llm(prompt).strip().lower()
        return respuesta.startswith("si")
    except Exception:
        return True


# =========================
# Limpieza
# =========================
def clean_text(x):
    if pd.isna(x):
        return ""

    text = str(x).strip().lower()
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text


def clean_money(x):
    if pd.isna(x):
        return 0.0

    text = str(x).strip()
    text = re.sub(r"[^0-9,]", "", text)

    if not text:
        return 0.0

    text = text.replace(",", ".")
    return float(text)


# =========================
# Utilidades
# =========================
def estimar_potencial_exportacion(valor_norte_america, valor_espana):
    poblacion_norte_america = 10_930_000.0
    poblacion_espana = 79_581.0

    gasto_norte_america = valor_norte_america / poblacion_norte_america
    gasto_espana = valor_espana / poblacion_espana if poblacion_espana else 0.0

    indice_actual = (
        gasto_espana / gasto_norte_america
        if gasto_norte_america
        else 0.0
    )

    indice_oportunidad = 1 - indice_actual

    valor_potencial = gasto_norte_america * poblacion_espana
    crecimiento_estimado = valor_potencial - valor_espana

    return {
        "gasto_norte_america": gasto_norte_america,
        "gasto_espana": gasto_espana,
        "indice_actual": indice_actual,
        "indice_oportunidad": indice_oportunidad,
        "valor_potencial": valor_potencial,
        "crecimiento_estimado": crecimiento_estimado,
    }


def obtener_valor_por_pais(datos_fraccion, pais_objetivo):
    pais_objetivo = clean_text(pais_objetivo)

    datos_pais = datos_fraccion[
        datos_fraccion["pais"] == pais_objetivo
    ]

    if datos_pais.empty:
        return 0.0

    return float(datos_pais["valor_exportado"].sum())


def buscar_exportaciones_por_fraccion(
    fraccion_consulta,
    exportaciones
):
    resultados = exportaciones[
        exportaciones["fraccion"] == fraccion_consulta
    ].copy()

    return resultados


# =========================
# Embeddings
# =========================
def generar_embeddings_tax(model, textos):
    print("Generando embeddings...")

    embeddings = model.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings


def cargar_o_generar_embeddings(model, textos):
    if EMB_PATH.exists():
        with open(EMB_PATH, "rb") as f:
            cache = pickle.load(f)

        textos_cache = cache.get("textos")
        embeddings_cache = cache.get("embeddings")

        if textos_cache == textos:
            print("Embeddings cargados desde cache.")
            return embeddings_cache

        print("El texto cambió. Regenerando embeddings...")

    embeddings = generar_embeddings_tax(model, textos)

    with open(EMB_PATH, "wb") as f:
        pickle.dump({
            "textos": textos,
            "embeddings": embeddings
        }, f)

    print("Embeddings guardados en cache.")

    return embeddings


# =========================
# Búsqueda semántica
# =========================
def buscar_con_sbert_tax(
    consulta,
    base,
    model,
    embeddings,
    top_k=5,
    umbral=0.65
):
    consulta_limpia = clean_text(consulta)

    if "texto_busqueda" in base.columns:
        textos = base["texto_busqueda"].map(clean_text).tolist()
    else:
        textos = [
            (
                f"{clean_text(base.iloc[i]['uso'])} "
                f"{clean_text(base.iloc[i]['fraccion'])}"
            )
            for i in range(len(base))
        ]

    usos = base["uso"].map(clean_text).tolist()

    # 1. Coincidencia exacta
    exact_idx = [
        i
        for i, uso in enumerate(usos)
        if uso == consulta_limpia
    ]

    if exact_idx:
        idx = exact_idx[:top_k]

        resultados = base.iloc[idx].copy()
        resultados["score"] = 1.0

        return resultados

    # 2. Coincidencia por palabra
    palabra_idx = [
        i
        for i, uso in enumerate(usos)
        if re.search(
            rf"\b{re.escape(consulta_limpia)}\b",
            uso
        )
    ]

    if palabra_idx:

        def prioridad(i):
            uso = usos[i]

            if uso == consulta_limpia:
                return 0

            if uso.startswith(consulta_limpia + " "):
                return 1

            if consulta_limpia in uso:
                return 2

            return 3

        idx = sorted(
            palabra_idx,
            key=prioridad
        )[:top_k]

        resultados = base.iloc[idx].copy()

        resultados["score"] = [
            0.98 - (0.01 * prioridad(i))
            for i in idx
        ]

        return resultados

    # 3. Coincidencia aproximada
    aprox_idx = [
        i
        for i, uso in enumerate(usos)
        if (
            uso.startswith(consulta_limpia)
            or uso.startswith(consulta_limpia + "s")
            or consulta_limpia in uso
        )
    ]

    if aprox_idx:

        def prioridad_aprox(i):
            uso = usos[i]

            if uso.startswith(consulta_limpia):
                return 0

            if consulta_limpia in uso:
                return 1

            return 2

        idx = sorted(
            aprox_idx,
            key=prioridad_aprox
        )[:top_k]

        resultados = base.iloc[idx].copy()

        resultados["score"] = [
            0.93 - (0.01 * prioridad_aprox(i))
            for i in idx
        ]

        return resultados

    # 4. Fallback semántico SBERT
    consulta_emb = model.encode(
        [consulta_limpia],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores = cosine_similarity(
        consulta_emb,
        embeddings
    )[0]

    idx_ordenado = np.argsort(scores)[::-1]

    idx_filtrado = [
        i
        for i in idx_ordenado
        if scores[i] >= umbral
    ]

    if not idx_filtrado:
        resultados = base.iloc[0:0].copy()
        resultados["score"] = []

        return resultados

    idx = idx_filtrado[:top_k]

    resultados = base.iloc[idx].copy()
    resultados["score"] = scores[idx]

    return resultados


# =========================
# Inicialización UNA vez
# =========================
def inicializar_rag():
    global RAG_INICIALIZADO
    global MODEL_REF
    global TAX_REF
    global EXPORTACIONES_REF
    global EMBEDDINGS_REF

    if RAG_INICIALIZADO:
        return

    print("Inicializando RAG...")

    CACHE_DIR.mkdir(exist_ok=True)

    if APIKEY_PATH.exists():
        cargar_api_key()

    # Import pesado SOLO cuando llega la primera consulta.
    # Así Render puede levantar FastAPI/Uvicorn sin cargar PyTorch.
    print("Cargando SentenceTransformer...")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    print("Cargando archivos de datos...")

    tax = pd.read_csv(
        TAX_PATH,
        sep=";"
    )

    exportaciones = pd.read_csv(
        EXPORT_PATH,
        sep=";"
    )

    tax.columns = [
        str(c).strip()
        for c in tax.columns
    ]

    exportaciones.columns = [
        str(c).strip()
        for c in exportaciones.columns
    ]

    tax = tax.rename(columns={
        "Fracción": "fraccion",
        "Uso": "uso",
        "Impuesto importación": "impuesto_importacion"
    })

    exportaciones = exportaciones.rename(columns={
        "Fracción": "fraccion",
        "Pais": "pais",
        "Gasto": "valor_exportado"
    })

    tax["fraccion"] = tax["fraccion"].map(clean_text)
    tax["uso"] = tax["uso"].map(clean_text)

    tax["impuesto_importacion"] = (
        tax["impuesto_importacion"].map(clean_text)
    )

    exportaciones["fraccion"] = (
        exportaciones["fraccion"].map(clean_text)
    )

    exportaciones["pais"] = (
        exportaciones["pais"].map(clean_text)
    )

    exportaciones["valor_exportado"] = (
        exportaciones["valor_exportado"].map(clean_money)
    )

    tax["texto_busqueda"] = (
        tax["uso"].map(clean_text)
    )

    textos_tax = (
        tax["texto_busqueda"].tolist()
    )

    embeddings_tax = cargar_o_generar_embeddings(
        model,
        textos_tax
    )

    MODEL_REF = model
    TAX_REF = tax
    EXPORTACIONES_REF = exportaciones
    EMBEDDINGS_REF = embeddings_tax

    RAG_INICIALIZADO = True

    print("RAG inicializado correctamente.")


# =========================
# Consulta principal
# =========================
def consultar_producto_web(consulta_usuario: str):
    inicializar_rag()

    if not es_producto_valido(consulta_usuario):
        return {
            "ok": False,
            "consulta_original": consulta_usuario,
            "consulta_interpretada": "",
            "mensaje": (
                "Tu consulta no parece corresponder a una fruta, "
                "verdura, hortaliza o bebida identificable en la base. "
                "Por favor, vuelve a escribir el nombre del producto "
                "que deseas analizar."
            ),
            "resultados": [],
            "fraccion_seleccionada": "",
            "valor_norte_america": 0.0,
            "valor_espana": 0.0,
            "metricas": None
        }

    try:
        consulta_mejorada = reescribir_consulta_con_llm(
            consulta_usuario
        )
    except Exception:
        consulta_mejorada = clean_text(
            consulta_usuario
        )

    tax = TAX_REF
    model = MODEL_REF
    embeddings_tax = EMBEDDINGS_REF
    exportaciones = EXPORTACIONES_REF

    resultados_tax = buscar_con_sbert_tax(
        consulta_mejorada,
        tax,
        model,
        embeddings_tax,
        top_k=5
    )

    if (
        resultados_tax.empty
        or float(resultados_tax.iloc[0]["score"]) < 0.65
    ):
        return {
            "ok": False,
            "consulta_original": consulta_usuario,
            "consulta_interpretada": consulta_mejorada,
            "mensaje": (
                "No se encontraron coincidencias "
                "suficientemente confiables."
            ),
            "resultados": [],
            "fraccion_seleccionada": "",
            "valor_norte_america": 0.0,
            "valor_espana": 0.0,
            "metricas": None
        }

    fraccion_seleccionada = (
        resultados_tax.iloc[0]["fraccion"]
    )

    datos_exportacion = buscar_exportaciones_por_fraccion(
        fraccion_seleccionada,
        exportaciones
    )

    valor_norte_america = obtener_valor_por_pais(
        datos_exportacion,
        "américa del norte"
    )

    valor_espana = obtener_valor_por_pais(
        datos_exportacion,
        "españa"
    )

    resultado_estimacion = estimar_potencial_exportacion(
        valor_norte_america,
        valor_espana
    )

    return {
        "ok": True,
        "consulta_original": consulta_usuario,
        "consulta_interpretada": consulta_mejorada,
        "mensaje": "Consulta procesada correctamente.",
        "resultados": resultados_tax.to_dict(
            orient="records"
        ),
        "fraccion_seleccionada": fraccion_seleccionada,
        "valor_norte_america": valor_norte_america,
        "valor_espana": valor_espana,
        "metricas": resultado_estimacion
    }


def responder(pregunta: str):
    return consultar_producto_web(pregunta)