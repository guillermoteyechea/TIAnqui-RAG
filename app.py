import streamlit as st
from rag import responder

IMAGENES_POR_CODIGO = {
    "0701": "Images/papa.jpeg",
    "0702": "Images/tomate.jpeg",
    "0703": "Images/ajo.jpeg",
    "0704": "Images/canasta.jpeg",
    "0705": "Images/canasta.jpeg",
    "0706": "Images/zanahoria.jpeg",
    "0707": "Images/canasta.jpeg",
    "0708": "Images/canasta.jpeg",
    "0709": "Images/canasta.jpeg",
    "0710": "Images/zanahoria.jpeg",
    "0711": "Images/zanahoria.jpeg",
    "0712": "Images/limon.jpeg",
    "0713": "Images/limon.jpeg",
    "0714": "Images/limon.jpeg",

    "0801": "Images/nuez.jpeg",
    "0802": "Images/nuez.jpeg",
    "0803": "Images/mandarina.jpeg",
    "0804": "Images/mandarina.jpeg",
    "0805": "Images/limon.jpeg",
    "0806": "Images/papaya.jpeg",
    "0807": "Images/papaya.jpeg",
    "0808": "Images/papaya.jpeg",
    "0809": "Images/papaya.jpeg",
    "0810": "Images/papaya.jpeg",
    "0811": "Images/papaya.jpeg",
    "0812": "Images/papaya.jpeg",
    "0813": "Images/papaya.jpeg",
    "0814": "Images/papaya.jpeg",

    "2201": "Images/Palonegro.jpeg",
    "2202": "Images/amaras.jpeg",
    "2203": "Images/dolores.jpeg",
    "2204": "Images/novia.jpeg",
    "2205": "Images/amaras.jpeg",
    "2206": "Images/dolores.jpeg",
    "2207": "Images/novia.jpeg",

    "220820": "Images/amaras.jpeg",
    "220830": "Images/dolores.jpeg",
    "220840": "Images/novia.jpeg",
    "220850": "Images/amaras.jpeg",
    "220860": "Images/dolores.jpeg",
    "220870": "Images/novia.jpeg",
    "2208900301": "Images/amaras.jpeg",
    "2208900391": "Images/dolores.jpeg",
    "2208900400": "Images/novia.jpeg",
    "2208900500": "Images/amaras.jpeg",
}

def obtener_imagen_por_fraccion(fraccion: str) -> str:
    codigo = str(fraccion).split()[0].replace("-", "").strip()

    for largo in [10, 6, 4]:
        clave = codigo[:largo]
        if clave in IMAGENES_POR_CODIGO:
            return IMAGENES_POR_CODIGO[clave]

    return "Images/canasta.jpeg"


st.set_page_config(page_title="Motor de Inteligencia Comercial", layout="centered")

st.title("Motor de Inteligencia Comercial")

st.caption(
    "Facilitando la internacionalización de productos mexicanos hacia España mediante IA."
)

st.image("Images/ramo.jpeg", use_container_width=True)

with st.expander("📖 Sobre el proyecto", expanded=True):

    st.markdown("""
### ¿Por qué nace este proyecto?

México depende en gran medida del comercio exterior y España representa un mercado con creciente potencial para los productos mexicanos. Durante la última década la población mexicana residente en España prácticamente se triplicó, pasando de poco más de 21.000 personas a alrededor de 61.000

Sin embargo, muchas PyMEs enfrentan dificultades para acceder a información sobre aranceles, impuestos, regulaciones y oportunidades comerciales.

Este proyecto utiliza Inteligencia Artificial para facilitar ese proceso y apoyar la toma de decisiones de exportación.

---

### ¿Qué puede hacer?

- Identificar la fracción arancelaria.
- Consultar impuestos de importación.
- Analizar exportaciones México–España.
- Estimar potencial comercial.
- Facilitar la internacionalización de PyMEs.

---

### Visión

Este prototipo evolucionará hacia una plataforma integral de inteligencia comercial que, además de proporcionar información estratégica para la exportación, facilitará la conexión entre productores mexicanos y potenciales compradores en España, generando oportunidades de negocio para las pequeñas y medianas empresas mexicanas interesadas en internacionalizar sus productos.
""")

pregunta = st.text_input("Escribe un producto:")

if st.button("Consultar") and pregunta:
    with st.spinner("Procesando consulta..."):
        r = responder(pregunta)

    if not r["ok"]:
        st.error(r["mensaje"])

    else:
        st.success("Consulta procesada correctamente. Si desea buscar otro producto, simplemente vuelva al buscador.")

        imagen = obtener_imagen_por_fraccion(r["fraccion_seleccionada"])
        st.image(imagen, width="stretch")

        st.subheader("🔎 Interpretación")
        st.write(f"**Original:** {r['consulta_original']}")
        st.write(f"**Interpretada:** {r['consulta_interpretada']}")
        
        # =========================
        # 📑 RESULTADOS SOBRE IMPUESTO A CONSIDERAR
        # =========================
        st.subheader("📑 Resultados encontrados sobre impuesto a considerar")

        for item in r["resultados"]:
            with st.container():
                st.markdown(f"""
                **Fracción:** {item['fraccion']}  
                **Uso:** {item['uso']}  
                **Impuesto:** {item['impuesto_importacion']}  
                **Score:** {item['score']:.3f}
                """)
                st.divider()

        st.subheader("📦 Fracción seleccionada")
        st.write(r["fraccion_seleccionada"])

        # =========================
        # 📊 EXPORTACIONES
        # =========================
        st.subheader("📊 Exportaciones")

        m = r["metricas"]
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Valor exportado a Norteamérica", f"${r['valor_norte_america']:,.0f}")
            st.metric("Población de referencia Norteamérica", f"{10_930_000:,.0f}")
            st.metric("Gasto por persona Norteamérica", f"${m['gasto_norte_america']:,.2f}")

        with col2:
            st.metric("Valor exportado a España", f"${r['valor_espana']:,.0f}")
            st.metric("Población de referencia España", f"{79_581:,.0f}")
            st.metric("Gasto por persona España", f"${m['gasto_espana']:,.2f}")

        # =========================
        # 📈 POTENCIAL (AQUÍ ESTABA EL ERROR)
        # =========================
        st.subheader("📈 Potencial")

        crecimiento = float(m["crecimiento_estimado"])
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Índice actual", f"{m['indice_actual']:.2f}")
            st.metric("Índice oportunidad", f"{m['indice_oportunidad']:.2f}")

        with col2:
            st.metric("Valor potencial", f"${m['valor_potencial']:,.0f}")

            if abs(crecimiento) < 1e-6:
                st.metric("Crecimiento estimado", "$0")
            else:
                st.metric("Crecimiento estimado", f"${crecimiento:,.0f}")

        # Mensaje interpretativo
        if abs(crecimiento) < 1e-6:
            st.info("No contamos con datos suficientes para poder estimar un valor que le pueda ser de utilidad.")
        elif crecimiento < 0:
            st.info("El producto presenta una penetración relevante en España, por lo que se trata de un mercado ya bien aceptado y con un nivel de madurez considerable.")
        else:
            st.success("Se observa una oportunidad de crecimiento estimado en el mercado español.")




