import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import platform
from gtts import gTTS
import io

# Configuración de estilos y colores personalizados (CSS)
st.markdown("""
    <style>
    /* Cambiar color de fondo y FORZAR color de texto oscuro */
    .st-response-box {
        background-color: #1e293b;
        color: #f8fafc;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #38bdf8;
        margin-top: 10px;
        font-size: 1.05em;
        line-height: 1.6;
    }
    /* Estilo para el contenedor de fuentes */
    .st-source-box {
        background-color: #0f172a;
        color: #94a3b8;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.9em;
        border: 1px solid #334155;
        margin-top: 8px;
    }
    .st-source-box b {
        color: #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# Título y presentación personalizados
st.title('📚 Asistente Inteligente de Lectura para Yos (RAG)')
st.caption(f"Motor ejecutado en Python v{platform.python_version()}")

# Cargar y mostrar imagen personalizada
try:
    image = Image.open('ia.jpg')
    st.image(image, width=320, caption="Consultas documentales interactivas")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Información de la barra lateral
with st.sidebar:
    st.header("⚙️ Configuración")
    st.subheader("Este Agente te ayudará a realizar análisis sobre el PDF cargado")
    ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Carga de archivo PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento del PDF conservando metadatos de página
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        documents = []

        # Extraer texto guardando la página de origen en los metadatos
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"page": i + 1}))

        st.info(f"Texto extraído: {sum(len(d.page_content) for d in documents)} caracteres en {len(pdf_reader.pages)} páginas.")

        # División en fragmentos manteniendo metadatos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        st.success(f"Documento dividido en {len(chunks)} fragmentos.")

        # Creación de la base de conocimiento en FAISS
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_documents(chunks, embeddings)

        # Interfaz de preguntas
        st.subheader("🔍 Realiza una consulta sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            # Búsqueda de similitud obteniendo los documentos con sus páginas
            docs = knowledge_base.similarity_search(user_question, k=3)

            llm = OpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            # Despliegue de la respuesta
            st.markdown("### 📝 Respuesta:")
            st.markdown(f'<div class="st-response-box">{response}</div>', unsafe_allow_html=True)

            # --- GENERACIÓN DE AUDIO ---
            try:
                tts = gTTS(text=response, lang='es')
                audio_bytes = io.BytesIO()
                tts.write_to_fp(audio_bytes)
                st.audio(audio_bytes.getvalue(), format="audio/mp3")
            except Exception as audio_err:
                st.error(f"No se pudo generar el audio: {audio_err}")

            # Identificación y despliegue de las páginas de origen
            pages = sorted(list(set(doc.metadata.get("page") for doc in docs if "page" in doc.metadata)))
            if pages:
                pages_str = ", ".join(f"Página {p}" for p in pages)
                st.markdown(f'<div class="st-source-box">📌 <b>Fuente:</b> Información obtenida de la(s) <b>{pages_str}</b> del documento PDF.</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("Por favor carga un archivo PDF para comenzar")
