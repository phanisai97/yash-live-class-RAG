import hashlib
import os
import tempfile
from pathlib import Path

import streamlit as st
from groq import Groq
from groq import APIConnectionError, APIStatusError, AuthenticationError, NotFoundError, RateLimitError

# Document loaders
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LLM
from langchain_groq import ChatGroq

# Pure langchain_core
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# PPTX custom loader
from pptx import Presentation


st.set_page_config(
    page_title="DocChat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Model IDs containing these terms are not suitable for ChatGroq text generation.
# The model catalog itself is always fetched live from the Groq API.
_NON_CHAT_MODEL_MARKERS = (
    "whisper",
    "distil-whisper",
    "tts",
    "playai-tts",
    "prompt-guard",
    "safeguard",
    "guard-2-",
    "orpheus",
)

_PREFERRED_MODEL_IDS = (
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
)


@st.cache_resource(show_spinner="Loading sentence-transformer (first run only)...")
def load_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _api_key_cache_token(api_key: str) -> str:
    """Create a non-reversible cache token so the raw API key is not a cache key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_groq_model_ids(_api_key: str, cache_token: str) -> list[str]:
    """Fetch the live Groq model catalog. cache_token changes when the key changes."""
    del cache_token
    client = Groq(api_key=_api_key)
    response = client.models.list()
    return sorted({model.id for model in response.data if getattr(model, "id", None)})


def get_groq_models(api_key: str) -> tuple[list[str], list[str]]:
    all_models = _fetch_groq_model_ids(api_key, _api_key_cache_token(api_key))
    chat_models = [
        model_id
        for model_id in all_models
        if not any(marker in model_id.lower() for marker in _NON_CHAT_MODEL_MARKERS)
    ]
    excluded_models = [model_id for model_id in all_models if model_id not in chat_models]
    return chat_models, excluded_models


def choose_default_model(models: list[str]) -> int:
    for preferred in _PREFERRED_MODEL_IDS:
        if preferred in models:
            return models.index(preferred)
    return 0


def _load_pdf(path: str) -> list[Document]:
    return PyPDFLoader(path).load()


def _load_docx(path: str) -> list[Document]:
    return Docx2txtLoader(path).load()


def _load_pptx(path: str) -> list[Document]:
    prs = Presentation(path)
    docs = []
    for idx, slide in enumerate(prs.slides, start=1):
        text = "\n".join(
            shape.text.strip()
            for shape in slide.shapes
            if hasattr(shape, "text") and shape.text.strip()
        )
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": path, "slide": idx},
                )
            )
    return docs


def ingest_uploaded_file(uploaded_file) -> list[Document]:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        path = tmp.name

    try:
        if suffix == ".pdf":
            docs = _load_pdf(path)
        elif suffix == ".docx":
            docs = _load_docx(path)
        elif suffix == ".pptx":
            docs = _load_pptx(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        os.unlink(path)

    if not docs or not any(doc.page_content.strip() for doc in docs):
        raise ValueError("No readable text was extracted from the uploaded document.")
    return docs


_QA_PROMPT = ChatPromptTemplate.from_template(
    "You are a helpful assistant answering questions about an uploaded document.\n"
    "Use only the context below to answer. If the answer is not in the context, say so.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_rag_components(docs: list[Document], groq_api_key: str, model_name: str):
    embeddings = load_embedding_model()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("The document did not produce any searchable chunks.")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_name,
        temperature=0.2,
        max_retries=2,
    )

    qa_chain = _QA_PROMPT | llm | StrOutputParser()
    return retriever, qa_chain, len(chunks)


def answer_question(retriever, qa_chain, question: str):
    source_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)
    answer = qa_chain.invoke({"context": context, "question": question})
    return answer, source_docs


def friendly_groq_error(exc: Exception, model_name: str) -> str:
    if isinstance(exc, AuthenticationError):
        return "Groq rejected the API key. Verify the key in the sidebar and try again."
    if isinstance(exc, NotFoundError):
        _fetch_groq_model_ids.clear()
        return (
            f"Groq reports that '{model_name}' is unavailable. Refresh the model list, "
            "select another live model, and retry."
        )
    if isinstance(exc, RateLimitError):
        return "Groq rate limit or token quota reached. Wait briefly or use a model with available quota."
    if isinstance(exc, APIConnectionError):
        return "Could not connect to Groq. Check network access and retry."
    if isinstance(exc, APIStatusError):
        return f"Groq API request failed with HTTP {exc.status_code}. Try another model or review the app logs."
    return "The model request failed. Review the Streamlit Cloud logs for the complete server-side error."


def render_stage(title: str, body: str):
    st.markdown(f"**{title}**")
    st.caption(body)


def page_architecture():
    st.title("Pipeline Architecture")
    st.caption(
        "A visual walkthrough of the data-ingestion and query pipelines that power this app."
    )

    st.markdown("### 1  Data Ingestion Pipeline")
    st.caption("Runs when a new document is uploaded.")

    i_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    ingestion = [
        ("Upload", "PDF, DOCX, or PPTX received through Streamlit."),
        ("Extract", "Text extracted per page, document, or slide."),
        ("Chunk", "Recursive splitting into 500-character windows with 50-character overlap."),
        ("Embed", "all-MiniLM-L6-v2 maps each chunk to a normalized 384-dimensional vector."),
        ("Index", "Vectors are stored in an in-memory FAISS index."),
    ]
    for col, (title, desc) in zip(i_cols[::2], ingestion):
        with col:
            render_stage(title, desc)
    for col in i_cols[1::2]:
        with col:
            st.write("→")

    st.divider()
    st.markdown("### 2  Query Pipeline")
    st.caption("Runs on every message.")

    q_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    query = [
        ("Question", "The user submits a question."),
        ("Retrieve", "FAISS returns the four closest chunks."),
        ("Prompt", "Retrieved chunks are inserted into a grounded prompt."),
        ("Groq LLM", "The selected live Groq chat model generates the answer."),
        ("Reply", "The answer and source-chunk previews are displayed."),
    ]
    for col, (title, desc) in zip(q_cols[::2], query):
        with col:
            render_stage(title, desc)
    for col in q_cols[1::2]:
        with col:
            st.write("→")

    st.divider()
    st.markdown("### 3  Component Details")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Embedding Model**")
        st.code(
            "sentence-transformers/all-MiniLM-L6-v2\n"
            "Package    : langchain-huggingface\n"
            "Dimensions : 384\n"
            "Device     : CPU\n"
            "Caching    : @st.cache_resource",
            language=None,
        )
        st.markdown("**Vector Store**")
        st.code(
            "Package : langchain-community + faiss-cpu\n"
            "Search  : k-NN, k=4\n"
            "Storage : in-memory per upload",
            language=None,
        )

    with c2:
        st.markdown("**Text Splitter**")
        st.code(
            "Package    : langchain-text-splitters\n"
            "Type       : RecursiveCharacterTextSplitter\n"
            "Chunk size : 500 characters\n"
            "Overlap    : 50 characters",
            language=None,
        )
        st.markdown("**RAG Chain (LCEL)**")
        st.code(
            "ChatPromptTemplate\n"
            "  | ChatGroq\n"
            "  | StrOutputParser\n\n"
            "No deprecated chains.\n"
            "Each question is answered independently.",
            language=None,
        )

    with c3:
        st.markdown("**LLM (BYOK via Groq)**")
        st.code(
            "Package  : langchain-groq\n"
            "Provider : Groq Cloud API\n"
            "Auth     : user API key, not persisted\n"
            "Models   : fetched live from /openai/v1/models\n"
            "Refresh  : every 5 minutes or on demand\n"
            "Filter   : chat-generation models only",
            language=None,
        )

    st.divider()
    st.markdown("### 4  Supported File Formats")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("**PDF**")
        st.caption("PyPDFLoader extracts text per page with page metadata.")
    with f2:
        st.markdown("**DOCX**")
        st.caption("Docx2txtLoader extracts Word document text.")
    with f3:
        st.markdown("**PPTX**")
        st.caption("python-pptx extracts text per slide with slide metadata.")


def page_chat(uploaded_file, groq_api_key: str, model_name: str | None):
    st.title("Chat with Your Document")

    defaults = {
        "messages": [],
        "retriever": None,
        "qa_chain": None,
        "loaded_signature": None,
        "stats": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if uploaded_file is not None:
        if not groq_api_key:
            st.warning("Enter your Groq API key in the sidebar to continue.")
            return
        if not model_name:
            st.warning("Select an available Groq chat model in the sidebar.")
            return

        file_hash = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        current_signature = (uploaded_file.name, file_hash, model_name)

        if st.session_state.loaded_signature != current_signature:
            try:
                with st.status("Processing document...", expanded=True) as status:
                    st.write("Loading the embedding model...")
                    load_embedding_model()

                    st.write(f"Extracting text from **{uploaded_file.name}**...")
                    docs = ingest_uploaded_file(uploaded_file)
                    st.write(f"Extracted {len(docs)} page(s) or slide(s).")

                    st.write(f"Building the FAISS index and configuring **{model_name}**...")
                    retriever, qa_chain, n_chunks = build_rag_components(
                        docs, groq_api_key, model_name
                    )

                    st.session_state.retriever = retriever
                    st.session_state.qa_chain = qa_chain
                    st.session_state.loaded_signature = current_signature
                    st.session_state.messages = []
                    st.session_state.stats = {
                        "file": uploaded_file.name,
                        "pages": len(docs),
                        "chunks": n_chunks,
                        "model": model_name,
                    }
                    status.update(
                        label="Ready. Start chatting below.",
                        state="complete",
                        expanded=False,
                    )
            except Exception as exc:
                st.error(f"Document processing failed: {exc}")
                return

    if st.session_state.stats:
        s = st.session_state.stats
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Document", s["file"])
        m2.metric("Pages / Slides", s["pages"])
        m3.metric("FAISS Chunks", s["chunks"])
        m4.metric("Groq Model", s["model"])
        st.divider()

    if st.session_state.retriever is None:
        st.info(
            "Enter a Groq API key, select a live chat model, and upload a PDF, DOCX, or PPTX."
        )
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if user_input := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        answer = None
        sources = []
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and generating..."):
                    answer, sources = answer_question(
                        st.session_state.retriever,
                        st.session_state.qa_chain,
                        user_input,
                    )
                st.write(answer)

                if sources:
                    with st.expander("Source chunks used", expanded=False):
                        for i, doc in enumerate(sources[:4], 1):
                            meta = doc.metadata
                            label = f"Chunk {i}"
                            if "page" in meta:
                                label += f" | Page {int(meta['page']) + 1}"
                            elif "slide" in meta:
                                label += f" | Slide {meta['slide']}"
                            st.caption(label)
                            preview = doc.page_content
                            if len(preview) > 350:
                                preview = preview[:350] + "..."
                            st.text(preview)
                            if i < min(4, len(sources)):
                                st.markdown("---")
            except Exception as exc:
                st.error(friendly_groq_error(exc, model_name or "unknown"))
                with st.expander("Technical details", expanded=False):
                    st.code(f"{type(exc).__name__}: {exc}")

        if answer is not None:
            st.session_state.messages.append({"role": "assistant", "content": answer})


def main():
    with st.sidebar:
        st.title("DocChat")
        st.caption("RAG over your own documents")
        st.divider()

        st.subheader("Groq API Key")
        groq_api_key = st.text_input(
            "Paste your key here",
            type="password",
            placeholder="gsk_...",
            help="Create a key at console.groq.com. The app does not persist it.",
            label_visibility="collapsed",
        )
        if groq_api_key:
            st.caption("Key received and used only for this session.")
        else:
            st.caption("Get a key at [console.groq.com](https://console.groq.com).")

        st.divider()
        st.subheader("Model")
        model_name = None

        if groq_api_key:
            refresh = st.button("Refresh Groq model list", use_container_width=True)
            if refresh:
                _fetch_groq_model_ids.clear()

            try:
                chat_models, excluded_models = get_groq_models(groq_api_key)
                if chat_models:
                    model_name = st.selectbox(
                        "Groq chat model",
                        options=chat_models,
                        index=choose_default_model(chat_models),
                        label_visibility="collapsed",
                        help="Fetched from your Groq account. Retired models disappear automatically.",
                    )
                    st.caption(f"{len(chat_models)} chat-capable model(s) available.")
                    if excluded_models:
                        with st.expander("Non-chat models excluded", expanded=False):
                            st.code("\n".join(excluded_models), language=None)
                else:
                    st.error("Groq returned no chat-generation models for this API key.")
            except AuthenticationError:
                st.error("Invalid Groq API key.")
            except Exception as exc:
                st.error("Could not load the live Groq model catalog.")
                with st.expander("Technical details", expanded=False):
                    st.code(f"{type(exc).__name__}: {exc}")
        else:
            st.caption("Enter the API key to load the current Groq model catalog.")

        st.divider()
        st.subheader("Navigation")
        page = st.radio(
            "Navigation",
            options=["Chat", "Architecture"],
            label_visibility="collapsed",
        )

        st.divider()
        uploaded_file = None
        if page == "Chat":
            st.subheader("Upload Document")
            uploaded_file = st.file_uploader(
                "Document",
                type=["pdf", "docx", "pptx"],
                help="PDF, Word (.docx), or PowerPoint (.pptx)",
                label_visibility="collapsed",
            )

        st.divider()
        st.caption("LangChain · FAISS · Hugging Face · Groq")

    if page == "Architecture":
        page_architecture()
    else:
        page_chat(uploaded_file, groq_api_key, model_name)


if __name__ == "__main__":
    main()

# import streamlit as st
# import os
# import tempfile
# from pathlib import Path

# # Document loaders
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain_community.vectorstores import FAISS

# # Text splitter (moved to own package in LangChain 1.0)
# from langchain_text_splitters import RecursiveCharacterTextSplitter

# # Embeddings
# from langchain_huggingface import HuggingFaceEmbeddings

# # LLM
# from langchain_groq import ChatGroq

# # Pure langchain_core — no deprecated chains, no langchain.chains at all
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.documents import Document

# # PPTX custom loader
# from pptx import Presentation

# # ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config(
#     page_title="DocChat",
#     page_icon="",
#     layout="wide",
#     initial_sidebar_state="expanded",
# )

# # ─── Cached Embedding Model ───────────────────────────────────────────────────
# @st.cache_resource(show_spinner="Loading sentence-transformer (first run only)...")
# def load_embedding_model() -> HuggingFaceEmbeddings:
#     return HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )


# # ─── Document Loaders ─────────────────────────────────────────────────────────

# def _load_pdf(path: str) -> list[Document]:
#     return PyPDFLoader(path).load()


# def _load_docx(path: str) -> list[Document]:
#     return Docx2txtLoader(path).load()


# def _load_pptx(path: str) -> list[Document]:
#     prs = Presentation(path)
#     docs = []
#     for idx, slide in enumerate(prs.slides, start=1):
#         text = "\n".join(
#             shape.text.strip()
#             for shape in slide.shapes
#             if hasattr(shape, "text") and shape.text.strip()
#         )
#         if text:
#             docs.append(Document(
#                 page_content=text,
#                 metadata={"source": path, "slide": idx},
#             ))
#     return docs


# def ingest_uploaded_file(uploaded_file) -> list[Document]:
#     suffix = Path(uploaded_file.name).suffix.lower()
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(uploaded_file.getvalue())
#         path = tmp.name
#     try:
#         if suffix == ".pdf":
#             docs = _load_pdf(path)
#         elif suffix in (".docx", ".doc"):
#             docs = _load_docx(path)
#         elif suffix in (".pptx", ".ppt"):
#             docs = _load_pptx(path)
#         else:
#             raise ValueError(f"Unsupported file type: {suffix}")
#     finally:
#         os.unlink(path)
#     return docs


# # ─── RAG Components ───────────────────────────────────────────────────────────
# # Plain LCEL — no langchain.chains, no deprecated imports, no conversation history.
# # Each question is answered independently: embed → retrieve → prompt → LLM.

# _QA_PROMPT = ChatPromptTemplate.from_template(
#     "You are a helpful assistant answering questions about an uploaded document.\n"
#     "Use only the context below to answer. If the answer is not in the context, say so.\n\n"
#     "Context:\n{context}\n\n"
#     "Question: {question}\n\n"
#     "Answer:"
# )


# def build_rag_components(docs: list[Document], groq_api_key: str, model_name: str):
#     embeddings = load_embedding_model()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", ".", " ", ""],
#     )
#     chunks = splitter.split_documents(docs)
#     vectorstore = FAISS.from_documents(chunks, embeddings)
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

#     llm = ChatGroq(
#         groq_api_key=groq_api_key,
#         model_name=model_name,
#         temperature=0.2,
#         max_tokens=1024,
#     )

#     # LCEL chain: fills prompt then calls LLM, returns plain string
#     qa_chain = _QA_PROMPT | llm | StrOutputParser()

#     return retriever, qa_chain, len(chunks)


# def answer_question(retriever, qa_chain, question: str):
#     source_docs = retriever.invoke(question)
#     context = "\n\n".join(doc.page_content for doc in source_docs)
#     answer = qa_chain.invoke({"context": context, "question": question})
#     return answer, source_docs


# # ─── Architecture Page ────────────────────────────────────────────────────────

# def render_stage(title: str, body: str):
#     st.markdown(f"**{title}**")
#     st.caption(body)


# def page_architecture():
#     st.title("Pipeline Architecture")
#     st.caption(
#         "A visual walkthrough of the two pipelines that power this app: "
#         "data ingestion (upload) and query (chat)."
#     )

#     # ── 1. Data Ingestion ──────────────────────────────────────────────────
#     st.markdown("### 1  Data Ingestion Pipeline")
#     st.caption("Runs once when you upload a document.")

#     i_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
#     ingestion = [
#         ("Upload",   "PDF / DOCX / PPTX received via Streamlit file uploader."),
#         ("Extract",  "Text pulled page-by-page (PDF), paragraph-by-paragraph (DOCX), or slide-by-slide (PPTX)."),
#         ("Chunk",    "RecursiveCharacterTextSplitter cuts text into 500-token windows with 50-token overlap."),
#         ("Embed",    "all-MiniLM-L6-v2 maps each chunk to a 384-dim dense vector (CPU, cached on VM)."),
#         ("Index",    "Vectors inserted into an in-memory FAISS flat index, ready for similarity search."),
#     ]
#     for col, (title, desc) in zip(i_cols[::2], ingestion):
#         with col:
#             render_stage(title, desc)
#     for col in i_cols[1::2]:
#         with col:
#             st.write("→")

#     st.divider()

#     # ── 2. Query Pipeline ─────────────────────────────────────────────────
#     st.markdown("### 2  Query Pipeline")
#     st.caption("Runs on every message you send.")

#     q_cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
#     query = [
#         ("Question", "User types a question in the chat input."),
#         ("Retrieve",  "FAISS returns top-4 chunks closest to the question vector."),
#         ("Prompt",    "Chunks joined as context and inserted into a ChatPromptTemplate."),
#         ("Groq LLM",  "LLaMA / Mixtral / Gemma on Groq Cloud generates the grounded answer."),
#         ("Reply",     "Answer shown in chat. Source chunks in the expander below the reply."),
#     ]
#     for col, (title, desc) in zip(q_cols[::2], query):
#         with col:
#             render_stage(title, desc)
#     for col in q_cols[1::2]:
#         with col:
#             st.write("→")

#     st.divider()

#     # ── 3. Component Details ──────────────────────────────────────────────
#     st.markdown("### 3  Component Details")

#     c1, c2, c3 = st.columns(3)

#     with c1:
#         st.markdown("**Embedding Model**")
#         st.code(
#             "sentence-transformers/all-MiniLM-L6-v2\n"
#             "Package    : langchain-huggingface\n"
#             "Dimensions : 384\n"
#             "Device     : CPU\n"
#             "Caching    : @st.cache_resource\n"
#             "             downloaded once, lives in\n"
#             "             ~/.cache/huggingface on VM",
#             language=None,
#         )

#         st.markdown("**Vector Store**")
#         st.code(
#             "Package : langchain-community + faiss-cpu\n"
#             "Index   : Flat L2\n"
#             "Search  : k-NN, k=4\n"
#             "Storage : in-memory (per upload)",
#             language=None,
#         )

#     with c2:
#         st.markdown("**Text Splitter**")
#         st.code(
#             "Package    : langchain-text-splitters\n"
#             "Type       : RecursiveCharacterTextSplitter\n"
#             "Chunk size : 500 tokens\n"
#             "Overlap    : 50 tokens\n"
#             "Separators : \\n\\n → \\n → . → ' '",
#             language=None,
#         )

#         st.markdown("**RAG Chain (LCEL)**")
#         st.code(
#             "ChatPromptTemplate   (langchain_core)\n"
#             "  | ChatGroq         (langchain_groq)\n"
#             "  | StrOutputParser  (langchain_core)\n\n"
#             "No deprecated chains.\n"
#             "No conversation history.\n"
#             "Each question answered independently.",
#             language=None,
#         )

#     with c3:
#         st.markdown("**LLM  (BYOK via Groq)**")
#         st.code(
#             "Package     : langchain-groq\n"
#             "Provider    : Groq Cloud API\n"
#             "Auth        : user API key (never stored)\n"
#             "Models      :\n"
#             "  llama-3.3-70b-versatile\n"
#             "  llama-3.1-8b-instant\n"
#             "  mixtral-8x7b-32768\n"
#             "  gemma2-9b-it\n"
#             "Temperature : 0.2\n"
#             "Max tokens  : 1024",
#             language=None,
#         )

#     st.divider()

#     # ── 4. File Format Support ────────────────────────────────────────────
#     st.markdown("### 4  Supported File Formats")
#     f1, f2, f3 = st.columns(3)
#     with f1:
#         st.markdown("**PDF**")
#         st.caption("PyPDFLoader (pypdf). Text extracted per page with page-number metadata.")
#     with f2:
#         st.markdown("**DOCX**")
#         st.caption("Docx2txtLoader (docx2txt). Full document text extracted in one pass.")
#     with f3:
#         st.markdown("**PPTX**")
#         st.caption("Custom loader (python-pptx). Text per slide with slide-number metadata. No unstructured dependency.")


# # ─── Chat Page ────────────────────────────────────────────────────────────────

# def page_chat(uploaded_file, groq_api_key: str, model_name: str):
#     st.title("Chat with Your Document")

#     for key, val in {
#         "messages": [],
#         "retriever": None,
#         "qa_chain": None,
#         "loaded_file": None,
#         "stats": None,
#     }.items():
#         if key not in st.session_state:
#             st.session_state[key] = val

#     # ── Process new / changed upload ───────────────────────────────────────
#     if uploaded_file is not None:
#         if not groq_api_key:
#             st.warning("Enter your Groq API key in the sidebar to continue.")
#             return

#         if st.session_state.loaded_file != uploaded_file.name:
#             with st.status("Processing document...", expanded=True) as status:
#                 st.write("Warming up embedding model (cached after first run)...")
#                 load_embedding_model()

#                 st.write(f"Extracting text from **{uploaded_file.name}** ...")
#                 docs = ingest_uploaded_file(uploaded_file)
#                 st.write(f"Extracted {len(docs)} page(s) / slide(s).")

#                 st.write("Chunking, embedding, and building FAISS index...")
#                 retriever, qa_chain, n_chunks = build_rag_components(docs, groq_api_key, model_name)
#                 st.write(f"Indexed {n_chunks} chunks. Ready.")

#                 st.session_state.retriever = retriever
#                 st.session_state.qa_chain = qa_chain
#                 st.session_state.loaded_file = uploaded_file.name
#                 st.session_state.messages = []
#                 st.session_state.stats = {
#                     "file": uploaded_file.name,
#                     "pages": len(docs),
#                     "chunks": n_chunks,
#                 }
#                 status.update(label="Ready — start chatting below.", state="complete", expanded=False)

#     # ── Stats bar ──────────────────────────────────────────────────────────
#     if st.session_state.stats:
#         s = st.session_state.stats
#         m1, m2, m3 = st.columns(3)
#         m1.metric("Document", s["file"])
#         m2.metric("Pages / Slides", s["pages"])
#         m3.metric("Chunks in FAISS", s["chunks"])
#         st.divider()

#     # ── Empty state ────────────────────────────────────────────────────────
#     if st.session_state.retriever is None:
#         st.info(
#             "Upload a PDF, DOCX, or PPTX in the sidebar and enter your Groq API key "
#             "to start chatting with your document."
#         )
#         return

#     # ── Chat history (display only) ────────────────────────────────────────
#     for msg in st.session_state.messages:
#         with st.chat_message(msg["role"]):
#             st.write(msg["content"])

#     # ── Chat input ─────────────────────────────────────────────────────────
#     if user_input := st.chat_input("Ask something about your document..."):
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.write(user_input)

#         with st.chat_message("assistant"):
#             with st.spinner("Retrieving and generating..."):
#                 answer, sources = answer_question(
#                     st.session_state.retriever,
#                     st.session_state.qa_chain,
#                     user_input,
#                 )

#             st.write(answer)

#             if sources:
#                 with st.expander("Source chunks used", expanded=False):
#                     for i, doc in enumerate(sources[:4], 1):
#                         meta = doc.metadata
#                         label = f"Chunk {i}"
#                         if "page" in meta:
#                             label += f"  —  Page {int(meta['page']) + 1}"
#                         elif "slide" in meta:
#                             label += f"  —  Slide {meta['slide']}"
#                         st.caption(label)
#                         preview = doc.page_content
#                         if len(preview) > 350:
#                             preview = preview[:350] + "..."
#                         st.text(preview)
#                         if i < min(4, len(sources)):
#                             st.markdown("---")

#         st.session_state.messages.append({"role": "assistant", "content": answer})


# # ─── Sidebar + Entry Point ────────────────────────────────────────────────────

# def main():
#     with st.sidebar:
#         st.title("DocChat")
#         st.caption("RAG over your own documents")

#         st.divider()

#         st.subheader("Groq API Key")
#         groq_api_key = st.text_input(
#             "Paste your key here",
#             type="password",
#             placeholder="gsk_...",
#             help="Free key at console.groq.com — never stored or logged.",
#             label_visibility="collapsed",
#         )
#         if groq_api_key:
#             st.caption("Key received. Used only for this session.")
#         else:
#             st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

#         st.divider()

#         st.subheader("Model")
#         model_name = st.selectbox(
#             "model",
#             options=[
#                 "llama-3.3-70b-versatile",
#                 "llama-3.1-8b-instant",
#                 "mixtral-8x7b-32768",
#                 "gemma2-9b-it",
#             ],
#             label_visibility="collapsed",
#         )

#         st.divider()

#         st.subheader("Navigation")
#         page = st.radio(
#             "nav",
#             options=["Chat", "Architecture"],
#             label_visibility="collapsed",
#         )

#         st.divider()

#         uploaded_file = None
#         if page == "Chat":
#             st.subheader("Upload Document")
#             uploaded_file = st.file_uploader(
#                 "file",
#                 type=["pdf", "docx", "pptx"],
#                 help="PDF, Word (.docx), or PowerPoint (.pptx)",
#                 label_visibility="collapsed",
#             )

#         st.divider()
#         st.caption("LangChain · FAISS · HuggingFace · Groq")

#     if page == "Architecture":
#         page_architecture()
#     else:
#         page_chat(uploaded_file, groq_api_key, model_name)


# if __name__ == "__main__":
#     main()
