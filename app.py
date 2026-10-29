import html
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.documents import Document

from src.chunker import chunk_data
from src.data_loader import load_pdf
from src.embedding import get_embeddings
from src.agent import stream_agent
from src.ocr_loader import load_image_text, load_pdf_with_ocr
from src.session_store import SessionVectorStore
from src.knowledge_base import QdrantKnowledgeBase, build_chat_collection_name
from llmops.guardrails import LLMGuardrails
from llmops.semantic_evaluator import SemanticRAGEvaluator
from llmops.telemetry import LLMOpsTelemetry
from llmops.text_utils import is_non_answer


APP_DIR = Path(__file__).resolve().parent
CHAT_FILE = APP_DIR / "chats.json"
LEGACY_CHAT_FILE = APP_DIR / "chat_history.json"
EVAL_CORPUS_PATH = APP_DIR / "llmops" / "data" / "eval_corpus.json"
CHATS_STORAGE_KEY = "rag_chats_v1"
CHATS_BRIDGE_PARAM = "st_chats_bridge"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

EXAMPLE_PROMPTS = [
    "What is RAG?",
    "Summarize the key points of my documents.",
]

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-color: #0a0f1c;
        --panel-bg: rgba(148, 163, 184, 0.06);
        --panel-bg-strong: rgba(30, 41, 59, 0.55);
        --panel-border: rgba(148, 163, 184, 0.14);
        --text-main: #e6edf7;
        --text-muted: #8b98ad;
        --accent: #6366f1;
        --accent-2: #a855f7;
        --accent-glow: rgba(99, 102, 241, 0.45);
        --user-bubble: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.12));
        --user-border: rgba(129, 140, 248, 0.32);
    }

    html, body, .stApp {
        background:
            radial-gradient(1100px 600px at 12% -5%, rgba(99, 102, 241, 0.14), transparent 55%),
            radial-gradient(900px 500px at 95% 10%, rgba(168, 85, 247, 0.12), transparent 55%),
            var(--bg-color) !important;
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    .block-container {
        max-width: 880px !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important;
    }

    h1, h2, h3, h4, p, label {
        color: var(--text-main) !important;
        font-family: 'Inter', sans-serif;
    }

    a {
        color: var(--accent) !important;
    }

    /* Welcome / empty state */
    .welcome-title {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.6rem 0 0.1rem;
    }

    .welcome-hint {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 0.5rem 0 0.7rem;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        background: rgba(148, 163, 184, 0.08) !important;
        color: var(--text-main) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 12px !important;
        min-height: 2.6rem !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: var(--user-border) !important;
        background: rgba(99, 102, 241, 0.12) !important;
        transform: translateY(-1px);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
        border: none !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 18px var(--accent-glow);
    }

    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 8px 26px var(--accent-glow);
        transform: translateY(-2px);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.9) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid var(--panel-border) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.2rem 0.1rem 0.9rem;
    }

    .brand-logo {
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
        border-radius: 9px;
        background: linear-gradient(135deg, var(--accent), #7c3aed);
        box-shadow: 0 3px 10px var(--accent-glow);
    }

    .brand-name {
        font-weight: 700;
        font-size: 0.98rem;
        letter-spacing: -0.01em;
        color: var(--text-main);
    }

    .brand-tag {
        font-size: 0.7rem;
        color: var(--text-muted);
    }

    .side-title {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin: 0.2rem 0 0.4rem;
    }

    .side-divider {
        border-top: 1px solid var(--panel-border);
        margin: 0.85rem 0;
    }

    .doc-chip {
        font-size: 0.9rem;
        color: var(--text-main);
        background: rgba(148, 163, 184, 0.06);
        border: 1px solid var(--panel-border);
        border-radius: 10px;
        padding: 0.5rem 0.75rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        font-family: 'Inter', sans-serif;
    }

    /* Citations */
    .cite-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.6rem;
    }

    .cite-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.8rem;
        color: var(--text-muted);
        background: rgba(148, 163, 184, 0.07);
        border: 1px solid var(--panel-border);
        border-radius: 999px;
        padding: 0.3rem 0.75rem;
        font-family: 'Inter', sans-serif;
    }

    .upload-note {
        font-size: 0.8rem;
        color: #4ade80;
        margin-top: 0.5rem;
    }

    /* Chat messages */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0.3rem 0 !important;
        margin-bottom: 0.2rem !important;
        backdrop-filter: none !important;
    }

    div[data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }

    div[data-testid="stChatMessageAvatarAssistant"] {
        background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.35) !important;
    }

    div[data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
        background: var(--user-bubble) !important;
        border: 1px solid var(--user-border) !important;
        border-radius: 18px 18px 6px 18px !important;
        padding: 0.5rem 0.9rem !important;
        max-width: 78% !important;
        margin-left: auto !important;
        flex: 0 1 auto !important;
    }

    [data-testid="stChatMessageContent"] {
        padding-top: 0.2rem !important;
    }

    [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessageContent"] li {
        color: var(--text-main);
        font-size: 0.97rem;
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] pre {
        background: rgba(10, 15, 28, 0.7) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 10px !important;
        padding: 0.7rem 0.9rem !important;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(15, 23, 42, 0.5) !important;
        border: 1.5px dashed rgba(148, 163, 184, 0.3) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--accent) !important;
        box-shadow: 0 0 18px var(--accent-glow) !important;
    }

    /* Expanders */
    [data-testid="stExpander"] {
        background: rgba(148, 163, 184, 0.04) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 14px !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-main) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
    }

    /* Alerts */
    [data-testid="stAlert"] {
        background: rgba(30, 41, 59, 0.6) !important;
        backdrop-filter: blur(10px) !important;
        border: 1px solid var(--panel-border) !important;
        border-left: 4px solid var(--accent) !important;
        border-radius: 12px !important;
        color: var(--text-main) !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid var(--panel-border) !important;
        border-radius: 16px !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--user-border) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.08) !important;
        margin: 1.5rem 0 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.1);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(148, 163, 184, 0.25);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(148, 163, 184, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def is_scanned_pdf(docs):
    watermarks = ["scanned by camscanner", "www.camscanner.com"]
    combined_text = " ".join(doc.page_content for doc in docs).lower()
    for watermark in watermarks:
        combined_text = combined_text.replace(watermark, "")
    return len(combined_text.strip()) < 50


def process_files(files):
    processed = []

    for file in files:
        suffix = Path(file.name).suffix.lower().lstrip(".")

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == "pdf":
                docs = load_pdf(tmp_path)
                if not docs or is_scanned_pdf(docs):
                    st.info(f"{file.name} appears scanned, so OCR is being used.")
                    docs = load_pdf_with_ocr(tmp_path)
                    if not docs:
                        st.warning(
                            f"OCR could not extract text from {file.name}. Check that Tesseract is installed; "
                            "the Groq fallback also needs a valid GROQ_API_KEY and vision model in .env."
                        )
            elif suffix == "docx":
                from src.data_loader import load_docx
                docs = load_docx(tmp_path)
            elif suffix == "txt":
                from src.data_loader import load_txt
                docs = load_txt(tmp_path)
            elif suffix == "csv":
                from src.data_loader import load_csv
                docs = load_csv(tmp_path)
            else:
                text = load_image_text(tmp_path)
                docs = (
                    [Document(page_content=text, metadata={"source": file.name})]
                    if text.strip()
                    else []
                )

            for doc in docs:
                doc.metadata["source"] = file.name

            if docs:
                processed.append((file.name, docs))
            elif suffix != "pdf":
                st.warning(f"No text could be extracted from {file.name}.")
        except Exception as exc:
            st.error(f"Error processing {file.name}: {exc}")
        finally:
            os.remove(tmp_path)

    return processed


def build_chat_history(messages):
    return [
        {"role": message["role"], "content": message["content"]}
        for message in messages[-8:]
    ]


def load_chats():
    bridge = st.query_params.get(CHATS_BRIDGE_PARAM)
    if bridge:
        st.session_state.persistence_bridge_rendered = True
        st.query_params.clear()
        try:
            chats = json.loads(bridge)
            if isinstance(chats, dict):
                return chats
        except (json.JSONDecodeError, TypeError):
            pass

    try:
        with CHAT_FILE.open("r", encoding="utf-8") as file:
            chats = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        chats = {}

    if chats:
        has_messages = any(chat.get("messages") for chat in chats.values())
        if has_messages:
            return chats

    legacy_chats = migrate_legacy_history()
    if legacy_chats:
        with CHAT_FILE.open("w", encoding="utf-8") as file:
            json.dump(legacy_chats, file, indent=2, ensure_ascii=False)
        return legacy_chats

    return chats


def save_chats():
    # Write atomically: a crash mid-write must not truncate the whole history.
    tmp = CHAT_FILE.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(st.session_state.all_chats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(CHAT_FILE)


def create_chat(title="New Chat", messages=None):
    chat_id = str(uuid.uuid4())[:6]
    chat_record = {
        "title": title,
        "messages": list(messages or []),
    }
    draft_store = st.session_state.get("draft_store")
    chat_record["document_collection"] = (
        draft_store.collection_name
        if draft_store is not None
        else build_chat_collection_name(chat_id)
    )
    st.session_state.all_chats[chat_id] = chat_record
    st.session_state.current_chat = chat_id
    # The draft uploader no longer exists once the draft becomes a real chat.
    # Clearing its signature stops the next draft from silently reusing it.
    st.session_state.chat_upload_sigs.pop("draft", None)
    save_chats()
    return chat_id


def ensure_current_chat():
    if st.session_state.current_chat not in st.session_state.all_chats:
        st.session_state.current_chat = None


def generate_title(query):
    cleaned = " ".join(query.strip().split())
    if not cleaned:
        return "New Chat"
    low = cleaned.lower()
    for lead in (
        "what is ", "what are ", "what's ", "whats ", "how to ", "how do i ",
        "how can i ", "why does ", "why do ", "can you ", "could you ",
        "please ", "tell me ", "summarize ", "summarise ", "explain ",
        "describe ", "give me ", "i need ", "help me ",
    ):
        if low.startswith(lead):
            cleaned = cleaned[len(lead):].strip()
            break
    short = " ".join(cleaned.split()[:6]).strip()
    if not short:
        short = cleaned
    if len(short) > 42:
        short = short[:39].rstrip() + "..."
    return short[0].upper() + short[1:]


def format_chat_title(chat_id):
    title = st.session_state.all_chats[chat_id]["title"]
    return title[:34] + "..." if len(title) > 34 else title


def migrate_legacy_history():
    try:
        with LEGACY_CHAT_FILE.open("r", encoding="utf-8") as file:
            legacy_items = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

    if not isinstance(legacy_items, list) or not legacy_items:
        return {}

    messages = []
    for item in legacy_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()

        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})

    if not messages:
        return {}

    first_user_message = next(
        (message["content"] for message in messages if message["role"] == "user"),
        "New Chat",
    )
    chat_id = str(uuid.uuid4())[:6]
    return {chat_id: {"title": generate_title(first_user_message), "messages": messages}}


def get_document_store():
    chat_id = st.session_state.current_chat
    if chat_id is None:
        return st.session_state.draft_store
    if chat_id and chat_id in st.session_state.chat_stores:
        for other_id, other_store in list(st.session_state.chat_stores.items()):
            if other_id != chat_id:
                close = getattr(other_store, "close", None)
                if close:
                    close()
                st.session_state.chat_stores.pop(other_id, None)
        draft_store = st.session_state.draft_store
        if draft_store is not None and draft_store is not st.session_state.chat_stores[chat_id]:
            close = getattr(draft_store, "close", None)
            if close:
                close()
            st.session_state.draft_store = None
        return st.session_state.chat_stores[chat_id]
    if chat_id:
        chat_meta = st.session_state.all_chats.get(chat_id, {})
        requested_collection = chat_meta.get("document_collection") or build_chat_collection_name(chat_id)
        if (
            st.session_state.draft_store is not None
            and st.session_state.draft_store.collection_name == requested_collection
        ):
            st.session_state.chat_stores[chat_id] = st.session_state.draft_store
            st.session_state.draft_store = None
            return st.session_state.chat_stores[chat_id]
        for other_id, other_store in list(st.session_state.chat_stores.items()):
            if other_id != chat_id:
                close = getattr(other_store, "close", None)
                if close:
                    close()
                st.session_state.chat_stores.pop(other_id, None)
        if st.session_state.draft_store is not None:
            close = getattr(st.session_state.draft_store, "close", None)
            if close:
                close()
            st.session_state.draft_store = None
        store = persistent_chat_store(chat_id)
        st.session_state.chat_stores[chat_id] = store
        return store
    return None


def persistent_chat_store(chat_id, mode="transformer"):
    """Open this chat's durable local Qdrant collection."""
    chat_meta = st.session_state.all_chats.get(chat_id, {})
    return QdrantKnowledgeBase(
        get_embeddings(mode=mode),
        embedding_mode=mode,
        base_dir=APP_DIR / "knowledge_base",
        collection_name_override=chat_meta.get("document_collection") or build_chat_collection_name(chat_id),
    )


def score_retrieval_cases(store, cases):
    """Score document-grounded questions against the top retrieved chunks."""
    results = []
    for case in cases:
        found = store.similarity_search(case["question"], k=3)
        expected_words = set(case["expected"].lower().split())
        retrieved_words = set(
            " ".join(doc.page_content for doc in found).lower().split()
        )
        coverage = len(expected_words & retrieved_words) / max(1, len(expected_words))
        results.append({
            **case,
            "recall": round(coverage, 4),
            "retrieved": bool(found),
        })
    return results


def build_eval_corpus_store():
    """Build a store from the shared corpus the sidebar chats were replayed on.

    Without this, reopening a replayed chat and asking a follow-up would run as
    a document-free chat: the answers would have no retrieved context and the
    turn would be excluded from the RAG aggregates. Loading the same corpus
    keeps those follow-ups grounded in the documents the transcript cites.
    """
    try:
        corpus = json.loads(EVAL_CORPUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if not isinstance(corpus, dict) or not corpus:
        return None

    store = SessionVectorStore(get_embeddings(mode="transformer"))
    for doc_id, passage in corpus.items():
        text = str(passage or "").strip()
        if text:
            store.add_documents(
                [Document(page_content=text, metadata={"source": doc_id})],
                source_name=doc_id,
            )
    return store if store.source_count() > 0 else None


def ensure_llmops_state():
    """Create the evaluator and telemetry logger once per session."""
    if "llmops_telemetry" not in st.session_state:
        st.session_state.llmops_eval = SemanticRAGEvaluator(min_faithfulness=0.50)
        st.session_state.llmops_telemetry = LLMOpsTelemetry("llmops/reports/telemetry_audit_log.jsonl")
    return st.session_state.llmops_eval, st.session_state.llmops_telemetry


def fit_drift_baseline(evaluator):
    """Fit the drift centroid on the documents currently in this chat.

    Keyed on the chat id plus a content signature, so switching chats or
    replacing a document without changing its chunk count still refits. When the
    active chat has no documents the baseline is dropped, otherwise drift would
    keep reporting the previous chat's centroid as a real measurement.
    """
    store = get_document_store()
    chat_key = st.session_state.current_chat or "draft"
    texts = store.all_texts() if store is not None else []
    signature = (chat_key, len(texts), sum(len(text) for text in texts))

    if not texts:
        if st.session_state.get("llmops_baseline_signature") is not None:
            evaluator.reset_baseline()
            st.session_state.llmops_baseline_signature = None
        return

    if st.session_state.get("llmops_baseline_signature") != signature:
        evaluator.fit_baseline_corpus(texts)
        st.session_state.llmops_baseline_signature = signature


def render_copy_button(text):
    btn_id = "btn_" + str(uuid.uuid4()).replace("-", "")
    payload = json.dumps(text).replace("</", "<\\/")
    html = f"""
    <div style="display:flex;justify-content:flex-end;margin-top:10px;margin-bottom:4px;">
        <button id="{btn_id}" onclick="copyAnswer_{btn_id}()" style="background:linear-gradient(135deg,var(--accent),#7c3aed);color:#fff;border:none;border-radius:10px;padding:8px 18px;font-size:0.85rem;font-weight:600;font-family:Inter,sans-serif;cursor:pointer;box-shadow:0 2px 8px var(--accent-glow);transition:transform 0.15s,box-shadow 0.15s;" onmouseover="this.style.transform='translateY(-1px)';this.style.boxShadow='0 4px 14px var(--accent-glow)'" onmouseout="this.style.transform='';this.style.boxShadow='0 2px 8px var(--accent-glow)'">Copy Answer</button>
    </div>
    <script>
    function copyAnswer_{btn_id}() {{
        var text = {payload};
        if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(text).then(function() {{
                var btn = document.getElementById('{btn_id}');
                if (btn) {{
                    var orig = btn.textContent;
                    btn.textContent = 'Copied!';
                    btn.style.background = 'linear-gradient(135deg, #22c55e, #16a34a)';
                    setTimeout(function() {{ btn.textContent = orig; btn.style.background = 'linear-gradient(135deg, var(--accent), #7c3aed)'; }}, 1500);
                }}
            }}).catch(function() {{ fallbackCopy_{btn_id}(text); }});
        }} else {{
            fallbackCopy_{btn_id}(text);
        }}
    }}
    function fallbackCopy_{btn_id}(text) {{
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        try {{ document.execCommand('copy'); }} catch (e) {{}}
        document.body.removeChild(ta);
    }}
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)


def render_chat_persistence_bridge():
    """If the browser has saved chats, hand them to the app once via a URL redirect."""
    if st.session_state.get("persistence_bridge_rendered"):
        return
    st.session_state.persistence_bridge_rendered = True
    
    # Embed the serialized chats as a JavaScript string literal. Passing the
    # object directly to localStorage.setItem coerces it to "[object Object]".
    payload = json.dumps(
        json.dumps(st.session_state.all_chats, ensure_ascii=False)
    ).replace("</", "<\\/")
    html = f"""
    <script>
    (function () {{
        try {{
            var raw = localStorage.getItem({json.dumps(CHATS_STORAGE_KEY)});
            if (!raw || raw.length > 80000) return;
            var url = new URL(window.location.href);
            if (url.searchParams.has({json.dumps(CHATS_BRIDGE_PARAM)})) return;
            if (raw === {payload}) return;
            url.searchParams.set({json.dumps(CHATS_BRIDGE_PARAM)}, raw);
            window.location.replace(url.toString());
        }} catch (e) {{}}
    }})();
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)


def persist_chats_localstorage():
    """Write the current chats into the browser's localStorage."""
    payload = json.dumps(
        json.dumps(st.session_state.all_chats, ensure_ascii=False)
    ).replace("</", "<\\/")
    html = f"""
    <script>
    try {{
        localStorage.setItem({json.dumps(CHATS_STORAGE_KEY)}, {payload});
    }} catch (e) {{}}
    </script>
    """
    st.html(html, unsafe_allow_javascript=True)


if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chats()

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "draft_messages" not in st.session_state:
    st.session_state.draft_messages = []

if "chat_stores" not in st.session_state:
    st.session_state.chat_stores = {}

if "draft_store" not in st.session_state:
    st.session_state.draft_store = None

if "chat_upload_sigs" not in st.session_state:
    st.session_state.chat_upload_sigs = {}

ensure_current_chat()

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <span class="brand-logo">🧠</span>
            <span>
                <div class="brand-name">RAG Assistant</div>
                <div class="brand-tag">Document Q&amp;A</div>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("New chat", type="primary", use_container_width=True):
        st.session_state.current_chat = None
        st.session_state.draft_messages = []
        st.session_state.draft_store = None
        st.session_state.chat_upload_sigs.pop("draft", None)
        st.rerun()

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "png", "jpg", "jpeg", "docx", "txt", "csv"],
        accept_multiple_files=True,
        key=f"doc_uploader_{st.session_state.current_chat}",
    )
    st.caption("PDF · Word · Text · CSV · Images")

    store = get_document_store()
    if store is None and st.session_state.current_chat:
        chat_meta = st.session_state.all_chats.get(st.session_state.current_chat) or {}
        try:
            store = persistent_chat_store(st.session_state.current_chat)
            if store.source_count() > 0:
                st.session_state.chat_stores[st.session_state.current_chat] = store
            else:
                store.close()
                store = None
        except Exception as exc:
            print(f"[storage] Could not reopen chat document store: {exc}", file=sys.stderr)
        if store is None and chat_meta.get("corpus"):
            corpus_name = Path(chat_meta["corpus"]).name
            st.caption(f"This transcript was replayed against {corpus_name}.")
            if st.button("Load shared corpus for follow-ups", use_container_width=True):
                with st.spinner("Loading shared corpus..."):
                    corpus_store = build_eval_corpus_store()
                if corpus_store is not None:
                    st.session_state.chat_stores[st.session_state.current_chat] = corpus_store
                    st.rerun()
                else:
                    st.error("The shared evaluation corpus could not be loaded.")
    if store and store.source_count() > 0:
        total_chunks = sum(s["chunk_count"] for s in store.list_sources())
        st.markdown(
            f'<div class="upload-note">✓ {store.source_count()} source(s) · {total_chunks} chunks in this chat</div>',
            unsafe_allow_html=True,
        )
        if st.button("Summarize Documents", type="primary", use_container_width=True):
            st.session_state.pending_query = "Please provide a comprehensive summary of the uploaded documents."
            st.rerun()

    st.markdown('<div class="side-title">Chats</div>', unsafe_allow_html=True)
    chat_ids = [
        chat_id
        for chat_id in reversed(list(st.session_state.all_chats.keys()))
        if (
            st.session_state.all_chats[chat_id].get("messages")
            or st.session_state.all_chats[chat_id].get("corpus")
            or st.session_state.all_chats[chat_id].get("document_collection")
        )
    ]
    if chat_ids:
        for chat_id in chat_ids:
            is_active = st.session_state.current_chat == chat_id
            row = st.columns([0.86, 0.14])
            with row[0]:
                marker = "● " if is_active else "   "
                label = f"{marker}{format_chat_title(chat_id)}"
                if st.button(label, key=f"chat_{chat_id}", use_container_width=True):
                    st.session_state.current_chat = chat_id
                    st.session_state.draft_messages = []
                    st.rerun()
            with row[1]:
                if st.button("✕", key=f"del_{chat_id}", help="Delete chat", use_container_width=True):
                    st.session_state.all_chats.pop(chat_id, None)
                    removed_store = st.session_state.chat_stores.pop(chat_id, None)
                    close = getattr(removed_store, "close", None)
                    if close:
                        close()
                    st.session_state.chat_upload_sigs.pop(chat_id, None)
                    save_chats()
                    if st.session_state.current_chat == chat_id:
                        st.session_state.current_chat = None
                        st.session_state.draft_messages = []
                        st.session_state.draft_store = None
                    st.rerun()
    else:
        st.caption("No conversations yet.")

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    if st.session_state.current_chat and st.session_state.all_chats.get(st.session_state.current_chat, {}).get("messages"):
        chat_data = st.session_state.all_chats[st.session_state.current_chat]
        export_text = f"# {chat_data['title']}\n\n"
        for msg in chat_data['messages']:
            role = "User" if msg["role"] == "user" else "Assistant"
            export_text += f"**{role}**:\n{msg['content']}\n\n"
        
        st.download_button(
            label="Download Chat",
            data=export_text,
            file_name=f"{chat_data['title'].replace(' ', '_').lower()}.md",
            mime="text/markdown",
            use_container_width=True
        )


if uploaded_files:
    # Start a real chat before indexing so this upload gets its own durable
    # collection and evaluation baseline, instead of sharing the draft store.
    if st.session_state.current_chat is None:
        create_chat(title="New Chat", messages=st.session_state.draft_messages)
        st.session_state.draft_messages = []
    sig_key = st.session_state.current_chat or "draft"
    current_signature = tuple(sorted((file.name, file.size) for file in uploaded_files))
    if st.session_state.chat_upload_sigs.get(sig_key) != current_signature:
        with st.spinner("Processing documents..."):
            processed = process_files(uploaded_files)

            if processed:
                store = get_document_store()
                if store is None:
                    store = persistent_chat_store(st.session_state.current_chat)
                    st.session_state.chat_stores[st.session_state.current_chat] = store

                total_chunks = 0
                for source, docs in processed:
                    chunks = chunk_data(docs)
                    result = store.add_documents(chunks, source_name=source)
                    total_chunks += result.get("chunk_count", 0)

                # Build and persist the retrieval baseline after indexing. Keep
                # evaluation failures isolated so a successful upload remains usable.
                try:
                    prior_baseline = (
                        store.get_evaluation_baseline()
                        if isinstance(store, QdrantKnowledgeBase) else None
                    )
                    new_sources = list(dict.fromkeys(source for source, _docs in processed))
                    new_document_cases = store.evaluation_questions(source_names=new_sources)
                    new_document_results = score_retrieval_cases(store, new_document_cases)
                    if prior_baseline:
                        regression_results = score_retrieval_cases(
                            store, prior_baseline.get("results", [])
                        )
                    else:
                        regression_results = score_retrieval_cases(
                            store, store.evaluation_questions()
                        )

                    if regression_results and isinstance(store, QdrantKnowledgeBase):
                        if prior_baseline:
                            store.save_evaluation_comparison(
                                regression_results, new_document_results=new_document_results
                            )
                        else:
                            store.save_evaluation_baseline(regression_results)
                    elif regression_results:
                        st.session_state[
                            f"eval_baseline_{st.session_state.current_chat or 'draft'}"
                        ] = {
                            "results": regression_results,
                            "mean_keyword_coverage": round(
                                sum(row["recall"] for row in regression_results) / len(regression_results), 4
                            ),
                        }
                    if new_document_results:
                        print(
                            f"[evaluation] Checked {len(new_document_results)} question(s) "
                            f"from newly uploaded source(s): {', '.join(new_sources)}"
                        )
                except Exception as exc:
                    print(f"[evaluation] Automatic upload baseline failed: {exc}", file=sys.stderr)

                # Only remember the signature after a successful ingest, so a
                # failed upload (e.g. a missing loader dependency) can be
                # retried by re-uploading the same file.
                st.session_state.chat_upload_sigs[sig_key] = current_signature
            st.rerun()
if st.session_state.current_chat is None:
    chat = {"title": "New Chat", "messages": st.session_state.draft_messages}
else:
    chat = st.session_state.all_chats[st.session_state.current_chat]

messages = chat["messages"]

if not messages:
    st.markdown(
        '<div class="welcome-title">What can I help you with?</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="welcome-hint">Try asking</div>', unsafe_allow_html=True)
    prompt_columns = st.columns(2)
    for index, prompt in enumerate(EXAMPLE_PROMPTS):
        with prompt_columns[index % 2]:
            if st.button(f"“{prompt}”", use_container_width=True):
                st.session_state.pending_query = prompt
                st.rerun()

for message in messages:
    avatar = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(message["role"], avatar=avatar):
        clean_content = message["content"].replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        st.markdown(clean_content)
        if message["role"] == "assistant":
            render_copy_button(clean_content)

pending_query = st.session_state.pop("pending_query", None)
query = st.chat_input("Ask a question about your documents")
if pending_query and not query:
    query = pending_query

if query:
    # --- LLMOps Guardrail: validate/sanitize input before RAG ---
    if "llmops_guard" not in st.session_state:
        st.session_state.llmops_guard = LLMGuardrails(block_injections=True, mask_pii=True)
    is_safe, sanitized_query, g_audit = st.session_state.llmops_guard.validate_input(query)
    if not is_safe:
        st.error(f"🚫 Request blocked by guardrail: {sanitized_query}")
        messages.append({"role": "user", "content": query})
        messages.append({"role": "assistant", "content": sanitized_query})
        if st.session_state.current_chat is None:
            blocked_chat_id = create_chat(title=generate_title(query), messages=messages)
            if st.session_state.draft_store is not None:
                st.session_state.chat_stores[blocked_chat_id] = st.session_state.draft_store
                st.session_state.draft_store = None
            st.session_state.draft_messages = []
        else:
            st.session_state.all_chats[st.session_state.current_chat] = chat
            save_chats()

        # Log blocked requests too. Otherwise the dashboard's guardrail counters
        # would read zero even while attacks were being stopped, and the blocked
        # turn would silently vanish from the audit trail.
        try:
            evaluator, telemetry = ensure_llmops_state()
            fit_drift_baseline(evaluator)
            drift_flag, drift_similarity = evaluator.detect_semantic_query_drift(query)
            active_store = get_document_store()
            mode = "rag" if active_store is not None and active_store.source_count() > 0 else "chat"
            telemetry.log_interaction(
                question=query,
                response=sanitized_query,
                context_count=0,
                latency_ms=0.0,
                eval_metrics={
                    "faithfulness": None,
                    "answer_relevance": None,
                    "context_precision": None,
                    "rag_composite_quality": None,
                    "hallucination_detected": None,
                    "semantic_query_drift": drift_flag,
                    "drift_similarity_to_corpus": drift_similarity,
                    "passed_quality_gate": None,
                    "skipped_reason": "guardrail_blocked",
                },
                guardrail_audit=g_audit,
                model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                mode=mode,
                is_answer=False,
                retrieved_context=[],
                tags={
                    "chat_id": st.session_state.current_chat or "draft",
                    "source": "streamlit",
                    "blocked": True,
                },
            )
        except Exception as exc:
            print(f"[telemetry] blocked turn could not be logged: {exc}", file=sys.stderr)
            st.warning(f"The blocked turn could not be added to the LLMOps audit log: {exc}")
        st.rerun()
    query = sanitized_query

    history = build_chat_history(messages)
    messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="user"):
        st.markdown(query)

    t0 = time.time()
    output_redacted = False
    out_audit = {}
    with st.chat_message("assistant", avatar="assistant"):
        store = get_document_store()
        has_kb = store is not None and store.source_count() > 0
        source_docs = []

        def answer_generator():
            for item in stream_agent(query, store if has_kb else None, history=history):
                if item[0] == "token":
                    yield item[1]
                else:
                    source_docs.extend(item[1])

        try:
            collected = []
            def display_generator():
                for token in answer_generator():
                    # Clean out stray raw HTML line breaks emitted by models
                    sanitized_token = token.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
                    collected.append(sanitized_token)
                    yield sanitized_token
            st.write_stream(display_generator())
            collected_text = "".join(collected).strip()
            answer = collected_text if collected_text else "Sorry, no answer was generated."
        except Exception as exc:
            st.error(f"Could not generate an answer. Check GROQ_API_KEY. ({exc})")
            answer = "Sorry, I could not generate an answer."

        # Redact PII the model may have reproduced from a document before the
        # answer is stored in chat history or the audit log. Streaming means the
        # raw text is briefly visible, so rerun below to re-render the sanitized
        # transcript when a redaction actually happened.
        _, safe_answer, out_audit = st.session_state.llmops_guard.validate_output(answer)
        if out_audit.get("output_pii_detected"):
            output_redacted = True
            answer = safe_answer
            st.warning(
                "The answer contained PII, so it was redacted before being saved "
                "to the chat and the LLMOps audit log."
            )

        local_sources = {}
        for doc in source_docs:
            meta = doc.metadata or {}
            src = meta.get("source")
            page = meta.get("page")

            if src:
                if src not in local_sources:
                    local_sources[src] = set()
                if page is not None:
                    local_sources[src].add(str(page))

        if local_sources:
            chip_htmls = []
            for src, pages in local_sources.items():
                if pages:
                    sorted_pages = sorted(list(pages), key=lambda x: int(x) if x.isdigit() else x)
                    page_str = f" (Page{'s' if len(pages) > 1 else ''} {', '.join(sorted_pages)})"
                else:
                    page_str = ""
                chip_htmls.append(
                    f'<span class="cite-chip">📄 {html.escape(str(src))}{html.escape(page_str)}</span>'
                )
            
            chips = "".join(chip_htmls)
            st.markdown(f'<div class="cite-row">{chips}</div>', unsafe_allow_html=True)

        render_copy_button(answer)

    messages.append({"role": "assistant", "content": answer})

    # Create the chat before logging so the audit row carries the real chat id
    # instead of the temporary "draft" id. The first turn used to be orphaned.
    new_chat_id = None
    if st.session_state.current_chat is None:
        new_chat_id = create_chat(title=generate_title(query), messages=messages)
        if st.session_state.draft_store is not None:
            st.session_state.chat_stores[new_chat_id] = st.session_state.draft_store
            st.session_state.draft_store = None
        st.session_state.draft_messages = []
        chat = st.session_state.all_chats[new_chat_id]

    # --- Real-Time LLMOps Automatic Telemetry Logging ---
    try:
        evaluator, telemetry = ensure_llmops_state()
        # Fit the out-of-domain baseline on the documents actually ingested in
        # this session, so drift is a real measurement against real corpus text
        # rather than a hardcoded reference set.
        fit_drift_baseline(evaluator)

        latency_ms = (time.time() - t0) * 1000
        ctx_chunks = [d.page_content for d in source_docs] if source_docs else []
        # Keep the audit from the original input. Re-validating the sanitized
        # query would erase the PII/injection signal the guardrail just found.
        # Merge in the output audit so PII reproduced by the model is counted too.
        guardrail_audit = {**g_audit, **out_audit}

        # Only evaluated, grounded document answers count toward the real RAG
        # hallucination rate. Refusals, empty answers, and conversational filler
        # are still logged as document-chat turns but excluded from aggregates.
        is_answer = bool(ctx_chunks) and bool(answer) and not is_non_answer(answer)
        active_store = get_document_store()
        mode = "rag" if active_store is not None and active_store.source_count() > 0 else "chat"

        if is_answer:
            eval_metrics = st.session_state.llmops_eval.evaluate_turn(query, answer, ctx_chunks)
        else:
            # Nothing was answered from documents, so every quality metric stays
            # null. Drift is still a real measurement of the question itself
            # (or null when no document baseline exists yet).
            drift_flag, drift_similarity = st.session_state.llmops_eval.detect_semantic_query_drift(query)
            eval_metrics = {
                "faithfulness": None,
                "answer_relevance": None,
                "context_precision": None,
                "rag_composite_quality": None,
                "hallucination_detected": None,
                "semantic_query_drift": drift_flag,
                "drift_similarity_to_corpus": drift_similarity,
                "passed_quality_gate": None,
                "skipped_reason": "no_document_context_or_refusal",
            }

        # Persist the actual chunks behind the answer so the turn can be
        # inspected (and re-scored) later from the observability dashboard.
        retrieved_context = []
        for doc in (source_docs or [])[:8]:
            meta = getattr(doc, "metadata", {}) or {}
            retrieved_context.append(
                {
                    "source": meta.get("source") or meta.get("source_name") or "unknown",
                    "page": meta.get("page"),
                    "text": doc.page_content,
                }
            )

        st.session_state.llmops_telemetry.log_interaction(
            question=query,
            response=answer,
            context_count=len(ctx_chunks),
            latency_ms=latency_ms,
            eval_metrics=eval_metrics,
            guardrail_audit=guardrail_audit,
            model_name=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            mode=mode,
            is_answer=is_answer,
            retrieved_context=retrieved_context,
            tags={
                "chat_id": st.session_state.current_chat or "draft",
                "source": "streamlit",
            },
        )
    except Exception as exc:
        # Never let a telemetry failure break the conversation, but never hide
        # it either: a silent pass would make missing metrics look like zeros.
        print(f"[telemetry] turn could not be logged: {exc}", file=sys.stderr)
        st.warning(f"This turn could not be added to the LLMOps audit log: {exc}")

    if new_chat_id is not None:
        # create_chat already persisted the transcript; rerun so the sidebar
        # and the sanitized history render from the saved chat.
        st.rerun()
    else:
        if chat["title"] == "New Chat":
            chat["title"] = generate_title(query)
        st.session_state.all_chats[st.session_state.current_chat] = chat
        save_chats()
        if output_redacted:
            # Replace the briefly streamed raw answer with the sanitized one.
            st.rerun()

render_chat_persistence_bridge()
persist_chats_localstorage()
