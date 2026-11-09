"""Streamlit UI: upload PDFs, then chat with the Corrective RAG workflow."""

import asyncio
import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from workflow import DEFAULT_LLM_MODEL, CorrectiveRAGWorkflow, build_index, build_llm

load_dotenv()
st.set_page_config(page_title="FireCrawl Agentic RAG", page_icon="🔥")
st.title("FireCrawl Agentic RAG")

firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
if not firecrawl_api_key:
    st.error("FIRECRAWL_API_KEY is not set. Add it to the .env file and restart.")
    st.stop()

st.session_state.setdefault("messages", [])
st.session_state.setdefault("index", None)

with st.sidebar:
    st.header("Configuration")
    model = st.text_input(
        "LLM model (LiteLLM name)",
        os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL),
        help='Examples: "ollama_chat/qwen3:8b", "openai/gpt-4o-mini", "lm_studio/<model>".',
    )
    api_base = st.text_input(
        "API base URL (optional)",
        os.getenv("LLM_API_BASE", ""),
        help="Only for local or self-hosted servers, e.g. http://localhost:11434 for Ollama.",
    )

    st.header("Documents")
    uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    if st.button("Index documents", disabled=not uploads):
        with st.spinner("Processing and indexing documents..."), tempfile.TemporaryDirectory() as tmp:
            paths = []
            for upload in uploads:
                # Keep only the file name: upload names are user-controlled.
                path = Path(tmp) / Path(upload.name).name
                path.write_bytes(upload.getvalue())
                paths.append(str(path))
            try:
                st.session_state.index = build_index(paths)
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
            else:
                st.success(f"Indexed {len(paths)} document(s).")
    if st.session_state.index is None:
        st.info("No documents indexed: answers come from web search only.")

    if st.button("Clear chat"):
        st.session_state.messages = []


async def run_workflow(workflow: CorrectiveRAGWorkflow, query: str) -> dict:
    # workflow.run() must be called inside a running event loop.
    return await workflow.run(query_str=query)


def render_details(result: dict) -> None:
    if result.get("web_error"):
        st.warning(f"Web search failed: {result['web_error']}")
    if result.get("sources"):
        with st.expander(f"Web sources (search: {result['web_query']})"):
            for source in result["sources"]:
                st.markdown(f"- [{source['title']}]({source['url']})")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_details(message.get("result", {}))

if query := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        workflow = CorrectiveRAGWorkflow(
            index=st.session_state.index,
            firecrawl_api_key=firecrawl_api_key,
            llm=build_llm(model, api_base),
            timeout=float(os.getenv("WORKFLOW_TIMEOUT", "120")),
        )
        try:
            with st.spinner("Retrieving, searching and answering..."):
                result = asyncio.run(run_workflow(workflow, query))
        except Exception as exc:
            st.error(f"Query failed: {type(exc).__name__}: {exc}")
        else:
            st.markdown(result["answer"])
            render_details(result)
            st.session_state.messages.append(
                {"role": "assistant", "content": result["answer"], "result": result}
            )
