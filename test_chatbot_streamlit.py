import streamlit as st
from openai import AzureOpenAI
from rag import *
 
st.set_page_config(page_title="Azure Chat", layout="centered")


# -------------------------------------------------------------------------------
# Sidebar
 
DEFAULTS = {
    "rag_enabled": False,
    "chunk_size": 800,
    "chunk_overlap": 200,
    "top_k": 4,
    "temperature": 0.7,
    "max_tokens": 512,
    "system_prompt": "Sei un assistente utile e conciso."
}
 
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)
 
# Funzione per testare la connessione
def test_connection(endpoint, deployment, api_key, api_version):
    try:
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=api_key,
        )
        # Test minimale: invio di un messaggio vuoto
        client.chat.completions.create(
            messages=[{"role": "system", "content": "Ping"}],
            model=deployment,
            temperature=0.0,
            max_tokens=10
        )
        return client
    except Exception as e:
        return str(e)
 
with st.sidebar:
    st.header("Impostazioni")
 
    # --- Documenti
    st.subheader("Documenti")
    uploaded_files = st.file_uploader(
        "Carica documenti",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        key="uploader"
    )
 
    if uploaded_files:
        st.write("**Documenti caricati:**")
        for f in uploaded_files:
            st.caption(f"• {f.name} ({f.type or 'tipo sconosciuto'})")
 
   
    if st.button("Svuota documenti"):
        st.session_state["uploader"] = None
        st.rerun()
 
    # --- Parametri RAG
    st.subheader("Parametri RAG")
    st.checkbox("Abilita RAG", key="rag_enabled")
    st.number_input("Chunk size", min_value=100, max_value=5000, step=50, key="chunk_size")
    st.number_input("Chunk overlap", min_value=0, max_value=2000, step=10, key="chunk_overlap")
    st.slider("Documenti da recuperare (top_k)", 1, 20, key="top_k")
 
   
    if st.button("Reset parametri"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

# -------------------------------------------------------------------------------
# Main Page

st.title("Chatbot GPT-4o-mini")

st.session_state.setdefault("messages", [])

if st.session_state.get("chain") is None:
    st.session_state['chain'] = setup()

# Mostra messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Scrivi qui..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Risposta completa
    answer = rag_answer(prompt, st.session_state.get('chain'))

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
