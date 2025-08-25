import streamlit as st
from openai import AzureOpenAI

st.set_page_config(page_title="Azure Chat", layout="centered")

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

# Se non abbiamo ancora una connessione valida
if "client" not in st.session_state:
    st.title("Connessione ad Azure OpenAI")

    with st.form("connessione_form"):
        endpoint = st.text_input("Endpoint Azure OpenAI")
        deployment = st.text_input("Nome Deployment")
        api_key = st.text_input("Chiave API", type="password")
        api_version = st.text_input("Versione API (es. 2023-07-01)")
        submitted = st.form_submit_button("Connetti")

        if submitted:
            result = test_connection(endpoint, deployment, api_key, api_version)
            if isinstance(result, AzureOpenAI):
                st.session_state.client = result
                st.session_state.endpoint = endpoint
                st.session_state.deployment = deployment
                st.session_state.api_version = api_version
                st.session_state.api_key = api_key
                st.session_state.messages = []
                st.rerun()
            else:
                st.error(f"Connessione fallita: {result}")

# Se la connessione è valida, mostra la chat
else:
    st.title("Chatbot GPT-4o-mini")

    # Mostra messaggi precedenti
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input utente
    if prompt := st.chat_input("Scrivi qui..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Streaming reale della risposta
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed_text = ""

            stream = st.session_state.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=st.session_state.deployment,
                temperature=1.0,
                max_tokens=4096,
                stream=True
            )

            for chunk in stream:
                if (
                    chunk.choices and
                    chunk.choices[0].delta and
                    hasattr(chunk.choices[0].delta, "content") and
                    chunk.choices[0].delta.content
                ):
                    streamed_text += chunk.choices[0].delta.content
                    placeholder.markdown(streamed_text)

        st.session_state.messages.append({"role": "assistant", "content": streamed_text})
