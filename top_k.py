import html

import streamlit as st
from openai import OpenAI
from pinecone import Pinecone

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# Must match the model the index was built with (1536 dimensions)
EMBEDDING_MODEL = "text-embedding-ada-002"

@st.cache_resource
def connect_to_openai():
    return OpenAI(api_key=OPENAI_API_KEY)

@st.cache_resource
def connect_to_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY).Index(PINECONE_INDEX_NAME)

def render_answer(container, text):
    # Escape the model output so it can't inject HTML into the page
    colored_text = f'<div style="background-color: #08346c; color: #FFFFFF; padding: 10px; border-radius: 5px;">{html.escape(text)}</div>'
    container.markdown(colored_text, unsafe_allow_html=True)

def run_chatbot_app():
    client = connect_to_openai()
    idx = connect_to_pinecone()

    # prompt template
    template = """You are a helpful chatbot assistant who is having a conversation with a human
                You know all the regulation from the Friedrich-Alexander-Universität Erlangen-Nürnberg
                You should answer the question or questions posed in German, based ONLY on the context provided.
                DON'T write about anything that isn't present in the context.

                Context:
                {context}

                Human: {human_input}
                """

    # Streamlit app
    st.title('Deine Fragen & Antworten zur FAU')

    query = st.text_input("Stelle hier deine Frage", key='input')

    if st.button('jetzt finden') or 'input' in st.session_state:
        if 'input' in st.session_state:
            query = st.session_state.input
        else:
            query = st.session_state.input = ''

        if query:
            query = query.lower()
            # Chatbot
            st.subheader('Antwort:')
            chat_box = st.empty()

            # Fetch results with minimum score of relevancy
            embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=query).data[0].embedding
            results = idx.query(vector=embedding, top_k=3, include_metadata=True)
            matches = [match for match in results.matches if match.score >= 0.85]

            context = "\n\n".join(match.metadata['content'] for match in matches)
            prompt = template.format(context=context, human_input=query)

            # Stream the answer into the chat box
            answer = ""
            stream = client.chat.completions.create(
                model=OPENAI_MODEL, temperature=0, stream=True,
                messages=[{"role": "user", "content": prompt}],
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    answer += chunk.choices[0].delta.content
                    render_answer(chat_box, answer)

            # Display reference expanders
            for match in matches:
                content = match.metadata['content']
                title = match.metadata['title']
                paragraph = match.metadata['paragraph']
                with st.expander(f"{title} - {paragraph}"):
                    st.markdown(content)

run_chatbot_app()
