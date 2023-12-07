#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from tqdm.auto import tqdm
import pinecone
from collections import Counter
import tiktoken
from transformers import AutoTokenizer
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer


# In[2]:


OPENAI_API_KEY = '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'
def connect_to_pinecone():
    pinecone.init(api_key="63a060f2-2e41-4854-a497-5866c8dd65b4", environment="gcp-starter")
    return pinecone.Index('grundordnung-fau')
idx = connect_to_pinecone()


# In[3]:


tik_tokenizer = tiktoken.get_encoding('cl100k_base')

# create the length function
def tiktoken_len(text):
    tokens = tik_tokenizer.encode(
        text#,
        #disallowed_special=()
    )
    return len(tokens)


# ### Models
# #### Sparse (lexical part)

# In[4]:


class SparseEncoder:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def build_dict(self, input_batch):
      # store a batch of sparse embeddings
      sparse_emb = []
      # iterate through input batch
      for token_ids in input_batch:
          # convert the input_ids list to a dictionary of key to frequency values
          d = dict(Counter(token_ids))
          # remove special tokens and append sparse vectors to sparse_emb list
          sparse_emb.append({key: d[key] for key in d if key not in [101, 102, 103, 0]})
      # return sparse_emb list
      return sparse_emb

    def generate_sparse_vectors(self, context_batch):
      # create batch of input_ids
      inputs = self.tokenizer(
        context_batch, padding=True,
        truncation=True,
        max_length=512
      )['input_ids']
      # create sparse dictionaries
      sparse_embeds = self.build_dict(inputs)
      return sparse_embeds

    def encode_queries(self, query):
      sparse_vector = self.generate_sparse_vectors([query])[0]
      # Convert the format of the sparse vector
      indices, values = zip(*sparse_vector.items())
      return {"indices": list(indices), "values": list(values)}


# In[5]:


model_id = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
sparse_encoder = SparseEncoder(model_id)


# #### Dense (semantic part)

# In[6]:


# embed = OpenAIEmbeddings(
#     model = 'text-embedding-ada-002',
#     openai_api_key= OPENAI_API_KEY
# )

embed = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2', device='cpu')


# In[7]:


def hybrid_scale(dense, sparse, alpha: float):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse


def hybrid_query(question, top_k, alpha, filter=None):
    def connect_to_pinecone():
        pinecone.init(api_key="63a060f2-2e41-4854-a497-5866c8dd65b4", environment="gcp-starter")
        return pinecone.Index('grundordnung-fau')
    idx = connect_to_pinecone()
    # convert the question into a sparse vector
    sparse_vec = sparse_encoder.generate_sparse_vectors([question])[0]
    sparse_vec = {
        'indices': list(sparse_vec.keys()),
        'values': [float(value) for value in sparse_vec.values()]
    }

    # convert the question into a dense vector
    dense_vec = embed.encode(question).tolist()

    # scale alpha with hybrid_scale
    dense_vec, sparse_vec = hybrid_scale(
    dense_vec, sparse_vec, alpha
    )
    # query pinecone with the query parameters
    result = idx.query(
    vector=dense_vec,
    sparse_vector=sparse_vec,
    top_k=top_k,
    include_metadata=True,
    filter=filter
    )
    # return search results as json
    return result


# Indexing db to a variable

# In[8]:


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


# In[9]:


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        display_function = getattr(self.container, self.display_method, None)
        if display_function is not None:
            # Wrap the text in a div with a custom background color
            colored_text = f'<div style="background-color: #08346c; padding: 10px; border-radius: 5px;">{self.text}</div>'
            display_function(colored_text, unsafe_allow_html=True)
        else:
            raise ValueError(f"Invalid display_method: {self.display_method}")


# In[13]:


def run_chatbot_app():
    # Initialize Pinecone
    @st.cache_resource
    def connect_to_pinecone():
        pinecone.init(api_key="63a060f2-2e41-4854-a497-5866c8dd65b4", environment="gcp-starter")
        return pinecone.Index('grundordnung-fau')
    idx = connect_to_pinecone()
        
    # prompt template
    template = """You are a helpful chatbot assistant that is having a conversation with a human
                You know all the regulstion from the Friedrich-Alexander-Universität Erlangen-Nürnberg
                You should answer the question or questions posed in German, based on the context provided.

                Context:
                {context}

                Human: {human_input}
                """

    prompt = PromptTemplate(
        input_variables=["human_input", "context"],
        template=template
    )

    # Streamlit app
    st.title('FAURegelBot')

    query = st.text_input("Stelle eine frage", key='input')

    if st.button('Suchen') or 'input' in st.session_state:
        if 'input' in st.session_state:
            query = st.session_state.input
        else:
            query = st.session_state.input = ''

        if query:
            query = query.lower()
            # Chatbot
            st.subheader('Antwort:')
            
            # llm setup
            chat_box = st.empty()
            stream_handler = StreamHandler(chat_box, display_method='write')
            llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo",
                    callbacks=[stream_handler], streaming=True
            )

            # Fetch results with minimum score of relevancy
            results = hybrid_query(query, top_k=3, alpha=1)
            filtered_list = [match for match in results['matches'] if match['score'] >= 0.95]
            combined_docs_chain = []
            for result in filtered_list:
                doc = Document(result['metadata']['content'], result['metadata'])  
                combined_docs_chain.append(doc)

            # Create the chain
            chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            response = chain.run(input_documents=combined_docs_chain, question=query, human_input=query)

            # Display reference expanders
            for doc in combined_docs_chain:
                content = doc.page_content
                title = doc.metadata['title']
                paragraph = doc.metadata['paragraph']
                content = doc.metadata['content']
                with st.expander(f"{title} - {paragraph}"):
                    st.markdown(content)


# In[14]:


run_chatbot_app()

