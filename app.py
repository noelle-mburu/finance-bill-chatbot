import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

# ── Page configuration ────────────────────────────────────────────────────────
# This must be the first Streamlit command in the file
st.set_page_config(
    page_title="Kenya Finance Bill 2025 Chatbot",
    page_icon="🇰🇪",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
# We inject CSS to:
# 1. Style the header banner with Kenya flag colours (black, red, green)
# 2. Pin the footer to the bottom so it doesn't float with the messages
# unsafe_allow_html=True is required whenever we inject raw HTML/CSS into Streamlit
st.markdown("""
    <style>
        /* Header banner styled with Kenya flag colours */
        .header-banner {
            background: linear-gradient(135deg, #006600 0%, #000000 50%, #BB0000 100%);
            padding: 24px 32px;
            border-radius: 12px;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        .header-banner h1 {
            color: white;
            font-size: 1.8rem;
            margin: 0;
            font-weight: 700;
        }
        .header-banner p {
            color: rgba(255,255,255,0.85);
            margin: 4px 0 0 0;
            font-size: 0.95rem;
        }
        .header-flag {
            font-size: 3rem;
            line-height: 1;
        }

        /* Pin footer to the bottom of the viewport */
        /* bottom: 60px leaves space above the chat input bar */
        .footer {
            position: fixed;
            bottom: 60px;
            left: 0;
            width: 100%;
            text-align: center;
            font-size: 0.78rem;
            color: grey;
            pointer-events: none;  /* so it doesnt block clicks */
        }

        /* Welcome message box */
        .welcome-box {
            background-color: #f0f7f0;
            border-left: 4px solid #006600;
            padding: 16px 20px;
            border-radius: 8px;
            margin: 16px 0;
        }
        .welcome-box p {
            margin: 4px 0;
            color: #333;
            font-size: 0.92rem;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load API key from Streamlit secrets ───────────────────────────────────────
# In Streamlit Cloud, secrets are stored in the app settings (not hardcoded)
# st.secrets["GROQ_API_KEY"] reads the key we set there
api_key = st.secrets["GROQ_API_KEY"]

# ── Embedding model ───────────────────────────────────────────────────────────
# We wrap SentenceTransformer in a class so LangChain can use it
# LangChain expects embed_documents() and embed_query() methods
class BGEEmbeddings:
    def __init__(self):
        # Load the pre-trained BGE embedding model
        # This converts text into vectors that capture meaning
        self.model = SentenceTransformer("BAAI/bge-base-en")

    def embed_documents(self, texts):
        """Embed a batch of document chunks — used when building the vector store"""
        return self.model.encode(
            texts,
            batch_size=8,
            normalize_embeddings=True  # normalizing improves search accuracy
        ).tolist()

    def embed_query(self, text):
        """Embed a single user question — used at query time"""
        return self.model.encode(
            [text],
            normalize_embeddings=True
        ).tolist()[0]

# ── RAG setup ─────────────────────────────────────────────────────────────────
# @st.cache_resource tells Streamlit to run this function only ONCE
# and reuse the result for every user — this is important because loading
# the PDF and building the vector store is slow and expensive
@st.cache_resource
def setup_rag():
    """
    Load the Finance Bill PDF, split it into chunks, embed them,
    and build the retriever. This runs once when the app starts.
    """

    # Step 1: Load the PDF
    # The PDF lives in the same folder as this app.py file
    loader = PyPDFLoader("The Finance Bill 2025.pdf")
    docs = loader.load()  # each page becomes a Document object

    # Step 2: Split the PDF into chunks
    # We split because LLMs can't process the full PDF at once
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   # each chunk is up to 1000 characters
        chunk_overlap=100  # 100 character overlap prevents cutting sentences mid-thought
    )
    chunks = splitter.split_documents(docs)

    # Step 3: Set up the embedding model
    embedding_function = BGEEmbeddings()

    # Step 4: Build the vector store
    # Chroma converts our chunks to vectors and stores them so we can search by meaning
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory="./chroma_store"  # saves to disk to avoid re-embedding on restart
    )

    # Step 5: Build the retriever
    # When a question comes in, this finds the 5 most relevant chunks
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}  # return top 5 most similar chunks
    )

    # Step 6: Set up the LLM (Groq's LLaMA 3.1)
    # temperature=0.2 keeps answers factual and consistent
    llm = ChatGroq(
        api_key=api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.2
    )

    # Step 7: Define the prompt template
    # This structures exactly what we send to the LLM:
    # the retrieved context, the conversation history, and the question
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="""You are a helpful assistant that answers questions about the Kenya Finance Bill 2025.
Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I could not find that information in the Finance Bill 2025."

Previous conversation:
{chat_history}

Context from the Finance Bill:
{context}

Question: {question}

Answer:"""
    )

    # Step 8: Create the LLM chain — links the prompt template to the LLM
    chain = LLMChain(llm=llm, prompt=prompt)

    return retriever, chain


def get_answer(question, retriever, chain, chat_history):
    """
    Given a question, retrieve relevant chunks from the PDF,
    build a prompt with conversation history, and return the LLM's answer.
    """
    # Find the most relevant chunks for this question
    retrieved_docs = retriever.invoke(question)

    # Combine all retrieved chunks into one context string
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Format conversation history so the LLM knows what was discussed before
    past_conversation = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history]
    ) if chat_history else "No previous conversation."

    # Run the chain — sends context + history + question to the LLM
    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": past_conversation
    })

    return response["text"]


# ── Header ────────────────────────────────────────────────────────────────────
# Custom HTML banner using Kenya flag colours (green, black, red)
st.markdown("""
    <div class="header-banner">
        <div class="header-flag">🇰🇪</div>
        <div>
            <h1>Kenya Finance Bill 2025 Chatbot</h1>
            <p>Ask questions about the official Finance Bill — answers sourced directly from the document</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# ── Load the RAG pipeline ─────────────────────────────────────────────────────
# Shows a spinner while setting up on first load
with st.spinner("Loading the Finance Bill... (this takes about 30 seconds on first load)"):
    retriever, chain = setup_rag()

# ── Initialise chat history ───────────────────────────────────────────────────
# session_state persists data across reruns (each time the user sends a message)
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome message ───────────────────────────────────────────────────────────
# Only shown when the chat is empty — disappears once the user starts chatting
if not st.session_state.messages:
    st.markdown("""
        <div class="welcome-box">
            <p><strong>👋 Welcome!</strong> I can answer questions about the Kenya Finance Bill 2025.</p>
            <p>Try asking things like:</p>
            <p>• <em>What is the digital service tax?</em></p>
            <p>• <em>What changes are proposed to income tax?</em></p>
            <p>• <em>What amendments are made to the VAT Act?</em></p>
        </div>
    """, unsafe_allow_html=True)

# ── Clear conversation button ─────────────────────────────────────────────────
# Only shown once the conversation has started
if st.session_state.messages:
    # Use columns to push the button to the right side
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear"):
            # Clear message history and rerun the app to reset the UI
            st.session_state.messages = []
            st.rerun()

# ── Display conversation history ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # "user" or "assistant"
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the Finance Bill 2025..."):

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message to session state
    st.session_state.messages.append({"role": "user", "content": question})

    # Get the answer from the RAG pipeline and display it
    with st.chat_message("assistant"):
        with st.spinner("Searching the Finance Bill..."):
            answer = get_answer(
                question,
                retriever,
                chain,
                st.session_state.messages[:-1]  # pass history excluding the current question
            )
        st.markdown(answer)

    # Save assistant answer to session state so it appears on future reruns
    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Pinned footer ─────────────────────────────────────────────────────────────
# Fixed to the bottom of the viewport using CSS defined at the top of the file
st.markdown(
    '<div class="footer">Built by Noelle · Powered by LangChain, Groq LLaMA 3.1 & ChromaDB</div>',
    unsafe_allow_html=True
)