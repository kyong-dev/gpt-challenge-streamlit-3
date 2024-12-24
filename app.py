from langchain_community.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
import requests
from bs4 import BeautifulSoup
import streamlit as st


st.set_page_config(
    page_title="SiteGPT",
    page_icon="❓",
)

st.title("SiteGPT")


answers_prompt = ChatPromptTemplate.from_template(
    """
    Using ONLY the following context answer the user's question. If you can't just say you don't know, don't make anything up.
                                                  
    Then, give a score to the answer between 0 and 5.

    If the answer answers the user question the score should be high, else it should be low.

    Make sure to always include the answer's score even if it's 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: How far away is the moon?
    Answer: The moon is 384,400 km away.
    Score: 5
                                                  
    Question: How far away is the sun?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)

openai_api_key = None
retriever=None
processing = False
query = None
llm = None

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm
    # answers = []
    # for doc in docs:
    #     result = answers_chain.invoke(
    #         {"question": question, "context": doc.page_content}
    #     )
    #     answers.append(result.content)
    print(docs)
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Use ONLY the following pre-existing answers to answer the user's question.

            Use the answers that have the highest score (more helpful) and favor the most recent ones.

            Cite sources and return the sources of the answers as they are, do not change them.

            Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )

allowed_paths = ['ai-gateway', 'vectorize', 'workers-ai']
def parse_page(soup):
    # Check if the URL starts with any of the allowed paths
    # if not any(url.startswith(f"https://developers.cloudflare.com/{path}") for path in allowed_paths):
    #     return ""
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return (
        str(soup.get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("CloseSearch Submit Blog", "")
    )


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )

    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    print(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cache_dir = LocalFileStore("./.cache/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    faiss_embeddings = None
    if cached_embeddings:
        faiss_embeddings = cached_embeddings
    else:
        faiss_embeddings = embeddings
    vector_store = FAISS.from_documents(docs, faiss_embeddings)
    return vector_store.as_retriever()


with st.sidebar:
    openai_api_key = st.text_input("Write down your OpenAI key", placeholder="sk-proj-NDE*********")

    search_button = st.button("Search")

    st.write("<a href='https://github.com/kyong-dev/gpt-challenge-streamlit-3'>https://github.com/kyong-dev/gpt-challenge-streamlit-3</a>", unsafe_allow_html=True)
    url = st.text_input("URL", value="https://developers.cloudflare.com/sitemap-0.xml")
    if openai_api_key and not processing:
        llm = ChatOpenAI(
            temperature=0.1,
            model="gpt-4o",
            streaming=True,
            openai_api_key=openai_api_key,
        )
        if url:
            if ".xml" not in url:
                with st.sidebar:
                    st.error("Please write down a Sitemap URL.")
            else:
                retriever = load_website(url)
    else:
        st.error("Please write down your OpenAI key.")


if openai_api_key and not processing:
    retriever = load_website(url)
    query = st.text_input("Ask a question to the website.")
    if query:
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        result = chain.invoke(query)
        st.markdown(result.content.replace("$", "dollars"))
