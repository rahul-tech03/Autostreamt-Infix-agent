from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

class AutoStreamRAG:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.0
        )

        self.vectorstore = self._build_vectorstore()

    def _build_vectorstore(self):
        # 1. Load knowledge base
        loader = TextLoader("data/knowledge_base.md")
        documents = loader.load()

        # 2. Chunk documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        # 3. Embeddings
        embeddings = OpenAIEmbeddings()

        # 4. FAISS Vector Store (in-memory)
        vectorstore = FAISS.from_documents(
            docs,
            embeddings
        )

        return vectorstore

    def answer(self, query: str) -> str:
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are a helpful assistant for AutoStream.

Answer the question strictly using the context below.
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}
"""

        response = self.llm.invoke(
            [HumanMessage(content=prompt)]
            )


        return response.content.strip()