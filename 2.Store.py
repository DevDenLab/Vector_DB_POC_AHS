from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.embeddings import HuggingFaceEmbeddings
loader = TextLoader("data\scraped_data_BFS.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5432/vector"

COLLECTION_NAME = "AHS_Scraped_test"

embeddings = HuggingFaceEmbeddings(model_name = "BAAI/bge-large-en-v1.5")

db = PGVector.from_documents(
    embedding=embeddings,
    documents=docs,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
query = "AHS supports our people by creating a culture"
docs_with_score = db.similarity_search_with_score(query,k=2)
for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)