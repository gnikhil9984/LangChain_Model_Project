from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Delhi is the capital of India",
    "Bhopal is the capital of MP",
    "Lucknow is the capital of UP",
    "Raipur is the capital of Chhattisgarh"
]

query = " tell me about UP"

query_embedding = embedding.embed_query(query)
doc_embeddings = embedding.embed_documents(documents)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index = np.argmax(scores)

print(query)
print(documents[index])
print(scores[index])
                 



