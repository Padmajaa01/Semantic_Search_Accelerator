import pinecone
from dotenv import load_dotenv
import os
import shutil

load_dotenv()
base_filepath = os.getenv('BASE_FILEPATH')
pinecone_key= os.getenv('PINECONE_API_KEY')
pinecone_env= os.getenv('PINECONE_ENV')
pinecone_index_name= os.getenv('PINECONE_INDEX_NAME')


pinecone.init(
    api_key=pinecone_key,  # find at app.pinecone.io
    environment=pinecone_env,  # next to api key in console
    )
index = pinecone.Index(index_name=pinecone_index_name)

# Function to query Pinecone index using cosine similarity
def query_pinecone_index(model,index,query_text,top_k):
        print("Inside query method : ")
        query_embedding = model.encode(query_text).tolist()
        return index.query(vector=query_embedding, top_k=top_k, include_metadata=True,namespace="")


def clearingTheIndex():
        print(index.delete(index=pinecone_index_name,deleteAll=True, namespace=''))
        print("After deletion : "+str(index.describe_index_stats()))

def upsertingTheData(ids,embeddings,metadatas):
        index.upsert(vectors=zip(ids,embeddings,metadatas))

