# import pinecone
# from dotenv import load_dotenv
import os
import shutil
from pinecone import Pinecone, ServerlessSpec

# load_dotenv()
# base_filepath = os.getenv('BASE_FILEPATH')
# pinecone_key= os.getenv('PINECONE_API_KEY')
# pinecone_env= os.getenv('PINECONE_ENV')
# pinecone_index_name= os.getenv('PINECONE_INDEX_NAME')

pinecone_key= "d4dbf30a-2fd0-417d-a555-5b9cfb07dca5"
pinecone_env= "us-west4-gcp-free"
pinecone_index_name= "semantic-search-pinecone"

pc = Pinecone(
        api_key=pinecone_key,
        environment=pinecone_env
    )

# pinecone.init(
#     api_key=pinecone_key,  # find at app.pinecone.io
#     environment=pinecone_env,  # next to api key in console
#     )
index = pc.Index("semantic-search-pinecone")

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

def clearingTheIndexFAISS_Chroma(folder_name):
    folder_path = r"C:\\Users\jayanth.kappala\Documents\GenerativeAI\IBM WatsonX\SemanticSearch-WatsonX_Copy\Py-Files\chunks\1"

    target_folder_path = os.path.join(folder_path, folder_name)

    if os.path.exists(target_folder_path) and os.path.isdir(target_folder_path):
        try:
            shutil.rmtree(target_folder_path)  # Remove folder and its contents
            print(f"The folder '{folder_name}' in '{folder_path}' was deleted successfully.")
        except OSError as e:
            print(f"Error while deleting the folder: {e}")
    else:
        print(f"The folder '{folder_name}' in '{folder_path}' does not exist or is not a directory.")  