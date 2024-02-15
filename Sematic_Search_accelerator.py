
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from sentence_transformers import SentenceTransformer
#from langchain.document_loaders import PyPDFLoader
#from langchain.document_loaders import UnstructuredFileLoader
from ibm_cloud_sdk_core import IAMTokenManager
import requests
from PIL import Image
#from uuid import uuid4
#from langchain.document_loaders import WebBaseLoader
import glob
import datetime
from PyPDF2 import PdfReader
#from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#from langchain.document_loaders import Docx2txtLoader
#from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.vectorstores import Chroma
#from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.llms.base import LLM as A_LLM
from langchain.document_loaders import UnstructuredHTMLLoader
from streamlit_option_menu import option_menu
from streamlit_chat import message
import random
sidebar_flag=0
import requests
import time

from database import clearingTheIndex, query_pinecone_index, upsertingTheData, clearingTheIndexFAISS_Chroma,index


# # load_dotenv()
# # base_filepath = os.getenv('BASE_FILEPATH')
# # base_filepath="<Project filepath/location where you keep the code <C:\Users\padmaja.a01\GenAI\IBM_WatsonX> and documents to be processed for example <C:\Users\padmaja.a01\GenAI\IBM_WatsonX\InputFiles>>"
# # huggingface_access_token =os.getenv('HUGGINGFACE_API_TOKEN')
# api_url = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
# # documents_directory = os.getenv('Documents_Directory')
# # database = ""
# # project_id=os.getenv('WATSON_PROJECT_ID')
# # endpoint_url=os.getenv('WML_URL')
# # api_key=os.getenv('API_KEY')


# # PINECONE_API_KEY=""
# # PINECONE_ENV=""
# # PINECONE_INDEX_NAME="semantic-search-pinecone"
# api_key=""
# endpoint_url="https://us-south.ml.cloud.ibm.com"
# project_id=""

# base_filepath="<Project filepath/location where you keep the code <C:\Users\padmaja.a01\GenAI\IBM_WatsonX> and documents to be processed for example <C:\Users\padmaja.a01\GenAI\IBM_WatsonX\InputFiles>>"
# Documents_Directory = "<FAISS / CHROMA INDEX file storage location for example C:\Users\padmaja.a01\GenAI\IBM_WatsonX\Local_vector_DB_store >"

PINECONE_API_KEY="d4dbf30a-2fd0-417d-a555-5b9cfb07dca5"
PINECONE_ENV="us-west4-gcp-free"
PINECONE_INDEX_NAME="semantic-search-pinecone"
API_KEY="TBfsGAwtWYaDs3ZSjNbjtFrqcMO69srAVJooMHWLp79k"
 
endpoint_url="https://us-south.ml.cloud.ibm.com"
project_id="5299d745-b240-4f41-8d05-2709ff226ef9"
#5299d745-b240-4f41-8d05-2709ff226ef9
huggingface_access_token="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"

# Function to generate embeddings for chunks
def generate_embeddings(model,chunks):
    chunk_text = list(map(lambda doc: doc.page_content, chunks))
    embeddings = model.encode(chunk_text)
    return embeddings
# Function to chunk the text into smaller parts
def chunk_text(text):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 750, chunk_overlap  = 30, length_function = len,)
    chunks = text_splitter.split_documents(text)
    return chunks

# Function to genearte access token for IBM 
access_token = IAMTokenManager(
    apikey = API_KEY,
    url = "https://iam.cloud.ibm.com/identity/token"
).get_token()
# Summarizing the response using IBM Watsonx foundation model
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-xxl"
headers = {"Authorization": "Bearer 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"}

def query1(payload):
    parameters = {
         "decoding_method": "greedy",
         "random_seed": 33,
         "repetition_penalty":1,
         "min_new_tokens": 50,
         "max_new_tokens": 300,
         "temperature":0.01
        }
    data = {
            # "model_id": model_id,
            "inputs": payload,
            "parameters": parameters
            #"project_id": project_id
        }
    response = requests.post(API_URL, json=data, headers=headers)
    print("Response : ", response.json())
	#response = requests.post(API_URL, headers=headers, json=data)
    return response.json()   

# main method
def main():
    if "chat_answers_history" not in st.session_state:
        st.session_state["chat_answers_history"] = []
    if "user_prompt_history" not in st.session_state:
        st.session_state["user_prompt_history"] = []
    if "database" not in st.session_state:
        st.session_state["database"] = "Pinecone"
    if "Response" not in st.session_state:
        st.session_state["Response"] = ""     

    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    
    st.set_page_config(
        # page_title="Ask Your Query",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        'About': "WatsonX Powered Semantic Search Accelerator"
        })
    st.markdown("""
                <style>
                .big-font {
                font-size:35px;
                font-weight: bold;
                }
                .action {
                font-size:25px;
                font-weight: bold;
                }
                .hd{text-align: center;
                font-size:25px;
                font-weight: bold;
                .pb{
                color:yellow
                }
                }
                </style>
                """, unsafe_allow_html=True)
    # Set up main page
    with st.sidebar:
        st.markdown('<p class="big-font">WatsonX Semantic Search Accelerator</p>', unsafe_allow_html=True)
        st.write("Powered by",st.session_state['database'])
        image2 =Image.open("watsonx_1000.png")
        st.image(image=image2, width=200)
        action = st.radio("What are you looking for", ("Submit your documents", "Ask about your document ??","Chat History","Settings") )       
    if action == 'Submit your documents':
        st.markdown('<p class="action">Submit your documents</p>', unsafe_allow_html=True)
        st.write("Browse your files from local directory and click the upload button to upload your files to vector DB")
        uploaded_files = st.file_uploader("", type=["pdf","html","xml","docx","txt","pptx"], accept_multiple_files=True)
        btn1 = st.button("Click here to upload")
        if btn1:
            print("Updated files :",uploaded_files)
            if uploaded_files:
                documents=[]
                i=0
                for file in uploaded_files:
                    # save the file 
                    if ".pdf" in file.name.lower():
                        loader = PyPDFLoader(file_path = base_filepath+file.name)
                        # loader = PyPDFLoader(stream=file.name)
                        documents=loader.load()
                    elif ".html" in file.name.lower() or ".htm" in file.name.lower() or ".xml" in file.name.lower():
                        loader = UnstructuredHTMLLoader(file_path = base_filepath+file.name,strategy="fast", mode="elements")
                        documents=loader.load()
                        # st.write(documents)
                    elif ".docx" in file.name.lower():
                        loader = Docx2txtLoader(file_path = base_filepath+file.name)
                        documents = loader.load()   
                    elif ".txt" in file.name.lower():
                        loader = UnstructuredFileLoader(file_path = base_filepath+file.name,strategy="fast", mode="elements")
                        documents=loader.load()
                    elif ".pptx" in file.name.lower():
                        loader = UnstructuredPowerPointLoader(file_path = base_filepath+file.name,strategy="fast", mode="elements")
                        documents=loader.load()
                    else:
                        st.write(" For now we only support PDF's, HTML and XML documents. ")   
                    if documents:
                        # Chunk the data
                        chunks = chunk_text(documents)
                        if st.session_state['database'] == "FAISS":
                            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                            if os.path.exists(documents_directory+"faiss_index"):
                                db2 = FAISS.from_documents(chunks, embedding_function)
                                db1 = FAISS.load_local(documents_directory+"faiss_index", embedding_function)
                                db1.merge_from(db2)
                                db1.save_local(documents_directory+"faiss_index")
                                print("ABDD:",documents_directory+"faiss_index")
                            else:
                                db = FAISS.from_documents(chunks, embedding_function)
                                db.save_local(documents_directory+"faiss_index")
                            st.session_state['embedding_function'] = embedding_function
                        elif st.session_state['database'] == "Pinecone":
                            embeddings = generate_embeddings(model,chunks)
                            ids = [str(uuid4()) for _ in range(len(chunks))]
                            metadatas = [{
                                "chunk": j, "text": doc.page_content, "source":doc.metadata['source']} for j, doc in enumerate(chunks)]
                            upsertingTheData(ids,embeddings.tolist(),metadatas)
                        elif st.session_state['database'] == "Chroma":
                            embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                            if os.path.exists(documents_directory+"chroma_db"):
                                db2_chroma = Chroma.from_documents(chunks, embedding_function,persist_directory=documents_directory+"chroma_db")
                                db1_chroma = Chroma(persist_directory=documents_directory+"chroma_db", embedding_function=embedding_function)
                                retriever1 = db2_chroma.as_retriever()
                                retriever2 = db1_chroma.as_retriever()
                                merger_retriever = MergerRetriever(retrievers=[retriever1, retriever2])
                                st.session_state['merger_retriever'] = merger_retriever
                            else:
                                db = Chroma.from_documents(chunks, embedding_function,persist_directory=documents_directory+"chroma_db")
                            st.session_state['embedding_function'] = embedding_function       
                    else:
                        st.info(" For now we only support PDF's, HTML and XML documents.") 
                st.success(" Successfully uploaded the data. ")                
    
    elif action == 'Ask about your document ??':
        try:
            st.markdown('<p class="action">Ask about your document ??</p>', unsafe_allow_html=True)
            #print("Database inside retrival:",database)
            user_question = st.text_input("")
            btn = st.button("Submit")
            top_3_search_results_text=""
            if 'database' in st.session_state:
                database =  st.session_state['database']
            else:
                print("db is not fetched from session state")
            if user_question:
                # Initialize chat history
                if "messages" not in st.session_state:
                    st.session_state.messages = []
                top_k_results = 3
                if database == "FAISS":
                    Embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    print("Inside retrival if condition")
                    if os.path.exists(documents_directory + "faiss_index"):
                        Embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        new_db = FAISS.load_local(documents_directory + "faiss_index", Embedding_function)
                        docs_and_scores = new_db.similarity_search_with_score(user_question)
                        top_3_search_results_text = [docs_and_scores[i] for i in range(0,2)]
                        st.session_state['FAISS_Context'] = top_3_search_results_text
                        print("Nearest Texts:", top_3_search_results_text) 
                    else:
                        raise FileNotFoundError  # Manually raise the FileNotFoundError
                elif database == "Chroma":
                    Embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                    print("Inside retrival if condition")
                    if os.path.exists(documents_directory+"chroma_db"):
                        Embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        new_db = Chroma(persist_directory=documents_directory+"chroma_db", embedding_function=Embedding_function)
                        docs_and_scores = new_db.similarity_search_with_score(user_question)
                    else:
                        raise FileNotFoundError   
                    top_3_search_results_text = [docs_and_scores[i] for i in range(0,2)]
                    st.session_state['Chroma_Context'] = top_3_search_results_text         
                elif database == "Pinecone":
                    stats = index.describe_index_stats()
                    total_vector_count = stats['total_vector_count']
                    top_k_results = 3
                    print("total_vector_count: ",total_vector_count)
                    if total_vector_count != 0:
                        result_ids = query_pinecone_index(model,index,user_question, top_k=top_k_results)
                        print("Results_ids", result_ids['matches'])
                        # Print the results
                        for i,idx in enumerate(result_ids['matches']):
                            text=idx['metadata']['text'].strip().replace('\n', ' ').replace('\r', '')
                            top_3_search_results_text = top_3_search_results_text+text
                            print("Chunck Text "+str(i)+" : "+text)
                            print("Similarity score : ",end="")
                            print(idx['score'])
                    else:
                        raise FileNotFoundError        
                
                prompt = f"""Context: You are a virtual assistant with access to a comprehensive document. Your task is to provide accurate responses to specific queries based on the content of the document.
                Documents : {top_3_search_results_text}
                Question: {user_question}
                Response Format: Please provide relevant answer to the Question based on the information provided in the Documents.
                Answer the question as 'I don't know.' If you can't find the answer from the above data. Also, please use professional tone to answer the query. Please end your response properly without leaving statements incomplete."""

                
                if btn:
                    class ModelLoadingError(Exception):
                        pass
                    try:
                        # response2 = {'error': 'Model google/flan-t5-xxl is currently loading', 'estimated_time': 1802.7086181640625}
                        response2 = query1(prompt)
                        print("Responsee:",response2)
                        if 'error' not in response2:
                            response = response2[0]['generated_text']
                            #response = response2[0]['generated_text']
                            st.session_state['Response'] = response 
                        else:
                            raise ModelLoadingError
                    except ModelLoadingError:
                        for i in range(0,6):
                            loading_message = st.empty()
                            loading_message.text("GPT Model is currently loading. Please wait for some time and try again")
                            time.sleep(45)
                            loading_message.empty()
                            response3 = query1(prompt)
                            if 'error' not in response3:
                                response = response3[0]['generated_text']
                                st.session_state['Response'] = response
                                break   
                    
                    # Write answer :
                    st.session_state["user_prompt_history"].append(user_question)
                    st.session_state["chat_answers_history"].append(response)
                    st.session_state.messages.append({"role": "user", "content": user_question})
                    # # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                Response = st.session_state['Response']
                st.markdown("Answer:")
                temp_res="Apologies, I was unable to locate an answer to your question. Kindly verify and upload the appropriate documents before rephrasing your query."
                print("Response from the model : "+Response,type(Response))
                print("Is response equals to 'I don't know' :"+" I don't know." in Response)
                if " I don't know." not in Response:
                    print("In if response")
                    st.markdown(f"<div style='max-height: 250px; overflow-y: scroll;'>{Response}</div>",unsafe_allow_html=True)
                else:
                    print("In else ")
                    st.markdown(f"<div style='max-height: 250px; overflow-y: scroll;margin-bottom: 50px;'>{temp_res}</div>",unsafe_allow_html=True)
                st.write("\n") 
                
                display_context = st.checkbox('Display the context')
                # print("Context",top_3_search_results_text)
                print("Database :", database)                  
                # Write the matching context details :
                if display_context:
                    st.write("Matching Context:")
                    if database == "FAISS":
                        st.json(top_3_search_results_text)
                    elif database == "Pinecone":
                        st.json(result_ids['matches'])
                    elif database == "Chroma":
                        st.json(top_3_search_results_text)
        except FileNotFoundError:    
            st.write(f"{database} Index file is not found. Please upload the documents to query")
    elif action == 'Chat History':
        st.markdown('<p class="hd">Chat History</p>', unsafe_allow_html=True)
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.write("No previous queries to display")
        for generated_response1, user_query in zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"]):
            message(user_query, is_user=True,key=f"button_{random.sample(range(10000), 1)[0]}",avatar_style="avataaars" ,seed=152)
            message(generated_response1,key=f"button_{random.sample(range(10000), 1)[0]}",avatar_style="bottts" , seed=134)
        st.write("\n")
    elif action == "Settings":
        st.markdown('<p class="big-font">Select vector DB of your choice</p>', unsafe_allow_html=True)
        action = st.radio(" ", ("Pinecone","FAISS","Chroma"))
        if action =='FAISS':
            database = "FAISS"
            st.session_state['database'] = database
            print(database)
        elif action == "Pinecone":
            database = "Pinecone"
            st.session_state['database'] = database
            print(database)  
        elif action == "Chroma":
            database = "Chroma"
            st.session_state['database'] = database 
            print(database)     
        database =  st.session_state['database']
        if database == "FAISS":
            if st.button('Clear Vector DB', help="Clear the FAISS index folder to data upload"):
                clearingTheIndexFAISS_Chroma("faiss_index")
                st.success("Deleted previous vector data from the index successfully !")
        elif database == "Pinecone":    
            if st.button('Clear Vector DB', help="Clear the Pinecone index prior to data upload"):
                clearingTheIndex()
                st.success("Deleted previous vector data from the index successfully !")
        elif database == "Chroma":
            if st.button('Clear Vector DB', help="Clear the Chroma index prior to data upload"):
                clearingTheIndexFAISS_Chroma("chroma_db")
                st.success("Deleted previous vector data from the index successfully !")     

if __name__ == '__main__':
    main()
