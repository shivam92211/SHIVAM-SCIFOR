import os
import time
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))




# Global variables for batch processing
BATCH_SIZE = 100
last_save_time = time.time()
SAVE_INTERVAL = 300  # 5 minutes in seconds
added_since_last_save = 0

# Global variable for our database
new_db = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the FAISS index
    global new_db
    embedding_model = GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
        # google_api_key="AIzaSyA_jMrk2zSlIJOUic6v-5zCscQoUUObNOM", 
    )
    new_db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)
    print("Database loaded.")
    
    yield  # This is where the application runs
    
    # Shutdown: Save any unsaved changes
    global added_since_last_save
    if added_since_last_save > 0:
        new_db.save_local("faiss_index")
    print("Application shutting down. Database saved.")

    

app = FastAPI(lifespan=lifespan)

class SentenceInput(BaseModel):
    sentence: str

def get_similarity_score(sentence: str):
    global added_since_last_save, last_save_time, new_db
    
    # Perform similarity search
    results = new_db.similarity_search_with_score(sentence)
    
    if results and results[0][1] < 0.5:
        most_similar, score = results[0]
        return {"message": f"Similar sentence exists: '{most_similar.page_content}' with score {score}"}
    else:
        # Add the new sentence to the database
        new_db.add_texts([sentence])
        added_since_last_save += 1
        
        # Check if we should save based on batch size or time interval
        current_time = time.time()
        if added_since_last_save >= BATCH_SIZE or (current_time - last_save_time) > SAVE_INTERVAL:
            new_db.save_local("faiss_index")
            added_since_last_save = 0
            last_save_time = current_time
        
        return {"message": f"New sentence added to the database: '{sentence}'"}

@app.post("/similarity")
async def similarity_score(input: SentenceInput):
    try:
        result = get_similarity_score(input.sentence)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)