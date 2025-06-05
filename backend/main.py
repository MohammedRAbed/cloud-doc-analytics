from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
from datetime import datetime
from fastapi import Query
import re
from fastapi import APIRouter, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List, Dict
import shutil
from fastapi.staticfiles import StaticFiles
from fastapi import Response
import time
from typing import Optional
from fastapi.responses import FileResponse


# Import document processor
from backend.services import document_processor
from collections import defaultdict

app = FastAPI(
    title="Document Analytics API",
    description="Cloud-based document processing and analytics system",
    version="0.1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.state.stats = {
    "total_documents": 0,
    "total_size_kb": 0,
    "documents_by_type": defaultdict(int),
    "processing_times": {
        "upload": [],
        "search": [],
        "sort": [],
        "classify": []
    },
    "last_updated": None
}

# Helper function to track processing time
def track_time(operation: str):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            
            processing_time = end_time - start_time
            app.state.stats["processing_times"][operation].append(processing_time)
            app.state.stats["last_updated"] = datetime.now().isoformat()
            
            return result
        return wrapper
    return decorator


# ------------------------------ Upload endpoint ------------------------------ 

documents = []


UPLOAD_DIR = "frontend/public/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files (for development)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# Serve static files (JS, CSS, images, etc.)
app.mount("/static", StaticFiles(directory="frontend/"), name="static")

# Serve HTML pages directly
@app.get("/")
def serve_index():
    return FileResponse("frontend/index.html")

@app.get("/upload")
def serve_upload():
    return FileResponse("frontend/upload.html")

@app.get("/classify")
def serve_classify():
    return FileResponse("frontend/classify.html")

@app.get("/search")
def serve_search():
    return FileResponse("frontend/search.html")


# Basic health check endpoint
@app.get("/")
def read_root():
    return {
        "message": "Document Analytics API",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# Document upload endpoint
@track_time("upload")
@app.post("/upload/")
async def upload_documents(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        ext = file.filename.split('.')[-1].lower()
        if ext not in ["pdf", "docx", "doc"]:
            continue  # skip unsupported files

        unique_id = uuid.uuid4()
        filename = f"{unique_id}.{ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        public_url = f"/uploads/{filename}"

        #temp_filename = f"{uuid.uuid4()}.{ext}"
        #file_path = os.path.join(UPLOAD_DIR, temp_filename)

        # Save file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Extract content
        try:
            if ext == "pdf":
                title, text = document_processor.extract_pdf_content(file_path)
            elif ext in ["docx", "doc"]:
                title, text = document_processor.extract_docx_content(file_path)
            else:
                os.remove(file_path)
                continue  # skip unsupported files
        except Exception as e:
            os.remove(file_path)
            continue  # skip problematic files


        doc_id = str(uuid.uuid4())


        # Store document metadata
        documents.append({
            "id": doc_id,
            "type": ext,
            "filename": file.filename,
            "title": title,
            "content": text,
            "word_count": len(text.split()),
            "file_path": public_url,
            "upload_time": datetime.now().isoformat()
        })

        results.append({
            "id": doc_id,
            "type": ext,
            "filename": file.filename,
            "title": title,
            "word_count": len(text.split()),
            "url": public_url,
            "content": text,

        })

        #os.remove(file_path)  # Cleanup

    # Update stats
    app.state.stats["total_documents"] += len(results)
    for doc in results:
        app.state.stats["documents_by_type"][doc['type']] += 1
        app.state.stats["total_size_kb"] += len(doc['content']) / 1024
   
    return {"uploaded": results}



# ------------------------------ Documents List endpoint ------------------------------ 


# Document list endpoint
#@app.get("/documents/")
#def get_all_documents():
#    return {"count":len(documents), "documents": documents}
#
## Sorted (by title) documents list
#@app.get("/documents/sorted/")
#def get_documents_sorted():
#    sorted_docs = sorted(documents, key=lambda d: d["title"].lower())
#    return {"count":len(documents), "documents": sorted_docs}
#

# Updated document list endpoint
@app.get("/documents/")
def get_all_documents():
    return {
        "count": len(documents),
        "documents": [{
            "id": doc["id"],
            "title": doc["title"],
            "type": doc["type"],
            "word_count": doc["word_count"],
            "url": doc["file_path"],
            "upload_time": doc["upload_time"]
        } for doc in documents]
    }

# Updated sorted documents endpoint
@track_time("sort")
@app.get("/documents/sorted/")
def get_documents_sorted():
    sorted_docs = sorted(documents, key=lambda d: d["title"].lower())
    return {
        "count": len(sorted_docs),
        "documents": [{
            "id": doc["id"],
            "title": doc["title"],
            "type": doc["type"],
            "word_count": doc["word_count"],
            "url": doc["file_path"],
            "upload_time": doc["upload_time"]
        } for doc in sorted_docs]
    }



# ------------------------------ Search endpoint ------------------------------ 

@track_time("search")
@app.get("/documents/search/")
def search_documents(
    keywords: str = Query(..., description="Space-separated keywords to search for"),
    case_sensitive: bool = False
):
    """
    Search documents for keywords and return matches with highlighted text
    """
    keyword_list = keywords.split()
    results = []
    
    for doc in documents:
        content = doc["content"]
        matches = []
        highlighted_content = content
        
        # Search for each keyword
        for keyword in keyword_list:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(re.escape(keyword), flags)
            
            # Find all matches and their positions
            for match in pattern.finditer(content):
                matches.append({
                    "keyword": keyword,
                    "start": match.start(),
                    "end": match.end()
                })
            
            # Add highlighting
            highlighted_content = pattern.sub(
                f'<span class="highlight">{match.group(0)}</span>',
                highlighted_content
            )
        
        if matches:
            results.append({
                "filename": doc["filename"],
                "title": doc["title"],
                "word_count": doc["word_count"],
                "match_count": len(matches),
                "highlighted_content": highlighted_content,
                "matches": matches
            })
    
    # Sort by most matches first
    results.sort(key=lambda x: x["match_count"], reverse=True)
    
    return {"results": results}


# ------------------------------ Classify endpoint ------------------------------ 

classify_router = APIRouter()

# Predefined classification categories
CLASSIFICATION_CATEGORIES = [
    "Academic",
    "Business",
    "Technical",
    "Other"
]

# Sample training data (you should replace with your actual training data)
TRAINING_DATA = {
    "texts": [
        "research study university education learning",
        "market sales business profit growth",
        "programming code software development",
        "random miscellaneous unknown text"
    ],
    "categories": [
        "Academic",
        "Business",
        "Technical",
        "Other"
    ]
}

# Initialize and train the classifier
classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Train the classifier with sample data
classifier.fit(TRAINING_DATA["texts"], TRAINING_DATA["categories"])

@track_time("classify")
@classify_router.post("/classify/")
async def classify_documents(document_ids: List[str]):
    """
    Classify documents based on their content.
    
    Args:
        document_ids: List of document IDs to classify
        
    Returns:
        List of classification results with confidence scores
    """
    try:
        results = []
        
        for doc_id in document_ids:
            # Find the document in your storage (replace with your actual document retrieval)
            document = next((d for d in documents if d.get("id") == doc_id), None)
            
            if not document:
                results.append({
                    "document_id": doc_id,
                    "error": "Document not found"
                })
                continue
            
            # Get document text (assuming you have 'content' field)
            text = document.get("content", "")
            
            # Predict category
            predicted = classifier.predict([text])[0]
            probabilities = classifier.predict_proba([text])[0]
            
            # Get confidence scores for all categories
            confidence_scores = {
                category: float(probabilities[i]) 
                for i, category in enumerate(classifier.classes_)
            }
            
            results.append({
                "document_id": doc_id,
                "title": document.get("title", ""),
                "predicted_category": predicted,
                "confidence_scores": confidence_scores
            })
        
        return {"results": results}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@classify_router.get("/classify/categories/")
async def get_classification_categories():
    """
    Get the list of predefined classification categories
    """
    return {"categories": CLASSIFICATION_CATEGORIES}

# Helper function to train with more data (optional)
@classify_router.post("/classify/train/")
async def train_classifier(new_texts: List[str], new_categories: List[str]):
    """
    Add more training data to improve the classifier
    """
    try:
        if len(new_texts) != len(new_categories):
            raise ValueError("Texts and categories must be of equal length")
            
        # Update training data
        TRAINING_DATA["texts"].extend(new_texts)
        TRAINING_DATA["categories"].extend(new_categories)
        
        # Retrain classifier
        classifier.fit(TRAINING_DATA["texts"], TRAINING_DATA["categories"])
        
        return {"message": "Classifier updated successfully", "training_samples": len(TRAINING_DATA["texts"])}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

app.include_router(classify_router, prefix="/api/v1")


# ------------------------------ Stats endpoint ------------------------------ 

# Statistics endpoint
@app.get("/stats/", response_model=dict)
async def get_statistics(
    include_processing_times: bool = True,
    max_samples: int = 10
):
    """Get system statistics and performance metrics"""
    stats = {
        "total_documents": app.state.stats["total_documents"],
        "total_size_mb": round(app.state.stats["total_size_kb"] / 1024, 2),
        "documents_by_type": dict(app.state.stats["documents_by_type"]),
        "last_updated": app.state.stats["last_updated"]
    }
    
    if include_processing_times:
        stats["processing_times"] = {
            op: {
                "count": len(times),
                "avg_ms": round((sum(times[-max_samples:]) / len(times[-max_samples:])) * 1000, 2) if times else 0,
                "last_ms": round(times[-1] * 1000, 2) if times else 0
            }
            for op, times in app.state.stats["processing_times"].items()
        }
    
    return stats