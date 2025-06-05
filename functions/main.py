# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`
from firebase_functions import https_fn
from fastapi import FastAPI
import uvicorn
import os
from mangum import Mangum

# Import your existing FastAPI app
from backend.main import app as fastapi_app

# Create handler for Firebase Functions
handler = Mangum(fastapi_app)

@https_fn.on_request()
def fastapi_function(req: https_fn.Request) -> https_fn.Response:
    # Convert Firebase request to ASGI scope
    scope = {
        "type": "http",
        "method": req.method,
        "path": req.path,
        "headers": [[k.lower(), v] for k, v in req.headers.items()],
        "query_string": req.query_string.encode(),
        "body": req.data
    }
    
    # Call FastAPI through Mangum adapter
    response = handler(scope, None)
    
    return https_fn.Response(
        response["body"],
        status=response["status"],
        headers=dict(response["headers"])
    )

# initialize_app()
#
#
# @https_fn.on_request()
# def on_request_example(req: https_fn.Request) -> https_fn.Response:
#     return https_fn.Response("Hello world!")