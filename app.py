# Import necessary libraries
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from azure.storage.blob import BlobServiceClient
from fastapi.responses import StreamingResponse, FileResponse
from pymongo import MongoClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import io
import mimetypes
import pandas as pd
from datetime import datetime, date
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4  # Import for generating unique ID
from model import download_csv, upload_bank_file, get_extracted_values,upload_pdf_to_blob, view_pdf
from pymongo import MongoClient
import certifi
from model import upload_bank_file, get_extracted_values, upload_pdf_to_blob, view_pdf





# Azure Blob Storage connection string
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=aisa0101;AccountKey=rISVuOQPHaSssHHv/dQsDSKBrywYnk6bNuXuutl4n+ILZNXx/CViS50NUn485kzsRxd5sfiVSsMi+AStga0t0g==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "bankimage"

# Azure Form Recognizer details
FORM_RECOGNIZER_ENDPOINT = "https://eastus.api.cognitive.microsoft.com/"
FORM_RECOGNIZER_KEY = "26cfaa6c7c314e9a8ad7a68587ca3ce9"

# Initialize the client
document_analysis_client = DocumentAnalysisClient(endpoint=FORM_RECOGNIZER_ENDPOINT, credential=AzureKeyCredential(FORM_RECOGNIZER_KEY))

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
client = MongoClient(
    "mongodb+srv://akratirathore21:Akrati12345@clusterar1.jjhs5.mongodb.net/?retryWrites=true&w=majority&appName=ClusterAR1",
    tls=True,
    tlsCAFile=certifi.where()
)
db = client["Akrati"]
form_collection = db["form_data"]
doc_collection = db["doc_data"]

global_bank_statement = None
uploaded_files = {}

@app.post("/submit")
async def submit_data(
    first_name: str = Form(...),
    last_name: str = Form(...),
    email: str = Form(...),
    dob: str = Form(...),
    phone: str = Form(...),
    aadhar: str = Form(...),
    pan: str = Form(...),
    aadhar_file: UploadFile = File(...),
    pan_file: UploadFile = File(...),
    bank_statement_file: UploadFile = File(...),
):
    # Generate a unique ID to link form and document data
    await upload_bank_file(bank_statement_file)
    unique_id = str(uuid4())
    uploaded_files['aadhar_file'] = aadhar_file.filename
    uploaded_files['pan_file'] = pan_file.filename

    # Store the bank statement in the global variable
    if bank_statement_file.content_type == "application/pdf":
        global_bank_statement = await bank_statement_file.read()
    else:
        return {"error": "Bank statement must be a PDF file."}

    # Save form data to MongoDB with unique ID
    form_data = {
        "unique_id": unique_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "dob": dob,
        "phone": phone,
        "aadhar": aadhar,
        "pan": pan,
    }
    form_result = form_collection.insert_one(form_data)

    # Analyze the uploaded files using Azure Form Recognizer
    def analyze_document(file_stream: io.BytesIO, mime_type: str, doc_type: str):
        if mime_type.startswith('image') or mime_type in ['image/jpeg', 'image/png', 'application/pdf']:
            poller = document_analysis_client.begin_analyze_document("prebuilt-idDocument", document=file_stream)
            id_documents = poller.result()
            return extract_data(id_documents, doc_type)
        return {"error": "Unsupported file type. Please upload PDF or image file."}

    def extract_data(id_documents, doc_type: str):
        data = {}
        for id_document in id_documents.documents:
            if "FirstName" in id_document.fields:
                data["first_name"] = id_document.fields["FirstName"].value
            if "LastName" in id_document.fields:
                data["last_name"] = id_document.fields["LastName"].value
            if "DateOfBirth" in id_document.fields:
                dob = id_document.fields["DateOfBirth"].value
                if isinstance(dob, date):
                    dob = datetime.combine(dob, datetime.min.time())
                if isinstance(dob, datetime):
                    data["dob"] = dob.strftime('%Y-%m-%d')
            if "DocumentNumber" in id_document.fields:
                if doc_type == "aadhar":
                    data["aadhar_no"] = id_document.fields["DocumentNumber"].value
                elif doc_type == "pan":
                    data["pan_no"] = id_document.fields["DocumentNumber"].value
        return data

    mime_type_aadhar, _ = mimetypes.guess_type(aadhar_file.filename)
    mime_type_pan, _ = mimetypes.guess_type(pan_file.filename)

    file_stream_aadhar = await aadhar_file.read()
    file_stream_pan = await pan_file.read()

    # Upload Aadhar and PAN files to Azure Blob Storage
    upload_pdf_to_blob('aadhar_file', file_stream_aadhar)
    upload_pdf_to_blob('pan_file', file_stream_pan)

    extracted_aadhar = analyze_document(io.BytesIO(file_stream_aadhar), mime_type_aadhar, "aadhar")
    extracted_pan = analyze_document(io.BytesIO(file_stream_pan), mime_type_pan, "pan")

    # Save extracted document data to MongoDB with the same unique ID
    account_name_str, account_no_str, address_str, branch_str = get_extracted_values()
    extracted_bank = {
        "account_name": account_name_str,
        "account_no": account_no_str,
        "address": address_str,
        "branch": branch_str
    }
    document_data = {
        "unique_id": unique_id,
        "extracted_aadhar": extracted_aadhar,
        "extracted_pan": extracted_pan,
        "extracted_bank": extracted_bank
    }
    doc_result = doc_collection.insert_one(document_data)

    # Compare the form and Aadhaar data
    aadhaar_comparison_result = {
        "first_name_match_aadhaar": first_name.strip().lower() == extracted_aadhar.get("first_name", "").strip().lower(),
        "last_name_match_aadhaar": last_name.strip().lower() == extracted_aadhar.get("last_name", "").strip().lower(),
        "dob_match_aadhaar": dob == extracted_aadhar.get("dob", ""),
        "aadhar_no_match_aadhaar": aadhar == extracted_aadhar.get("aadhar_no", "")
    }

    # Compare the form and PAN data
    pan_comparison_result = {
        "first_name_match_pan": first_name.strip().lower() == extracted_pan.get("first_name", "").strip().lower(),
        "last_name_match_pan": last_name.strip().lower() == extracted_pan.get("last_name", "").strip().lower(),
        "dob_match_pan": dob == extracted_pan.get("dob", ""),
        "pan_no_match_pan": pan == extracted_pan.get("pan_no", "")
    }

    # Overall verification status based on matching all details
    aadhaar_verified = all(aadhaar_comparison_result.values())
    pan_verified = all(pan_comparison_result.values())
    overall_verified = aadhaar_verified and pan_verified

    return {
        "message": "Form data and documents uploaded successfully",
        # "form_id": str(form_result.inserted_id),
        # "doc_id": str(doc_result.inserted_id),
        "aadhaar_comparison_result": aadhaar_comparison_result,
        "pan_comparison_result": pan_comparison_result,
        "verification_status": "verified" if overall_verified else "not verified"
    }

@app.get("/view_adhar_file")
def view_adhar_file():
    """Route to view the Aadhaar PDF in the browser directly from Azure Blob."""
    try:
        return view_pdf("aadhar_file", "pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Aadhaar PDF: {str(e)}")

@app.get("/view_pan_file")
def view_pan_file():
    """Route to view the PAN PDF in the browser directly from Azure Blob."""
    try:
        return view_pdf("pan_file", "pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching PAN PDF: {str(e)}")


@app.get("/view_bank_statement")
def view_bank_statement():
    """Route to download and view the bank statement CSV."""
    try:
        file_path = download_csv("Bank_Statement", "csv")  # Update blob name if necessary
        return FileResponse(file_path, media_type="text/csv", filename="Bank_Statement.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching Bank Statement CSV: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
