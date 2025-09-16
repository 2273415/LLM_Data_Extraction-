from PIL import Image, ImageOps,ImageFilter
import os
import pdfplumber
import pytesseract
import pandas as pd
import requests
import json
from langchain_ollama import OllamaLLM
import json
import zipfile
import pandas as pd
import os
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
import zipfile
import pandas as pd
import time
from typing import Optional, Dict, Type,Union
from datetime import date
from pydantic import BaseModel
from ollama import chat
app = FastAPI()



tesseract_path=r"/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd=tesseract_path

llm="gemma2:2b"

def process_zip_memory(zip_memory: BytesIO):
    """
    Processes PDF files inside an in-memory ZIP archive,
    extracts content, classifies them, and returns a final DataFrame.
    """
    try:
        all_data = []
        
        with zipfile.ZipFile(zip_memory) as zip_file:
            # List all PDF files in the ZIP
            pdf_filenames = [f for f in zip_file.namelist() if f.lower().endswith(".pdf")]

            for pdf_name in pdf_filenames:
                with zip_file.open(pdf_name) as pdf_file:
                    # Read PDF file as bytes and pass it to your PDF extraction logic
                    pdf_bytes = pdf_file.read()
                    pdf_data = pdf_to_content(BytesIO(pdf_bytes),pdf_name)  # Assuming pdf_to_content accepts BytesIO

                    if pdf_data:
                        all_data.append(pdf_data)
                time.sleep(5)
                
        final_df = pd.DataFrame(all_data)
        print(final_df)
        return final_df
    
    except Exception as e:
        print("Error occurred:", e)
        return None


def pdf_to_content(file_path: str,filename: str = "uploaded.pdf"):
    """
    High-accuracy OCR from a PDF using image preprocessing + Tesseract config.
    """
    if not file_path:
        raise ValueError("Invalid file path provided.")

    try:
        content_string = ""
        with pdfplumber.open(file_path) as pdf:
            print(f"\n🔍 Processing file: {filename}\n")

            metadata = pdf.metadata or {}
            meta_text = "\n".join([f"{k}: {v}" for k, v in metadata.items() if v])
            content_string += meta_text + "\n\n"

            for i, page in enumerate(pdf.pages):
                try:
                    # Convert to image at higher resolution
                    image = page.to_image(resolution=400).original

                    # --- OCR Image Preprocessing ---
                    image = image.convert("L")  # Grayscale
                    image = image.point(lambda x: 0 if x < 180 else 255, '1')  # Binarize
                    image = image.filter(ImageFilter.MedianFilter(size=3))  # Denoise

                    # Run OCR with advanced config
                    custom_config = r'--oem 3 --psm 6'  # Use LSTM OCR Engine, assume block of text
                    ocr_text = pytesseract.image_to_string(image, lang="eng", config=custom_config)

                    content_string += f"\n--- Page {i + 1} ---\n{ocr_text}\n|next page|\n\n"

                except Exception as e:
                    print(f"⚠️ OCR failed on page {i + 1}: {e}")

        content_string += "\n|end of file|\n"

        # Classification (your logic)
        category, score, fields= classify_and_extraction(doc=content_string)

        return {
            "filename": filename.split("/")[-1],
            "category": category,
            "classification_confidence": score,
            "extracted_fields": fields,
            "content": content_string
        }
    except Exception as e:
        print(f"🚨 Failed to process PDF '{file_path}': {e}")
        return None


        
def category_finder(doc: str):
    
    class Category(BaseModel):
      category: str
      confidence_score:int

    document_category=['Claim Form', 'iReport', 'Work Permit/Employment Pass', 'Identification Document', 'Medical Bills', 'Medical Certificates', 'Discharge Summary', 'Wage Vouchers', 'Death Certificate', 'Sub-Contractor Agreement & Work Injury Compensation Insurance']
    prompt=f"""document : {str(doc)}
            Classify the document into one of these categories: {str(document_category)}use  this list and search in the document if you can't classify example search claim form in text document if you find then classify as claim form.
            - Provide a confidence score (0-100) for the classification."""
    
    response = chat(messages=[
            {
            'role': 'user',
            'content':prompt,
            }
        ],
        model=llm,
        format=Category.model_json_schema(),
        )
    category = Category.model_validate_json(response.message.content)
   
    return category.category,category.confidence_score



def data_extraction(doc: str, category: str):
    

    document_fields = {
        "Claim Form": ["claim number", "policyholder name", "accident date", "claim amount", "insurance provider", "claim status"],
        "iReport": ["report number", "incident date", "injured party", "injury description", "location", "witnesses"],
        "Work Permit/Employment Pass": ["permit number", "worker name", "nationality", "employer name", "validity period", "work sector"],
        "Identification Document": ["document type", "full name", "date of birth", "nationality", "document number", "expiry date"],
        "Medical Bills": ["bill number", "hospital/clinic name", "patient name", "treatment date", "total amount", "insurance coverage"],
        "Medical Certificates": ["certificate number", "patient name", "issue date", "doctor's name", "validity period", "reason for leave"],
        "Discharge Summary": ["patient name", "hospital name", "admission date", "discharge date", "diagnosis", "treatment provided"],
        "Wage Vouchers": ["employee name", "month/year", "basic salary", "overtime pay", "deductions", "net salary"],
        "Death Certificate": ["deceased name", "date of death", "cause of death", "place of death", "certificate number", "issued by"],
        "Sub-Contractor Agreement & Work Injury Compensation Insurance": ["contractor name", "subcontractor name", "agreement date", "insurance provider", "policy number", "coverage details"]
    }

    class ClaimForm(BaseModel):
        claim_number: Optional[int]
        claim_number_confidence_score: int

        policyholder_name: Optional[str]
        policyholder_name_confidence_score: int

        accident_date: Optional[date]
        accident_date_confidence_score: int

        claim_amount: Optional[float]
        claim_amount_confidence_score: int

        insurance_provider: Optional[str]
        insurance_provider_confidence_score: int

        claim_status: Optional[str]
        claim_status_confidence_score: int

        overall_data_extraction_confidence_score: int

    class IReport(BaseModel):
        report_number: Optional[int]
        report_number_confidence_score: int

        incident_date: Optional[date]
        incident_date_confidence_score: int

        injured_party: Optional[str]
        injured_party_confidence_score: int

        injury_description: Optional[str]
        injury_description_confidence_score: int

        location: Optional[str]
        location_confidence_score: int

        witnesses: Optional[str]
        witnesses_confidence_score: int

        overall_data_extraction_confidence_score: int

    class WorkPermitEmploymentPass(BaseModel):
        permit_number: Optional[str]
        permit_number_confidence_score: int

        worker_name: Optional[str]
        worker_name_confidence_score: int

        nationality: Optional[str]
        nationality_confidence_score: int

        employer_name: Optional[str]
        employer_name_confidence_score: int

        validity_period: Optional[str]
        validity_period_confidence_score: int

        work_sector: Optional[str]
        work_sector_confidence_score: int

        overall_data_extraction_confidence_score: int

    class IdentificationDocument(BaseModel):
        document_type: Optional[str]
        document_type_confidence_score: int

        full_name: Optional[str]
        full_name_confidence_score: int

        date_of_birth: Optional[date]
        date_of_birth_confidence_score: int

        nationality: Optional[str]
        nationality_confidence_score: int

        document_number: Optional[str]
        document_number_confidence_score: int

        expiry_date: Optional[date]
        expiry_date_confidence_score: int

        overall_data_extraction_confidence_score: int

    class MedicalBills(BaseModel):
        bill_number: Optional[Union[str,int]]
        bill_number_confidence_score: int

        hospital_clinic_name: Optional[str]
        hospital_clinic_name_confidence_score: int

        patient_name: Optional[str]
        patient_name_confidence_score: int

        treatment_date: Optional[Union[str,int,date]]
        treatment_date_confidence_score: int

        total_amount: Optional[float]
        total_amount_confidence_score: int

        insurance_coverage: Optional[str]
        insurance_coverage_confidence_score: int

    class MedicalCertificates(BaseModel):
        certificate_number: Optional[str]
        certificate_number_confidence_score: int

        patient_name: Optional[str]
        patient_name_confidence_score: int

        issue_date: Optional[date]
        issue_date_confidence_score: int

        doctors_name: Optional[str]
        doctors_name_confidence_score: int

        validity_period: Optional[str]
        validity_period_confidence_score: int

        reason_for_leave: Optional[str]
        reason_for_leave_confidence_score: int

        overall_data_extraction_confidence_score: int

    class DischargeSummary(BaseModel):
        patient_name: Optional[str]
        patient_name_confidence_score: int

        hospital_name: Optional[str]
        hospital_name_confidence_score: int

        admission_date: Optional[date]
        admission_date_confidence_score: int

        discharge_date: Optional[date]
        discharge_date_confidence_score: int

        diagnosis: Optional[str]
        diagnosis_confidence_score: int

        treatment_provided: Optional[str]
        treatment_provided_confidence_score: int

        overall_data_extraction_confidence_score: int

    class WageVouchers(BaseModel):
        employee_name: Optional[str]
        employee_name_confidence_score: int

        month_year: Optional[str]
        month_year_confidence_score: int

        basic_salary: Optional[float]
        basic_salary_confidence_score: int

        overtime_pay: Optional[float]
        overtime_pay_confidence_score: int

        deductions: Optional[float]
        deductions_confidence_score: int

        net_salary: Optional[float]
        net_salary_confidence_score: int

        overall_data_extraction_confidence_score: int

    class DeathCertificate(BaseModel):
        deceased_name: Optional[str]
        deceased_name_confidence_score: int

        date_of_death: Optional[date]
        date_of_death_confidence_score: int

        cause_of_death: Optional[str]
        cause_of_death_confidence_score: int

        place_of_death: Optional[str]
        place_of_death_confidence_score: int

        certificate_number: Optional[str]
        certificate_number_confidence_score: int

        issued_by: Optional[str]
        issued_by_confidence_score: int

        overall_data_extraction_confidence_score: int

    class SubContractorAgreementAndWICInsurance(BaseModel):
        contractor_name: Optional[str]
        contractor_name_confidence_score: int

        subcontractor_name: Optional[str]
        subcontractor_name_confidence_score: int

        agreement_date: Optional[date]
        agreement_date_confidence_score: int

        insurance_provider: Optional[str]
        insurance_provider_confidence_score: int

        policy_number: Optional[str]
        policy_number_confidence_score: int

        coverage_details: Optional[str]
        coverage_details_confidence_score: int

        overall_data_extraction_confidence_score: int

    model_mapping: Dict[str, Type[BaseModel]] = {
        "Claim Form": ClaimForm,
        "iReport": IReport,
        "Work Permit/Employment Pass": WorkPermitEmploymentPass,
        "Identification Document": IdentificationDocument,
        "Medical Bills": MedicalBills,
        "Medical Certificates": MedicalCertificates,
        "Discharge Summary": DischargeSummary,
        "Wage Vouchers": WageVouchers,
        "Death Certificate": DeathCertificate,
        "Sub-Contractor Agreement & Work Injury Compensation Insurance": SubContractorAgreementAndWICInsurance,
    }

    if category not in model_mapping:
        raise ValueError(f"Unsupported category: {category}")

    model_cls = model_mapping[category]
    response = chat(
        messages=[
            {
                'role': 'user',
                'content': f"""
                document : {str(doc)}
                gather the information such as {str(document_fields[category])} from the above document along with the confidence_score for each data which is nothing but how confident you are about the given output from (0-100)
                """
            }
        ],
        # model="llama3.2:latest",
        model=llm,
        format=model_cls.model_json_schema()
    )
    data = model_cls.model_validate_json(response.message.content)
    df = pd.DataFrame([data.model_dump()])
    return df

def classify_and_extraction(doc:str):
    
    actual_category,category_confidence=category_finder(doc)
    data=data_extraction(doc,actual_category)
    return actual_category,category_confidence,data




@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed.")

    try:
        zip_memory = BytesIO(await file.read())
        df = process_zip_memory(zip_memory)  # Your function that returns a DataFrame

        if df.empty:
            raise HTTPException(status_code=400, detail="No PDF data extracted.")

        # Write CSV to buffer
        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # This is critical to reset buffer before reading

        return StreamingResponse(
            content=csv_buffer,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=result.csv"}
        )


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



