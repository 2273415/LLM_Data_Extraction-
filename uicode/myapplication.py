import os
import io
import zipfile
import tempfile
import pandas as pd
from PIL import Image
import pytesseract
import pdfplumber
import gradio as gr
from pydantic import BaseModel
from typing import Optional, Dict, Type, Union
from datetime import date, datetime
from ollama import chat
import json

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"

# --- CATEGORY FINDER ---
def category_finder(doc: str, filename: str):
    class Category(BaseModel):
        category: str
        confidence_score: int

    document_category = [
        'Claim Form', 'iReport', 'Work Permit/Employment Pass', 'Incident Report', 'Identification Document',
        'Medical Bills', 'Medical Certificates', 'Discharge Summary', 'Wage Vouchers', 'Death Certificate',
        'Sub-Contractor Agreement & Work Injury Compensation Insurance', "Others"
    ]

    response = chat(
        messages=[
            {
                'role': 'user',
                'content': f"""
                filename : {filename}
                document content: {doc}
                Analyze and classify the document into one of these categories: {document_category}.
                Provide a confidence score (0-100) for the classification.
                """
            }
        ],
        model="gemma2:2b",
        format=Category.model_json_schema(),
    )
    category = Category.model_validate_json(response.message.content)
    return {"category": category.category, "confidence_score": category.confidence_score}

# --- DATA EXTRACTION ---
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
        "Sub-Contractor Agreement & Work Injury Compensation Insurance": ["contractor name", "subcontractor name", "agreement date", "insurance provider", "policy number", "coverage details"],
        "Others": ["name", "date of birth", "nationality"]
    }

    class Others(BaseModel):
        name: Optional[str]
        name_confidence_score: int
        date_of_birth: Optional[Union[date, str]]
        date_of_birth_confidence_score: int
        nationality: Optional[str]
        nationality_confidence_score: int
        overall_data_extraction_confidence_score: int

    class ClaimForm(BaseModel):
        claim_number: Optional[int]
        claim_number_confidence_score: int
        policyholder_name: Optional[str]
        policyholder_name_confidence_score: int
        accident_date: Optional[Union[date, str]]
        accident_date_confidence_score: int
        claim_amount: Optional[float]
        claim_amount_confidence_score: int
        insurance_provider: Optional[str]
        insurance_provider_confidence_score: int
        claim_status: Optional[str]
        claim_status_confidence_score: int
        overall_data_extraction_confidence_score: int

        claim_status_confidence_score: int
        overall_data_extraction_confidence_score: int

    class IReport(BaseModel):
        report_number: Optional[int]
        report_number_confidence_score: int
        incident_date: Optional[Union[date, str]]
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
        date_of_birth: Optional[Union[date, str]]
        date_of_birth_confidence_score: int
        nationality: Optional[str]
        nationality_confidence_score: int
        document_number: Optional[str]
        document_number_confidence_score: int
        expiry_date: Optional[Union[date, str]]
        expiry_date_confidence_score: int
        overall_data_extraction_confidence_score: int

    class MedicalBills(BaseModel):
        bill_number: Optional[Union[str, int]]
        bill_number_confidence_score: int
        hospital_clinic_name: Optional[str]
        hospital_clinic_name_confidence_score: int
        patient_name: Optional[str]
        patient_name_confidence_score: int
        treatment_date: Optional[Union[str, int, date]]
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
        issue_date: Optional[Union[date, str]]
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
        admission_date: Optional[Union[date, str]]
        admission_date_confidence_score: int
        discharge_date: Optional[Union[date, str]]
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
        date_of_death: Optional[Union[date, str]]
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
        agreement_date: Optional[Union[date, str]]
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
        "Others": Others
    }

    category = category.strip()
    if category not in document_fields:
        print(f"Unsupported category: '{category}'. Supported: {list(document_fields.keys())}")
        return pd.DataFrame()

    model_cls = model_mapping[category]

    response = chat(
        messages=[
            {
                'role': 'user',
                'content': f"""
                document : {doc}
                gather the information such as {document_fields[category]} from the above document along with the confidence_score for each data which is nothing but how confident you are about the given output from (0-100)
                """
            }
        ],
        model="gemma2:2b",
        format=model_cls.model_json_schema()
    )

    try:
        data = model_cls.model_validate_json(response.message.content)
    except Exception as e:
        print(f"Failed to parse LLM response: {e}")
        print("Raw response:", response.message.content)
        return pd.DataFrame()

    df = pd.DataFrame([data.model_dump()])
    return df

# --- INTERMEDIATE STORAGE ---
file_results = {}

last_extracted_df = {"df": pd.DataFrame()}
last_files_table_df = {"df": pd.DataFrame()}

# --- FILE EXTRACTION & CLASSIFICATION ---
def extract_and_classify(zip_file):
    if zip_file is None:
        last_files_table_df["df"] = pd.DataFrame()
        return pd.DataFrame()
    table = []
    file_results.clear()
    with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
        for info in zip_ref.infolist():
            if info.is_dir():
                continue
            filename = os.path.basename(info.filename)
            ext = os.path.splitext(info.filename)[1].lower()
            size_kb = round(info.file_size / 1024, 2)
            file_bytes = zip_ref.read(info.filename)
            text = ""
            no_of_pages = 1
            try:
                if ext == ".pdf":
                    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                        no_of_pages = len(pdf.pages)
                        for page in pdf.pages:
                            img = page.to_image(resolution=200).original
                            text += pytesseract.image_to_string(img) + "\n"
                elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                    img = Image.open(io.BytesIO(file_bytes))
                    text = pytesseract.image_to_string(img)
                else:
                    text = file_bytes.decode('utf-8', errors='replace')
            except Exception:
                text = ""
            cat_result = category_finder(text, filename)
            category = cat_result["category"]
            confidence = cat_result["confidence_score"]
            table.append([filename, size_kb, ext, category, confidence, text[:1000]])
            file_results[filename] = {
                "text": text,
                "category": category,
                "category_confidence_score": confidence,
                "no_of_pages": no_of_pages
            }
    df = pd.DataFrame(table, columns=["Name", "Size (KB)", "Extension", "Category", "Confidence", "Text Preview"])
    last_files_table_df["df"] = df
    return df


def collect_data():
    dfs = []
    for filename, info in file_results.items():
        df = data_extraction(info["text"], info["category"])
        if not df.empty:
            if "filename" not in df.columns:
                df.insert(0, "filename", filename)
            dfs.append(df)
    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        last_extracted_df["df"] = result_df
        return result_df
    else:
        last_extracted_df["df"] = pd.DataFrame()
        return pd.DataFrame()

def download_csv():
    df = last_extracted_df["df"]
    if df.empty:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        df.to_csv(f.name, index=False)
        return f.name

def download_json():
    df = last_extracted_df["df"]
    if df.empty:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
        df.to_json(f.name, orient="records", force_ascii=False)
        return f.name

def download_files_table_csv():
    df = last_files_table_df["df"]
    if df.empty:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="w", encoding="utf-8") as f:
        df.to_csv(f.name, index=False)
        return f.name

def download_files_table_json():
    df = last_files_table_df["df"]
    if df.empty:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
        df.to_json(f.name, orient="records", force_ascii=False)
        return f.name

def convert_dates(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: convert_dates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_dates(i) for i in obj]
    return obj

def download_custom_json():
    result = []
    for filename, info in file_results.items():
        df = data_extraction(info["text"], info["category"])
        if not df.empty:
            extracted_fields = df.iloc[0].to_dict()
            extracted_fields.pop("filename", None)
            # Convert all date/datetime objects to strings
            extracted_fields = convert_dates(extracted_fields)
            result.append({
                "file_name": filename,
                "no_of_pages": info.get("no_of_pages", 1),
                "category": info["category"],
                "category_confidence_score": info["category_confidence_score"],
                "extracted_fields": extracted_fields
            })
    if not result:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        return f.name

with gr.Blocks(theme=gr.themes.Soft(), title="Document Analyzer") as demo:
    gr.Markdown(
        """
        #  Document Category & Data Extractor
        Upload a ZIP file containing your documents.  
        The app will classify each file and extract structured data using AI.
        """
    )
    with gr.Row():
        zip_file = gr.File(label="Upload ZIP", file_types=[".zip"])
        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
        collect_fields = gr.Button("📝 Collect Data", variant="secondary")
    with gr.Tab("File Overview"):
        files_table = gr.Dataframe(
            label="Files in ZIP with Category",
            interactive=False,
            wrap=True
        )
        with gr.Row():
            download_files_table_csv_btn = gr.Button("⬇️ Download Table CSV")
            download_files_table_json_btn = gr.Button("⬇️ Download Table JSON")
        with gr.Row():
            files_table_csv_file = gr.File(label="ZIP Table CSV Download")
            files_table_json_file = gr.File(label="ZIP Table JSON Download")
    with gr.Tab("Extracted Data"):
        output_df = gr.Dataframe(
            label="Extracted Data (all files)",
            interactive=False,
            wrap=True
        )
        with gr.Row():
            download_csv_btn = gr.Button("⬇️ Download Extracted CSV")
            download_json_btn = gr.Button("⬇️ Download Extracted JSON")
            download_custom_json_btn = gr.Button("⬇️ Download Custom JSON")
        with gr.Row():
            csv_file = gr.File(label="CSV Download")
            json_file = gr.File(label="JSON Download")
            custom_json_file = gr.File(label="Custom JSON Download")

    analyze_btn.click(
        extract_and_classify,
        inputs=zip_file,
        outputs=files_table
    )
    collect_fields.click(
        collect_data,
        inputs=None,
        outputs=output_df
    )
    download_files_table_csv_btn.click(
        download_files_table_csv,
        inputs=None,
        outputs=files_table_csv_file
    )
    download_files_table_json_btn.click(
        download_files_table_json,
        inputs=None,
        outputs=files_table_json_file
    )
    download_csv_btn.click(
        download_csv,
        inputs=None,
        outputs=csv_file
    )
    download_json_btn.click(
        download_json,
        inputs=None,
        outputs=json_file
    )
    download_custom_json_btn.click(
        download_custom_json,
        inputs=None,
        outputs=custom_json_file
    )

if __name__ == "__main__":
    demo.launch(share=True,inbrowser=True)
