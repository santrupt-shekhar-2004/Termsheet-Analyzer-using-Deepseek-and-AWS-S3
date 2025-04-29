import os
import re
import PyPDF2
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import pandas as pd
import logging
import datetime
from tabulate import tabulate
import json
import spacy
from dateutil import parser as dateparser  # For free-form date parsing
import os
import json
import openai 
from openai import OpenAI 
from tabulate import tabulate# Add this import
from rich.console import Console
from rich.table import Table
from rich import box
import tempfile
import imaplib
import email
from email.header import decode_header
from typing import Dict, Any, Set,List
import pprint
import tempfile
from typing import Any, Dict, Optional
from telethon.sync import TelegramClient
from telethon.tl.types import InputMessagesFilterDocument
import re
import email
import imaplib
import os
import tempfile
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from enum import Enum
from email.header import decode_header


import pdfplumber  # For PDF extraction
import pytesseract  # OCR engine
from PIL import Image  # For image processing
from pdf2image import convert_from_path  # For converting PDF pages to images
from docx import Document  # For DOCX extraction
import spacy  # NLP library
nlp = spacy.load("en_core_web_sm")

import pandas as pd
from transformers import pipeline  # For more advanced NLP tasks

from typing import Tuple, Dict, Any
import boto3
from botocore.exceptions import ClientError


import json
import boto3
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")

def check_aws_credentials():
    """
    Comprehensive check of AWS credentials and configuration
    """
    print("\n🔍 AWS Credentials Check:")
    print("=" * 30)
    
    try:
        # Check session credentials
        session = boto3.Session()
        credentials = session.get_credentials()
        
        if credentials:
            print("✅ AWS Credentials Found:")
            print(f"Access Key ID: {credentials.access_key[:4]}{'*' * (len(credentials.access_key) - 8)}{credentials.access_key[-4:]}")
            print("Credential Source:", credentials.method)
        else:
            print("❌ No AWS Credentials Found")
        
        # Check current region
        current_region = session.region_name
        print(f"\n🌐 Current Region: {current_region}")
        
        # List available profiles
        print("\n📋 Available AWS Profiles:")
        profiles = session.available_profiles
        for profile in profiles:
            print(f"- {profile}")
    
    except Exception as e:
        print(f"❌ Error checking credentials: {e}")


def get_secret():
    """
    Retrieve secret from AWS Secrets Manager with comprehensive logging
    """
    secret_name = "hackathon-2025-secret"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        # Retrieve the secret
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
        # Check if SecretString is in the response
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            
            # Attempt to parse the secret
            try:
                # Try parsing as JSON
                parsed_secret = json.loads(secret)
                
                # Comprehensive logging
                print("🔐 Secret Retrieved Successfully:")
                print("=" * 40)
                
                # Pretty print all keys and values
                for key, value in parsed_secret.items():
                    # Mask sensitive information
                    display_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '****'
                    print(f"{key}: {display_value}")
                
                print("=" * 40)
                
                return parsed_secret
            
            except json.JSONDecodeError:
                # If it's not a JSON, print the raw string
                print("⚠️ Secret is not a valid JSON. Raw content:")
                print(secret)
                return {"raw_secret": secret}
        
        else:
            print("❌ No SecretString found in the secret")
            return None

    except ClientError as e:
        # Detailed error handling
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        print("❌ Error retrieving secret:")
        print(f"Error Code: {error_code}")
        print(f"Error Message: {error_message}")
        
        # Additional context based on common error codes
        if error_code == 'ResourceNotFoundException':
            print("The secret does not exist in Secrets Manager.")
        elif error_code == 'InvalidParameterException':
            print("The request had invalid parameters.")
        elif error_code == 'InvalidRequestException':
            print("The request was invalid due to a problem with the secret.")
        elif error_code == 'DecryptionFailureException':
            print("The requested secret can't be decrypted.")
        
        raise

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

# Run the checks
# Run the checks
if __name__ == '__main__':
    # Check AWS credentials first
    check_aws_credentials()
    
    # Retrieve and print secret
    try:
        secret = get_secret()
        print("\nRetrieved Secret:")
        print(secret)
    except Exception as e:
        print(f"Error retrieving secret: {e}")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# instantiate DeepSeek client
client = OpenAI(
    api_key= "sk-215d925b64e44893acc2d2ed3f3aa8cf",
    base_url='https://api.deepseek.com'
)
def validate_with_claude_bedrock(
    parameters: dict[str, Any],
    master_path: str = "master_values.xlsx"
) -> dict[str, Any]:
    """
    Uses AWS Bedrock (Claude 3.5 Haiku) to validate extracted parameters.
    """

    master = load_master_values(master_path)

    validation_input = []
    for key, extracted_value in parameters.items():
        master_values = list(master.get(key, set()))
        validation_input.append({
            "parameter": key,
            "extracted_value": extracted_value,
            "master_values": master_values
        })

    prompt = (
        "You are a senior derivatives analyst at Barclays, reviewing extracted financial parameters "
        "from a transaction term sheet related to Foreign Exchange (FX) Options or structured products.\n"
        "For each parameter:\n"
        "1. Normalize both the extracted value and master values.\n"
        "2. Determine if the extracted value matches any master value.\n"
        "3. Provide a detailed, domain-specific comment.\n"
        "Output must be in JSON format like:\n"
        "{\n"
        "  \"validation_results\": {\n"
        "    \"parameter_name\": { \"match\": true/false, \"comment\": \"...\" }\n"
        "  }\n"
        "}"
    )

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": 1,
        "top_p": 0.999,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt + "\n\n" + json.dumps(validation_input, indent=2)
                    }
                ]
            }
        ]
    }

    try:
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-5-haiku-20241022-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        response_body = json.loads(response['body'].read())

        # Claude responses have 'content' inside a message
        content_text = response_body['content'][0]['text']

        # Cleanup if needed
        cleaned = content_text.strip().lstrip('⁠  json').rstrip('  ⁠').strip()

        results = json.loads(cleaned).get("validation_results", {})

        # Structure final output
        validation_results = {}
        for param in parameters:
            extracted_val = parameters[param]
            master_vals = sorted(master.get(param, set()))
            deep = results.get(param, {})

            validation_results[param] = {
                "value": extracted_val,
                "master_list": master_vals,
                "match": deep.get("match", False),
                "comment": deep.get("comment", "No response")
            }

        return {"validation_results": validation_results}

    except Exception as e:
        logger.error(f"Claude Bedrock validation failed: {e}")
        return {
            "validation_results": {
                k: {
                    "value": parameters[k],
                    "master_list": sorted(master.get(k, set())),
                    "match": False,
                    "comment": "Claude service unavailable"
                } for k in parameters
            }
        }


import os
import json
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import tempfile
import re

'''class S3Handler:
    def __init__(self, bucket_name=None, region_name=None):
        """
        Initialize S3 handler with bucket info
        Uses AWS environment or configured credentials
        """
        self.bucket_name = bucket_name or os.environ.get('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3_BUCKET_NAME environment variable is not set")

        # Use default credentials from AWS configuration
        self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_file(self, file_path: str, s3_key: str) -> bool:
        """Upload a file to S3"""
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"Successfully uploaded file to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error uploading file to S3: {e}")
            return False

    def upload_data(self, data: dict, s3_key: str) -> bool:
        """Upload JSON data to S3"""
        try:
            json_data = json.dumps(data, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Successfully uploaded data to s3://{self.bucket_name}/{s3_key}")
            return True
        except ClientError as e:
            print(f"Error uploading data to S3: {e}")
            return False

def generate_safe_filename(filename):
    """Generate a safe filename by removing special characters"""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', filename)

def upload_to_s3(s3_handler, result, original_file_path=None):
    """
    Upload extraction results and original file to S3
    
    Args:
        s3_handler (S3Handler): S3 handler instance
        result (dict): Extracted data dictionary
        original_file_path (str, optional): Path to original file
    
    Returns:
        dict: Updated result with S3 metadata
    """
    try:
        # Ensure S3 handler exists
        if not isinstance(s3_handler, S3Handler):
            s3_handler = S3Handler()
        
        # Generate timestamp and safe filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = generate_safe_filename("termsheet.pdf")

        # Generate S3 keys
        base_path = f"termsheets/{timestamp}"
        original_key = f"{base_path}/original/{safe_filename}"
        extracted_key = f"{base_path}/extracted/termsheet_extracted.json"

        # Upload original file if provided
        uploaded_file = False
        if original_file_path and os.path.exists(original_file_path):
            uploaded_file = s3_handler.upload_file(original_file_path, original_key)

        # Upload extracted data
        uploaded_json = s3_handler.upload_data(result, extracted_key)

        # Add S3 metadata to result
        if uploaded_file and uploaded_json:
            result["s3_metadata"] = {
                "original_file": {
                    "bucket": s3_handler.bucket_name,
                    "key": original_key,
                    "filename": safe_filename
                },
                "extracted_data": {
                    "bucket": s3_handler.bucket_name,
                    "key": extracted_key,
                    "filename": "termsheet_extracted.json"
                }
            }
        else:
            result["s3_upload_error"] = "One or more S3 uploads failed"

    except Exception as e:
        print(f"Error uploading to S3: {str(e)}")
        result["s3_upload_error"] = str(e)

    return result

# Modify your existing methods to use this upload_to_s3 function

def find_and_extract_termsheet_from_pdf(
    email_address: str,
    password: str,
    termsheet_id: str,
    mail_server: str = "imap.gmail.com",
    s3_handler: S3Handler = None
) -> Dict[str, Any]:
    """
    Enhanced method to extract termsheet and optionally upload to S3
    """
    # Your existing extraction logic here
    
    # After extraction, upload to S3
    if s3_handler and extracted:
        extracted = upload_to_s3(s3_handler, extracted, original_file_path)
    
    return extracted

def find_and_extract_termsheet_from_telegram(
    api_id: int,
    api_hash: str,
    chat_identifier: str,
    termsheet_keyword: str,
    s3_handler: S3Handler = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Enhanced Telegram extraction method with S3 upload
    """
    # Your existing extraction logic here
    
    # After extraction, upload to S3
    if s3_handler and extracted:
        extracted = upload_to_s3(s3_handler, extracted, original_file_path)
    
    return extracted

def process_direct_upload(file_path, s3_handler=None):
    """
    Process direct PDF upload with optional S3 upload
    """
    try:
        # Extract data from PDF
        pdf_data = extract_data_from_pdf(file_path)
        
        # Validate extracted data
        validator = TermSheetValidator(pdf_data)
        validation_results = validator.validate()

        result = {
            "extracted_data": pdf_data,
            "validation_results": validation_results,
            "file_metadata": {
                "filename": os.path.basename(file_path),
                "file_path": file_path
            }
        }

        # Optional S3 upload
        if s3_handler:
            result = upload_to_s3(s3_handler, result, file_path)

        return result, None

    except Exception as e:
        return None, str(e)

# Initialize S3 handler
s3_handler = S3Handler(bucket_name='your-bucket-name')

# Email extraction with S3 upload
result = find_and_extract_termsheet_from_pdf(
    email, password, termsheet_id, 
    s3_handler=s3_handler
)

# Telegram extraction with S3 upload
result = find_and_extract_termsheet_from_telegram(
    api_id, api_hash, chat_id, keyword,
    s3_handler=s3_handler
)

# Direct file upload with S3 upload
result, error = process_direct_upload(
    file_path, 
    s3_handler=s3_handler
)'''
import os
import json
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import tempfile
import re

import os
import re
import json
from datetime import datetime

import boto3
from botocore.exceptions import ClientError

# At the top of your file
import boto3
import os
import json

class S3Handler:
    def __init__(self, bucket_name=None, region_name=None):
        self.bucket_name = os.environ.get('S3_BUCKET_NAME')
        if not self.bucket_name:
            raise ValueError("S3 bucket name must be provided")

        self.s3_client = boto3.client('s3', region_name=region_name)

    def upload_file(self, file_path: str, s3_key: str) -> bool:
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"Uploaded file to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False

    def upload_data(self, data: dict, s3_key: str) -> bool:
        try:
            json_data = json.dumps(data, indent=2, default=str)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=json_data.encode('utf-8'),
                ContentType='application/json'
            )
            print(f"Uploaded JSON to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            print(f"Error uploading data to S3: {e}")
            return False

def enhance_extraction_with_s3(extraction_func):
    """
    Decorator to automatically upload extracted files and data to S3
    
    Args:
        extraction_func (callable): Function that extracts data from a source
    
    Returns:
        callable: Enhanced function with S3 upload capability
    """
    def wrapper(*args, **kwargs):
        # Check if S3 upload is desired
        s3_handler = kwargs.pop('s3_handler', None)
        
        # Call original extraction function
        result = extraction_func(*args, **kwargs)
        
        # If S3 handler is provided and extraction was successful
        if s3_handler and isinstance(s3_handler, S3Handler) and result:
            # Upload original file if available
            original_file = result.get('file_path') or kwargs.get('file_path')
            if original_file and os.path.exists(original_file):
                original_s3_metadata = s3_handler.upload_file(original_file, s3_prefix='original/')
                result['s3_original_file'] = original_s3_metadata
            
            # Upload extracted data
            extracted_data = result.get('extracted_data', result)
            extracted_s3_metadata = s3_handler.upload_json_data(extracted_data)
            result['s3_extracted_data'] = extracted_s3_metadata
        
        return result
    
    return wrapper

# Example usage in your extraction functions:

@enhance_extraction_with_s3
def find_and_extract_termsheet_from_pdf(email_address, password, termsheet_id, **kwargs):
    # Your existing PDF extraction logic
    # ...
    # You can now optionally pass an s3_handler
    pass

@enhance_extraction_with_s3
def find_and_extract_termsheet_from_telegram(api_id, api_hash, chat_identifier, termsheet_keyword, **kwargs):
    # Your existing Telegram extraction logic
    # ...
    # You can now optionally pass an s3_handler
    pass

@enhance_extraction_with_s3
def process_direct_upload(file_path, **kwargs):
    # Your existing direct upload logic
    # ...
    # You can now optionally pass an s3_handler
    pass

'''# Example of how to use with S3 upload:
if __name__ == '__main__':
    # Initialize S3 handler
    s3_handler = S3Handler(bucket_name='your-derivative-termsheets')

    # Example usages with S3 upload
    pdf_result = find_and_extract_termsheet_from_pdf(
        email, password, termsheet_id, 
        s3_handler=s3_handler
    )

    telegram_result = find_and_extract_termsheet_from_telegram(
        api_id, api_hash, chat_id, keyword,
        s3_handler=s3_handler
    )

    direct_upload_result = process_direct_upload(
        file_path, 
        s3_handler=s3_handler
    )'''

    
class DerivativeTradeExtractor:
    def __init__(self):
        # Load spaCy NLP model (used for both date and parameter extraction fallback)
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Successfully loaded spaCy model en_core_web_sm")
        except Exception as e:
            logger.warning("en_core_web_sm not found. Installing...")
            import subprocess
            subprocess.call("python -m spacy download en_core_web_sm", shell=True)
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define mapping for date fields to context keywords for NLP extraction
        self.date_map = {
            'trade_date': ['trade date', 'trade', 'transaction date'],
            'expiry_date': ['expiry date', 'expiry', 'expiration', 'maturity'],
            'delivery_date': ['delivery date', 'delivery', 'settlement', 'payment date']
        }
        
        # Define regex patterns for each required parameter (general and custom formats)
        self.parameters = {
            # Custom date pattern for "DD Month YYYY" format
            'trade_date': r'(?i)trade\s*date\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'ref_spot_price': r'(?i)(?:reference\s*spot\s*price|ref\s*spot\s*price|spot\s*price|reference\s*price)[\s:]+([\d.,]+)',
            'notional_amount': r'(?i)notional\s*amount\s*((?:[A-Z]{3}\s*)?[\d,]+(?:,\d{3})*(?:\.\d+)?)',
            'strike_price': r'(?i)strike\s*price\s*([\d.,]+)',
            'option_type': r'(?i)option\s*type\s*(European\s*(?:Call|Put)|American\s*(?:Call|Put)|Call|Put)',
            # Custom date pattern for "DD Month YYYY" format
            'expiry_date': r'(?i)expiry\s*date\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'business_calendar': r'(?i)business\s*calendar\s*([A-Za-z\s&]+)',
            # Custom date pattern for "DD Month YYYY" format
            'delivery_date': r'(?i)delivery\s*date\s*(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'premium_rate': r'(?i)premium\s*rate\s*([^\n\r]*?)(?:of\s*Notional|\(|$)',
            'transaction_currency': r'(?i)transaction\s*currency\s*([A-Z]{3})',
            'counter_currency': r'(?i)counter\s*currency\s*([A-Z]{3})'
        }
        
        # Custom specific patterns tailored to your document format
        self.custom_formats = {
            'trade_date': r'(?:^|\n)Trade\s+Date\s+(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'ref_spot_price': r'(?:^|\n)Reference\s+Spot\s+Price\s+([\d.,]+)',
            'notional_amount': r'(?:^|\n)Notional\s+Amount\s+([^\n\r]*)',
            'strike_price': r'(?:^|\n)Strike\s+Price\s+([\d.,]+)',
            'option_type': r'(?:^|\n)Option\s+Type\s+([^\n\r]*)',
            'expiry_date': r'(?:^|\n)Expiry\s+Date\s+(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'delivery_date': r'(?:^|\n)Delivery\s+Date\s+(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            'business_calendar': r'(?:^|\n)Business\s+Calendar\s+([^\n\r]*)',
            'premium_rate': r'(?:^|\n)Premium\s+Rate\s+([^\n\r]*)',
            'transaction_currency': r'(?:^|\n)Transaction\s+Currency\s+([A-Z]{3})',
            'counter_currency': r'(?:^|\n)Counter\s+Currency\s+([A-Z]{3})'
        }
        
        # Patterns for dates in the format DD MMMM YYYY
        self.date_patterns = [
            r'\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        ]
    
    def discover_undefined_terms(self, text, existing_parameters):
        """
        Use DeepSeek to discover new or undefined terms in the termsheet
        
        Args:
            text (str): Full text of the termsheet
            existing_parameters (dict): Currently known extracted parameters
        
        Returns:
            list: Newly discovered terms
        """
        prompt = f"""
        You are an advanced financial document analysis AI specializing in extracting 
        novel terms from derivative term sheets.

        Existing Known Parameters: {json.dumps(list(existing_parameters.keys()))}

        Task:
        1. Scan the following text and identify:
           - New financial terms not in the existing parameters
           - Potential synonyms or alternative names for known parameters
           - Contextual meanings of these terms
           - Potential data type or format of the term's value

        Text to Analyze:
        {text[:5000]}  # Limit text to first 5000 chars

        Output Format:
        {{
            "new_terms": [
                {{
                    "term": "Term Name",
                    "context": "Sentence or context where term was found",
                    "potential_meaning": "Interpretation of the term",
                    "suggested_parameter_name": "snake_case recommended parameter name",
                    "data_type": "string/date/numeric/etc",
                    "example_value": "Sample extracted value"
                }}
            ]
        }}
        """

        try:
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                temperature=0.5,
                max_tokens=1500,
                messages=[
                    {"role": "system", "content": "You are an expert in financial document analysis."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw_response = response.choices[0].message.content
            cleaned_response = re.sub(r"^```(?:json)?\s*|\s*```\$", "", raw_response).strip()
            
            new_terms_data = json.loads(cleaned_response)
            return new_terms_data.get("new_terms", [])

        except Exception as e:
            logger.error(f"Error extracting undefined terms: {e}")
            return []

    def extract_from_pdf(self, pdf_path):
        """
        Modified extraction method to include term discovery
        """
        # Existing extraction logic
        extracted = super().extract_from_pdf(pdf_path)
        
        # Extract full text for undefined term analysis
        full_text = self._extract_text_from_pdf(pdf_path)
        
        # Discover new terms
        new_terms = self.discover_undefined_terms(full_text, extracted)
        
        # Optional: Log or process new terms
        if new_terms:
            logger.info("New terms discovered:")
            for term in new_terms:
                logger.info(f"Term: {term['term']}")
                logger.info(f"Potential Meaning: {term['potential_meaning']}")
        
        # You could extend the extracted dictionary with new terms
        extracted['discovered_terms'] = new_terms
        
        return extracted

    def extract_from_pdf(self, pdf_path):
        """Main method to extract parameters from a PDF"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Try direct text extraction first
        text = self._extract_text_from_pdf(pdf_path)
        
        # If minimal text was extracted, it might be a scanned PDF
        if len(text.strip()) < 100:
            logger.info("Little text found, attempting OCR processing")
            text = self._perform_ocr(pdf_path)
            
        logger.info(f"Extracted text sample: {text[:200]}...")
        
        # Extract parameters from text using regex
        results = self._extract_parameters(text)
        
        # For date fields that were not found, try using NLP-based extraction
        for field in ['trade_date', 'expiry_date', 'delivery_date']:
            if field not in results or not results[field]:
                nlp_date = self._extract_date_with_nlp(field, text)
                if nlp_date:
                    results[field] = nlp_date
                    logger.info(f"NLP provided {field}: {nlp_date}")
        
        # For non-date parameters missing from regex extraction, use NLP fallback
        fallback_fields = ["ref_spot_price", "notional_amount", "strike_price", "option_type", "transaction_currency", "counter_currency"]
        for field in fallback_fields:
            # Map field keys to our standardized key names
            key = field if field not in ['transaction_currency', 'counter_currency'] else field
            if key not in results or not results[key]:
                nlp_value = self._extract_param_with_nlp(key, text)
                if nlp_value:
                    results[key] = nlp_value
                    logger.info(f"NLP fallback extracted {key}: {nlp_value}")
        
        # Post-process results for consistency and standardization
        results = self._postprocess_results(results)
        
        # Print the results in a nicely formatted table
        self.print_extraction_results(results, pdf_path)
        return results
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text using both PyPDF2 and pdfplumber for best results"""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page_text = reader.pages[page_num].extract_text()
                    if page_text:
                        text += page_text + "\n"
            logger.info(f"Extracted {len(text)} characters with PyPDF2")
        except Exception as e:
            logger.error(f"Error extracting text with PyPDF2: {e}")
        
        if len(text.strip()) < 100:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text() or ""
                        text += page_text + "\n"
                        tables = page.extract_tables()
                        for table in tables:
                            for row in table:
                                if len(row) >= 2 and row[0] and row[1]:
                                    text += f"{row[0]}: {row[1]}\n"
                                else:
                                    row_text = " ".join([str(cell) for cell in row if cell is not None])
                                    text += row_text + "\n"
                logger.info(f"Extracted {len(text)} characters with pdfplumber")
            except Exception as e:
                logger.error(f"Error extracting text with pdfplumber: {e}")
        return text
    
    def _perform_ocr(self, pdf_path):
        """Perform OCR on PDF pages"""
        logger.info("Starting OCR processing")
        text = ""
        try:
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1} with OCR")
                page_text = pytesseract.image_to_string(image)
                text += page_text + "\n"
            logger.info(f"OCR completed, extracted {len(text)} characters")
        except Exception as e:
            logger.error(f"Error during OCR processing: {e}")
        return text
    
    def _extract_parameters(self, text):
        """Extract financial parameters from text using regex patterns"""
        results = {}
        text = self._preprocess_text(text)
        dates = self._extract_dates_dd_month_yyyy(text)
        logger.info(f"Extracted dates: {dates}")
        
        # Use custom formats first
        for param_name, pattern in self.custom_formats.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if matches:
                results[param_name] = matches[0].strip()
                logger.info(f"Found {param_name}: {results[param_name]} using custom format")
        
        # Then use general patterns for missing fields
        for param_name, pattern in self.parameters.items():
            if param_name not in results:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    results[param_name] = matches[0].strip()
                    logger.info(f"Found {param_name}: {results[param_name]} using general pattern")
        
        # Map date context using our date_map
        for field, keywords in self.date_map.items():
            if field not in results and dates:
                for date_val, date_ctx in dates:
                    if any(kw in date_ctx.lower() for kw in keywords):
                        results[field] = date_val
                        logger.info(f"Assigned date {date_val} to {field} based on context: {date_ctx}")
                        break
        
        return results
    
    def _extract_dates_dd_month_yyyy(self, text):
        """Extract dates in the format DD Month YYYY and return with context"""
        dates_with_context = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                day = match.group(1).zfill(2)
                month = match.group(2)
                year = match.group(3)
                month_num = self._month_to_number(month)
                formatted_date = f"{year}-{month_num}-{day}"
                start_pos = max(0, match.start() - 30)
                end_pos = min(len(text), match.end() + 30)
                context = text[start_pos:end_pos]
                dates_with_context.append((formatted_date, context))
                logger.info(f"Extracted date: {formatted_date} with context: {context}")
        return dates_with_context

    def _extract_date_with_nlp(self, field, text):
        """
        Use spaCy NLP to extract a date for the given field if missing.
        Scans sentences containing context keywords and looks for DATE entities.
        """
        doc = self.nlp(text)
        keywords = self.date_map.get(field, [])
        for sent in doc.sents:
            if any(kw in sent.text.lower() for kw in keywords):
                for ent in sent.ents:
                    if ent.label_ == "DATE":
                        try:
                            parsed_date = dateparser.parse(ent.text, fuzzy=True)
                            if parsed_date:
                                formatted_date = parsed_date.strftime("%Y-%m-%d")
                                logger.info(f"NLP extracted {field}: {formatted_date} from sentence: {sent.text}")
                                return formatted_date
                        except Exception as ex:
                            logger.error(f"Error parsing date '{ent.text}': {ex}")
        return None
    
    def _extract_param_with_nlp(self, field, text):
        """
        Fallback NLP method for extracting non-date parameters.
        Scans sentences for keywords specific to the field and uses simple regex extraction.
        """
        fallback_map = {
            'ref_spot_price': ["reference spot price", "ref spot price", "spot price"],
            'notional_amount': ["notional amount"],
            'strike_price': ["strike price"],
            'option_type': ["option type"],
            'transaction_currency': ["transaction currency"],
            'counter_currency': ["counter currency"]
        }
        keywords = fallback_map.get(field, [])
        doc = self.nlp(text)
        for sent in doc.sents:
            if any(kw in sent.text.lower() for kw in keywords):
                if field in ['ref_spot_price', 'notional_amount', 'strike_price']:
                    match = re.search(r'([\d,]+(?:\.\d+)?)', sent.text)
                    if match:
                        return match.group(1)
                elif field == 'option_type':
                    match = re.search(r'option\s*type\s*([\w\s]+)', sent.text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip()
                elif field in ['transaction_currency', 'counter_currency']:
                    match = re.search(r'([A-Z]{3})', sent.text)
                    if match:
                        return match.group(1)
        return None

    def _month_to_number(self, month_name):
        """Convert month name to number (with leading zero)"""
        months = {
            'january': '01', 'jan': '01',
            'february': '02', 'feb': '02',
            'march': '03', 'mar': '03',
            'april': '04', 'apr': '04',
            'may': '05',
            'june': '06', 'jun': '06',
            'july': '07', 'jul': '07',
            'august': '08', 'aug': '08',
            'september': '09', 'sep': '09',
            'october': '10', 'oct': '10',
            'november': '11', 'nov': '11',
            'december': '12', 'dec': '12'
        }
        return months.get(month_name.lower(), '01')
    
    def _preprocess_text(self, text):
        """Clean and normalize text for processing"""
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\t', ' ')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text
    
    def _postprocess_results(self, results):
        """Clean and format the extracted parameters"""
        # Standardize currency keys
        if 'transaction_currency' in results and 'transaction_ccy' not in results:
            results['transaction_ccy'] = results.pop('transaction_currency')
        if 'counter_currency' in results and 'counter_ccy' not in results:
            results['counter_ccy'] = results.pop('counter_currency')
        
        # Clean monetary values
        money_fields = ['ref_spot_price', 'notional_amount', 'strike_price', 'premium_rate']
        for field in money_fields:
            if field in results:
                value = results[field]
                currency_code = ""
                currency_match = re.search(r'\b([A-Z]{3})\b', value)
                if currency_match:
                    currency_code = currency_match.group(1) + " "
                numeric_part = re.search(r'([\d,]+(?:\.\d+)?)', value)
                if numeric_part:
                    results[field] = currency_code + numeric_part.group(1)
                else:
                    results[field] = value
                if field == 'premium_rate' and '%' in value:
                    percentage = re.search(r'([\d\.]+)\s*%', value)
                    if percentage:
                        results[field] = percentage.group(1) + "%"
        
        # Standardize option type
        if 'option_type' in results:
            option_type = results['option_type'].lower()
            if 'call' in option_type:
                if 'european' in option_type:
                    results['option_type'] = 'European Call'
                elif 'american' in option_type:
                    results['option_type'] = 'American Call'
                else:
                    results['option_type'] = 'Call'
            elif 'put' in option_type:
                if 'european' in option_type:
                    results['option_type'] = 'European Put'
                elif 'american' in option_type:
                    results['option_type'] = 'American Put'
                else:
                    results['option_type'] = 'Put'
        
        for field in ['transaction_ccy', 'counter_ccy']:
            if field in results:
                results[field] = results[field].upper()
        
        return results
    
    def print_extraction_results(self, results, filename=None):
        """Print extraction results in a nicely formatted way"""
        print("\n" + "=" * 80)
        print(f"EXTRACTION RESULTS" + (f" for {os.path.basename(filename)}" if filename else ""))
        print("=" * 80)

        print("\nALL EXTRACTED DATES (for verification):")
        for date_param in ['trade_date', 'expiry_date', 'delivery_date']:
            if date_param in results:
                raw_date = results[date_param]
                if re.match(r'\d{4}-\d{2}-\d{2}', raw_date):
                    try:
                        # Change this line
                        date_obj = datetime.strptime(raw_date, '%Y-%m-%d')
                        formatted_date = date_obj.strftime('%d %B %Y')
                        print(f"  {date_param.replace('_', ' ').title()}: {raw_date} => {formatted_date}")
                    except ValueError:
                        print(f"  {date_param.replace('_', ' ').title()}: {raw_date}")
                else:
                    print(f"  {date_param.replace('_', ' ').title()}: {raw_date}")

        
        data = []
        expected_fields = ["trade_date", "expiry_date", "delivery_date", "ref_spot_price", 
                           "notional_amount", "strike_price", "option_type", "premium_rate", 
                           "transaction_ccy", "counter_ccy", "business_calendar"]
        
        for field in expected_fields:
            if field in results:
                value = results[field]
                if field.endswith('_date') and re.match(r'\d{4}-\d{2}-\d{2}', value):
                    try:
                        date_obj = datetime.strptime(value, '%Y-%m-%d')
                        value = date_obj.strftime('%d %B %Y')
                    except ValueError:
                        pass
                data.append([field, value])
            else:
                data.append([field, "NOT FOUND"])
                
        print("\nEXTRACTED PARAMETERS:")
        print(tabulate(data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
        
        print("\nJSON Format:")
        print(json.dumps({field: value for field, value in data if "NOT FOUND" not in value}, indent=2))
        print("=" * 80 + "\n")
def find_and_extract_termsheet_from_pdf(
    email_address: str,
    password: str,
    termsheet_id: str,
    mail_server: str = "imap.gmail.com"
) -> Dict[str, Any]:
    """
    Connects to the inbox, finds the latest email whose SUBJECT contains termsheet_id,
    downloads its PDF attachment(s), runs your PDF extractor on it, then returns
    both the raw data and its DeepSeek validation + email metadata.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        mail = imaplib.IMAP4_SSL(mail_server)
        mail.login(email_address, password)
        mail.select("inbox")

        status, messages = mail.search(None, f'SUBJECT "{termsheet_id}"')
        if status != 'OK' or not messages[0]:
            print(f"No emails found with termsheet ID: {termsheet_id}")
            return {}

        email_ids = messages[0].split()
        latest = email_ids[-1]
        status, msg_data = mail.fetch(latest, "(RFC822)")
        raw_email = msg_data[0][1]
        msg = email.message_from_bytes(raw_email)

        # decode subject & sender/date
        subject = decode_header(msg["Subject"])[0][0]
        if isinstance(subject, bytes):
            subject = subject.decode()
        sender = msg.get("From")
        date   = msg.get("Date")

        extracted = {}
        attachments = []
        # walk attachments
        for part in msg.walk():
            cd = str(part.get("Content-Disposition"))
            if "attachment" in cd:
                fn = part.get_filename()
                if fn and fn.lower().endswith(".pdf"):
                    # decode filename
                    fn = decode_header(fn)[0][0]
                    if isinstance(fn, bytes):
                        fn = fn.decode()
                    path = os.path.join(temp_dir, fn)
                    with open(path, "wb") as f:
                        f.write(part.get_payload(decode=True))
                    attachments.append(fn)

                    # use your extractor
                    extractor = DerivativeTradeExtractor()
                    data = extractor.extract_from_pdf(path)
                    # pick the file with most fields found
                    if not extracted or len(data) > len(extracted):
                        extracted = data

        mail.close()
        mail.logout()

        if not extracted:
            print("No PDF data extracted.")
            return {}

        # run DeepSeek validation
        validation = validate_with_deepseek(extracted)
        print("\n🧠 DEEPSEEK VALIDATION REPORT:")
        print_validation_report(validation, extracted)
        collect_feedback(validation)

        return {
            "extracted_data": extracted,
            "validation_results": validation,
            "email_metadata": {
                "subject": subject,
                "sender": sender,
                "date": date,
                "attachments": attachments
            }
        }
    except Exception as e:
        print(f"Error fetching termsheet from email: {e}")
        return {}
    finally:
        # cleanup
        try:
            os.rmdir(temp_dir)
        except:
            pass

def find_and_extract_termsheet_from_telegram(
    api_id: int,
    api_hash: str,
    chat_identifier: str,
    termsheet_keyword: str,
    session_name: str = "session_telegram",
    phone_number: Optional[str] = None,
    otp: Optional[str] = None,
    login_required: bool = False,
    download_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Connects to Telegram via Telethon. On first login, uses phone and otp.
    On subsequent access, uses saved session file.

    Returns:
      - extracted_data: extracted termsheet fields
      - telegram_metadata: info about the message and source
    """

    if not download_dir:
        download_dir = tempfile.mkdtemp()

    # Check if session file exists
    session_file = f"{session_name}.session"
    client = TelegramClient(session_name, api_id, api_hash)

    if login_required or not os.path.exists(session_file):
        if not phone_number or not otp:
            raise ValueError("Phone number and OTP are required for first-time login.")
        logger.info("Starting Telegram login with phone and OTP...")
        client.connect()
        if not client.is_user_authorized():
            client.send_code_request(phone_number)
            client.sign_in(phone=phone_number, code=otp)
        client.disconnect()

    # Now reuse the session to extract termsheet
    extracted = {}
    metadata: Dict[str, Any] = {}

    with TelegramClient(session_name, api_id, api_hash) as client:
        logger.info("Connected to Telegram via saved session.")

        for message in client.iter_messages(chat_identifier, filter=InputMessagesFilterDocument):
            caption = (message.message or "").lower()
            if message.document and message.document.mime_type == 'application/pdf' and termsheet_keyword.lower() in caption:
                file_path = message.download_media(file=download_dir)
                extractor = DerivativeTradeExtractor()
                extracted = extractor.extract_from_pdf(file_path)

                # Upload to S3
                s3_handler.upload_json(extracted, "extracted_data/telegram_result.json")

                # Collect Telegram metadata
                metadata = {
                "message_id": message.id,
                "chat": chat_identifier,
                "date": message.date.isoformat(),
                "sender_id": message.sender_id,
                "file_path": file_path
                }
                break  # exit after first match

    return {
        "extracted_data": extracted,
        "telegram_metadata": metadata
    }

def load_master_values(path: str = "master_values.xlsx") -> Dict[str, Set[str]]:
    """
    Reads an Excel file (default: master_values.xlsx) and returns a map
    param_key -> set of allowed values.  It will auto‑detect which two
    columns in the sheet represent “parameter name” and “allowed value.”
    """
    try:
        df = pd.read_excel(path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Failed to read master sheet '{path}': {e}")
        return {}

    # Normalize column names
    cols = [c.strip().lower() for c in df.columns]

    # Attempt to find the parameter column
    param_candidates = ["parameter", "parameters", "param", "field", "name"]
    param_col = None
    for cand in param_candidates:
        if cand in cols:
            param_col = df.columns[cols.index(cand)]
            break
    if not param_col:
        # fallback to first column
        param_col = df.columns[0]

    # Attempt to find the value column
    value_candidates = ["value", "allowed", "allowedvalue", "allowed_value", "allowed values"]
    value_col = None
    for cand in value_candidates:
        if cand in cols:
            value_col = df.columns[cols.index(cand)]
            break
    if not value_col and len(df.columns) > 1:
        # fallback to second column
        value_col = df.columns[1]
    if not value_col:
        value_col = param_col  # worst case, they'll both be same

    master: Dict[str, Set[str]] = {}
    for _, row in df.iterrows():
        raw_param = str(row.get(param_col, "")).strip()
        raw_val   = str(row.get(value_col, "")).strip()
        if not raw_param or not raw_val:
            continue

        # normalize e.g. "Trade Date" -> "trade_date"
        key = re.sub(r"[^\w\s]", "", raw_param)       # drop punctuation
        key = re.sub(r"\s+", "_", key).lower()        # spaces -> underscore

        master.setdefault(key, set()).add(raw_val)

    logger.info(f"Loaded master values for parameters: {list(master.keys())}")
    return master
# def infer_column_mapping_via_deepseek(
#     extractor_keys: list[str],
#     master_columns: list[str]
# ) -> dict[str,str]:
#     """
#     Use DeepSeek to map each snake_case extractor key to the best-matching
#     column header from the master sheet.
#     """
#     prompt = f"""
# You are a data-mapping assistant.  I have two lists:
# 1) My extractor parameters (snake_case):
# {json.dumps(extractor_keys, indent=2)}

# 2) The raw column headers from my master Excel sheet:
# {json.dumps(master_columns, indent=2)}

# For each extractor parameter, pick the one master‐sheet header that best
# matches (allowing for slight naming differences), or return an empty
# string if none matches.  Reply *only* with a JSON object of the form:

# {{"trade_date": "Trade Date", "ref_spot_price": "Reference Spot Price", …}}
# """
#     resp = client.chat.completions.create(
#         model="deepseek-reasoner",
#         temperature=0.0,
#         max_tokens=1500,
#         messages=[
#             {"role":"system", "content":"Output pure JSON only."},
#             {"role":"user",   "content":prompt}
#         ]
#     )
#     raw = resp.choices[0].message.content
#     cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
#     try:
#         return json.loads(cleaned)
#     except json.JSONDecodeError:
#         logger.error("Failed to parse mapping JSON from DeepSeek:\n" + raw)
#         return {}
#     return mapping      
def validate_with_deepseek(
    parameters: dict[str, Any],
    master_path: str = "master_values.xlsx"
) -> dict[str, Any]:
    """
    Uses DeepSeek to:
    1. Normalize extracted and master values.
    2. Determine match status.
    3. Provide expert comment.
    """

    master = load_master_values(master_path)

    # Build prompt input
    deepseek_input = []
    for key, extracted_value in parameters.items():
        master_values = list(master.get(key, set()))
        deepseek_input.append({
            "parameter": key,
            "extracted_value": extracted_value,
            "master_values": master_values
        })

    prompt = (
        "You are a senior derivatives analyst at Barclays, reviewing extracted financial parameters "
    "from a transaction term sheet related to Foreign Exchange (FX) Options or structured products. "
    "Your task is to evaluate each extracted parameter by comparing it to known reference/master values. "
    "For each parameter:\n"
    "\n"
    "1. Normalize both the extracted value and master values (e.g., for format, currency symbols, date formats).\n"
    "2. Determine if the normalized extracted value exactly matches any of the normalized master values.\n"
    "3. Provide a highly descriptive, domain-specific comment for each parameter. Your comment should:\n"
    "   - Explain the relevance of the extracted value in a real FX trade context.\n"
    "   - Mention whether the value is standard, acceptable, or unusual.\n"
    "   - If the value does not match the master list, briefly explain what is different or concerning.\n"
    "   - Include a benchmark, threshold, or rationale when possible (e.g., settlement T+2, 3-month expiry tenor, etc).\n"
    "   - Use financial terminology common on the Barclays trading floor.\n"
    "\n"
    "Avoid vague phrases like 'okay', 'looks good', or 'seems valid'. Your tone should be like an internal expert "
    "flagging and explaining risks, inconsistencies, or confirmations.\n"
    "\n"
    "Return only a JSON object in the following format:\n"
    "{\n"
    "  \"validation_results\": {\n"
    "    \"parameter_name\": {\n"
    "      \"match\": true or false,\n"
    "      \"comment\": \"Detailed financial validation comment... (✔ or ✘ based on master match)\"\n"
    "    },\n"
    "    ...\n"
    "  }\n"
    "}"
    )

    try:
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            temperature=0.4,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": "Output JSON only. Do not wrap in markdown or explanations."},
                {"role": "user", "content": prompt + "\n\n" + json.dumps(deepseek_input, indent=2)}
            ]
        )

        raw = response.choices[0].message.content
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        results = json.loads(cleaned).get("validation_results", {})

        # Combine DeepSeek output with master data for display
        validation_results = {}
        for param in parameters:
            extracted_val = parameters[param]
            master_vals = sorted(master.get(param, set()))
            deep = results.get(param, {})

            validation_results[param] = {
                "value": extracted_val,
                "master_list": master_vals,
                "match": deep.get("match", False),
                "comment": deep.get("comment", "No response")
            }

        return {"validation_results": validation_results}

    except Exception as e:
        logger.error(f"DeepSeek validation failed: {e}")
        # Fallback with only extracted + master, no comments
        return {
            "validation_results": {
                k: {
                    "value": parameters[k],
                    "master_list": sorted(master.get(k, set())),
                    "match": False,
                    "comment": "DeepSeek unavailable"
                } for k in parameters
            }
        }



def collect_feedback(report):
    """
    Walk the user through each parameter and record whether they agree
    with the validation status, plus any comment if they disagree.
    Saves feedback to a timestamped JSON file.
    """
    if not isinstance(report, dict):
        print("\n⚠️ Cannot collect feedback: validation output is not structured JSON.\n")
        return None

    print("\n" + "-"*30 + " Feedback " + "-"*30)
    feedback = {}
    for param, info in report.items():
        # handle both dict and fallback-string cases
        if isinstance(info, dict):
            val    = info.get("value", "")
            status = info.get("status", "")
        else:
            val    = ""
            status = str(info)

        print(f"\nParameter: {param}")
        print(f"  Extracted value : {val}")
        print(f"  Validation status: {status}")

        # prompt for agreement
        while True:
            ans = input("   Do you agree with this status? (y/n): ").strip().lower()
            if ans in ("y", "n"):
                break
            print("   Please enter 'y' or 'n'.")

        if ans == "y":
            feedback[param] = {"agree": True, "comment": ""}
        else:
            comment = input("   Enter your correction or comment: ").strip()
            feedback[param] = {"agree": False, "comment": comment}

    # save feedback
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"validation_feedback_{ts}.json"
    with open(fname, "w") as f:
        json.dump(feedback, f, indent=2)
    print(f"\n✅ Feedback saved to {fname}\n")
    return feedback

def process_multiple_pdfs(directory):
    """Process all PDFs in a directory and return aggregated results"""
    extractor = DerivativeTradeExtractor()
    all_results = []
    
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            try:
                results = extractor.extract_from_pdf(pdf_path)
                results['filename'] = filename
                all_results.append(results)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
    
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(directory, "extracted_derivative_parameters.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        print("\n" + "=" * 80)
        print(f"SUMMARY OF PROCESSED FILES ({len(all_results)} files)")
        print("=" * 80)
        
        summary_data = []
        for result in all_results:
            row = [
                result.get('filename', 'Unknown'),
                result.get('trade_date', 'N/A'),
                result.get('expiry_date', 'N/A'),
                result.get('option_type', 'N/A'),
                f"{result.get('transaction_ccy', 'N/A')}/{result.get('counter_ccy', 'N/A')}"
            ]
            summary_data.append(row)
            
        print(tabulate(summary_data, 
                       headers=["Filename", "Trade Date", "Expiry Date", "Option Type", "Currency Pair"], 
                       tablefmt="fancy_grid"))
        print(f"\nDetailed results saved to: {csv_path}")
        print("=" * 80 + "\n")
        
        return df
    else:
        logger.warning("No results extracted from PDFs")
        return None

from rich.console import Console
from rich.table import Table
from rich import box

def print_validation_report(
    report: dict,
    extracted: dict,
    master_path: str = "master_values.xlsx"
):
    console = Console()

    # 1) Extracted Values
    table1 = Table(title="1) Extracted Values", box=box.ROUNDED)
    table1.add_column("Parameter", style="cyan", no_wrap=True)
    table1.add_column("Extracted", overflow="fold")
    for param, val in extracted.items():
        table1.add_row(param, str(val))
    console.print(table1)

    # Load master‐sheet values
    master_map = load_master_values(master_path)

    # 2) Master-sheet Allowed Values
    table2 = Table(title="2) Master-sheet Allowed Values", box=box.ROUNDED)
    table2.add_column("Parameter", style="cyan", no_wrap=True)
    table2.add_column("Allowed (master_list)", overflow="fold", style="yellow")
    for param in extracted.keys():
        allowed = master_map.get(param, set())
        allowed_str = ", ".join(sorted(allowed)) if allowed else "—"
        table2.add_row(param, allowed_str)
    console.print(table2)

    # 3) Match / Mismatch
    vr = report.get("validation_results", {})
    table3 = Table(title="3) Match / Mismatch", box=box.ROUNDED)
    table3.add_column("Parameter", style="cyan", no_wrap=True)
    table3.add_column("Match?", justify="center")
    for param in extracted.keys():
        info = vr.get(param, {})
        # check either map_match or match boolean
        matched = bool(info.get("map_match") or info.get("match"))
        icon = "[green]✔[/green]" if matched else "[red]✘[/red]"
        table3.add_row(param, icon)
    console.print(table3)

    # 4) DeepSeek Comments
    table4 = Table(title="4) DeepSeek Comments", box=box.ROUNDED)
    table4.add_column("Parameter", style="cyan", no_wrap=True)
    table4.add_column("DS Comment", overflow="fold")
    table4.add_column("Map Comment", overflow="fold")
    for param in extracted.keys():
        info = vr.get(param, {})
        ds_comment  = info.get("comment") or info.get("message") or "—"
        map_comment = info.get("map_comment") or "—"
        table4.add_row(param, ds_comment, map_comment)
    console.print(table4)

    # 5) Errors
    if errors := report.get("errors"):
        err_tbl = Table(title="Errors", box=box.ROUNDED, header_style="bold red")
        err_tbl.add_column("Field")
        err_tbl.add_column("Message", overflow="fold")
        for e in errors:
            err_tbl.add_row(e.get("field","—"), e.get("message",""))
        console.print(err_tbl)

    # 6) Warnings
    if warnings := report.get("warnings") or report.get("validation_notes",{}).get("warnings"):
        warn_tbl = Table(title="Warnings", box=box.ROUNDED, header_style="bold yellow")
        warn_tbl.add_column("Field")
        warn_tbl.add_column("Message", overflow="fold")
        for w in warnings:
            if isinstance(w, dict):
                fld = w.get("field","—"); msg = w.get("message","")
            else:
                fld, msg = "—", str(w)
            warn_tbl.add_row(fld, msg)
        console.print(warn_tbl)




if __name__ == "__main__":
    import getpass
    from pathlib import Path
    print("\n" + "*" * 80)
    print("PDF DERIVATIVE PARAMETER EXTRACTOR WITH DeepSeek VALIDATION")
    print("*" * 80)

    # Initialize S3
    try:
        s3_handler = S3Handler(bucket_name='your-derivative-termsheets')
    except Exception as s3_error:
        print(f"❌ S3 Initialization Error: {s3_error}")
        s3_handler = None

    # Menu
    print("\nSelect input mode:")
    print("  1) Single PDF file")
    print("  2) Folder of PDFs")
    print("  3) Email attachment lookup")
    print("  4) Telegram chat lookup")
    choice = input("Enter choice (1/2/3/4): ").strip()

    def print_discovered_terms(extracted):
        """Helper function to print discovered terms"""
        if 'discovered_terms' in extracted and extracted['discovered_terms']:
            print("\n🔍 New Terms Discovered:")
            for term in extracted['discovered_terms']:
                print(f"Term: {term.get('term', 'N/A')}")
                print(f"Context: {term.get('context', 'N/A')}")
                print(f"Potential Meaning: {term.get('potential_meaning', 'N/A')}\n")

    try:
        if choice == "1":  # Single PDF file
            pdf_path = input("Enter the full path to the PDF file: ").strip()
            if not os.path.isfile(pdf_path):
                print(f"File not found: {pdf_path}")
                exit(1)

            # Extract PDF parameters with term discovery built-in
            extractor = DerivativeTradeExtractor()
            extracted = extractor.extract_from_pdf(pdf_path)

            # Print discovered terms
            print_discovered_terms(extracted)

            # Upload extracted data to S3
            if s3_handler:
                s3_handler.upload_data(extracted, "extracted_data/single_file_result.json")

            # Perform DeepSeek validation
            validation = validate_with_claude_bedrock(extracted)

            # Print validation report
            print_validation_report(validation, extracted)

            # Collect feedback
            collect_feedback(validation)

        elif choice == "2":  # Folder of PDFs
            pdf_dir = input("Enter the directory containing PDFs: ").strip()
            if not os.path.isdir(pdf_dir):
                print(f"Directory not found: {pdf_dir}")
                exit(1)

            # Process multiple PDFs
            results = []
            for filename in os.listdir(pdf_dir):
                if filename.endswith('.pdf'):
                    full_path = os.path.join(pdf_dir, filename)
                    try:
                        # Extract parameters
                        extractor = DerivativeTradeExtractor()
                        extracted = extractor.extract_from_pdf(full_path)

                        # Print discovered terms
                        print(f"\n📄 Processing {filename}")
                        print_discovered_terms(extracted)

                        # Validate
                        validation = validate_with_deepseek(extracted)

                        # Prepare result entry
                        result_entry = {
                            "filename": filename,
                            "extracted_data": extracted,
                            "validation": validation,
                            "discovered_terms": extracted.get('discovered_terms', [])
                        }
                        results.append(result_entry)

                        # Upload to S3
                        if s3_handler:
                            s3_key = f"extracted_data/{Path(full_path).stem}_result.json"
                            s3_handler.upload_data(result_entry, s3_key)

                        # Print validation for each file
                        print(f"\n📄 Validation for {filename}:")
                        print_validation_report(validation, extracted)

                    except Exception as file_error:
                        print(f"❌ Error processing {filename}: {file_error}")

            # Save all results to a consolidated file
            if results:
                with open("batch_extraction_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                print("\n💾 All results saved to batch_extraction_results.json")

        elif choice == "3":  # Email attachment lookup
            email = input("Enter email: ")
            password = getpass.getpass("Enter password: ")
            termsheet_id = input("Enter termsheet ID: ")

            # Extract from email
            result = find_and_extract_termsheet_from_pdf(email, password, termsheet_id)

            # Print discovered terms
            if result and 'extracted_data' in result:
                print_discovered_terms(result['extracted_data'])

                # Upload extracted data to S3
                if s3_handler:
                    s3_handler.upload_data(result["extracted_data"], "extracted_data/email_result.json")

                # Validate extracted data
                validation = validate_with_deepseek(result["extracted_data"])
                print_validation_report(validation, result["extracted_data"])
                collect_feedback(validation)

        elif choice == "4":  # Telegram chat lookup
            # Telegram extraction
            api_id = input("Enter Telegram API ID: ")
            api_hash = input("Enter Telegram API Hash: ")
            chat_identifier = input("Enter Chat Identifier: ")
            termsheet_keyword = input("Enter Termsheet Keyword: ")

            # Extract from Telegram
            result = find_and_extract_termsheet_from_telegram(
                int(api_id),
                api_hash,
                chat_identifier,
                termsheet_keyword
            )

            # Print discovered terms
            if result and result.get("extracted_data"):
                print_discovered_terms(result["extracted_data"])

                # Upload extracted data to S3
                if s3_handler:
                    s3_handler.upload_data(result["extracted_data"], "extracted_data/telegram_result.json")

                # Validate extracted data
                validation = validate_with_claude_bedrock(result["extracted_data"])
                print_validation_report(validation, result["extracted_data"])
                collect_feedback(validation)

        else:
            print("Invalid choice. Exiting.")
            exit(1)

    except Exception as main_error:
        print(f"❌ An unexpected error occurred: {main_error}")
        import traceback
        traceback.print_exc()