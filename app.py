from flask import Flask, request, jsonify, send_from_directory
import os
import json
import tempfile
import logging
import traceback
from datetime import datetime

# Import required modules
from test5 import (
    DerivativeTradeExtractor, 
    validate_with_deepseek, 
    find_and_extract_termsheet_from_pdf,
    find_and_extract_termsheet_from_telegram
)
from flask_cors import CORS
from s3_handler import S3Handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__, 
            static_folder="../frontend",
            template_folder="../frontend")

# Enable CORS with more comprehensive settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize S3 Handler using environment variable
try:
    # Get bucket name from environment, with a fallback
    bucket_name = os.environ.get('S3_BUCKET_NAME', 'derivative-termsheets-default')
    s3_handler = S3Handler(bucket_name=bucket_name)
    logger.info(f"S3 Handler initialized with bucket: {s3_handler.bucket_name}")
except Exception as e:
    logger.error(f"Failed to initialize S3 Handler: {e}")
    s3_handler = None

@app.route('/')
def index():
    """Serve the main index.html file"""
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {e}")
        return "Server error", 500

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory"""
    try:
        return send_from_directory(app.static_folder, path)
    except Exception as e:
        logger.error(f"Error serving static file {path}: {e}")
        return "File not found", 404

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload with comprehensive error handling"""
    try:
        # Validate file presence
        if 'file' not in request.files:
            logger.warning("No file part in the request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.warning("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Unsupported file type: {file_ext}")
            return jsonify({'error': 'Invalid file type'}), 400

        # Create temporary directory for file processing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            
            # S3 upload handling
            s3_upload_successful = False
            s3_key = f"termsheet_pdfs/{file.filename}"
            if s3_handler:
                try:
                    s3_upload_successful = s3_handler.upload_file(file_path, s3_key)
                except Exception as s3_error:
                    logger.error(f"S3 upload failed: {s3_error}")
            
            # Extract parameters
            extractor = DerivativeTradeExtractor()
            extracted_data = extractor.extract_from_pdf(file_path)
            
            # Validate extracted data
            validation_results = validate_with_deepseek(extracted_data)
            
            # Additional S3 upload of extracted data
            if s3_handler and s3_upload_successful:
                try:
                    extracted_data_key = f"extracted_data/{file.filename}.json"
                    s3_handler.upload_data({
                        "extracted_data": extracted_data,
                        "validation_results": validation_results
                    }, extracted_data_key)
                except Exception as s3_upload_error:
                    logger.error(f"Failed to upload extracted data: {s3_upload_error}")
            
            # Return results
            return jsonify({
                'success': True,
                'extracted_data': extracted_data,
                'validation_results': validation_results
            })

    except Exception as e:
        logger.error(f"Upload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/email-access', methods=['POST'])
def email_access():
    """Handle email-based termsheet extraction"""
    try:
        data = request.json
        email = data.get('email')
        password = data.get('password')
        termsheet_id = data.get('termsheetId')
        
        # Validate input
        if not all([email, password, termsheet_id]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Extract termsheet
        result = find_and_extract_termsheet_from_pdf(email, password, termsheet_id)
        
        if not result:
            return jsonify({'error': 'No termsheet found or extraction failed'}), 404
        
        # Optional: Upload to S3 if handler is available
        if s3_handler:
            try:
                s3_key = f"email_extractions/{termsheet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                s3_handler.upload_data(result, s3_key)
            except Exception as s3_error:
                logger.error(f"Failed to upload email extraction: {s3_error}")
        
        return jsonify({
            'success': True,
            'data': result
        })
    
    except Exception as e:
        logger.error(f"Email access error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/chat-access', methods=['POST'])
def chat_access():
    """Handle Telegram-based termsheet extraction"""
    try:
        data = request.json
        phone = data.get('phone')
        otp = data.get('otp')
        termsheet_id = data.get('termsheetId')

        # Validate input
        if not termsheet_id:
            return jsonify({'error': 'Termsheet ID is required'}), 400

        # Telegram credentials (should be from environment in production)
        api_id = os.environ.get('TELEGRAM_API_ID', 23364361)
        api_hash = os.environ.get('TELEGRAM_API_HASH', '5c452d22c18f86ae2013cfa66f140d97')
        chat_identifier = os.environ.get('TELEGRAM_CHAT_ID', 6163289640)
        session_name = "barclays_session"

        # Determine login method
        login_required = bool(phone and otp)

        try:
            # Check if session already exists (means user is logged in)
            if os.path.exists(f"{session_name}.session") or (not login_required):
                # Session exists or only termsheetId is sent
                result = find_and_extract_termsheet_from_telegram(
                    api_id=api_id,
                    api_hash=api_hash,
                    chat_identifier=chat_identifier,
                    termsheet_keyword=termsheet_id,
                    session_name=session_name,
                    login_required=False
                )
            else:
                # First-time login (requires phone & OTP)
                if not phone or not otp:
                    return jsonify({'error': 'Phone and OTP required for first-time login'}), 400

                result = find_and_extract_termsheet_from_telegram(
                    api_id=api_id,
                    api_hash=api_hash,
                    chat_identifier=chat_identifier,
                    termsheet_keyword=termsheet_id,
                    session_name=session_name,
                    phone_number=phone,
                    otp=otp,
                    login_required=True
                )

            # Validate result
            if not result or not result.get("extracted_data"):
                return jsonify({'error': 'No termsheet found or extraction failed'}), 404

            # Optional: Upload to S3 if handler is available
            if s3_handler:
                try:
                    s3_key = f"telegram_extractions/{termsheet_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    s3_handler.upload_data(result, s3_key)
                except Exception as s3_error:
                    logger.error(f"Failed to upload Telegram extraction: {s3_error}")

            # Validate extracted data
            try:
                validation_results = validate_with_deepseek(result.get('extracted_data', {}))
                result['validation_results'] = validation_results
            except Exception as validation_error:
                logger.error(f"Validation error: {validation_error}")
                result['validation_results'] = {"error": str(validation_error)}

            return jsonify({
                'success': True,
                'data': result
            })

        except Exception as telegram_error:
            logger.error(f"Telegram extraction error: {telegram_error}")
            logger.error(traceback.format_exc())
            return jsonify({
                'error': 'Telegram extraction failed',
                'details': str(telegram_error)
            }), 500

    except Exception as e:
        logger.error(f"Chat access error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/save-feedback', methods=['POST'])
def save_feedback():
    """Save user feedback with optional S3 upload"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No feedback data provided'}), 400
        
        # Generate a timestamp for the filename
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_feedback_{ts}.json"
        
        # Ensure feedback directory exists
        feedback_dir = os.path.join(os.path.dirname(__file__), 'feedback')
        os.makedirs(feedback_dir, exist_ok=True)
        
        # Local file path
        filepath = os.path.join(feedback_dir, filename)
        
        # Enrich feedback with metadata
        enriched_feedback = {
            "timestamp": ts,
            "received_at": datetime.now().isoformat(),
            "source_ip": request.remote_addr,
            "feedback_data": data
        }
        
        # Save locally
        with open(filepath, "w") as f:
            json.dump(enriched_feedback, f, indent=2)
        
        # Optional: Upload to S3
        if s3_handler:
            try:
                s3_key = f"user_feedback/{ts}_feedback.json"
                s3_handler.upload_data(enriched_feedback, s3_key)
                logger.info(f"Feedback uploaded to S3: {s3_key}")
            except Exception as s3_error:
                logger.error(f"Failed to upload feedback to S3: {s3_error}")
        
        return jsonify({
            'success': True,
            'message': f'Feedback saved to {filename}',
            'local_path': filepath
        })
    
    except Exception as e:
        logger.error(f"Feedback save error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    # Add port configuration from environment
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',  # Listen on all available interfaces
        port=port,
        debug=True  # Set to False in production
    )