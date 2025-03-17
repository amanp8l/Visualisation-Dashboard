from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dewasCsvReader import dewasCsvReader, generate_file_summary, generate_chart_data
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
public_path = "https://chartaiapi.amanpatel.in/images/"
upload_folder = os.path.join(os.path.dirname(__file__), 'upload')
public_folder = os.path.join(os.path.dirname(__file__), 'public')
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

# Create upload and public folders if they don't exist
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(public_folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(upload_folder, filename)
        file.save(filepath)
        return jsonify({'message': 'File uploaded successfully', 'filename': filename})
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate', methods=['POST'])
def generate():
    print("Generate API called")
    data = request.json
    finalResponse = {}
    prompt = data.get('prompt')
    filename = data.get('filename')  # Get filename from request
    
    if not filename:
        return jsonify({'error': 'Filename not provided'}), 400
        
    filepath = os.path.join(upload_folder, secure_filename(filename))
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
        
    res = dewasCsvReader(filePath=filepath, user_query=prompt)
    response = res["response"]
    plot_path = res["plot"]
    finalResponse["response"] = response
    
    # Handle plot_path which could be a string or a list of strings
    if plot_path is not None:
        if isinstance(plot_path, list):
            # If we have multiple plots, create a list of full URLs
            finalResponse["plot"] = [public_path + path for path in plot_path]
        else:
            # If we have a single plot, create a single URL
            finalResponse["plot"] = public_path + plot_path
    
    # If analyses are provided (for complex queries), include them with proper plot URLs
    if "analyses" in res:
        analyses = res["analyses"]
        for analysis in analyses:
            if analysis["plot"]:
                analysis["plot"] = public_path + analysis["plot"]
        finalResponse["analyses"] = analyses
            
    return jsonify(finalResponse)

@app.route('/summary/<filename>', methods=['GET'])
def get_file_summary(filename):
    """Get summary information about an uploaded file for the UI dashboard"""
    print(f"Summary API called for file: {filename}")
    filepath = os.path.join(upload_folder, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Generate summary data
        summary_data = generate_file_summary(filepath)
        return jsonify(summary_data)
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate summary: {str(e)}'}), 500

@app.route('/chart-data/<filename>', methods=['GET'])
def get_chart_data(filename):
    """Get chart data for a specific visualization type"""
    chart_type = request.args.get('type', 'overview')  # Default to overview if not specified
    filepath = os.path.join(upload_folder, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # Generate chart data for the frontend
        chart_data = generate_chart_data(filepath, chart_type)
        if chart_data:
            return jsonify(chart_data)
        else:
            return jsonify({'error': 'Could not generate chart data'}), 500
    except Exception as e:
        print(f"Error generating chart data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate chart data: {str(e)}'}), 500

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """Delete an uploaded file"""
    filepath = os.path.join(upload_folder, secure_filename(filename))
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        os.remove(filepath)
        return jsonify({'success': True, 'message': 'File deleted successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to delete file: {str(e)}'}), 500

# Route to serve images from the 'public' folder
@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(public_folder, filename)

if __name__ == '__main__':
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 8080)),
        debug=True
    )