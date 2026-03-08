import os
import uuid
import threading
import time
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from processor import process_manuscript

app = Flask(__name__, static_folder='static')
CORS(app)

# Use /tmp on cloud containers (ephemeral but always writable);
# fall back to a local subdir when running on a dev machine.
_BASE = '/tmp' if os.path.isdir('/tmp') else os.path.dirname(__file__)
UPLOAD_DIR = os.path.join(_BASE, 'calligraphy_uploads')
OUTPUT_DIR = os.path.join(_BASE, 'calligraphy_outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 100 MB upload limit — large scanned PDFs can be 30–50 MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_EXT = {'.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.heic', '.heif'}


# ── Background cleanup ────────────────────────────────────────────────────────
# Delete output files older than 1 hour to avoid filling container disk.
def _cleanup_loop():
    while True:
        time.sleep(1800)          # run every 30 minutes
        cutoff = time.time() - 3600
        for d in (UPLOAD_DIR, OUTPUT_DIR):
            for fname in os.listdir(d):
                fp = os.path.join(d, fname)
                try:
                    if os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                except OSError:
                    pass

threading.Thread(target=_cleanup_loop, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_file(os.path.join(app.static_folder, 'index.html'))


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    grid_style = request.form.get('grid_style', 'tian')
    try:
        manual_rows = int(request.form.get('rows', 0)) or None
        manual_cols = int(request.form.get('cols', 0)) or None
    except ValueError:
        manual_rows = manual_cols = None

    params = {
        'grid_style': grid_style,
        'manual_rows': manual_rows,
        'manual_cols': manual_cols,
    }

    file_id    = str(uuid.uuid4())
    upload_path  = os.path.join(UPLOAD_DIR, f'{file_id}{ext}')
    output_pdf   = os.path.join(OUTPUT_DIR,  f'{file_id}_practice.pdf')
    preview_path = os.path.join(OUTPUT_DIR,  f'{file_id}_preview.jpg')

    file.save(upload_path)

    try:
        result = process_manuscript(upload_path, output_pdf, preview_path, params)
        return jsonify({
            'file_id':      file_id,
            'preview_url':  f'/preview/{file_id}',
            'download_url': f'/download/{file_id}',
            **result,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(upload_path):
            os.remove(upload_path)


@app.route('/preview/<file_id>')
def preview(file_id):
    if '/' in file_id or '..' in file_id:
        return jsonify({'error': 'Invalid id'}), 400
    path = os.path.join(OUTPUT_DIR, f'{file_id}_preview.jpg')
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404
    return send_file(path, mimetype='image/jpeg')


@app.route('/download/<file_id>')
def download(file_id):
    if '/' in file_id or '..' in file_id:
        return jsonify({'error': 'Invalid id'}), 400
    path = os.path.join(OUTPUT_DIR, f'{file_id}_practice.pdf')
    if not os.path.exists(path):
        return jsonify({'error': 'Not found'}), 404

    response = send_file(path, as_attachment=True,
                         download_name='calligraphy_practice.pdf',
                         mimetype='application/pdf')
    # Schedule deletion of both output files after sending
    @response.call_on_close
    def _cleanup():
        for suffix in ('_practice.pdf', '_preview.jpg'):
            fp = os.path.join(OUTPUT_DIR, f'{file_id}{suffix}')
            try:
                os.remove(fp)
            except OSError:
                pass

    return response


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large (max 100 MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # debug=True only when run directly (dev); gunicorn ignores this
    app.run(host='0.0.0.0', port=port, debug=True)
