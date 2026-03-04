import os
import sys
from pathlib import Path
import json
import tempfile
import base64

# ensure project root in path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dotenv import load_dotenv
load_dotenv()

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')

try:
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client['objectify_db']
    tr = db['test_results']
except Exception as e:
    print('ERROR: Could not connect to MongoDB:', e)
    sys.exit(1)

latest = tr.find().sort('timestamp', -1).limit(1)
latest = list(latest)
if not latest:
    print('No saved tests found in DB')
    sys.exit(0)

test = latest[0]
print('\n=== Latest saved test ===')
print('id:', test.get('_id'))
print('primary_object:', test.get('primary_object'))
print('timestamp:', test.get('timestamp'))
print('detection_method:', test.get('detection_method'))
print('existing_ai_detection:', json.dumps(test.get('detection_results', {}).get('AI Detection'), indent=2))

# get image
image_data = test.get('image_path') or test.get('image_data')
if not image_data:
    print('No image stored for this test')
    sys.exit(1)

tmp_file = None
try:
    if isinstance(image_data, str) and image_data.startswith('data:'):
        header, encoded = image_data.split(',', 1)
        ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
        tmp = tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False)
        tmp.write(base64.b64decode(encoded))
        tmp.flush()
        tmp.close()
        tmp_file = tmp.name
    elif isinstance(image_data, str) and os.path.exists(image_data):
        tmp_file = image_data
    else:
        print('Unsupported image storage format')
        sys.exit(1)

    print('Saved image to:', tmp_file)

    # Run AI detector
    print('\n== Running AI detector (hybrid, high sensitivity) ==')
    from src.ai_image_detector import AIImageDetector
    det = AIImageDetector(method='hybrid', sensitivity='high')
    res = det.predict(tmp_file)
    print(json.dumps(res, indent=2))

    # If artifact metrics exist, print them clearly
    if res.get('metrics'):
        print('\n-- Artifact metrics --')
        for k,v in res['metrics'].items():
            print(f"{k}: {v}")

finally:
    if tmp_file and tmp_file.startswith(tempfile.gettempdir()):
        try:
            os.remove(tmp_file)
        except Exception:
            pass

print('\nDone.')
