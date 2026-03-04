import os
import sys
from pathlib import Path
import json

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
    print('No saved tests found')
    sys.exit(0)

test = latest[0]
print('Updating test:', test.get('_id'))

from src.ai_image_detector import AIImageDetector

det = AIImageDetector(method='hybrid', sensitivity='high')
# Prepare image path (handle stored path or data URL)
img = test.get('image_path') or test.get('image_data')
img_file = None
if isinstance(img, str) and img.startswith('data:'):
    import tempfile, base64
    header, encoded = img.split(',', 1)
    ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
    tmp = tempfile.NamedTemporaryFile(suffix=f'.{ext}', delete=False)
    tmp.write(base64.b64decode(encoded))
    tmp.flush()
    tmp.close()
    img_file = tmp.name
elif isinstance(img, str) and os.path.exists(img):
    img_file = img
else:
    print('Unsupported image format for update, aborting')
    sys.exit(1)

res = det.predict(img_file)
print('New detection:', json.dumps(res, indent=2))

# Save to DB
tr.update_one({'_id': test['_id']}, {'$set': {'detection_results.AI Detection': res}})
print('Saved new detection to DB')

# Cleanup temporary file
try:
    if img_file and img_file.startswith(tempfile.gettempdir()):
        os.remove(img_file)
except Exception:
    pass
