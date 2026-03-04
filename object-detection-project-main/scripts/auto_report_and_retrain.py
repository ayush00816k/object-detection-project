import os
import sys
from pathlib import Path
import tempfile
import base64
import shutil
import random
from datetime import datetime

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
    retrain_collection = db['retrain_queue']
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
print('existing_ai_detection:', test.get('detection_results', {}))

# get image
image_data = test.get('image_path') or test.get('image_data')
if not image_data:
    print('No image stored for this test')
    sys.exit(1)

# prepare ai_detector directories
DATA_DIR = ROOT / 'data' / 'ai_detector'
TRAIN_AI = DATA_DIR / 'train' / 'ai'
TRAIN_REAL = DATA_DIR / 'train' / 'real'
VAL_AI = DATA_DIR / 'val' / 'ai'
VAL_REAL = DATA_DIR / 'val' / 'real'

for p in [TRAIN_AI, TRAIN_REAL, VAL_AI, VAL_REAL]:
    p.mkdir(parents=True, exist_ok=True)

reported_name = f"reported_{test.get('_id')}.jpg"
reported_train_path = TRAIN_AI / reported_name
reported_val_path = VAL_AI / reported_name

# Save image
tmp_file = None
try:
    if isinstance(image_data, str) and image_data.startswith('data:'):
        header, encoded = image_data.split(',', 1)
        ext = 'jpg' if 'jpeg' in header or 'jpg' in header else 'png'
        # Save a base reported image
        with open(reported_train_path, 'wb') as f:
            f.write(base64.b64decode(encoded))
        print('Saved reported image to:', reported_train_path)
    elif isinstance(image_data, str) and os.path.exists(image_data):
        shutil.copy(image_data, reported_train_path)
        print('Copied reported image to:', reported_train_path)
    else:
        print('Unsupported image storage format')
        sys.exit(1)

    # Create augmented variants for training and a distinct validation variant
    from PIL import Image, ImageFilter, ImageEnhance
    def augment_and_save(src_path, dest_path, seed=0, mode='train'):
        im = Image.open(src_path).convert('RGB')
        random.seed(seed)
        if mode == 'train':
            # apply random crop/resize
            w,h = im.size
            crop_box = (int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95))
            im = im.crop(crop_box).resize((224,224))
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.6:
                im = im.rotate(random.uniform(-15,15))
            if random.random() < 0.6:
                im = im.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            enhancer = ImageEnhance.Color(im)
            im = enhancer.enhance(random.uniform(0.8,1.2))
        else:
            im = im.resize((224,224))
            if random.random() < 0.5:
                im = im.rotate(random.uniform(-8,8))
        im.save(dest_path, quality=95)

    # Generate multiple train augmentations
    for i in range(8):
        dest = TRAIN_AI / f"reported_{test.get('_id')}_{i}.jpg"
        if not dest.exists():
            augment_and_save(reported_train_path, dest, seed=i, mode='train')
    # Create a distinct validation variant
    val_dest = VAL_AI / f"reported_{test.get('_id')}_val.jpg"
    if not val_dest.exists():
        augment_and_save(reported_train_path, val_dest, seed=999, mode='val')
    print('Created augmented variants for train/val')

    # Seed real images (copy from data/images/train/*)
    IM_DIR = ROOT / 'data' / 'images' / 'train'
    all_real_imgs = []
    if IM_DIR.exists():
        for cls in IM_DIR.iterdir():
            if cls.is_dir():
                for img in cls.iterdir():
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.jfif']:
                        all_real_imgs.append(img)
    random.shuffle(all_real_imgs)

    # copy up to 40 for train, 10 for val
    train_real_target = list(TRAIN_REAL.iterdir()) if any(TRAIN_REAL.iterdir()) else []
    if len(train_real_target) < 5:
        to_train = all_real_imgs[:40]
        to_val = all_real_imgs[40:50]
        for i,p in enumerate(to_train):
            dest = TRAIN_REAL / f"real_{i}{p.suffix}"
            if not dest.exists():
                shutil.copy(p, dest)
        for i,p in enumerate(to_val):
            dest = VAL_REAL / f"real_{i}{p.suffix}"
            if not dest.exists():
                shutil.copy(p, dest)
        print(f'Seeded {min(len(to_train),40)} real images for training and {len(to_val)} for val')
    else:
        print('Real images already present in train/real; skipping seeding')

    # Insert retrain queue document
    retrain_doc = {
        'test_id': str(test.get('_id')),
        'image_path': str(reported_train_path),
        'reported_at': datetime.utcnow(),
        'reported_by': 'auto_script'
    }
    retrain_collection.insert_one(retrain_doc)
    print('Inserted retrain queue document')

    # Start retraining (background)
    import subprocess
    LOG_PATH = ROOT / 'checkpoints' / 'ai_retrain_manual.log'
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, 'src/train_ai_detector.py', '--data_dir', str(DATA_DIR), '--epochs', '16', '--batch_size', '16', '--save_path', 'checkpoints/ai_detector.pth']
    print('Starting retrain subprocess:', ' '.join(cmd))
    with open(LOG_PATH, 'ab') as lf:
        # Use Popen to run in background; do not wait here
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT)
    print('Retrain started with PID', proc.pid, ' — logs at', LOG_PATH)

except Exception as e:
    print('Error during auto-report/retrain:', e)
    sys.exit(1)

print('\nDone. Model retrain is running in the background.')
