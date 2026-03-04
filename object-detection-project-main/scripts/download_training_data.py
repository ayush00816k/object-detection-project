import os
import sys
import shutil
import random
import subprocess
from pathlib import Path

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"📦 Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

def setup_data():
    print("🚀 Setting up training data...")
    
    # Install dependencies if missing
    install_and_import("datasets")
    install_and_import("tqdm")
    from datasets import load_dataset
    from tqdm import tqdm

    # Define paths
    base_dir = Path("data/ai_detector")
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    
    # Clean up old data if exists
    if base_dir.exists():
        print("Cleaning up old data...")
        shutil.rmtree(base_dir)
    
    for d in [train_dir/"real", train_dir/"ai", val_dir/"real", val_dir/"ai"]:
        d.mkdir(parents=True, exist_ok=True)

    print("📥 Downloading training data from Hugging Face...")
    
    datasets_to_try = [
        ("Parveshiiii/AI-vs-Real", "image"), # Dataset, image_key
        ("artix/ai-vs-real-images", "image"),
    ]
    
    success = False
    
    from datasets import load_dataset
    
    for ds_name, img_key in datasets_to_try:
        if success: break
        try:
            print(f"\nTrying dataset: {ds_name}...")
            # Load in streaming mode first to peek
            ds = load_dataset(ds_name, split="train", streaming=True)
            
            counts = {"real": 0, "ai": 0}
            target_per_class = 800 # Reduced goal for speed/stability
            
            pbar = tqdm(total=target_per_class*2)
            
            for i, item in enumerate(ds):
                # Debug first few items to understand structure
                if i < 5:
                    print(f"[DEBUG] Item {i} keys: {list(item.keys())}, Label: {item.get('label')}")

                if counts["real"] >= target_per_class and counts["ai"] >= target_per_class:
                    success = True
                    break
                
                try:
                    image = item.get(img_key) or item.get('jpg') or item.get('file') or item.get('img')
                    if not image: continue
                    
                    # Convert to PIL if not already
                    if not hasattr(image, 'save'):
                        continue

                    label = item.get('label') or item.get('binary_label') or item.get('class_label')
                    
                    # Improved Label Logic
                    is_ai = False
                    if label is not None:
                        # integer check
                        if isinstance(label, int):
                            is_ai = (label == 1) # Standard 0=Real, 1=Fake
                        # string check
                        else:
                            l_str = str(label).lower()
                            is_ai = any(x in l_str for x in ['ai', 'fake', 'synth', 'gen', '1'])
                            
                    cat = "ai" if is_ai else "real"
                    
                    if counts[cat] >= target_per_class:
                        continue
                        
                    # Save
                    save_dir = train_dir if random.random() > 0.2 else val_dir
                    file_path = save_dir / cat / f"{ds_name.replace('/','_')}_{i}.jpg"
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    image.thumbnail((512, 512))
                    image.save(file_path, "JPEG", quality=85)
                    
                    counts[cat] += 1
                    pbar.update(1)
                    
                except Exception as e:
                    continue
            
            if success:
                print(f"\n✅ Successfully downloaded from {ds_name}")
                break
                
        except Exception as e:
            print(f"⚠️ Failed with {ds_name}: {e}")
            continue

    if not success:
        print("\n❌ Could not download sufficient data from any source.")
        print("Please check internet connection or firewall.")
        return

    print("\n✅ Dataset setup complete!")
    print(f"Train Real: {len(list((train_dir/'real').glob('*')))}")
    print(f"Train AI:   {len(list((train_dir/'ai').glob('*')))}")
    print(f"Val Real:   {len(list((val_dir/'real').glob('*')))}")
    print(f"Val AI:     {len(list((val_dir/'ai').glob('*')))}")

if __name__ == "__main__":
    setup_data()
