#!/usr/bin/env python3
"""
Configuration Manager for Objectify Application
Helps with MongoDB and environment setup
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path('.env')
    
    if env_path.exists():
        print("✓ .env file already exists")
        return True
    
    env_content = """# Flask Configuration
FLASK_ENV=development
FLASK_DEBUG=True

# Secret Key for JWT (Change this to a random string in production!)
SECRET_KEY=your-super-secret-key-change-this-in-production-12345

# MongoDB Configuration
# Local MongoDB
MONGO_URI=mongodb://localhost:27017/

# For MongoDB Atlas (cloud), use this format:
# MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/objectify_db?retryWrites=true&w=majority
"""
    
    try:
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("✓ Created .env file successfully")
        return True
    except Exception as e:
        print(f"✗ Error creating .env file: {e}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'flask',
        'flask_cors',
        'pymongo',
        'jwt',
        'torch',
        'torchvision',
        'numpy',
        'opencv',
        'PIL',
        'joblib',
        'scikit_learn',
        'ultralytics'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing.append(package)
    
    if missing:
        print(f"\nInstall missing packages with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def test_mongodb_connection(connection_string):
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        
        print("Testing MongoDB connection...")
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✓ MongoDB connection successful!")
        
        # Test database
        db = client['objectify_db']
        print(f"✓ Database 'objectify_db' accessible")
        
        # Check collections
        collections = db.list_collection_names()
        print(f"✓ Collections in database: {collections if collections else 'None (will be created)'}")
        
        client.close()
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MongoDB is running")
        print("2. Check connection string in .env")
        print("3. For local: mongodb://localhost:27017/")
        print("4. For Atlas: Check credentials and IP whitelist")
        return False

def setup_wizard():
    """Interactive setup wizard"""
    print("\n" + "="*50)
    print("   Objectify Setup Wizard")
    print("="*50 + "\n")
    
    # Step 1: Check .env
    print("Step 1: Environment Configuration")
    print("-" * 50)
    create_env_file()
    
    # Step 2: Check dependencies
    print("\nStep 2: Checking Dependencies")
    print("-" * 50)
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\nPlease install missing dependencies first:")
        print("pip install -r requirements.txt")
        return False
    
    # Step 3: Test MongoDB
    print("\nStep 3: MongoDB Connection Test")
    print("-" * 50)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    print(f"Using connection string: {mongo_uri}")
    
    if not test_mongodb_connection(mongo_uri):
        print("\nChoose an option:")
        print("1. Retry connection")
        print("2. Update MongoDB URI in .env")
        print("3. Skip (MongoDB will be required to run the app)")
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            return setup_wizard()
        elif choice == '2':
            print("\nUpdate MONGO_URI in .env file and run this script again")
            return False
    
    print("\n" + "="*50)
    print("✓ Setup Complete!")
    print("="*50)
    print("\nTo run the application:")
    print("python app.py")
    print("\nApplication will be available at:")
    print("http://localhost:5000")
    
    return True

def print_configuration():
    """Print current configuration"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("\n" + "="*50)
    print("   Current Configuration")
    print("="*50 + "\n")
    
    print(f"FLASK_ENV: {os.getenv('FLASK_ENV', 'Not set')}")
    print(f"FLASK_DEBUG: {os.getenv('FLASK_DEBUG', 'Not set')}")
    print(f"MONGO_URI: {os.getenv('MONGO_URI', 'Not set')[:50]}...")
    print(f"SECRET_KEY: {'Set' if os.getenv('SECRET_KEY') else 'Not set'}")
    
    print("\nTo update configuration:")
    print("Edit the .env file in the project root")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            from dotenv import load_dotenv
            load_dotenv()
            mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
            test_mongodb_connection(mongo_uri)
        elif command == 'config':
            print_configuration()
        elif command == 'setup':
            setup_wizard()
        else:
            print("Usage:")
            print("  python config.py setup   - Run setup wizard")
            print("  python config.py test    - Test MongoDB connection")
            print("  python config.py config  - Show current configuration")
    else:
        setup_wizard()
