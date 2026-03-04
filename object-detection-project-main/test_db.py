from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import datetime

try:
    client = MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print("✓ MongoDB is running")
    
    db = client['objectify_db']
    users_collection = db['users']
    print("✓ Connected to objectify_db")
    
    # Check existing users
    count = users_collection.count_documents({})
    print(f"✓ Total users in database: {count}")
    
    # List all users
    users = list(users_collection.find({}, {'password': 0}))
    if users:
        print("\nExisting users:")
        for user in users:
            print(f"  - {user['email']}")
    else:
        print("✓ No users found (database is clean)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
