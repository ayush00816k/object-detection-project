"""Test signup endpoint to see actual errors"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"

test_data = {
    "fullName": "Test User",
    "email": "testuser@example.com",
    "password": "Test12345",
    "confirmPassword": "Test12345"
}

print(f"Testing signup endpoint: {BASE_URL}/api/signup")
print(f"Sending data: {json.dumps(test_data, indent=2)}")
print("-" * 60)

try:
    response = requests.post(
        f"{BASE_URL}/api/signup",
        json=test_data,
        headers={'Content-Type': 'application/json'},
        timeout=10
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type')}")
    print(f"Response Text (first 500 chars): {response.text[:500]}")
    print("-" * 60)
    
    try:
        data = response.json()
        print(f"Parsed JSON:\n{json.dumps(data, indent=2)}")
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not parse JSON response!")
        print(f"Reason: {e}")
        print(f"Raw response:\n{response.text}")
        
except requests.exceptions.ConnectionError:
    print("ERROR: Could not connect to server!")
    print(f"Make sure Flask is running on {BASE_URL}")
except Exception as e:
    print(f"ERROR: {e}")
