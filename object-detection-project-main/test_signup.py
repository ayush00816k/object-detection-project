"""Simple curl test for signup"""
import subprocess
import json
import sys

test_data = {
    "fullName": "Test User",
    "email": f"testuser{int(__import__('time').time())}@example.com",
    "password": "Test123456",
    "confirmPassword": "Test123456"
}

print("Testing signup endpoint...")
print(f"Data: {json.dumps(test_data, indent=2)}")
print("-" * 60)

# Build curl command
curl_cmd = [
    'curl', 
    '-X', 'POST',
    '-H', 'Content-Type: application/json',
    '-d', json.dumps(test_data),
    'http://127.0.0.1:5000/api/signup'
]

try:
    result = subprocess.run(curl_cmd, capture_output=True, text=True, timeout=5)
    print(f"Status: {result.returncode}")
    print(f"Output:\n{result.stdout}")
    if result.stderr:
        print(f"Error:\n{result.stderr}")
except Exception as e:
    print(f"Error: {e}")
