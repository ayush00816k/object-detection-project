"""
Test the authentication flow
"""
import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_signup():
    """Test user signup"""
    print("\n" + "="*50)
    print("TEST 1: User Signup")
    print("="*50)
    
    signup_data = {
        "fullName": "Test User",
        "email": "testuser@example.com",
        "password": "Test12345",
        "confirmPassword": "Test12345"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/signup",
            json=signup_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 201:
            print("✓ Signup successful!")
            return response.json().get('token')
        else:
            print(f"✗ Signup failed: {response.json().get('message')}")
            return None
            
    except Exception as e:
        print(f"✗ Error during signup: {e}")
        return None

def test_login():
    """Test user login"""
    print("\n" + "="*50)
    print("TEST 2: User Login")
    print("="*50)
    
    login_data = {
        "email": "testuser@example.com",
        "password": "Test12345"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/login",
            json=login_data,
            headers={'Content-Type': 'application/json'},
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Login successful!")
            return response.json().get('token')
        else:
            print(f"✗ Login failed: {response.json().get('message')}")
            return None
            
    except Exception as e:
        print(f"✗ Error during login: {e}")
        return None

def test_with_token(token):
    """Test protected endpoint with token"""
    print("\n" + "="*50)
    print("TEST 3: Access Protected Endpoint with Token")
    print("="*50)
    
    try:
        response = requests.get(
            f"{BASE_URL}/api/user/profile",
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            },
            timeout=5
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Token validation successful!")
        else:
            print(f"✗ Token validation failed")
            
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("Starting authentication flow tests...")
    print(f"Base URL: {BASE_URL}")
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    for i in range(10):
        try:
            requests.get(f"{BASE_URL}/", timeout=1)
            print("✓ Server is ready!")
            break
        except:
            if i == 9:
                print("✗ Server did not respond. Make sure Flask is running on port 5000")
                exit(1)
            time.sleep(1)
    
    # Run tests
    token = test_signup()
    
    if token:
        print(f"\n✓ Token received: {token[:50]}...")
        test_with_token(token)
    
    test_login()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)
