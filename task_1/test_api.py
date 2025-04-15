# test_api.py
import requests
import json

# Base URL - change if you deploy elsewhere
BASE_URL = "http://localhost:8000"

# Test the API
def test_api():
    # Test root endpoint
    print("Testing API root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"API Status: {response.json()}")
    
    # Test transaction listing
    print("\nTesting transaction listing...")
    response = requests.get(f"{BASE_URL}/transactions/", params={"limit": 5})
    if response.status_code == 200:
        transactions = response.json()
        print(f"Success: {len(transactions)} transactions retrieved")
        if transactions:
            print("Sample transaction:")
            print(json.dumps(transactions[0], indent=2))
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test transaction summary
    print("\nTesting transaction summary...")
    response = requests.get(f"{BASE_URL}/transactions/summary/")
    if response.status_code == 200:
        summary = response.json()
        print(f"Success: Summary retrieved")
        print(f"Total spending: ${summary['total_spending']:.2f}")
        print(f"Total income: ${summary['total_income']:.2f}")
        print(f"Net cashflow: ${summary['net_cashflow']:.2f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test risk assessment for a user
    print("\nTesting risk assessment...")
    # Get first user ID from transactions
    try:
        user_id = "USER_0001"  # Default ID to try
        response = requests.get(f"{BASE_URL}/transactions/", params={"limit": 1})
        if response.status_code == 200 and response.json():
            # Try to get a user ID if available
            transactions = response.json()
            if 'user_id' in transactions[0]:
                user_id = transactions[0]['user_id']
    except:
        pass
    
    response = requests.get(f"{BASE_URL}/risk/{user_id}")
    if response.status_code == 200:
        risk = response.json()
        print(f"Success: Risk assessment for {user_id} retrieved")
        print(f"Risk level: {risk['risk_assessment']['risk_label']}")
        print(f"Risk probability: {risk['risk_assessment']['risk_probability']:.2f}")
        print("\nTop risk factors:")
        for factor in risk['risk_assessment']['top_factors'][:3]:
            print(f"- {factor['factor']}: {factor['value']} ({factor['direction']} risk)")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    
    # Test transaction categorization
    print("\nTesting transaction categorization...")
    sample_transaction = {
        "description": "Monthly payment to Landlord",
        "amount": -1200,
        "date": "2023-04-01"
    }
    response = requests.post(f"{BASE_URL}/categorize/", json=sample_transaction)
    if response.status_code == 200:
        result = response.json()
        print(f"Success: Transaction categorized as '{result['predicted_category']}'")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_api()