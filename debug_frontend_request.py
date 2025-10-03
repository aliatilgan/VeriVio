#!/usr/bin/env python3
"""
Frontend'den gönderilen request'i test etmek için script
"""

import requests
import json

def test_frontend_request():
    """Frontend'in gönderdiği gibi bir request test et"""
    
    # Frontend'in gönderdiği gibi test verisi
    test_data = [
        {"name": "John", "age": 25, "city": "New York"},
        {"name": "Jane", "age": 30, "city": "Los Angeles"},
        {"name": "Bob", "age": 35, "city": "Chicago"},
        {"name": "Alice", "age": 28, "city": "Miami"},
        {"name": "Charlie", "age": 32, "city": "Seattle"}
    ]
    
    # Frontend'in gönderdiği request formatı
    analysis_request = {
        "data": test_data,
        "analysis_type": "descriptive",
        "clean_data": True,
        "cleaning_options": {
            "remove_duplicates": True,
            "handle_missing": "drop",
            "remove_outliers": False,
            "normalize_data": False
        },
        "confidence_level": 0.95
    }
    
    print("Testing frontend-like request...")
    print(f"Data: {len(test_data)} rows")
    print(f"Request: {json.dumps(analysis_request, indent=2)}")
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"Analysis ID: {result.get('analysis_id', 'N/A')}")
            print(f"Results keys: {list(result.get('results', {}).keys())}")
            
            # Check descriptive_stats content
            descriptive_stats = result.get('results', {}).get('descriptive_stats')
            print(f"\nDescriptive Stats Content:")
            print(f"Type: {type(descriptive_stats)}")
            print(f"Value: {descriptive_stats}")
            
            if descriptive_stats:
                print(f"Descriptive Stats Keys: {list(descriptive_stats.keys()) if isinstance(descriptive_stats, dict) else 'Not a dict'}")
                if isinstance(descriptive_stats, dict):
                    for key, value in descriptive_stats.items():
                        print(f"  {key}: {type(value)} - {value}")
            else:
                print("❌ descriptive_stats is None or empty!")
        else:
            print("❌ Request failed!")
            try:
                error_data = response.json()
                print(f"Error data: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Request exception: {e}")

if __name__ == "__main__":
    test_frontend_request()