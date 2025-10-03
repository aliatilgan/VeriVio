#!/usr/bin/env python3
"""
Frontend analiz akışını test etmek için script
Bu script backend API'yi kullanarak analiz işlemini test eder
"""

import requests
import json
import time

# Test verisi
test_data = [
    {"age": 25, "income": 45000, "education": "Bachelor", "experience": 2},
    {"age": 30, "income": 55000, "education": "Master", "experience": 5},
    {"age": 35, "income": 65000, "education": "PhD", "experience": 8},
    {"age": 28, "income": 50000, "education": "Bachelor", "experience": 3},
    {"age": 32, "income": 60000, "education": "Master", "experience": 6},
    {"age": 40, "income": 75000, "education": "PhD", "experience": 12},
    {"age": 26, "income": 47000, "education": "Bachelor", "experience": 2},
    {"age": 29, "income": 52000, "education": "Master", "experience": 4},
    {"age": 38, "income": 70000, "education": "PhD", "experience": 10},
    {"age": 31, "income": 58000, "education": "Master", "experience": 5}
]

def test_backend_health():
    """Backend sağlık kontrolü"""
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"Health check status: {response.status_code}")
        if response.status_code == 200:
            print(f"Health response: {response.json()}")
            return True
        return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_analysis(analysis_type="descriptive"):
    """Analiz testi"""
    try:
        # Analiz isteği hazırla
        analysis_request = {
            "data": test_data,
            "analysis_type": analysis_type,
            "clean_data": True,
            "cleaning_options": {
                "remove_duplicates": True,
                "handle_missing": "drop",
                "remove_outliers": False,
                "normalize_data": False
            },
            "confidence_level": 0.95
        }
        
        print(f"Testing {analysis_type} analysis...")
        print(f"Sending request with {len(test_data)} rows")
        
        # API'ye istek gönder
        response = requests.post(
            "http://localhost:8000/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Analysis response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Analysis successful!")
            print(f"Analysis ID: {result.get('analysis_id', 'N/A')}")
            print(f"Results keys: {list(result.get('results', {}).keys())}")
            
            # Sonuçları detaylı yazdır
            if 'results' in result:
                results = result['results']
                if 'summary' in results:
                    print(f"Summary: {results['summary']}")
                if 'statistics' in results:
                    print(f"Statistics keys: {list(results['statistics'].keys())}")
                if 'dataset_info' in results:
                    print(f"Dataset info: {results['dataset_info']}")
            
            return result
        else:
            print(f"Analysis failed: {response.text}")
            return None
            
    except Exception as e:
        print(f"Analysis test failed: {e}")
        return None

def main():
    print("=== Backend API Test ===")
    
    # 1. Sağlık kontrolü
    if not test_backend_health():
        print("Backend health check failed!")
        return
    
    print("\n=== Analysis Test ===")
    
    # 2. Analiz testi
    result = test_analysis("descriptive")
    
    if result:
        print("\n=== Test Successful ===")
        print("Backend API is working correctly!")
        print("The issue might be in the frontend state management or component rendering.")
    else:
        print("\n=== Test Failed ===")
        print("Backend API has issues!")

if __name__ == "__main__":
    main()