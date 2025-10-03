#!/usr/bin/env python3
"""
Frontend veri akışını test etmek için script
"""

import requests
import json

def test_frontend_flow():
    """Frontend'in tam veri akışını test et"""
    
    # Test verisi - yaş sütunu ile
    test_data = [
        {"name": "John", "age": 25, "score": 85},
        {"name": "Jane", "age": 30, "score": 92},
        {"name": "Bob", "age": 35, "score": 78},
        {"name": "Alice", "age": 28, "score": 88},
        {"name": "Charlie", "age": 32, "score": 95},
        {"name": "Diana", "age": 27, "score": 82},
        {"name": "Eve", "age": 29, "score": 90},
        {"name": "Frank", "age": 33, "score": 76}
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
    
    print("=== Frontend Flow Test ===")
    print(f"Test Data: {len(test_data)} rows")
    print(f"Columns: {list(test_data[0].keys())}")
    print(f"Age values: {[row['age'] for row in test_data]}")
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\n=== Backend Response ===")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend request successful!")
            
            # Backend response yapısını analiz et
            print(f"\n=== Response Structure ===")
            print(f"Top level keys: {list(result.keys())}")
            
            if 'results' in result:
                results = result['results']
                print(f"Results keys: {list(results.keys())}")
                
                if 'descriptive_stats' in results:
                    desc_stats = results['descriptive_stats']
                    print(f"\n=== Descriptive Stats ===")
                    print(f"Type: {type(desc_stats)}")
                    
                    if isinstance(desc_stats, dict):
                        print(f"Variables: {list(desc_stats.keys())}")
                        
                        # Age sütunu için detayları göster
                        if 'age' in desc_stats:
                            age_stats = desc_stats['age']
                            print(f"\n=== Age Statistics ===")
                            print(f"Age stats type: {type(age_stats)}")
                            print(f"Age stats keys: {list(age_stats.keys()) if isinstance(age_stats, dict) else 'Not a dict'}")
                            
                            if isinstance(age_stats, dict):
                                for key, value in age_stats.items():
                                    print(f"  {key}: {value} ({type(value)})")
                        else:
                            print("❌ No 'age' column found in descriptive_stats")
                    else:
                        print(f"❌ descriptive_stats is not a dict: {desc_stats}")
                else:
                    print("❌ No 'descriptive_stats' in results")
            else:
                print("❌ No 'results' in response")
                
            # Frontend'in beklediği yapıyı simüle et
            print(f"\n=== Frontend Expected Structure ===")
            frontend_data = result.get('results', {})
            print(f"Frontend data keys: {list(frontend_data.keys())}")
            print(f"Has descriptive_stats: {'descriptive_stats' in frontend_data}")
            
            if 'descriptive_stats' in frontend_data:
                print("✅ Frontend should be able to access descriptive_stats")
                desc_stats = frontend_data['descriptive_stats']
                if isinstance(desc_stats, dict) and 'age' in desc_stats:
                    age_stats = desc_stats['age']
                    print(f"Age stats available: {list(age_stats.keys()) if isinstance(age_stats, dict) else 'Invalid'}")
                    
                    # Türkçe istatistik isimleri için mapping
                    expected_stats = ['count', 'mean', 'median', 'std', 'var', 'min', 'max']
                    available_stats = list(age_stats.keys()) if isinstance(age_stats, dict) else []
                    
                    print(f"\n=== Statistics Mapping ===")
                    for stat in expected_stats:
                        if stat in available_stats:
                            value = age_stats[stat]
                            print(f"✅ {stat}: {value}")
                        else:
                            print(f"❌ Missing: {stat}")
            else:
                print("❌ Frontend will not find descriptive_stats")
                
        else:
            print("❌ Backend request failed!")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Request exception: {e}")

if __name__ == "__main__":
    test_frontend_flow()