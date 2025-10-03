#!/usr/bin/env python3
"""
Frontend'in CSV parsing işlemini simüle eden test
"""

import pandas as pd
import json
import requests

def test_csv_parsing():
    """CSV dosyasını okuyup frontend formatına dönüştür ve test et"""
    
    # CSV dosyasını oku
    csv_file = "test_data_simple.csv"
    df = pd.read_csv(csv_file)
    
    print(f"CSV dosyası okundu: {len(df)} satır, {len(df.columns)} sütun")
    print(f"Sütunlar: {list(df.columns)}")
    print(f"İlk 3 satır:\n{df.head(3)}")
    
    # Frontend'in yaptığı gibi dictionary listesine dönüştür
    parsed_data = df.to_dict('records')
    
    print(f"\nParsed data (frontend format):")
    print(f"Type: {type(parsed_data)}")
    print(f"Length: {len(parsed_data)}")
    print(f"First 3 records: {json.dumps(parsed_data[:3], indent=2)}")
    
    # Backend'e gönderilecek request'i hazırla
    analysis_request = {
        "data": parsed_data,
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
    
    print(f"\nTesting with backend...")
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json=analysis_request,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Response Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CSV parsing test successful!")
            print(f"Analysis ID: {result.get('analysis_id', 'N/A')}")
        else:
            print("❌ CSV parsing test failed!")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error text: {response.text}")
                
    except Exception as e:
        print(f"❌ Request exception: {e}")

if __name__ == "__main__":
    test_csv_parsing()