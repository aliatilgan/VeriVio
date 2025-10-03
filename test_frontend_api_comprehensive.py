#!/usr/bin/env python3
"""
Frontend API'sini farklı veri türleri ile test eden kapsamlı script
"""

import requests
import json
import pandas as pd
import numpy as np

# Backend URL
BASE_URL = "http://localhost:8000"

def test_api_with_data(data, test_name):
    """API'yi belirli veri ile test et"""
    print(f"\n=== {test_name} ===")
    
    # Veriyi frontend formatına çevir (list of dictionaries)
    if isinstance(data, pd.DataFrame):
        data_dict = data.to_dict('records')
    else:
        data_dict = data
    
    print(f"Veri boyutu: {len(data_dict)} satır")
    print(f"İlk satır: {data_dict[0] if data_dict else 'Boş'}")
    
    # API isteği
    request_data = {
        "data": data_dict,
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
    
    try:
        response = requests.post(f"{BASE_URL}/analyze", json=request_data)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Başarılı! Analysis ID: {result.get('analysis_id', 'N/A')}")
            return True
        else:
            error_data = response.json()
            print(f"Hata: {error_data}")
            return False
            
    except Exception as e:
        print(f"İstek hatası: {e}")
        return False

def main():
    print("Frontend API Kapsamlı Test")
    print("=" * 50)
    
    # Test 1: Normal veri
    normal_data = [
        {"age": 25, "income": 50000, "education": "Bachelor", "experience": 3},
        {"age": 30, "income": 60000, "education": "Master", "experience": 5},
        {"age": 35, "income": 70000, "education": "PhD", "experience": 10}
    ]
    test_api_with_data(normal_data, "Normal Veri")
    
    # Test 2: Null değerler içeren veri
    null_data = [
        {"age": 25, "income": None, "education": "Bachelor", "experience": 3},
        {"age": None, "income": 60000, "education": "", "experience": 5},
        {"age": 35, "income": 70000, "education": "PhD", "experience": None}
    ]
    test_api_with_data(null_data, "Null Değerler İçeren Veri")
    
    # Test 3: Boş string'ler içeren veri
    empty_string_data = [
        {"age": 25, "income": 50000, "education": "", "experience": 3},
        {"age": 30, "income": "", "education": "Master", "experience": ""},
        {"age": "", "income": 70000, "education": "PhD", "experience": 10}
    ]
    test_api_with_data(empty_string_data, "Boş String'ler İçeren Veri")
    
    # Test 4: Karışık veri türleri
    mixed_data = [
        {"age": "25", "income": 50000.5, "education": "Bachelor", "experience": 3},
        {"age": 30, "income": "60000", "education": "Master", "experience": "5"},
        {"age": 35.0, "income": 70000, "education": "PhD", "experience": 10}
    ]
    test_api_with_data(mixed_data, "Karışık Veri Türleri")
    
    # Test 5: Özel karakterler içeren veri
    special_char_data = [
        {"name": "Ahmet Öz", "city": "İstanbul", "salary": 50000},
        {"name": "Mehmet Ç.", "city": "Ankara", "salary": 60000},
        {"name": "Ayşe Ğ.", "city": "İzmir", "salary": 70000}
    ]
    test_api_with_data(special_char_data, "Özel Karakterler İçeren Veri")
    
    # Test 6: Çok büyük sayılar
    large_numbers_data = [
        {"id": 1, "value": 999999999999999, "score": 1.7976931348623157e+308},
        {"id": 2, "value": -999999999999999, "score": -1.7976931348623157e+308},
        {"id": 3, "value": 0, "score": 0.0}
    ]
    test_api_with_data(large_numbers_data, "Çok Büyük Sayılar")
    
    # Test 7: Boş veri
    empty_data = []
    test_api_with_data(empty_data, "Boş Veri")
    
    # Test 8: Tek satır veri
    single_row_data = [
        {"age": 25, "income": 50000, "education": "Bachelor", "experience": 3}
    ]
    test_api_with_data(single_row_data, "Tek Satır Veri")

if __name__ == "__main__":
    main()