#!/usr/bin/env python3
"""
VeriVio Modül Test Scripti

Bu script tüm entegre edilmiş modüllerin doğru çalıştığını test eder.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from io import StringIO

# Test verileri oluştur
def create_test_data():
    """Test için örnek veri seti oluşturur"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'score': np.random.normal(75, 10, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def test_api_endpoints():
    """API endpoint'lerini test eder"""
    base_url = "http://localhost:8000"
    
    print("🔍 API Endpoint Testleri")
    print("=" * 50)
    
    # Health check
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health endpoint: OK")
            health_data = response.json()
            print(f"   Modüller: {list(health_data.get('modules', {}).keys())}")
        else:
            print(f"❌ Health endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Health endpoint hatası: {e}")
    
    # Root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Root endpoint: OK")
        else:
            print(f"❌ Root endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint hatası: {e}")

def test_data_upload():
    """Veri yükleme fonksiyonalitesini test eder"""
    print("\n📊 Veri Yükleme Testi")
    print("=" * 50)
    
    # Test verisi oluştur
    df = create_test_data()
    csv_data = df.to_csv(index=False)
    
    try:
        # CSV dosyası olarak yükle
        files = {'file': ('test_data.csv', csv_data, 'text/csv')}
        response = requests.post("http://localhost:8000/upload", files=files)
        
        if response.status_code == 200:
            print("✅ Veri yükleme: OK")
            upload_result = response.json()
            print(f"   Dosya ID: {upload_result.get('file_id', 'N/A')}")
            print(f"   Satır sayısı: {upload_result.get('rows', 'N/A')}")
            print(f"   Sütun sayısı: {upload_result.get('columns', 'N/A')}")
            return upload_result.get('file_id')
        else:
            print(f"❌ Veri yükleme hatası: {response.status_code}")
            print(f"   Hata: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Veri yükleme exception: {e}")
        return None

def test_descriptive_stats(file_id):
    """Betimsel istatistik modülünü test eder"""
    if not file_id:
        print("\n❌ Betimsel istatistik testi atlandı (dosya yok)")
        return
    
    print("\n📈 Betimsel İstatistik Testi")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "descriptive",
                "columns": ["age", "income", "score"]
            }
        )
        
        if response.status_code == 200:
            print("✅ Betimsel istatistikler: OK")
            stats = response.json()
            print(f"   Analiz türü: {stats.get('analysis_type', 'N/A')}")
            print(f"   Sonuç: {len(stats.get('results', {}).get('descriptive_stats', {})) if stats.get('results') else 0} sütun analiz edildi")
        else:
            print(f"❌ Betimsel istatistik hatası: {response.status_code}")
            print(f"   Hata: {response.text}")
    except Exception as e:
        print(f"❌ Betimsel istatistik exception: {e}")

def test_hypothesis_testing(file_id):
    """Hipotez testi modülünü test eder"""
    if not file_id:
        print("\n❌ Hipotez testi atlandı (dosya yok)")
        return
    
    print("\n🧪 Hipotez Testi")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/analyze",
            json={
                "file_id": file_id,
                "analysis_type": "hypothesis_test",
                "test_type": "t_test_one_sample",
                "columns": ["score"],
                "test_value": 75
            }
        )
        
        if response.status_code == 200:
            print("✅ Hipotez testi: OK")
            result = response.json()
            print(f"   Analiz türü: {result.get('analysis_type', 'N/A')}")
            print(f"   Test sonucu mevcut: {'results' in result}")
        else:
            print(f"❌ Hipotez testi hatası: {response.status_code}")
            print(f"   Hata: {response.text}")
    except Exception as e:
        print(f"❌ Hipotez testi exception: {e}")

def main():
    """Ana test fonksiyonu"""
    print("🚀 VeriVio Modül Test Süreci Başlatılıyor...")
    print("=" * 60)
    
    # API endpoint testleri
    test_api_endpoints()
    
    # Veri yükleme testi
    file_id = test_data_upload()
    
    # Modül testleri
    test_descriptive_stats(file_id)
    test_hypothesis_testing(file_id)
    
    print("\n" + "=" * 60)
    print("🎉 Test süreci tamamlandı!")

if __name__ == "__main__":
    main()