import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_paired_ttest():
    """Paired t-test API'sini test et"""
    print("=== Paired t-test Test ===")
    
    # 1. Dosya upload
    with open("test_data.csv", "rb") as f:
        files = {"file": f}
        upload_response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if upload_response.status_code != 200:
        print(f"Upload failed: {upload_response.text}")
        return
    
    file_id = upload_response.json()["file_id"]
    print(f"File uploaded with ID: {file_id}")
    
    # 2. Paired t-test analizi
    analysis_data = {
        "file_id": file_id,
        "analysis_type": "hypothesis",
        "parameters": {
            "test_type": "t_test_paired",
            "paired_col_1": "pre",
            "paired_col_2": "post"
        }
    }
    
    analysis_response = requests.post(
        f"{BASE_URL}/analyze",
        json=analysis_data
    )
    
    if analysis_response.status_code == 200:
        result = analysis_response.json()
        print("✅ Paired t-test başarılı!")
        print(f"Test istatistiği: {result.get('results', {}).get('test_statistic', 'N/A')}")
        print(f"P-değeri: {result.get('results', {}).get('p_value', 'N/A')}")
    else:
        print(f"❌ Paired t-test başarısız: {analysis_response.text}")

def test_manova():
    """MANOVA API'sini test et"""
    print("\n=== MANOVA Test ===")
    
    # 1. Dosya upload
    with open("test_data.csv", "rb") as f:
        files = {"file": f}
        upload_response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if upload_response.status_code != 200:
        print(f"Upload failed: {upload_response.text}")
        return
    
    file_id = upload_response.json()["file_id"]
    print(f"File uploaded with ID: {file_id}")
    
    # 2. MANOVA analizi
    analysis_data = {
        "file_id": file_id,
        "analysis_type": "hypothesis",
        "parameters": {
            "test_type": "manova",
            "dependent_columns": ["y1", "y2"],
            "independent_formula": "group"
        }
    }
    
    analysis_response = requests.post(
        f"{BASE_URL}/analyze",
        json=analysis_data
    )
    
    if analysis_response.status_code == 200:
        result = analysis_response.json()
        print("✅ MANOVA başarılı!")
        print(f"Wilks' Lambda: {result.get('results', {}).get('wilks_lambda', 'N/A')}")
        print(f"P-değeri: {result.get('results', {}).get('p_value', 'N/A')}")
    else:
        print(f"❌ MANOVA başarısız: {analysis_response.text}")

def test_mixed_anova():
    """Mixed ANOVA API'sini test et"""
    print("\n=== Mixed ANOVA Test ===")
    
    # 1. Dosya upload
    with open("test_data.csv", "rb") as f:
        files = {"file": f}
        upload_response = requests.post(f"{BASE_URL}/upload", files=files)
    
    if upload_response.status_code != 200:
        print(f"Upload failed: {upload_response.text}")
        return
    
    file_id = upload_response.json()["file_id"]
    print(f"File uploaded with ID: {file_id}")
    
    # 2. Mixed ANOVA analizi
    analysis_data = {
        "file_id": file_id,
        "analysis_type": "hypothesis",
        "parameters": {
            "test_type": "mixed_anova",
            "subject_column": "id",
            "within_column": "time",
            "between_column": "group",
            "dv_column": "score"
        }
    }
    
    analysis_response = requests.post(
        f"{BASE_URL}/analyze",
        json=analysis_data
    )
    
    if analysis_response.status_code == 200:
        result = analysis_response.json()
        print("✅ Mixed ANOVA başarılı!")
        print(f"F istatistiği: {result.get('results', {}).get('f_statistic', 'N/A')}")
        print(f"P-değeri: {result.get('results', {}).get('p_value', 'N/A')}")
    else:
        print(f"❌ Mixed ANOVA başarısız: {analysis_response.text}")

if __name__ == "__main__":
    print("API Test Başlıyor...")
    
    try:
        test_paired_ttest()
        test_manova()
        test_mixed_anova()
        print("\n🎉 Tüm testler tamamlandı!")
    except Exception as e:
        print(f"❌ Test hatası: {e}")