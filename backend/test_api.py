"""
VeriVio Backend API Test Scripti
FastAPI endpoint'lerini test eder ve dokümantasyon oluşturur
"""

import requests
import json
import os
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from datetime import datetime


class VeriVioAPITester:
    """VeriVio API test sınıfı"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.test_results = {}
        self.uploaded_files = {}
        
    def create_test_data(self) -> str:
        """Test için örnek veri dosyası oluşturur"""
        # Örnek veri seti oluştur
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'id': range(1, n_samples + 1),
            'age': np.random.normal(35, 10, n_samples),
            'income': np.random.normal(50000, 15000, n_samples),
            'education_years': np.random.normal(14, 3, n_samples),
            'experience': np.random.normal(10, 5, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'department': np.random.choice(['IT', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
            'performance_score': np.random.normal(75, 15, n_samples),
            'satisfaction': np.random.normal(7, 2, n_samples),
            'promotion': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Bazı missing values ekle
        missing_indices = np.random.choice(df.index, size=50, replace=False)
        df.loc[missing_indices, 'satisfaction'] = np.nan
        
        # Test dosyasını kaydet
        test_file_path = "test_data.csv"
        df.to_csv(test_file_path, index=False)
        
        return test_file_path
    
    def test_health_check(self) -> Dict[str, Any]:
        """Health check endpoint'ini test eder"""
        print("🔍 Health check testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            
            result = {
                'endpoint': '/health',
                'method': 'GET',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                print("✅ Health check başarılı")
            else:
                print(f"❌ Health check başarısız: {response.status_code}")
                
        except Exception as e:
            result = {
                'endpoint': '/health',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Health check hatası: {str(e)}")
        
        self.test_results['health_check'] = result
        return result
    
    def test_root_endpoint(self) -> Dict[str, Any]:
        """Root endpoint'ini test eder"""
        print("🔍 Root endpoint testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/")
            
            result = {
                'endpoint': '/',
                'method': 'GET',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                print("✅ Root endpoint başarılı")
            else:
                print(f"❌ Root endpoint başarısız: {response.status_code}")
                
        except Exception as e:
            result = {
                'endpoint': '/',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Root endpoint hatası: {str(e)}")
        
        self.test_results['root_endpoint'] = result
        return result
    
    def test_file_upload(self, file_path: str) -> Dict[str, Any]:
        """Dosya yükleme endpoint'ini test eder"""
        print("🔍 Dosya yükleme testi...")
        
        try:
            with open(file_path, 'rb') as f:
                files = {'file': (os.path.basename(file_path), f, 'text/csv')}
                data = {'file_type': 'csv'}
                
                response = self.session.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data
                )
            
            result = {
                'endpoint': '/upload',
                'method': 'POST',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                file_id = response.json().get('file_id')
                self.uploaded_files['test_data'] = file_id
                print(f"✅ Dosya yükleme başarılı (ID: {file_id})")
            else:
                print(f"❌ Dosya yükleme başarısız: {response.status_code}")
                if response.text:
                    print(f"Hata detayı: {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': '/upload',
                'method': 'POST',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Dosya yükleme hatası: {str(e)}")
        
        self.test_results['file_upload'] = result
        return result
    
    def test_list_files(self) -> Dict[str, Any]:
        """Dosya listeleme endpoint'ini test eder"""
        print("🔍 Dosya listeleme testi...")
        
        try:
            response = self.session.get(f"{self.base_url}/files")
            
            result = {
                'endpoint': '/files',
                'method': 'GET',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                files_count = response.json().get('count', 0)
                print(f"✅ Dosya listeleme başarılı ({files_count} dosya)")
            else:
                print(f"❌ Dosya listeleme başarısız: {response.status_code}")
                
        except Exception as e:
            result = {
                'endpoint': '/files',
                'method': 'GET',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Dosya listeleme hatası: {str(e)}")
        
        self.test_results['list_files'] = result
        return result
    
    def test_descriptive_analysis(self) -> Dict[str, Any]:
        """Betimsel istatistik analizi endpoint'ini test eder"""
        print("🔍 Betimsel istatistik analizi testi...")
        
        if 'test_data' not in self.uploaded_files:
            return {'error': 'Test dosyası yüklenmemiş'}
        
        try:
            request_data = {
                'file_id': self.uploaded_files['test_data'],
                'analysis_type': 'descriptive',
                'parameters': {
                    'columns': ['age', 'income', 'performance_score'],
                    'include_plots': True
                },
                'clean_data': True,
                'cleaning_options': {
                    'remove_duplicates': True,
                    'handle_missing': 'drop',
                    'outlier_method': 'iqr'
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=request_data
            )
            
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'descriptive',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                analysis_id = response.json().get('analysis_id')
                print(f"✅ Betimsel analiz başarılı (ID: {analysis_id})")
            else:
                print(f"❌ Betimsel analiz başarısız: {response.status_code}")
                if response.text:
                    print(f"Hata detayı: {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'descriptive',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Betimsel analiz hatası: {str(e)}")
        
        self.test_results['descriptive_analysis'] = result
        return result
    
    def test_visualization(self) -> Dict[str, Any]:
        """Görselleştirme endpoint'ini test eder"""
        print("🔍 Görselleştirme testi...")
        
        if 'test_data' not in self.uploaded_files:
            return {'error': 'Test dosyası yüklenmemiş'}
        
        try:
            request_data = {
                'file_id': self.uploaded_files['test_data'],
                'analysis_type': 'visualization',
                'parameters': {
                    'visualization_type': 'histogram',
                    'columns': ['age', 'income'],
                    'plot_style': 'seaborn'
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=request_data
            )
            
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'visualization',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                analysis_id = response.json().get('analysis_id')
                print(f"✅ Görselleştirme başarılı (ID: {analysis_id})")
            else:
                print(f"❌ Görselleştirme başarısız: {response.status_code}")
                if response.text:
                    print(f"Hata detayı: {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'visualization',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Görselleştirme hatası: {str(e)}")
        
        self.test_results['visualization'] = result
        return result
    
    def test_hypothesis_test(self) -> Dict[str, Any]:
        """Hipotez testi endpoint'ini test eder"""
        print("🔍 Hipotez testi...")
        
        if 'test_data' not in self.uploaded_files:
            return {'error': 'Test dosyası yüklenmemiş'}
        
        try:
            request_data = {
                'file_id': self.uploaded_files['test_data'],
                'analysis_type': 'hypothesis_test',
                'parameters': {
                    'test_type': 't_test_two_sample',
                    'columns': ['performance_score'],
                    'group_column': 'gender',
                    'confidence_level': 0.95
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=request_data
            )
            
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'hypothesis_test',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                analysis_id = response.json().get('analysis_id')
                print(f"✅ Hipotez testi başarılı (ID: {analysis_id})")
            else:
                print(f"❌ Hipotez testi başarısız: {response.status_code}")
                if response.text:
                    print(f"Hata detayı: {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'hypothesis_test',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Hipotez testi hatası: {str(e)}")
        
        self.test_results['hypothesis_test'] = result
        return result
    
    def test_regression_analysis(self) -> Dict[str, Any]:
        """Regresyon analizi endpoint'ini test eder"""
        print("🔍 Regresyon analizi testi...")
        
        if 'test_data' not in self.uploaded_files:
            return {'error': 'Test dosyası yüklenmemiş'}
        
        try:
            request_data = {
                'file_id': self.uploaded_files['test_data'],
                'analysis_type': 'regression',
                'parameters': {
                    'regression_type': 'linear',
                    'target_column': 'performance_score',
                    'columns': ['age', 'income', 'education_years', 'experience'],
                    'include_plots': True
                }
            }
            
            response = self.session.post(
                f"{self.base_url}/analyze",
                json=request_data
            )
            
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'regression',
                'status_code': response.status_code,
                'success': response.status_code == 200,
                'response_time': response.elapsed.total_seconds(),
                'response_data': response.json() if response.status_code == 200 else None,
                'error': None
            }
            
            if response.status_code == 200:
                analysis_id = response.json().get('analysis_id')
                print(f"✅ Regresyon analizi başarılı (ID: {analysis_id})")
            else:
                print(f"❌ Regresyon analizi başarısız: {response.status_code}")
                if response.text:
                    print(f"Hata detayı: {response.text}")
                
        except Exception as e:
            result = {
                'endpoint': '/analyze',
                'method': 'POST',
                'analysis_type': 'regression',
                'success': False,
                'error': str(e)
            }
            print(f"❌ Regresyon analizi hatası: {str(e)}")
        
        self.test_results['regression_analysis'] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Tüm testleri çalıştırır"""
        print("🚀 VeriVio API Testleri Başlıyor...\n")
        
        # Test verisi oluştur
        test_file = self.create_test_data()
        print(f"📊 Test verisi oluşturuldu: {test_file}\n")
        
        # Testleri sırayla çalıştır
        tests = [
            self.test_root_endpoint,
            self.test_health_check,
            lambda: self.test_file_upload(test_file),
            self.test_list_files,
            self.test_descriptive_analysis,
            self.test_visualization,
            self.test_hypothesis_test,
            self.test_regression_analysis
        ]
        
        for test in tests:
            try:
                test()
                time.sleep(1)  # Testler arası kısa bekleme
            except Exception as e:
                print(f"❌ Test hatası: {str(e)}")
            print()
        
        # Test dosyasını temizle
        if os.path.exists(test_file):
            os.remove(test_file)
        
        # Özet rapor
        self.generate_test_report()
        
        return self.test_results
    
    def generate_test_report(self):
        """Test raporu oluşturur"""
        print("📋 TEST RAPORU")
        print("=" * 50)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get('success', False))
        
        print(f"Toplam Test: {total_tests}")
        print(f"Başarılı: {successful_tests}")
        print(f"Başarısız: {total_tests - successful_tests}")
        print(f"Başarı Oranı: %{(successful_tests/total_tests*100):.1f}")
        print()
        
        # Detaylı sonuçlar
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅" if result.get('success', False) else "❌"
                endpoint = result.get('endpoint', 'N/A')
                method = result.get('method', 'N/A')
                response_time = result.get('response_time', 0)
                
                print(f"{status} {test_name}")
                print(f"   Endpoint: {method} {endpoint}")
                if response_time:
                    print(f"   Yanıt Süresi: {response_time:.3f}s")
                if result.get('error'):
                    print(f"   Hata: {result['error']}")
                print()
        
        # JSON raporu kaydet
        report_file = f"api_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Detaylı rapor kaydedildi: {report_file}")


def main():
    """Ana test fonksiyonu"""
    # API sunucusunun çalıştığını kontrol et
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API sunucusu çalışmıyor. Lütfen önce 'npm run dev' ile sunucuyu başlatın.")
            return
    except requests.exceptions.RequestException:
        print("❌ API sunucusuna bağlanılamıyor. Lütfen önce 'npm run dev' ile sunucuyu başlatın.")
        return
    
    # Testleri çalıştır
    tester = VeriVioAPITester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()