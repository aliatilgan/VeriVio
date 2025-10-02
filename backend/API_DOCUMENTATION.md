# VeriVio Backend API Dokümantasyonu

## Genel Bilgiler

VeriVio Backend, FastAPI kullanılarak geliştirilmiş modüler bir veri analizi sistemidir. Bu API, veri yükleme, temizleme, analiz ve görselleştirme işlemlerini destekler.

**Base URL:** `http://localhost:8000`

## Endpoint'ler

### 1. Sistem Durumu

#### GET `/health`
Sistem sağlık durumunu kontrol eder.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET `/`
Ana sayfa endpoint'i.

**Response:**
```json
{
  "message": "VeriVio Analytics API",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### 2. Dosya İşlemleri

#### POST `/upload`
Veri dosyası yükler.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (CSV, Excel, JSON formatları desteklenir)

**Response:**
```json
{
  "file_id": "uuid-string",
  "filename": "data.csv",
  "size": 1024,
  "upload_time": "2024-01-01T12:00:00Z"
}
```

#### GET `/files`
Yüklenmiş dosyaları listeler.

**Response:**
```json
{
  "files": [
    {
      "file_id": "uuid-string",
      "filename": "data.csv",
      "size": 1024,
      "upload_time": "2024-01-01T12:00:00Z"
    }
  ]
}
```

#### DELETE `/files/{file_id}`
Belirtilen dosyayı siler.

**Parameters:**
- `file_id`: Silinecek dosyanın ID'si

**Response:**
```json
{
  "message": "File deleted successfully"
}
```

### 3. Analiz İşlemleri

#### POST `/analyze`
Veri analizi gerçekleştirir.

**Request Body:**
```json
{
  "file_id": "uuid-string",
  "analysis_type": "descriptive|visualization|hypothesis_test|regression|finance|marketing|healthcare",
  "cleaning_options": {
    "remove_duplicates": true,
    "handle_missing": "drop|fill_mean|fill_median|fill_mode|forward_fill|backward_fill",
    "outlier_method": "iqr|zscore|isolation_forest|none",
    "normalize": false,
    "standardize": false,
    "encoding": "utf-8"
  },
  "analysis_parameters": {
    // Analiz tipine göre değişen parametreler
  }
}
```

**Analysis Types:**

1. **descriptive**: Betimsel istatistikler
2. **visualization**: Veri görselleştirme
3. **hypothesis_test**: Hipotez testleri
4. **regression**: Regresyon analizi
5. **finance**: Finansal analiz
6. **marketing**: Pazarlama analizi
7. **healthcare**: Sağlık analizi

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### GET `/results/{analysis_id}`
Analiz sonuçlarını getirir.

**Parameters:**
- `analysis_id`: Analiz ID'si

**Response:**
```json
{
  "analysis_id": "uuid-string",
  "analysis_type": "descriptive",
  "results": {
    // Analiz sonuçları
  },
  "metadata": {
    "execution_time": 1.23,
    "data_shape": [1000, 10],
    "created_at": "2024-01-01T12:00:00Z"
  }
}
```

#### GET `/visualizations/{analysis_id}`
Analiz görselleştirmelerini getirir.

**Parameters:**
- `analysis_id`: Analiz ID'si

**Response:**
PNG formatında görselleştirme dosyası

## Analiz Modülleri

### 1. Betimsel İstatistikler (descriptive)
- Temel istatistikler (ortalama, medyan, standart sapma)
- Dağılım bilgileri
- Korelasyon matrisi
- Eksik veri analizi

### 2. Görselleştirme (visualization)
- Histogram
- Scatter plot
- Box plot
- Korelasyon ısı haritası
- Zaman serisi grafikleri

### 3. Hipotez Testleri (hypothesis_test)
- T-test
- Chi-square test
- ANOVA
- Kolmogorov-Smirnov test

### 4. Regresyon Analizi (regression)
- Linear regression
- Logistic regression
- Polynomial regression
- Model değerlendirme metrikleri

### 5. Finansal Analiz (finance)
- Portföy analizi
- Risk analizi
- Finansal oranlar
- Performans metrikleri

### 6. Pazarlama Analizi (marketing)
- Müşteri segmentasyonu
- Kampanya analizi
- Kohort analizi
- RFM analizi

### 7. Sağlık Analizi (healthcare)
- Klinik sonuç analizi
- Biyobelirteç analizi
- Epidemiyolojik analiz
- Risk faktörü analizi

## Hata Kodları

- `400`: Bad Request - Geçersiz istek
- `404`: Not Found - Kaynak bulunamadı
- `422`: Validation Error - Veri doğrulama hatası
- `500`: Internal Server Error - Sunucu hatası

## Örnek Kullanım

### 1. Dosya Yükleme
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data.csv"
```

### 2. Betimsel Analiz
```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "file_id": "your-file-id",
    "analysis_type": "descriptive",
    "cleaning_options": {
      "remove_duplicates": true,
      "handle_missing": "drop"
    }
  }'
```

### 3. Sonuçları Alma
```bash
curl -X GET "http://localhost:8000/results/your-analysis-id"
```

## Geliştirici Notları

- API, CORS desteği ile geliştirilmiştir
- Dosya yüklemeleri `uploads/` dizininde saklanır
- Görselleştirmeler `static/visualizations/` dizininde saklanır
- Analiz sonuçları bellekte cache'lenir
- Tüm endpoint'ler JSON formatında yanıt verir

## Güvenlik

- Dosya türü doğrulaması yapılır
- Dosya boyutu sınırlaması mevcuttur
- Input validation uygulanır
- Error handling implementasyonu vardır