// Browser Console'da çalıştırılacak test scripti
// Bu scripti browser'da Developer Tools > Console'a yapıştırın

console.log('=== VeriVio Frontend Test Başlıyor ===');

// Test verisi
const testData = [
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
];

// Simulated uploaded data
const simulatedUploadedData = {
    fileName: "test_data.csv",
    size: 1024,
    parsedData: testData,
    columns: ["age", "income", "education", "experience"],
    rowCount: testData.length,
    uploadedAt: new Date(),
    fileType: "csv"
};

console.log('Test verisi hazırlandı:', simulatedUploadedData);

// AppContext'e erişim için React DevTools gerekli
// Alternatif olarak, window objesine context'i expose edebiliriz

// Manuel test için:
console.log('=== Manuel Test Adımları ===');
console.log('1. Ana sayfada CSV dosyası yükleyin');
console.log('2. Analyses sayfasına gidin');
console.log('3. Descriptive Analysis seçin');
console.log('4. Console log\'larını kontrol edin');
console.log('5. AnalysisResults component\'inin render olup olmadığını kontrol edin');

// Backend API test
async function testBackendAPI() {
    console.log('=== Backend API Test ===');
    
    try {
        // Health check
        const healthResponse = await fetch('http://localhost:8000/health');
        console.log('Health check:', healthResponse.status, await healthResponse.json());
        
        // Analysis test
        const analysisRequest = {
            data: testData,
            analysis_type: "descriptive",
            clean_data: true,
            cleaning_options: {
                remove_duplicates: true,
                handle_missing: "drop",
                remove_outliers: false,
                normalize_data: false
            },
            confidence_level: 0.95
        };
        
        const analysisResponse = await fetch('http://localhost:8000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(analysisRequest)
        });
        
        console.log('Analysis response status:', analysisResponse.status);
        const analysisResult = await analysisResponse.json();
        console.log('Analysis result:', analysisResult);
        
        return analysisResult;
        
    } catch (error) {
        console.error('Backend API test failed:', error);
        return null;
    }
}

// Test'i çalıştır
testBackendAPI().then(result => {
    if (result) {
        console.log('✅ Backend API çalışıyor');
        console.log('🔍 Şimdi frontend state management\'ı kontrol edin');
    } else {
        console.log('❌ Backend API sorunu var');
    }
});

console.log('=== Test scripti hazır ===');
console.log('testBackendAPI() fonksiyonunu çağırarak backend\'i test edebilirsiniz');