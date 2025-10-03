// Frontend analiz işlemini test etmek için script
// Bu script browser console'da çalıştırılabilir

// Test verisi
const testData = [
  {age: 25, income: 45000, education: 'Bachelor', experience: 2},
  {age: 30, income: 55000, education: 'Master', experience: 5},
  {age: 35, income: 65000, education: 'PhD', experience: 8},
  {age: 28, income: 50000, education: 'Bachelor', experience: 3},
  {age: 32, income: 60000, education: 'Master', experience: 6},
  {age: 40, income: 75000, education: 'PhD', experience: 12},
  {age: 26, income: 47000, education: 'Bachelor', experience: 2},
  {age: 29, income: 52000, education: 'Master', experience: 4},
  {age: 38, income: 70000, education: 'PhD', experience: 10},
  {age: 31, income: 58000, education: 'Master', experience: 5}
];

// Test için uploadedData simülasyonu
const mockUploadedData = {
  fileName: 'test_data.csv',
  size: 1024,
  parsedData: testData,
  columns: ['age', 'income', 'education', 'experience'],
  rowCount: testData.length,
  uploadedAt: new Date(),
  fileType: 'csv'
};

console.log('Test verisi hazırlandı:', mockUploadedData);

// Bu script'i browser console'da çalıştırmak için:
// 1. Browser'da http://localhost:8081 açın
// 2. F12 ile Developer Tools açın
// 3. Console sekmesine gidin
// 4. Bu script'i kopyalayıp yapıştırın
// 5. Enter'a basın

// Sonra AppContext'e erişim için:
// window.React ile React context'e erişebilirsiniz