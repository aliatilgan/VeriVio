// Browser Console Test Script for VeriVio Frontend
// Bu script'i browser console'da çalıştırarak frontend'i test edebilirsiniz

console.log('=== VeriVio Frontend Test Starting ===');

// Test verisi
const testData = [
  {name: "John", age: 25, score: 85, salary: 45000},
  {name: "Jane", age: 30, score: 92, salary: 55000},
  {name: "Bob", age: 35, score: 78, salary: 65000},
  {name: "Alice", age: 28, score: 88, salary: 50000},
  {name: "Charlie", age: 32, score: 95, salary: 60000},
  {name: "Diana", age: 27, score: 82, salary: 47000},
  {name: "Eve", age: 29, score: 90, salary: 52000},
  {name: "Frank", age: 33, score: 76, salary: 58000}
];

// Mock uploaded data
const mockUploadedData = {
  fileName: 'test_data.csv',
  size: 1024,
  parsedData: testData,
  columns: ['name', 'age', 'score', 'salary'],
  rowCount: testData.length,
  uploadedAt: new Date(),
  fileType: 'csv'
};

console.log('Test data prepared:', mockUploadedData);

// Test function
async function testFrontendAnalysis() {
  try {
    console.log('=== Testing Frontend Analysis ===');
    
    // Direct API call to test backend
    const response = await fetch('http://localhost:8000/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        data: testData,
        analysis_type: 'descriptive',
        clean_data: true,
        cleaning_options: {
          remove_duplicates: true,
          handle_missing: 'drop',
          remove_outliers: false,
          normalize_data: false
        },
        confidence_level: 0.95
      })
    });
    
    console.log('API Response Status:', response.status);
    
    if (response.ok) {
      const result = await response.json();
      console.log('✅ API Response Success!');
      console.log('Response structure:', {
        analysis_id: result.analysis_id,
        results_keys: Object.keys(result.results || {}),
        has_descriptive_stats: !!result.results?.descriptive_stats
      });
      
      if (result.results?.descriptive_stats) {
        console.log('✅ Descriptive stats found!');
        console.log('Variables:', Object.keys(result.results.descriptive_stats));
        
        // Age statistics detail
        if (result.results.descriptive_stats.age) {
          console.log('Age statistics:', result.results.descriptive_stats.age);
        }
      } else {
        console.log('❌ No descriptive stats in response');
      }
      
      return result;
    } else {
      console.log('❌ API call failed:', response.status);
      const errorText = await response.text();
      console.log('Error:', errorText);
      return null;
    }
  } catch (error) {
    console.log('❌ Test failed:', error);
    return null;
  }
}

// React Context test (if available)
function testReactContext() {
  console.log('=== Testing React Context ===');
  
  // Try to access React DevTools
  if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
    console.log('✅ React DevTools detected');
  } else {
    console.log('❌ React DevTools not found');
  }
  
  // Try to access window.React
  if (window.React) {
    console.log('✅ React available on window');
  } else {
    console.log('❌ React not available on window');
  }
}

// Run tests
console.log('Starting tests...');
testReactContext();

// Run API test
testFrontendAnalysis().then(result => {
  if (result) {
    console.log('=== Test Completed Successfully ===');
    console.log('You can now check the frontend UI to see if descriptive stats are displayed correctly.');
  } else {
    console.log('=== Test Failed ===');
    console.log('Check the console for error details.');
  }
});

console.log('=== Instructions ===');
console.log('1. Open http://localhost:8081/ in your browser');
console.log('2. Open Developer Tools (F12) and go to Console tab');
console.log('3. Copy and paste this entire script');
console.log('4. Press Enter to run the test');
console.log('5. Check the console output for test results');
console.log('6. Then manually test by uploading test_descriptive.csv file');