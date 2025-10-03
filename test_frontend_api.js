// Frontend API Test Script
// Bu script browser console'da çalıştırılabilir

async function testFrontendAPI() {
  console.log("🚀 Frontend API Test Başlıyor...");
  
  try {
    // Test verisi oluştur
    const csvContent = `id,group,time,score,pre,post,y1,y2
1,A,1,85,75,85,12.5,8.2
2,A,1,88,78,88,13.1,8.8
3,A,1,82,72,82,11.9,7.9
4,A,2,90,75,90,14.2,9.1
5,A,2,92,78,92,14.8,9.5
6,B,1,87,76,87,13.2,8.5
7,B,1,89,79,89,13.8,9.0
8,B,1,84,73,84,12.1,8.1
9,B,2,91,77,91,14.5,9.3
10,B,2,93,80,93,15.1,9.7
11,C,1,86,74,86,12.8,8.3
12,C,1,90,81,90,14.0,9.2`;
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const file = new File([blob], 'test_data.csv', { type: 'text/csv' });
    
    console.log("📁 Test dosyası oluşturuldu:", file.name);
    
    // Test parametreleri
    const testCases = [
      {
        name: "Paired t-test",
        testType: "t_test_paired",
        params: {
          paired_col_1: "pre",
          paired_col_2: "post"
        }
      },
      {
        name: "MANOVA",
        testType: "manova",
        params: {
          dependent_columns: ["y1", "y2"],
          independent_formula: "group"
        }
      }
    ];
    
    // Her test case'i çalıştır
    for (const testCase of testCases) {
      console.log(`\n=== ${testCase.name} Test ===`);
      
      try {
        // Step 1: Upload file
        const formData = new FormData();
        formData.append("file", file);

        const uploadRes = await fetch("http://localhost:8000/upload", {
          method: "POST",
          body: formData,
        });
        
        if (!uploadRes.ok) {
          throw new Error(`Upload failed: ${uploadRes.status}`);
        }
        
        const uploadData = await uploadRes.json();
        console.log("✅ File uploaded with ID:", uploadData.file_id);

        // Step 2: Run analysis
        const analysisPayload = {
          file_id: uploadData.file_id,
          analysis_type: "hypothesis",
          parameters: { 
            test_type: testCase.testType, 
            ...testCase.params 
          },
        };
        
        console.log("📤 Sending payload:", JSON.stringify(analysisPayload, null, 2));

        const analysisRes = await fetch("http://localhost:8000/analyze", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(analysisPayload),
        });
        
        if (!analysisRes.ok) {
          const errorText = await analysisRes.text();
          console.error("❌ Analysis failed:", analysisRes.status, errorText);
          continue;
        }
        
        const result = await analysisRes.json();
        console.log("✅ Analysis successful!");
        console.log("📊 Result keys:", Object.keys(result));
        
      } catch (error) {
        console.error(`❌ ${testCase.name} failed:`, error.message);
      }
    }
    
    console.log("\n🎉 Frontend API Test Tamamlandı!");
    
  } catch (error) {
    console.error("💥 Test failed:", error);
  }
}

// Test'i çalıştır
testFrontendAPI();