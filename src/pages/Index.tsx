import { useState } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import DataUpload from "@/components/DataUpload";
import AnalysisSelection from "@/components/AnalysisSelection";
import ResultsPanel from "@/components/ResultsPanel";

const Index = () => {
  const [uploadedData, setUploadedData] = useState<{ fileName: string; size: number; parsedData: any[] } | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string>("");
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  return (
    <div className="min-h-screen bg-gradient-subtle">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <Hero />
        <div className="mt-12 space-y-8">
          <DataUpload onDataUploaded={setUploadedData} />
          {uploadedData && uploadedData.parsedData.length > 0 && (
            <AnalysisSelection 
              onAnalysisSelected={(analysis) => {
                setSelectedAnalysis(analysis);
                // Simulate analysis results
                setAnalysisResults({
                  type: analysis,
                  data: uploadedData.parsedData,
                  summary: {
                    count: uploadedData.parsedData.length,
                    columns: Object.keys(uploadedData.parsedData[0] || {}).length,
                  }
                });
              }}
              selectedAnalysis={selectedAnalysis}
            />
          )}
          {analysisResults && (
            <ResultsPanel results={analysisResults} fileName={uploadedData?.fileName || ''} />
          )}
        </div>
      </main>
    </div>
  );
};

export default Index;
