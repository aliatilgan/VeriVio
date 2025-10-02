import { useState } from "react";
import Header from "@/components/Header";
import Hero from "@/components/Hero";
import DataUpload from "@/components/DataUpload";
import AnalysisSelection from "@/components/AnalysisSelection";
import ResultsPanel from "@/components/ResultsPanel";
import UserGuide from "@/components/UserGuide";
import { LanguageSwitcher } from "@/components/LanguageSwitcher";
import { SidebarLayout, ResponsiveContainer, useResponsive } from "@/components/ResponsiveLayout";
import { useApp } from "@/contexts/AppContext";
import { Button } from "@/components/ui/button";
import { HelpCircle, X } from "lucide-react";

const Index = () => {
  const { 
    state,
    setUploadedData, 
    setSelectedAnalysis, 
    performAnalysis,
    showNotification 
  } = useApp();
  
  const { uploadedData, selectedAnalysis, analysisResults } = state;
  
  const [showUserGuide, setShowUserGuide] = useState(false);
  const { isMobile } = useResponsive();

  const handleAnalysisSelected = async (analysis: string) => {
    setSelectedAnalysis(analysis);
    
    if (uploadedData) {
      try {
        await performAnalysis(analysis, uploadedData.parsedData);
        showNotification('success', 'Analysis Complete', 'Your data analysis has been completed successfully.');
      } catch (error) {
        showNotification('error', 'Analysis Failed', 'There was an error processing your data. Please try again.');
      }
    }
  };

  const handleDataUploaded = (data: { fileName: string; size: number; parsedData: any[] }) => {
    const columns = data.parsedData.length > 0 ? Object.keys(data.parsedData[0]) : [];
    setUploadedData({
      ...data,
      columns,
      rowCount: data.parsedData.length,
      uploadedAt: new Date(),
      fileType: data.fileName.split('.').pop() || 'unknown'
    });
    showNotification('success', 'File Uploaded', `Successfully uploaded ${data.fileName}`);
  };

  const sidebar = (
    <div className="h-full flex flex-col">
      {/* Language Switcher */}
      <div className="p-4 border-b border-gray-200">
        <LanguageSwitcher variant="compact" />
      </div>
      
      {/* Help Button */}
      <div className="p-4 border-b border-gray-200">
        <Button
          onClick={() => setShowUserGuide(true)}
          variant="outline"
          className="w-full justify-start gap-2"
        >
          <HelpCircle className="w-4 h-4" />
          User Guide
        </Button>
      </div>
      
      {/* Navigation or additional content can go here */}
      <div className="flex-1 p-4">
        <div className="text-sm text-gray-600">
          <h3 className="font-semibold mb-2">Quick Actions</h3>
          <ul className="space-y-1">
            <li>• Upload your data file</li>
            <li>• Select analysis type</li>
            <li>• View results</li>
            <li>• Export findings</li>
          </ul>
        </div>
      </div>
    </div>
  );

  const mainContent = (
    <div className="h-full overflow-auto">
      <Header />
      <ResponsiveContainer maxWidth="xl" padding="md">
        <Hero />
        <div className="mt-12 space-y-8">
          <DataUpload onDataUploaded={handleDataUploaded} />
          {uploadedData && uploadedData.parsedData.length > 0 && (
            <AnalysisSelection 
              onAnalysisSelected={handleAnalysisSelected}
              selectedAnalysis={selectedAnalysis}
            />
          )}
          {analysisResults && (
            <ResultsPanel results={analysisResults} fileName={uploadedData?.fileName || ''} />
          )}
        </div>
      </ResponsiveContainer>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-subtle">
      {isMobile ? (
        // Mobile layout - no sidebar
        <div className="relative">
          {/* Mobile language switcher */}
          <div className="absolute top-4 right-4 z-10">
            <LanguageSwitcher variant="icon-only" />
          </div>
          
          {/* Mobile help button */}
          <div className="absolute top-4 right-16 z-10">
            <Button
              onClick={() => setShowUserGuide(true)}
              variant="outline"
              size="sm"
              className="p-2"
            >
              <HelpCircle className="w-4 h-4" />
            </Button>
          </div>
          
          {mainContent}
        </div>
      ) : (
        // Desktop layout - with sidebar
        <SidebarLayout
          sidebar={sidebar}
          sidebarWidth="sm"
          collapsible={true}
          defaultCollapsed={false}
        >
          {mainContent}
        </SidebarLayout>
      )}

      {/* User Guide Modal */}
      {showUserGuide && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-lg shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden">
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <h2 className="text-xl font-semibold">User Guide</h2>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowUserGuide(false)}
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
            <div className="overflow-auto max-h-[calc(90vh-80px)]">
              <UserGuide isOpen={showUserGuide} onClose={() => setShowUserGuide(false)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Index;
