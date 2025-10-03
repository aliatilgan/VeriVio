import React from 'react';
import { useNavigate } from 'react-router-dom';
import Header from '@/components/Header';
import AnalysisSelection from '@/components/AnalysisSelection';
import AnalysisResults from '@/components/AnalysisResults';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useApp } from '@/contexts/AppContext';
import { BarChart3, Users, Calendar, FileText, Database, Upload } from 'lucide-react';
import { useTranslation } from 'react-i18next';

const Analyses = () => {
  const { t } = useTranslation();
  const { state, setSelectedAnalysis, performAnalysis, showNotification } = useApp();
  const { uploadedData, selectedAnalysis, analysisResults } = state;
  const navigate = useNavigate();

  // Debug logs
  console.log('Analyses - uploadedData:', uploadedData);
  console.log('Analyses - selectedAnalysis:', selectedAnalysis);
  console.log('Analyses - analysisResults:', analysisResults);

  const handleAnalysisSelected = async (analysis: string) => {
    console.log('handleAnalysisSelected - analysis:', analysis);
    setSelectedAnalysis(analysis);
    
    if (uploadedData) {
      try {
        console.log('handleAnalysisSelected - calling performAnalysis with:', uploadedData.parsedData);
        const result = await performAnalysis(analysis, uploadedData.parsedData);
        console.log('handleAnalysisSelected - performAnalysis result:', result);
        showNotification('success', 'Analiz Tamamlandı', 'Veri analiziniz başarıyla tamamlandı.');
      } catch (error) {
        console.error('handleAnalysisSelected - error:', error);
        showNotification('error', 'Analiz Başarısız', 'Verilerinizi işlerken bir hata oluştu. Lütfen tekrar deneyin.');
      }
    } else {
      console.log('handleAnalysisSelected - no uploadedData');
    }
  };

  const handleBackToUpload = () => {
    navigate('/');
  };

  const handleNewAnalysis = () => {
    setSelectedAnalysis(null);
  };



  return (
    <div className="min-h-screen bg-background">
      <Header />
      <div className="container mx-auto px-4 py-8">
        {/* Uploaded Data Info */}
        {uploadedData ? (
          <div className="mb-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Database className="w-5 h-5" />
                  Yüklenen Veri
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-2">
                    <FileText className="w-4 h-4 text-blue-500" />
                    <div>
                      <p className="text-sm text-muted-foreground">Dosya Adı</p>
                      <p className="font-medium">{uploadedData.fileName}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Users className="w-4 h-4 text-green-500" />
                    <div>
                      <p className="text-sm text-muted-foreground">Satır Sayısı</p>
                      <p className="font-medium">{uploadedData.rowCount.toLocaleString()}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-purple-500" />
                    <div>
                      <p className="text-sm text-muted-foreground">Sütun Sayısı</p>
                      <p className="font-medium">{uploadedData.columns.length}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-orange-500" />
                    <div>
                      <p className="text-sm text-muted-foreground">Yüklenme Tarihi</p>
                      <p className="font-medium">{uploadedData.uploadedAt.toLocaleDateString('tr-TR')}</p>
                    </div>
                  </div>
                </div>
                <div className="mt-4">
                  <p className="text-sm text-muted-foreground mb-2">Sütunlar:</p>
                  <div className="flex flex-wrap gap-2">
                    {uploadedData.columns.map((column, index) => (
                      <Badge key={index} variant="secondary">
                        {column}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="mb-8">
            <Card>
              <CardContent className="text-center py-8">
                <Upload className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="text-lg font-semibold mb-2">Henüz veri yüklenmedi</h3>
                <p className="text-muted-foreground mb-4">
                  Analiz yapmak için önce veri yüklemeniz gerekiyor.
                </p>
                <Button onClick={handleBackToUpload}>
                  Ana Sayfaya Dön
                </Button>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Analysis Results */}
        {uploadedData && analysisResults && (
          <AnalysisResults 
            results={analysisResults}
            analysisType={selectedAnalysis || 'unknown'}
            onNewAnalysis={handleNewAnalysis}
          />
        )}

        {/* Analysis Selection */}
        {uploadedData && !analysisResults && (
          <div>
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold mb-4">Analiz Türü Seçin</h1>
              <p className="text-muted-foreground">
                Verilerinizi analiz etmek için aşağıdaki seçeneklerden birini seçin
              </p>
            </div>
            <AnalysisSelection 
               onAnalysisSelected={handleAnalysisSelected} 
               selectedAnalysis={selectedAnalysis}
             />
          </div>
        )}
      </div>
    </div>
  );
};

export default Analyses;