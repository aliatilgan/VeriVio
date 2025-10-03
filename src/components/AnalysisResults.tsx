import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  CheckCircle, 
  BarChart3, 
  FileText, 
  Download, 
  TrendingUp,
  Activity,
  Brain,
  Clock
} from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface AnalysisResultsProps {
  results: any;
  analysisType: string;
  onNewAnalysis: () => void;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ 
  results, 
  analysisType, 
  onNewAnalysis 
}) => {
  const { t } = useTranslation();

  // Debug logs - keep minimal for troubleshooting
  console.log('AnalysisResults - received results:', !!results);
  console.log('AnalysisResults - data available:', !!results?.data);
  console.log('AnalysisResults - data keys:', results?.data ? Object.keys(results.data) : 'no data');

  if (!results) {
    console.log('AnalysisResults - No results provided, returning null');
    return null;
  }

  const getAnalysisIcon = (type: string) => {
    switch (type) {
      case 'descriptive':
        return <BarChart3 className="w-5 h-5" />;
      case 'hypothesis':
        return <Activity className="w-5 h-5" />;
      case 'regression':
        return <TrendingUp className="w-5 h-5" />;
      case 'clustering':
      case 'factor_analysis':
        return <Brain className="w-5 h-5" />;
      default:
        return <FileText className="w-5 h-5" />;
    }
  };

  const getAnalysisTitle = (type: string) => {
    const titles: { [key: string]: string } = {
      'descriptive': 'Betimsel İstatistik Analizi',
      'hypothesis': 'Hipotez Testi',
      'regression': 'Regresyon Analizi',
      'clustering': 'Kümeleme Analizi',
      'factor_analysis': 'Faktör Analizi',
      'machine_learning': 'Makine Öğrenmesi',
      'time_series': 'Zaman Serisi Analizi',
      'advanced': 'İleri Düzey Analiz'
    };
    return titles[type] || 'Analiz Sonuçları';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-100 dark:bg-green-900/20">
                <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <CardTitle className="flex items-center gap-2">
                  {getAnalysisIcon(analysisType)}
                  {getAnalysisTitle(analysisType)}
                </CardTitle>
                <CardDescription>
                  Analiz başarıyla tamamlandı
                </CardDescription>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {results.timestamp ? new Date(results.timestamp).toLocaleString('tr-TR') : 'Şimdi'}
              </Badge>
              <Button variant="outline" size="sm" onClick={onNewAnalysis}>
                Yeni Analiz
              </Button>
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Results Tabs */}
      <Tabs defaultValue="summary" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="summary">Özet</TabsTrigger>
          <TabsTrigger value="details">Detaylar</TabsTrigger>
          <TabsTrigger value="interpretation">Yorumlama</TabsTrigger>
          <TabsTrigger value="export">Dışa Aktar</TabsTrigger>
        </TabsList>

        <TabsContent value="summary" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Analiz Özeti</CardTitle>
              <CardDescription>
                Temel bulgular ve sonuçlar
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results.data?.summary && (
                <div className="space-y-6">
                  {/* Dataset Bilgileri */}
                  {results.data.summary.dataset_info && (
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                      <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-3">
                        Veri Seti Bilgileri
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Toplam Satır:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.total_rows?.toLocaleString('tr-TR')}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Toplam Sütun:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.total_columns}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Sayısal Sütun:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.numeric_columns}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Kategorik Sütun:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.categorical_columns}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Eksik Değer:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.missing_values_total}
                          </span>
                        </div>
                        <div>
                          <span className="text-blue-700 dark:text-blue-300 font-medium">Bellek Kullanımı:</span>
                          <span className="ml-2 text-blue-900 dark:text-blue-100">
                            {results.data.summary.dataset_info.memory_usage_mb} MB
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Veri Kalitesi */}
                  {results.data.summary.data_quality && (
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                      <h4 className="font-semibold text-green-900 dark:text-green-100 mb-3">
                        Veri Kalitesi
                      </h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="text-green-700 dark:text-green-300 font-medium">Tamlık Skoru:</span>
                          <span className="ml-2 text-green-900 dark:text-green-100">
                            %{results.data.summary.data_quality.completeness_score}
                          </span>
                        </div>
                        <div>
                          <span className="text-green-700 dark:text-green-300 font-medium">Tekrarlanan Satır:</span>
                          <span className="ml-2 text-green-900 dark:text-green-100">
                            {results.data.summary.data_quality.duplicate_rows}
                          </span>
                        </div>
                        {results.data.summary.data_quality.columns_with_missing?.length > 0 && (
                          <div className="col-span-full">
                            <span className="text-green-700 dark:text-green-300 font-medium">Eksik Değerli Sütunlar:</span>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {results.data.summary.data_quality.columns_with_missing.map((col: string, idx: number) => (
                                <Badge key={idx} variant="outline" className="text-xs">
                                  {col}
                                </Badge>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Önemli Bulgular */}
                  {results.data.summary.key_findings && results.data.summary.key_findings.length > 0 && (
                    <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg border border-yellow-200 dark:border-yellow-800">
                      <h4 className="font-semibold text-yellow-900 dark:text-yellow-100 mb-3">
                        Önemli Bulgular
                      </h4>
                      <ul className="space-y-2 text-sm">
                        {results.data.summary.key_findings.map((finding: string, idx: number) => (
                          <li key={idx} className="text-yellow-800 dark:text-yellow-200 flex items-start">
                            <span className="w-2 h-2 bg-yellow-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                            {finding}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
              
              {results.data?.interpretation && (
                <div className="mt-4">
                  <h4 className="font-semibold mb-2">Temel Bulgular:</h4>
                  <div className="bg-muted p-4 rounded-lg">
                    <pre className="whitespace-pre-wrap text-sm">
                      {results.data.interpretation}
                    </pre>
                  </div>
                </div>
              )}

              {!results.data?.summary && !results.data?.interpretation && (
                <div className="text-center py-8 text-muted-foreground">
                  <FileText className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Özet bilgisi mevcut değil</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="details" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Detaylı Sonuçlar</CardTitle>
              <CardDescription>
                Analiz sonuçlarının detaylı görünümü
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(results.data || {}).map(([key, value]) => {
                  if (key === 'interpretation' || key === 'summary' || value === null) {
                    return null;
                  }
                  
                  // Special handling for descriptive_stats
                  if (key === 'descriptive_stats' && typeof value === 'object') {
                    return (
                      <div key={key} className="border rounded-lg p-4">
                        <h4 className="font-semibold mb-4">Betimsel İstatistikler</h4>
                        <div className="grid gap-4">
                          {Object.entries(value as Record<string, any>).map(([variable, stats]) => (
                            <div key={variable} className="bg-muted p-4 rounded-lg">
                              <h5 className="font-medium mb-3 text-lg capitalize">{variable}</h5>
                              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                {Object.entries(stats as Record<string, any>).map(([statName, statValue]) => (
                                  <div key={statName} className="bg-background p-2 rounded border">
                                    <div className="text-xs text-muted-foreground uppercase tracking-wide">
                                      {statName.replace(/_/g, ' ')}
                                    </div>
                                    <div className="font-medium">
                                      {typeof statValue === 'number' 
                                        ? statValue.toLocaleString('tr-TR', { 
                                            maximumFractionDigits: 2,
                                            minimumFractionDigits: 0
                                          })
                                        : String(statValue)
                                      }
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  }
                  
                  return (
                    <div key={key} className="border rounded-lg p-4">
                      <h4 className="font-semibold mb-2 capitalize">
                        {key.replace(/_/g, ' ')}
                      </h4>
                      <div className="bg-muted p-3 rounded text-sm">
                        <pre className="whitespace-pre-wrap">
                          {typeof value === 'object' 
                            ? JSON.stringify(value, null, 2)
                            : String(value)
                          }
                        </pre>
                      </div>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="interpretation" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Sonuçların Yorumlanması</CardTitle>
              <CardDescription>
                Analiz sonuçlarının anlamı ve önerileri
              </CardDescription>
            </CardHeader>
            <CardContent>
              {results.data?.interpretation ? (
                <div className="space-y-4">
                  <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
                    <h4 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">
                      Analiz Yorumu:
                    </h4>
                    <div className="text-blue-800 dark:text-blue-200">
                      <pre className="whitespace-pre-wrap text-sm">
                        {results.data.interpretation}
                      </pre>
                    </div>
                  </div>

                  {results.data?.recommendations && (
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border border-green-200 dark:border-green-800">
                      <h4 className="font-semibold text-green-900 dark:text-green-100 mb-2">
                        Öneriler:
                      </h4>
                      <div className="text-green-800 dark:text-green-200">
                        <pre className="whitespace-pre-wrap text-sm">
                          {results.data.recommendations}
                        </pre>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  <Brain className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Yorumlama bilgisi mevcut değil</p>
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="export" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Sonuçları Dışa Aktar</CardTitle>
              <CardDescription>
                Analiz sonuçlarını farklı formatlarda kaydedin
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Button variant="outline" className="h-20 flex-col gap-2">
                  <Download className="w-6 h-6" />
                  <span>PDF Raporu</span>
                </Button>
                <Button variant="outline" className="h-20 flex-col gap-2">
                  <Download className="w-6 h-6" />
                  <span>Excel Dosyası</span>
                </Button>
              </div>
              <div className="mt-4 p-4 bg-muted rounded-lg">
                <p className="text-sm text-muted-foreground">
                  <strong>Not:</strong> Dışa aktarma özelliği yakında eklenecektir.
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalysisResults;