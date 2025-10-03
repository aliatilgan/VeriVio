import Header from "@/components/Header";
import { ResponsiveContainer } from "@/components/ResponsiveLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { BookOpen, Download, ExternalLink, FileText, Video, Code } from "lucide-react";
import { useTranslation } from 'react-i18next';

const Documentation = () => {
  const { t } = useTranslation();

  const quickStart = [
    "Veri dosyanızı yükleyin (CSV, Excel, JSON formatları desteklenir)",
    "Analiz türünü seçin (Tanımlayıcı, Korelasyon, Regresyon vb.)",
    "Parametreleri ayarlayın ve analizi başlatın",
    "Sonuçları görüntüleyin ve dışa aktarın"
  ];

  const supportedFormats = [
    { name: "CSV", description: "Virgülle ayrılmış değerler", extension: ".csv" },
    { name: "Excel", description: "Microsoft Excel dosyaları", extension: ".xlsx, .xls" },
    { name: "JSON", description: "JavaScript Object Notation", extension: ".json" },
    { name: "TSV", description: "Sekmeyle ayrılmış değerler", extension: ".tsv" }
  ];

  const analysisTypes = [
    { name: "Tanımlayıcı İstatistikler", description: "Ortalama, medyan, standart sapma, çeyrekler" },
    { name: "Korelasyon Analizi", description: "Pearson, Spearman korelasyon katsayıları" },
    { name: "Regresyon Analizi", description: "Doğrusal ve çoklu regresyon modelleri" },
    { name: "Kümeleme", description: "K-means, hiyerarşik kümeleme algoritmaları" },
    { name: "Hipotez Testleri", description: "T-test, Ki-kare, ANOVA testleri" },
    { name: "Zaman Serisi", description: "Trend analizi, mevsimsellik tespiti" }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <ResponsiveContainer maxWidth="xl" padding="md">
        <div className="py-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
              Dokümantasyon
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              VeriVio platformunu etkili bir şekilde kullanmak için kapsamlı rehber ve dokümantasyon
            </p>
          </div>

          <Tabs defaultValue="getting-started" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="getting-started">Başlangıç</TabsTrigger>
              <TabsTrigger value="data-formats">Veri Formatları</TabsTrigger>
              <TabsTrigger value="analysis-types">Analiz Türleri</TabsTrigger>
              <TabsTrigger value="api-docs">API Dokümantasyonu</TabsTrigger>
            </TabsList>

            <TabsContent value="getting-started" className="mt-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BookOpen className="w-5 h-5" />
                      Hızlı Başlangıç
                    </CardTitle>
                    <CardDescription>
                      VeriVio'yu kullanmaya başlamak için temel adımlar
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ol className="list-decimal list-inside space-y-3">
                      {quickStart.map((step, index) => (
                        <li key={index} className="text-sm text-muted-foreground">
                          {step}
                        </li>
                      ))}
                    </ol>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Video className="w-5 h-5" />
                      Video Eğitimleri
                    </CardTitle>
                    <CardDescription>
                      Adım adım video rehberleri
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button variant="outline" className="w-full justify-start">
                      <Video className="w-4 h-4 mr-2" />
                      Veri Yükleme ve İlk Analiz
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Video className="w-4 h-4 mr-2" />
                      Gelişmiş Analiz Teknikleri
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Video className="w-4 h-4 mr-2" />
                      Sonuçları Yorumlama
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="data-formats" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Desteklenen Veri Formatları</CardTitle>
                  <CardDescription>
                    VeriVio'ya yükleyebileceğiniz dosya türleri ve özellikleri
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {supportedFormats.map((format, index) => (
                      <div key={index} className="border rounded-lg p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-semibold">{format.name}</h3>
                          <Badge variant="secondary">{format.extension}</Badge>
                        </div>
                        <p className="text-sm text-muted-foreground">{format.description}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analysis-types" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Analiz Türleri</CardTitle>
                  <CardDescription>
                    Mevcut istatistiksel analiz yöntemleri ve açıklamaları
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {analysisTypes.map((analysis, index) => (
                      <div key={index} className="border-l-4 border-primary pl-4">
                        <h3 className="font-semibold text-lg">{analysis.name}</h3>
                        <p className="text-muted-foreground">{analysis.description}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="api-docs" className="mt-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Code className="w-5 h-5" />
                      API Referansı
                    </CardTitle>
                    <CardDescription>
                      Geliştiriciler için API dokümantasyonu
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button variant="outline" className="w-full justify-start">
                      <FileText className="w-4 h-4 mr-2" />
                      REST API Endpoints
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Download className="w-4 h-4 mr-2" />
                      OpenAPI Specification
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <ExternalLink className="w-4 h-4 mr-2" />
                      Postman Collection
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>Örnek Kodlar</CardTitle>
                    <CardDescription>
                      Farklı programlama dilleri için örnek implementasyonlar
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="w-4 h-4 mr-2" />
                      Python SDK
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="w-4 h-4 mr-2" />
                      JavaScript/Node.js
                    </Button>
                    <Button variant="outline" className="w-full justify-start">
                      <Code className="w-4 h-4 mr-2" />
                      R Integration
                    </Button>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </ResponsiveContainer>
    </div>
  );
};

export default Documentation;