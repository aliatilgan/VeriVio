import { FileDown, Share2, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ResultsPanelProps {
  results: any;
}

const ResultsPanel = ({ results }: ResultsPanelProps) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground mb-2">Analiz Sonuçları</h2>
          <p className="text-muted-foreground">
            Aşağıda analiz sonuçlarınızı görüntüleyebilir ve indirebilirsiniz
          </p>
        </div>
        
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Share2 className="h-4 w-4 mr-2" />
            Paylaş
          </Button>
          <Button className="bg-gradient-primary hover:opacity-90" size="sm">
            <FileDown className="h-4 w-4 mr-2" />
            Rapor İndir
          </Button>
        </div>
      </div>

      <Card className="p-6">
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="summary">
              <Eye className="h-4 w-4 mr-2" />
              Özet
            </TabsTrigger>
            <TabsTrigger value="charts">Grafikler</TabsTrigger>
            <TabsTrigger value="data">Veri Tablosu</TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary" className="space-y-4 mt-6">
            <div className="p-6 rounded-lg bg-gradient-hero border border-border">
              <h3 className="text-lg font-semibold text-foreground mb-4">
                İstatistiksel Özet
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-lg bg-card/50 backdrop-blur-sm">
                  <p className="text-sm text-muted-foreground mb-1">Örnek Sayısı</p>
                  <p className="text-2xl font-bold text-foreground">1,234</p>
                </div>
                <div className="p-4 rounded-lg bg-card/50 backdrop-blur-sm">
                  <p className="text-sm text-muted-foreground mb-1">Ortalama</p>
                  <p className="text-2xl font-bold text-foreground">45.67</p>
                </div>
                <div className="p-4 rounded-lg bg-card/50 backdrop-blur-sm">
                  <p className="text-sm text-muted-foreground mb-1">Std. Sapma</p>
                  <p className="text-2xl font-bold text-foreground">12.34</p>
                </div>
                <div className="p-4 rounded-lg bg-card/50 backdrop-blur-sm">
                  <p className="text-sm text-muted-foreground mb-1">Güven Aralığı</p>
                  <p className="text-2xl font-bold text-foreground">95%</p>
                </div>
              </div>
            </div>

            <div className="p-6 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-3">Yorum</h4>
              <p className="text-muted-foreground leading-relaxed">
                Analiz sonuçlarına göre, verileriniz normal dağılım göstermektedir. 
                Ortalama değer 45.67 olup, standart sapma 12.34'tür. Bu değerler, 
                verilerinizin merkezi eğilimini ve dağılımını yansıtmaktadır.
              </p>
            </div>
          </TabsContent>
          
          <TabsContent value="charts" className="mt-6">
            <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
              <p className="text-muted-foreground">Grafik görünümü geliştirilme aşamasında</p>
            </div>
          </TabsContent>
          
          <TabsContent value="data" className="mt-6">
            <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
              <p className="text-muted-foreground">Veri tablosu görünümü geliştirilme aşamasında</p>
            </div>
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
};

export default ResultsPanel;
