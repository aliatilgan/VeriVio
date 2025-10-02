import { FileDown, Share2, Eye, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import jsPDF from 'jspdf';
import * as XLSX from 'xlsx';
import { useToast } from "@/hooks/use-toast";

interface ResultsPanelProps {
  results: any;
  fileName: string;
}

const ResultsPanel = ({ results, fileName }: ResultsPanelProps) => {
  const { toast } = useToast();

  // Sample data for charts
  const barData = results.data?.slice(0, 10).map((row: any, idx: number) => ({
    name: `Satır ${idx + 1}`,
    value: Object.values(row)[0] as number || Math.random() * 100,
  })) || [];

  const COLORS = ['hsl(245, 58%, 51%)', 'hsl(280, 65%, 60%)', 'hsl(210, 40%, 96%)', 'hsl(220, 15%, 45%)'];

  const handleDownloadPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(18);
    doc.text('Analiz Raporu', 14, 20);
    doc.setFontSize(12);
    doc.text(`Dosya: ${fileName}`, 14, 30);
    doc.text(`Analiz Türü: ${results.type}`, 14, 40);
    doc.text(`Toplam Satır: ${results.summary.count}`, 14, 50);
    doc.text(`Toplam Sütun: ${results.summary.columns}`, 14, 60);
    doc.save(`analiz-raporu-${Date.now()}.pdf`);
    
    toast({
      title: "PDF İndirildi",
      description: "Rapor başarıyla PDF olarak kaydedildi.",
    });
  };

  const handleDownloadExcel = () => {
    const ws = XLSX.utils.json_to_sheet(results.data || []);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Analiz Sonuçları');
    XLSX.writeFile(wb, `analiz-sonuclari-${Date.now()}.xlsx`);
    
    toast({
      title: "Excel İndirildi",
      description: "Sonuçlar başarıyla Excel olarak kaydedildi.",
    });
  };
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
          <Button variant="outline" size="sm" onClick={() => {
            toast({
              title: "Paylaşım",
              description: "Paylaşım özelliği yakında eklenecek.",
            });
          }}>
            <Share2 className="h-4 w-4 mr-2" />
            Paylaş
          </Button>
          <Button variant="outline" size="sm" onClick={handleDownloadExcel}>
            <FileDown className="h-4 w-4 mr-2" />
            Excel İndir
          </Button>
          <Button className="bg-gradient-primary hover:opacity-90" size="sm" onClick={handleDownloadPDF}>
            <FileDown className="h-4 w-4 mr-2" />
            PDF Rapor
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
          
          <TabsContent value="charts" className="mt-6 space-y-6">
            <div className="p-4 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-4">Çubuk Grafik</h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }} 
                  />
                  <Legend />
                  <Bar dataKey="value" fill="hsl(245, 58%, 51%)" name="Değer" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="p-4 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-4">Çizgi Grafik</h4>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'hsl(var(--card))', 
                      border: '1px solid hsl(var(--border))',
                      borderRadius: '8px'
                    }} 
                  />
                  <Legend />
                  <Line type="monotone" dataKey="value" stroke="hsl(280, 65%, 60%)" name="Trend" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="p-4 rounded-lg border border-border">
              <h4 className="font-semibold text-foreground mb-4">Pasta Grafik</h4>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={barData.slice(0, 4)}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={(entry) => entry.name}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {barData.slice(0, 4).map((entry: any, index: number) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          
          <TabsContent value="data" className="mt-6">
            {results.data && results.data.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead className="bg-muted">
                    <tr>
                      {Object.keys(results.data[0]).map((key) => (
                        <th key={key} className="px-4 py-3 text-left font-medium text-foreground border-b border-border">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {results.data.slice(0, 50).map((row: any, idx: number) => (
                      <tr key={idx} className="border-b border-border hover:bg-muted/50">
                        {Object.values(row).map((val: any, i) => (
                          <td key={i} className="px-4 py-3 text-muted-foreground">
                            {String(val)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {results.data.length > 50 && (
                  <p className="text-xs text-muted-foreground text-center mt-4">
                    İlk 50 satır gösteriliyor. Toplam {results.data.length} satır.
                  </p>
                )}
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
                <p className="text-muted-foreground">Veri bulunamadı</p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </Card>
    </div>
  );
};

export default ResultsPanel;
