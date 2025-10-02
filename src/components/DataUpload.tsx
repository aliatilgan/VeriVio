import { Upload, FileText, CheckCircle, Eye } from "lucide-react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import Papa from "papaparse";
import * as XLSX from "xlsx";

interface DataUploadProps {
  onDataUploaded: (data: { fileName: string; size: number; parsedData: any[] }) => void;
}

const DataUpload = ({ onDataUploaded }: DataUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  const { toast } = useToast();

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleFile = async (file: File) => {
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json'
    ];

    if (!validTypes.includes(file.type) && !file.name.endsWith('.csv')) {
      toast({
        title: "Geçersiz dosya formatı",
        description: "Lütfen CSV, Excel veya JSON dosyası yükleyin.",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setUploadedFile(file);

    try {
      let parsedData: any[] = [];

      if (file.name.endsWith('.csv') || file.type === 'text/csv') {
        // CSV parsing
        Papa.parse(file, {
          header: true,
          complete: (results) => {
            parsedData = results.data;
            setPreviewData(parsedData.slice(0, 10)); // First 10 rows for preview
            onDataUploaded({ fileName: file.name, size: file.size, parsedData });
            setIsProcessing(false);
            toast({
              title: "Dosya başarıyla yüklendi!",
              description: `${file.name} - ${parsedData.length} satır veri`,
            });
          },
          error: () => {
            setIsProcessing(false);
            toast({
              title: "Hata",
              description: "Dosya işlenirken bir hata oluştu.",
              variant: "destructive",
            });
          }
        });
      } else if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
        // Excel parsing
        const data = await file.arrayBuffer();
        const workbook = XLSX.read(data);
        const sheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[sheetName];
        parsedData = XLSX.utils.sheet_to_json(worksheet);
        setPreviewData(parsedData.slice(0, 10));
        onDataUploaded({ fileName: file.name, size: file.size, parsedData });
        setIsProcessing(false);
        toast({
          title: "Dosya başarıyla yüklendi!",
          description: `${file.name} - ${parsedData.length} satır veri`,
        });
      } else if (file.type === 'application/json') {
        // JSON parsing
        const text = await file.text();
        parsedData = JSON.parse(text);
        if (!Array.isArray(parsedData)) {
          parsedData = [parsedData];
        }
        setPreviewData(parsedData.slice(0, 10));
        onDataUploaded({ fileName: file.name, size: file.size, parsedData });
        setIsProcessing(false);
        toast({
          title: "Dosya başarıyla yüklendi!",
          description: `${file.name} - ${parsedData.length} satır veri`,
        });
      }
    } catch (error) {
      setIsProcessing(false);
      toast({
        title: "Hata",
        description: "Dosya işlenirken bir hata oluştu.",
        variant: "destructive",
      });
    }
  };

  return (
    <Card className="p-8 border-dashed border-2 transition-all hover:border-primary hover:shadow-glow">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative transition-all ${
          isDragging ? 'scale-105 border-primary' : ''
        }`}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".csv,.xlsx,.xls,.json"
          onChange={handleFileInput}
        />
        
        <div className="flex flex-col items-center justify-center gap-4 py-8">
          {uploadedFile ? (
            <>
              <div className="p-4 rounded-full bg-primary/10">
                <CheckCircle className="h-12 w-12 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold text-foreground mb-1">
                  Dosya Yüklendi
                </p>
                <p className="text-sm text-muted-foreground flex items-center gap-2 justify-center">
                  <FileText className="h-4 w-4" />
                  {uploadedFile.name}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {(uploadedFile.size / 1024).toFixed(2)} KB - {previewData.length > 0 ? `${previewData.length}+ satır` : ''}
                </p>
              </div>
              <div className="flex gap-2">
                {previewData.length > 0 && (
                  <Button
                    variant="outline"
                    onClick={() => setShowPreview(!showPreview)}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    {showPreview ? 'Önizlemeyi Gizle' : 'Veri Önizleme'}
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={() => {
                    setUploadedFile(null);
                    setPreviewData([]);
                    setShowPreview(false);
                    onDataUploaded({ fileName: '', size: 0, parsedData: [] });
                  }}
                >
                  Farklı Dosya Seç
                </Button>
              </div>
            </>
          ) : (
            <>
              <div className="p-4 rounded-full bg-gradient-primary/10">
                <Upload className="h-12 w-12 text-primary" />
              </div>
              <div className="text-center">
                <p className="text-lg font-semibold text-foreground mb-2">
                  Veri Dosyası Yükleyin
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  CSV, Excel veya JSON formatında dosyanızı sürükleyip bırakın
                </p>
              </div>
              <Button
                onClick={() => document.getElementById('file-upload')?.click()}
                className="bg-gradient-primary hover:opacity-90"
                disabled={isProcessing}
              >
                {isProcessing ? 'İşleniyor...' : 'Dosya Seç'}
              </Button>
              <p className="text-xs text-muted-foreground">
                veya dosyayı buraya sürükleyin
              </p>
            </>
          )}
        </div>
      </div>

      {/* Data Preview Table */}
      {showPreview && previewData.length > 0 && (
        <div className="mt-6 overflow-x-auto">
          <h3 className="text-sm font-semibold text-foreground mb-3">Veri Önizleme (İlk 10 Satır)</h3>
          <div className="border border-border rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted">
                <tr>
                  {Object.keys(previewData[0] || {}).map((key) => (
                    <th key={key} className="px-4 py-2 text-left font-medium text-foreground">
                      {key}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewData.map((row, idx) => (
                  <tr key={idx} className="border-t border-border hover:bg-muted/50">
                    {Object.values(row).map((val: any, i) => (
                      <td key={i} className="px-4 py-2 text-muted-foreground">
                        {String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card>
  );
};

export default DataUpload;
