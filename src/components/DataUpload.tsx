import { Upload, FileText, CheckCircle } from "lucide-react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";

interface DataUploadProps {
  onDataUploaded: (data: any) => void;
}

const DataUpload = ({ onDataUploaded }: DataUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
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

  const handleFile = (file: File) => {
    const validTypes = [
      'text/csv',
      'application/vnd.ms-excel',
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      'application/json'
    ];

    if (!validTypes.includes(file.type)) {
      toast({
        title: "Geçersiz dosya formatı",
        description: "Lütfen CSV, Excel veya JSON dosyası yükleyin.",
        variant: "destructive",
      });
      return;
    }

    setUploadedFile(file);
    onDataUploaded({ fileName: file.name, size: file.size });
    
    toast({
      title: "Dosya başarıyla yüklendi!",
      description: `${file.name} analiz için hazır.`,
    });
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
                  {(uploadedFile.size / 1024).toFixed(2)} KB
                </p>
              </div>
              <Button
                variant="outline"
                onClick={() => {
                  setUploadedFile(null);
                  onDataUploaded(null);
                }}
              >
                Farklı Dosya Seç
              </Button>
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
              >
                Dosya Seç
              </Button>
              <p className="text-xs text-muted-foreground">
                veya dosyayı buraya sürükleyin
              </p>
            </>
          )}
        </div>
      </div>
    </Card>
  );
};

export default DataUpload;
