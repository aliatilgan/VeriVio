import { Upload, FileText, CheckCircle, Eye, X, AlertCircle } from "lucide-react";
import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { useTranslation } from 'react-i18next';
import { useApp } from "@/contexts/AppContext";
import Papa from "papaparse";
import * as XLSX from "xlsx";

interface DataUploadProps {
  onDataUploaded?: (data: { fileName: string; size: number; parsedData: any[]; columns: string[]; rowCount: number }) => void;
}

const DataUpload = ({ onDataUploaded }: DataUploadProps) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [previewData, setPreviewData] = useState<any[]>([]);
  const [showPreview, setShowPreview] = useState(false);
  const [fileStats, setFileStats] = useState<{ columns: string[]; rowCount: number } | null>(null);
  const { toast } = useToast();
  const { t } = useTranslation();
  const { setUploadedData, setError } = useApp();

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

    const maxSize = 50 * 1024 * 1024; // 50MB limit

    if (!validTypes.includes(file.type) && !file.name.endsWith('.csv')) {
      const errorMsg = t('dataUpload.invalidFormat');
      setError(errorMsg);
      toast({
        title: t('common.error'),
        description: errorMsg,
        variant: "destructive",
      });
      return;
    }

    if (file.size > maxSize) {
      const errorMsg = t('dataUpload.fileTooLarge');
      setError(errorMsg);
      toast({
        title: t('common.error'),
        description: errorMsg,
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);
    setUploadProgress(0);
    setUploadedFile(file);
    setError(null);

    try {
      let parsedData: any[] = [];
      let columns: string[] = [];

      // Simulate progress for better UX
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => Math.min(prev + 10, 90));
      }, 100);

      if (file.name.endsWith('.csv') || file.type === 'text/csv') {
        // CSV parsing
        Papa.parse(file, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            clearInterval(progressInterval);
            setUploadProgress(100);
            
            parsedData = results.data.filter(row => Object.values(row).some(val => val !== ''));
            columns = results.meta.fields || [];
            
            const stats = { columns, rowCount: parsedData.length };
            setFileStats(stats);
            setPreviewData(parsedData.slice(0, 10));
            
            const uploadedData = {
              fileName: file.name,
              size: file.size,
              parsedData: parsedData,
              columns,
              rowCount: parsedData.length
            };
            
            console.log('DataUpload - CSV parsed, calling setUploadedData with:', uploadedData);
            setUploadedData(uploadedData);
            console.log('DataUpload - calling onDataUploaded callback with:', uploadedData);
            onDataUploaded?.(uploadedData);
            
            setIsProcessing(false);
            toast({
              title: t('dataUpload.success'),
              description: t('dataUpload.successDescription', { 
                fileName: file.name, 
                rowCount: parsedData.length 
              }),
            });
          },
          error: (error) => {
            clearInterval(progressInterval);
            setIsProcessing(false);
            const errorMsg = t('dataUpload.processingError');
            setError(errorMsg);
            toast({
              title: t('common.error'),
              description: errorMsg,
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
        
        if (parsedData.length > 0) {
          columns = Object.keys(parsedData[0]);
        }
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        const stats = { columns, rowCount: parsedData.length };
        setFileStats(stats);
        setPreviewData(parsedData.slice(0, 10));
        
        const uploadedData = {
          fileName: file.name,
          size: file.size,
          parsedData,
          columns,
          rowCount: parsedData.length
        };
        
        setUploadedData(uploadedData);
        onDataUploaded?.(uploadedData);
        
        setIsProcessing(false);
        toast({
          title: t('dataUpload.success'),
          description: t('dataUpload.successDescription', { 
            fileName: file.name, 
            rowCount: parsedData.length 
          }),
        });
      } else if (file.type === 'application/json') {
        // JSON parsing
        const text = await file.text();
        parsedData = JSON.parse(text);
        if (!Array.isArray(parsedData)) {
          parsedData = [parsedData];
        }
        
        if (parsedData.length > 0) {
          columns = Object.keys(parsedData[0]);
        }
        
        clearInterval(progressInterval);
        setUploadProgress(100);
        
        const stats = { columns, rowCount: parsedData.length };
        setFileStats(stats);
        setPreviewData(parsedData.slice(0, 10));
        
        const uploadedData = {
          fileName: file.name,
          size: file.size,
          parsedData,
          columns,
          rowCount: parsedData.length
        };
        
        setUploadedData(uploadedData);
        onDataUploaded?.(uploadedData);
        
        setIsProcessing(false);
        toast({
          title: t('dataUpload.success'),
          description: t('dataUpload.successDescription', { 
            fileName: file.name, 
            rowCount: parsedData.length 
          }),
        });
      }
    } catch (error) {
      setIsProcessing(false);
      setUploadProgress(0);
      const errorMsg = t('dataUpload.processingError');
      setError(errorMsg);
      toast({
        title: t('common.error'),
        description: errorMsg,
        variant: "destructive",
      });
    }
  };

  const clearFile = () => {
    setUploadedFile(null);
    setPreviewData([]);
    setShowPreview(false);
    setFileStats(null);
    setUploadProgress(0);
    setError(null);
    setUploadedData({
      fileName: '',
      size: 0,
      parsedData: [],
      columns: [],
      rowCount: 0
    });
  };

  const getFileTypeIcon = (fileName: string) => {
    if (fileName.endsWith('.csv')) return '📊';
    if (fileName.endsWith('.xlsx') || fileName.endsWith('.xls')) return '📈';
    if (fileName.endsWith('.json')) return '📋';
    return '📄';
  };

  return (
    <Card className="p-6 border-dashed border-2 transition-all hover:border-primary hover:shadow-lg">
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative transition-all ${
          isDragging ? 'scale-105 border-primary bg-primary/5' : ''
        }`}
      >
        <input
          type="file"
          id="file-upload"
          className="hidden"
          accept=".csv,.xlsx,.xls,.json"
          onChange={handleFileInput}
        />
        
        <div className="flex flex-col items-center justify-center gap-4 py-6">
          {uploadedFile ? (
            <>
              <div className="p-4 rounded-full bg-primary/10 relative">
                <CheckCircle className="h-12 w-12 text-primary" />
                <Button
                  variant="ghost"
                  size="icon"
                  className="absolute -top-2 -right-2 h-6 w-6 rounded-full bg-destructive text-destructive-foreground hover:bg-destructive/90"
                  onClick={clearFile}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
              
              <div className="text-center space-y-2">
                <p className="text-lg font-semibold text-foreground">
                  {t('dataUpload.fileUploaded')}
                </p>
                <div className="flex items-center gap-2 justify-center">
                  <span className="text-2xl">{getFileTypeIcon(uploadedFile.name)}</span>
                  <div className="text-left">
                    <p className="text-sm font-medium text-foreground">
                      {uploadedFile.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                
                {fileStats && (
                  <div className="flex gap-2 justify-center mt-3">
                    <Badge variant="secondary">
                      {t('dataUpload.rows', { count: fileStats.rowCount })}
                    </Badge>
                    <Badge variant="secondary">
                      {t('dataUpload.columns', { count: fileStats.columns.length })}
                    </Badge>
                  </div>
                )}
              </div>
              
              {isProcessing && (
                <div className="w-full max-w-xs space-y-2">
                  <Progress value={uploadProgress} className="h-2" />
                  <p className="text-xs text-center text-muted-foreground">
                    {t('dataUpload.processing')} {uploadProgress}%
                  </p>
                </div>
              )}
              
              <div className="flex gap-2 flex-wrap justify-center">
                {previewData.length > 0 && !isProcessing && (
                  <Button
                    variant="outline"
                    onClick={() => setShowPreview(!showPreview)}
                  >
                    <Eye className="h-4 w-4 mr-2" />
                    {showPreview ? t('dataUpload.hidePreview') : t('dataUpload.showPreview')}
                  </Button>
                )}
                {!isProcessing && (
                  <Button variant="outline" onClick={clearFile}>
                    <Upload className="h-4 w-4 mr-2" />
                    {t('dataUpload.selectDifferent')}
                  </Button>
                )}
              </div>
            </>
          ) : (
            <>
              <div className="p-4 rounded-full bg-gradient-to-br from-primary/10 to-primary/5">
                <Upload className="h-12 w-12 text-primary" />
              </div>
              <div className="text-center space-y-2">
                <p className="text-lg font-semibold text-foreground">
                  {t('dataUpload.title')}
                </p>
                <p className="text-sm text-muted-foreground max-w-md">
                  {t('dataUpload.description')}
                </p>
                <div className="flex gap-2 justify-center mt-2">
                  <Badge variant="outline">CSV</Badge>
                  <Badge variant="outline">Excel</Badge>
                  <Badge variant="outline">JSON</Badge>
                </div>
              </div>
              <Button
                onClick={() => document.getElementById('file-upload')?.click()}
                className="bg-gradient-to-r from-primary to-primary/80 hover:opacity-90"
                disabled={isProcessing}
              >
                {isProcessing ? t('dataUpload.processing') : t('dataUpload.selectFile')}
              </Button>
              <p className="text-xs text-muted-foreground">
                {t('dataUpload.dragDrop')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('dataUpload.maxSize')}
              </p>
            </>
          )}
        </div>
      </div>

      {/* Data Preview Table */}
      {showPreview && previewData.length > 0 && (
        <div className="mt-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-foreground">
              {t('dataUpload.previewTitle')}
            </h3>
            <Badge variant="outline" className="text-xs">
              {t('dataUpload.previewRows', { count: Math.min(previewData.length, 10) })}
            </Badge>
          </div>
          
          <div className="border border-border rounded-lg overflow-hidden bg-card">
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 sticky top-0">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium text-muted-foreground text-xs w-12">
                      #
                    </th>
                    {Object.keys(previewData[0] || {}).map((key) => (
                      <th key={key} className="px-4 py-2 text-left font-medium text-foreground min-w-[120px]">
                        <div className="flex items-center gap-2">
                          <span className="truncate">{key}</span>
                          <Badge variant="secondary" className="text-xs px-1 py-0">
                            {typeof previewData[0][key] === 'number' ? 'NUM' : 'TEXT'}
                          </Badge>
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {previewData.map((row, idx) => (
                    <tr key={idx} className="border-t border-border hover:bg-muted/30 transition-colors">
                      <td className="px-3 py-2 text-xs text-muted-foreground font-mono">
                        {idx + 1}
                      </td>
                      {Object.values(row).map((val: any, i) => (
                        <td key={i} className="px-4 py-2 text-foreground">
                          <div className="max-w-[200px] truncate" title={String(val)}>
                            {val === null || val === undefined || val === '' ? (
                              <span className="text-muted-foreground italic">
                                {t('common.empty')}
                              </span>
                            ) : (
                              String(val)
                            )}
                          </div>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
          
          {fileStats && fileStats.rowCount > 10 && (
            <div className="text-center">
              <p className="text-xs text-muted-foreground">
                {t('dataUpload.moreRows', { 
                  remaining: fileStats.rowCount - 10,
                  total: fileStats.rowCount 
                })}
              </p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};

export default DataUpload;
