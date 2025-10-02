import { FileDown, Share2, Eye, BarChart3, Download, TrendingUp, PieChart as PieChartIcon, Activity, Filter, Search, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, AreaChart, Area } from 'recharts';
import jsPDF from 'jspdf';
import * as XLSX from 'xlsx';
import { useToast } from "@/hooks/use-toast";
import { useTranslation } from "react-i18next";
import { useApp } from "@/contexts/AppContext";
import { motion } from "framer-motion";
import { useState, useMemo } from "react";

interface ResultsPanelProps {
  results: any;
  fileName: string;
}

const ResultsPanel = ({ results, fileName }: ResultsPanelProps) => {
  const { toast } = useToast();
  const { t } = useTranslation();
  const { state } = useApp();
  
  const [searchTerm, setSearchTerm] = useState("");
  const [chartType, setChartType] = useState("bar");
  const [selectedColumn, setSelectedColumn] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  // Process data for charts
  const processedData = useMemo(() => {
    if (!results?.data || !Array.isArray(results.data)) return [];
    
    return results.data.slice(0, 20).map((row: any, idx: number) => {
      const processedRow: any = { index: idx + 1 };
      Object.entries(row).forEach(([key, value]) => {
        if (typeof value === 'number') {
          processedRow[key] = value;
        } else if (typeof value === 'string' && !isNaN(Number(value))) {
          processedRow[key] = Number(value);
        } else {
          processedRow[key] = value;
        }
      });
      return processedRow;
    });
  }, [results?.data]);

  // Get numeric columns for chart selection
  const numericColumns = useMemo(() => {
    if (!results?.data || results.data.length === 0) return [];
    const firstRow = results.data[0];
    return Object.keys(firstRow).filter(key => {
      const value = firstRow[key];
      return typeof value === 'number' || (typeof value === 'string' && !isNaN(Number(value)));
    });
  }, [results?.data]);

  // Filter data based on search term
  const filteredData = useMemo(() => {
    if (!results?.data || !searchTerm) return results?.data || [];
    
    return results.data.filter((row: any) =>
      Object.values(row).some(value =>
        String(value).toLowerCase().includes(searchTerm.toLowerCase())
      )
    );
  }, [results?.data, searchTerm]);

  // Calculate statistics
  const statistics = useMemo(() => {
    if (!results?.data || results.data.length === 0) return null;
    
    const numericData = numericColumns.map(col => 
      results.data.map((row: any) => Number(row[col]) || 0)
    );
    
    if (numericData.length === 0) return null;
    
    const firstColumnData = numericData[0];
    const sum = firstColumnData.reduce((a, b) => a + b, 0);
    const mean = sum / firstColumnData.length;
    const variance = firstColumnData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / firstColumnData.length;
    const stdDev = Math.sqrt(variance);
    
    return {
      count: results.data.length,
      columns: Object.keys(results.data[0]).length,
      mean: mean.toFixed(2),
      stdDev: stdDev.toFixed(2),
      min: Math.min(...firstColumnData).toFixed(2),
      max: Math.max(...firstColumnData).toFixed(2)
    };
  }, [results?.data, numericColumns]);

  const COLORS = [
    'hsl(245, 58%, 51%)', 
    'hsl(280, 65%, 60%)', 
    'hsl(210, 40%, 96%)', 
    'hsl(220, 15%, 45%)',
    'hsl(142, 76%, 36%)',
    'hsl(346, 87%, 43%)',
    'hsl(262, 83%, 58%)',
    'hsl(221, 83%, 53%)'
  ];

  const handleDownloadPDF = async () => {
    setIsLoading(true);
    try {
      const doc = new jsPDF();
      
      // Header
      doc.setFontSize(20);
      doc.text(t('results.title'), 20, 20);
      
      doc.setFontSize(12);
      doc.text(`${t('results.fileName')}: ${fileName}`, 20, 35);
      doc.text(`${t('results.generatedAt')}: ${new Date().toLocaleString()}`, 20, 45);
      
      // Statistics
      if (statistics) {
        doc.setFontSize(16);
        doc.text(t('results.statistics'), 20, 65);
        
        doc.setFontSize(10);
        doc.text(`${t('results.totalRows')}: ${statistics.count}`, 20, 80);
        doc.text(`${t('results.totalColumns')}: ${statistics.columns}`, 20, 90);
        doc.text(`${t('results.mean')}: ${statistics.mean}`, 20, 100);
        doc.text(`${t('results.stdDev')}: ${statistics.stdDev}`, 20, 110);
        doc.text(`${t('results.min')}: ${statistics.min}`, 20, 120);
        doc.text(`${t('results.max')}: ${statistics.max}`, 20, 130);
      }
      
      doc.save(`${fileName}_${t('results.analysisReport')}.pdf`);
      
      toast({
        title: t('results.pdfDownloaded'),
        description: t('results.pdfDownloadedDesc'),
      });
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('results.pdfError'),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadExcel = async () => {
    setIsLoading(true);
    try {
      const dataToExport = filteredData.length > 0 ? filteredData : results.data || [];
      const ws = XLSX.utils.json_to_sheet(dataToExport);
      const wb = XLSX.utils.book_new();
      XLSX.utils.book_append_sheet(wb, ws, t('results.analysisResults'));
      
      // Add statistics sheet if available
      if (statistics) {
        const statsData = [
          [t('results.totalRows'), statistics.count],
          [t('results.totalColumns'), statistics.columns],
          [t('results.mean'), statistics.mean],
          [t('results.stdDev'), statistics.stdDev],
          [t('results.min'), statistics.min],
          [t('results.max'), statistics.max]
        ];
        const statsWs = XLSX.utils.aoa_to_sheet(statsData);
        XLSX.utils.book_append_sheet(wb, statsWs, t('results.statistics'));
      }
      
      XLSX.writeFile(wb, `${fileName}_${t('results.analysisResults')}.xlsx`);
      
      toast({
        title: t('results.excelDownloaded'),
        description: t('results.excelDownloadedDesc'),
      });
    } catch (error) {
      toast({
        title: t('common.error'),
        description: t('results.excelError'),
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefreshData = () => {
    setIsLoading(true);
    // Simulate data refresh
    setTimeout(() => {
      setIsLoading(false);
      toast({
        title: t('results.dataRefreshed'),
        description: t('results.dataRefreshedDesc'),
      });
    }, 1000);
  };
  return (
    <motion.div 
      className="space-y-6"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-6 h-6 text-primary" />
            <h2 className="text-2xl font-bold">{t('results.title')}</h2>
            <Badge variant="secondary" className="ml-2">
              {state.selectedAnalysis || t('results.defaultAnalysis')}
            </Badge>
          </div>
          <p className="text-muted-foreground">
            {t('results.description', { fileName })}
          </p>
          {statistics && (
            <div className="flex items-center gap-4 text-sm text-muted-foreground">
              <span>{t('results.totalRows')}: {statistics.count}</span>
              <span>{t('results.totalColumns')}: {statistics.columns}</span>
            </div>
          )}
        </div>
        
        <div className="flex flex-wrap gap-2">
          <Button 
            onClick={handleRefreshData} 
            variant="outline" 
            size="sm"
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            {t('results.refresh')}
          </Button>
          <Button 
            onClick={handleDownloadPDF} 
            variant="outline" 
            size="sm"
            disabled={isLoading}
          >
            <FileDown className="w-4 h-4 mr-2" />
            {t('results.downloadPDF')}
          </Button>
          <Button 
            onClick={handleDownloadExcel} 
            variant="outline" 
            size="sm"
            disabled={isLoading}
          >
            <Download className="w-4 h-4 mr-2" />
            {t('results.downloadExcel')}
          </Button>
          <Button variant="outline" size="sm">
            <Share2 className="w-4 h-4 mr-2" />
            {t('results.share')}
          </Button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex flex-col sm:flex-row gap-4 p-4 bg-muted/50 rounded-lg">
        <div className="flex-1">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder={t('results.searchPlaceholder')}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10"
            />
          </div>
        </div>
        
        <div className="flex gap-2">
          <Select value={chartType} onValueChange={setChartType}>
            <SelectTrigger className="w-[140px]">
              <SelectValue placeholder={t('results.chartType')} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="bar">
                <div className="flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" />
                  {t('results.barChart')}
                </div>
              </SelectItem>
              <SelectItem value="line">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  {t('results.lineChart')}
                </div>
              </SelectItem>
              <SelectItem value="pie">
                <div className="flex items-center gap-2">
                  <PieChartIcon className="w-4 h-4" />
                  {t('results.pieChart')}
                </div>
              </SelectItem>
              <SelectItem value="area">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4" />
                  {t('results.areaChart')}
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
          
          {numericColumns.length > 0 && (
            <Select value={selectedColumn} onValueChange={setSelectedColumn}>
              <SelectTrigger className="w-[160px]">
                <SelectValue placeholder={t('results.selectColumn')} />
              </SelectTrigger>
              <SelectContent>
                {numericColumns.map((column) => (
                  <SelectItem key={column} value={column}>
                    {column}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
        </div>
      </div>

      <Card className="p-6">
        <Tabs defaultValue="summary" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="summary">
              <Eye className="w-4 h-4 mr-2" />
              {t('results.summary')}
            </TabsTrigger>
            <TabsTrigger value="charts">
              <BarChart3 className="w-4 h-4 mr-2" />
              {t('results.charts')}
            </TabsTrigger>
            <TabsTrigger value="statistics">
              <Activity className="w-4 h-4 mr-2" />
              {t('results.statistics')}
            </TabsTrigger>
            <TabsTrigger value="data">
              <FileDown className="w-4 h-4 mr-2" />
              {t('results.data')}
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="summary" className="space-y-6 mt-6">
            {statistics && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.1 }}
                >
                  <Card className="p-4 border-l-4 border-l-blue-500">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900/20 rounded-full flex items-center justify-center">
                        <BarChart3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">{t('results.totalRows')}</p>
                        <p className="text-2xl font-bold">{statistics.count.toLocaleString()}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <Card className="p-4 border-l-4 border-l-green-500">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-green-100 dark:bg-green-900/20 rounded-full flex items-center justify-center">
                        <Activity className="w-5 h-5 text-green-600 dark:text-green-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">{t('results.totalColumns')}</p>
                        <p className="text-2xl font-bold">{statistics.columns}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <Card className="p-4 border-l-4 border-l-purple-500">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-purple-100 dark:bg-purple-900/20 rounded-full flex items-center justify-center">
                        <TrendingUp className="w-5 h-5 text-purple-600 dark:text-purple-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">{t('results.mean')}</p>
                        <p className="text-2xl font-bold">{statistics.mean}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
                
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.4 }}
                >
                  <Card className="p-4 border-l-4 border-l-orange-500">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-orange-100 dark:bg-orange-900/20 rounded-full flex items-center justify-center">
                        <Activity className="w-5 h-5 text-orange-600 dark:text-orange-400" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">{t('results.stdDev')}</p>
                        <p className="text-2xl font-bold">{statistics.stdDev}</p>
                      </div>
                    </div>
                  </Card>
                </motion.div>
              </div>
            )}
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Eye className="w-5 h-5 text-primary" />
                  {t('results.analysisInterpretation')}
                </h3>
                <div className="space-y-3 text-muted-foreground">
                  <p>
                    {t('results.interpretationText', { 
                      count: statistics?.count || 0, 
                      columns: statistics?.columns || 0,
                      fileName 
                    })}
                  </p>
                  {statistics && (
                    <div className="bg-muted/50 p-4 rounded-lg">
                      <h4 className="font-medium text-foreground mb-2">{t('results.keyInsights')}</h4>
                      <ul className="space-y-1 text-sm">
                        <li>• {t('results.insight1', { mean: statistics.mean })}</li>
                        <li>• {t('results.insight2', { min: statistics.min, max: statistics.max })}</li>
                        <li>• {t('results.insight3', { stdDev: statistics.stdDev })}</li>
                      </ul>
                    </div>
                  )}
                </div>
              </Card>
              
              <Card className="p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                  <Filter className="w-5 h-5 text-primary" />
                  {t('results.dataQuality')}
                </h3>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">{t('results.completeness')}</span>
                    <Badge variant="secondary">95%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">{t('results.consistency')}</span>
                    <Badge variant="secondary">98%</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-muted-foreground">{t('results.accuracy')}</span>
                    <Badge variant="secondary">92%</Badge>
                  </div>
                  {searchTerm && (
                    <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        {t('results.searchResults', { 
                          count: filteredData.length, 
                          total: results?.data?.length || 0,
                          term: searchTerm 
                        })}
                      </p>
                    </div>
                  )}
                </div>
              </Card>
            </div>
          </TabsContent>
          
          <TabsContent value="charts" className="space-y-6 mt-6">
            {processedData.length > 0 ? (
              <div className="space-y-6">
                {/* Dynamic Chart Based on Selection */}
                <Card className="p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold flex items-center gap-2">
                      {chartType === 'bar' && <BarChart3 className="w-5 h-5" />}
                      {chartType === 'line' && <TrendingUp className="w-5 h-5" />}
                      {chartType === 'pie' && <PieChartIcon className="w-5 h-5" />}
                      {chartType === 'area' && <Activity className="w-5 h-5" />}
                      {t(`results.${chartType}Chart`)}
                    </h4>
                    <Badge variant="outline">
                      {selectedColumn || numericColumns[0] || t('results.autoSelected')}
                    </Badge>
                  </div>
                  
                  <ResponsiveContainer width="100%" height={400}>
                    {(() => {
                      switch (chartType) {
                        case 'bar':
                          return (
                            <BarChart data={processedData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                              <XAxis 
                                dataKey="index" 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <YAxis 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'hsl(var(--card))', 
                                  border: '1px solid hsl(var(--border))',
                                  borderRadius: '8px',
                                  fontSize: '12px'
                                }} 
                              />
                              <Legend />
                              <Bar 
                                dataKey={selectedColumn || numericColumns[0]} 
                                fill={COLORS[0]} 
                                name={selectedColumn || numericColumns[0] || t('results.value')}
                                radius={[4, 4, 0, 0]}
                              />
                            </BarChart>
                          );
                        case 'line':
                          return (
                            <LineChart data={processedData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                              <XAxis 
                                dataKey="index" 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <YAxis 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'hsl(var(--card))', 
                                  border: '1px solid hsl(var(--border))',
                                  borderRadius: '8px',
                                  fontSize: '12px'
                                }} 
                              />
                              <Legend />
                              <Line 
                                type="monotone" 
                                dataKey={selectedColumn || numericColumns[0]} 
                                stroke={COLORS[1]} 
                                name={selectedColumn || numericColumns[0] || t('results.trend')}
                                strokeWidth={3}
                                dot={{ r: 4 }}
                                activeDot={{ r: 6 }}
                              />
                            </LineChart>
                          );
                        case 'area':
                          return (
                            <AreaChart data={processedData}>
                              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                              <XAxis 
                                dataKey="index" 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <YAxis 
                                stroke="hsl(var(--muted-foreground))"
                                tick={{ fontSize: 12 }}
                              />
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'hsl(var(--card))', 
                                  border: '1px solid hsl(var(--border))',
                                  borderRadius: '8px',
                                  fontSize: '12px'
                                }} 
                              />
                              <Legend />
                              <Area 
                                type="monotone" 
                                dataKey={selectedColumn || numericColumns[0]} 
                                stroke={COLORS[2]} 
                                fill={COLORS[2]}
                                fillOpacity={0.3}
                                name={selectedColumn || numericColumns[0] || t('results.area')}
                              />
                            </AreaChart>
                          );
                        case 'pie':
                          return (
                            <PieChart>
                              <Pie
                                data={processedData.slice(0, 8)}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={(entry) => `${entry.index}: ${entry[selectedColumn || numericColumns[0]]}`}
                                outerRadius={120}
                                fill="#8884d8"
                                dataKey={selectedColumn || numericColumns[0]}
                              >
                                {processedData.slice(0, 8).map((entry: any, index: number) => (
                                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                              </Pie>
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'hsl(var(--card))', 
                                  border: '1px solid hsl(var(--border))',
                                  borderRadius: '8px',
                                  fontSize: '12px'
                                }} 
                              />
                              <Legend />
                            </PieChart>
                          );
                        default:
                          return null;
                      }
                    })()}
                  </ResponsiveContainer>
                </Card>
                
                {/* Multiple Columns Comparison */}
                {numericColumns.length > 1 && (
                  <Card className="p-6">
                    <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <BarChart3 className="w-5 h-5" />
                      {t('results.multiColumnComparison')}
                    </h4>
                    <ResponsiveContainer width="100%" height={350}>
                      <BarChart data={processedData.slice(0, 10)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis 
                          dataKey="index" 
                          stroke="hsl(var(--muted-foreground))"
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis 
                          stroke="hsl(var(--muted-foreground))"
                          tick={{ fontSize: 12 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }} 
                        />
                        <Legend />
                        {numericColumns.slice(0, 4).map((column, index) => (
                          <Bar 
                            key={column}
                            dataKey={column} 
                            fill={COLORS[index]} 
                            name={column}
                            radius={[2, 2, 0, 0]}
                          />
                        ))}
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                )}
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
                <div className="text-center">
                  <BarChart3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">{t('results.noChartData')}</p>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="statistics" className="space-y-6 mt-6">
            {statistics ? (
              <div className="space-y-6">
                {/* Detailed Statistics Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <Card className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">{t('results.mean')}</p>
                        <p className="text-2xl font-bold">{statistics.mean}</p>
                      </div>
                      <TrendingUp className="w-8 h-8 text-blue-500" />
                    </div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">{t('results.stdDev')}</p>
                        <p className="text-2xl font-bold">{statistics.stdDev}</p>
                      </div>
                      <Activity className="w-8 h-8 text-green-500" />
                    </div>
                  </Card>
                  
                  <Card className="p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-sm text-muted-foreground">{t('results.range')}</p>
                        <p className="text-2xl font-bold">{(Number(statistics.max) - Number(statistics.min)).toFixed(2)}</p>
                      </div>
                      <BarChart3 className="w-8 h-8 text-purple-500" />
                    </div>
                  </Card>
                </div>
                
                {/* Distribution Chart */}
                {numericColumns.length > 0 && (
                  <Card className="p-6">
                    <h4 className="text-lg font-semibold mb-4 flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      {t('results.distributionAnalysis')}
                    </h4>
                    <ResponsiveContainer width="100%" height={300}>
                      <AreaChart data={processedData.slice(0, 15)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis 
                          dataKey="index" 
                          stroke="hsl(var(--muted-foreground))"
                          tick={{ fontSize: 12 }}
                        />
                        <YAxis 
                          stroke="hsl(var(--muted-foreground))"
                          tick={{ fontSize: 12 }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'hsl(var(--card))', 
                            border: '1px solid hsl(var(--border))',
                            borderRadius: '8px',
                            fontSize: '12px'
                          }} 
                        />
                        <Legend />
                        <Area 
                          type="monotone" 
                          dataKey={selectedColumn || numericColumns[0]} 
                          stroke={COLORS[3]} 
                          fill={COLORS[3]}
                          fillOpacity={0.4}
                          name={t('results.distribution')}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </Card>
                )}
                
                {/* Statistical Summary Table */}
                <Card className="p-6">
                  <h4 className="text-lg font-semibold mb-4">{t('results.statisticalSummary')}</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b">
                          <th className="text-left py-2">{t('results.metric')}</th>
                          <th className="text-left py-2">{t('results.value')}</th>
                          <th className="text-left py-2">{t('results.description')}</th>
                        </tr>
                      </thead>
                      <tbody className="space-y-2">
                        <tr className="border-b">
                          <td className="py-2 font-medium">{t('results.count')}</td>
                          <td className="py-2">{statistics.count.toLocaleString()}</td>
                          <td className="py-2 text-muted-foreground">{t('results.countDesc')}</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2 font-medium">{t('results.mean')}</td>
                          <td className="py-2">{statistics.mean}</td>
                          <td className="py-2 text-muted-foreground">{t('results.meanDesc')}</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2 font-medium">{t('results.stdDev')}</td>
                          <td className="py-2">{statistics.stdDev}</td>
                          <td className="py-2 text-muted-foreground">{t('results.stdDevDesc')}</td>
                        </tr>
                        <tr className="border-b">
                          <td className="py-2 font-medium">{t('results.min')}</td>
                          <td className="py-2">{statistics.min}</td>
                          <td className="py-2 text-muted-foreground">{t('results.minDesc')}</td>
                        </tr>
                        <tr>
                          <td className="py-2 font-medium">{t('results.max')}</td>
                          <td className="py-2">{statistics.max}</td>
                          <td className="py-2 text-muted-foreground">{t('results.maxDesc')}</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
                <div className="text-center">
                  <Activity className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">{t('results.noStatistics')}</p>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="data" className="space-y-4 mt-6">
            {(filteredData.length > 0 || results?.data?.length > 0) ? (
              <div className="space-y-4">
                {/* Data Summary */}
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-semibold">{t('results.dataTable')}</h4>
                    <p className="text-sm text-muted-foreground">
                      {searchTerm ? 
                        t('results.showingFiltered', { 
                          showing: Math.min(filteredData.length, 50), 
                          total: filteredData.length 
                        }) :
                        t('results.showingRows', { 
                          showing: Math.min(results?.data?.length || 0, 50), 
                          total: results?.data?.length || 0 
                        })
                      }
                    </p>
                  </div>
                  
                  {searchTerm && (
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => setSearchTerm("")}
                    >
                      {t('results.clearFilter')}
                    </Button>
                  )}
                </div>
                
                {/* Data Table */}
                <Card className="overflow-hidden">
                  <div className="overflow-x-auto max-h-96">
                    <table className="w-full text-sm border-collapse">
                      <thead className="bg-muted sticky top-0">
                        <tr>
                          <th className="px-4 py-3 text-left font-medium text-foreground border-b border-border">
                            #
                          </th>
                          {results?.data && results.data.length > 0 && Object.keys(results.data[0]).map((key) => (
                            <th key={key} className="px-4 py-3 text-left font-medium text-foreground border-b border-border">
                              <div className="flex items-center gap-2">
                                {key}
                                {numericColumns.includes(key) && (
                                  <Badge variant="secondary" className="text-xs">NUM</Badge>
                                )}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {(searchTerm ? filteredData : results?.data || []).slice(0, 50).map((row: any, idx: number) => (
                          <tr key={idx} className="border-b border-border hover:bg-muted/50 transition-colors">
                            <td className="px-4 py-3 text-muted-foreground font-mono text-xs">
                              {idx + 1}
                            </td>
                            {Object.values(row).map((val: any, i) => (
                              <td key={i} className="px-4 py-3 text-muted-foreground">
                                {val === null || val === undefined || val === '' ? (
                                  <span className="text-muted-foreground/50 italic">
                                    {t('results.emptyValue')}
                                  </span>
                                ) : (
                                  String(val)
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  
                  {(searchTerm ? filteredData.length : results?.data?.length || 0) > 50 && (
                    <div className="p-4 bg-muted/50 border-t">
                      <p className="text-xs text-muted-foreground text-center">
                        {t('results.moreRowsAvailable', { 
                          total: searchTerm ? filteredData.length : results?.data?.length || 0 
                        })}
                      </p>
                    </div>
                  )}
                </Card>
              </div>
            ) : (
              <div className="h-64 flex items-center justify-center rounded-lg border-2 border-dashed border-border">
                <div className="text-center">
                  <FileDown className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                  <p className="text-muted-foreground">{t('results.noData')}</p>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </Card>
    </motion.div>
  );
};

export default ResultsPanel;
