// API Service for VeriVio
import { AnalysisResult, UploadedData } from '@/contexts/AppContext';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001/api';

// API Response types
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

interface AnalysisRequest {
  data: any[];
  analysisType: string;
  fileName: string;
  options?: Record<string, any>;
}

interface AnalysisProgress {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message?: string;
}

class ApiService {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  // Generic request method
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = `${this.baseUrl}${endpoint}`;
    
    const defaultOptions: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, defaultOptions);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred'
      };
    }
  }

  // File upload
  async uploadFile(file: File): Promise<ApiResponse<UploadedData>> {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${this.baseUrl}/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Upload failed'
      };
    }
  }

  // Start analysis
  async startAnalysis(request: AnalysisRequest): Promise<ApiResponse<{ analysisId: string }>> {
    return this.request<{ analysisId: string }>('/analysis/start', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Get analysis progress
  async getAnalysisProgress(analysisId: string): Promise<ApiResponse<AnalysisProgress>> {
    return this.request<AnalysisProgress>(`/analysis/progress/${analysisId}`);
  }

  // Get analysis results
  async getAnalysisResults(analysisId: string): Promise<ApiResponse<AnalysisResult>> {
    return this.request<AnalysisResult>(`/analysis/results/${analysisId}`);
  }

  // Get available analysis types
  async getAnalysisTypes(): Promise<ApiResponse<Array<{
    id: string;
    name: string;
    description: string;
    category: string;
    requirements: string[];
  }>>> {
    return this.request<Array<{
      id: string;
      name: string;
      description: string;
      category: string;
      requirements: string[];
    }>>('/analysis/types');
  }

  // Export results
  async exportResults(
    analysisId: string, 
    format: 'pdf' | 'excel' | 'json'
  ): Promise<ApiResponse<{ downloadUrl: string }>> {
    return this.request<{ downloadUrl: string }>(`/analysis/export/${analysisId}`, {
      method: 'POST',
      body: JSON.stringify({ format }),
    });
  }

  // Get user analysis history
  async getAnalysisHistory(limit = 10): Promise<ApiResponse<AnalysisResult[]>> {
    return this.request<AnalysisResult[]>(`/analysis/history?limit=${limit}`);
  }

  // Delete analysis
  async deleteAnalysis(analysisId: string): Promise<ApiResponse<void>> {
    return this.request<void>(`/analysis/${analysisId}`, {
      method: 'DELETE',
    });
  }

  // Health check
  async healthCheck(): Promise<ApiResponse<{ status: string; timestamp: string }>> {
    return this.request<{ status: string; timestamp: string }>('/health');
  }

  // Get system statistics
  async getSystemStats(): Promise<ApiResponse<{
    totalAnalyses: number;
    activeUsers: number;
    systemLoad: number;
    uptime: string;
  }>> {
    return this.request<{
      totalAnalyses: number;
      activeUsers: number;
      systemLoad: number;
      uptime: string;
    }>('/stats');
  }
}

// Create singleton instance
export const apiService = new ApiService();

// Utility functions for common operations
export const uploadAndAnalyze = async (
  file: File,
  analysisType: string,
  options?: Record<string, any>
): Promise<{
  uploadResult: ApiResponse<UploadedData>;
  analysisResult?: ApiResponse<AnalysisResult>;
}> => {
  // Upload file first
  const uploadResult = await apiService.uploadFile(file);
  
  if (!uploadResult.success || !uploadResult.data) {
    return { uploadResult };
  }

  // Start analysis
  const analysisRequest: AnalysisRequest = {
    data: uploadResult.data.parsedData,
    analysisType,
    fileName: uploadResult.data.fileName,
    options,
  };

  const startResult = await apiService.startAnalysis(analysisRequest);
  
  if (!startResult.success || !startResult.data) {
    return { 
      uploadResult, 
      analysisResult: {
        success: false,
        error: 'Failed to start analysis'
      }
    };
  }

  // Poll for results
  const analysisId = startResult.data.analysisId;
  let attempts = 0;
  const maxAttempts = 30; // 30 seconds timeout
  
  while (attempts < maxAttempts) {
    await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
    
    const progressResult = await apiService.getAnalysisProgress(analysisId);
    
    if (progressResult.success && progressResult.data) {
      const { status } = progressResult.data;
      
      if (status === 'completed') {
        const analysisResult = await apiService.getAnalysisResults(analysisId);
        return { uploadResult, analysisResult };
      } else if (status === 'failed') {
        return {
          uploadResult,
          analysisResult: {
            success: false,
            error: progressResult.data.message || 'Analysis failed'
          }
        };
      }
    }
    
    attempts++;
  }

  return {
    uploadResult,
    analysisResult: {
      success: false,
      error: 'Analysis timeout'
    }
  };
};

export default apiService;