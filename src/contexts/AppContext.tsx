import React, { createContext, useContext, useReducer, ReactNode, useEffect } from 'react';
import { toast } from 'sonner';

// Types
export interface UploadedData {
  fileName: string;
  size: number;
  parsedData: any[];
  columns: string[];
  rowCount: number;
  uploadedAt: Date;
  fileType: string;
}

export interface NotificationState {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  duration?: number;
}

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: 'tr' | 'en';
  autoSave: boolean;
  showTutorial: boolean;
  chartPreferences: {
    defaultType: 'bar' | 'line' | 'area' | 'pie';
    colorScheme: string;
    showLegend: boolean;
  };
}

export interface AnalysisResult {
  type: string;
  data: any;
  summary: any;
  charts?: any[];
  statistics?: any;
  timestamp: Date;
  id: string;
  fileName: string;
}

export interface AnalysisHistory {
  id: string;
  fileName: string;
  analysisType: string;
  timestamp: Date;
  results: AnalysisResult;
}

interface AppState {
  uploadedData: UploadedData | null;
  selectedAnalysis: string;
  analysisResults: AnalysisResult | null;
  analysisHistory: AnalysisHistory[];
  isLoading: boolean;
  error: string | null;
  notifications: NotificationState[];
  userPreferences: UserPreferences;
  isOnline: boolean;
  currentStep: 'upload' | 'analysis' | 'results';
}

type AppAction =
  | { type: 'SET_UPLOADED_DATA'; payload: UploadedData }
  | { type: 'SET_SELECTED_ANALYSIS'; payload: string }
  | { type: 'SET_ANALYSIS_RESULTS'; payload: AnalysisResult }
  | { type: 'ADD_TO_HISTORY'; payload: AnalysisHistory }
  | { type: 'CLEAR_HISTORY' }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'ADD_NOTIFICATION'; payload: NotificationState }
  | { type: 'REMOVE_NOTIFICATION'; payload: string }
  | { type: 'CLEAR_NOTIFICATIONS' }
  | { type: 'UPDATE_PREFERENCES'; payload: Partial<UserPreferences> }
  | { type: 'SET_ONLINE_STATUS'; payload: boolean }
  | { type: 'SET_CURRENT_STEP'; payload: 'upload' | 'analysis' | 'results' }
  | { type: 'CLEAR_DATA' }
  | { type: 'CLEAR_RESULTS' }
  | { type: 'RESET_APP' };

const initialState: AppState = {
  uploadedData: null,
  selectedAnalysis: '',
  analysisResults: null,
  analysisHistory: [],
  isLoading: false,
  error: null,
  notifications: [],
  userPreferences: {
    theme: 'system',
    language: 'tr',
    autoSave: true,
    showTutorial: true,
    chartPreferences: {
      defaultType: 'bar',
      colorScheme: 'default',
      showLegend: true
    }
  },
  isOnline: navigator.onLine,
  currentStep: 'upload'
};

const appReducer = (state: AppState, action: AppAction): AppState => {
  switch (action.type) {
    case 'SET_UPLOADED_DATA':
      return {
        ...state,
        uploadedData: action.payload,
        error: null,
        currentStep: 'analysis'
      };
    case 'SET_SELECTED_ANALYSIS':
      return {
        ...state,
        selectedAnalysis: action.payload
      };
    case 'SET_ANALYSIS_RESULTS':
      return {
        ...state,
        analysisResults: action.payload,
        isLoading: false,
        error: null,
        currentStep: 'results'
      };
    case 'ADD_TO_HISTORY':
      return {
        ...state,
        analysisHistory: [action.payload, ...state.analysisHistory.slice(0, 9)] // Keep last 10
      };
    case 'CLEAR_HISTORY':
      return {
        ...state,
        analysisHistory: []
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false
      };
    case 'ADD_NOTIFICATION':
      return {
        ...state,
        notifications: [action.payload, ...state.notifications]
      };
    case 'REMOVE_NOTIFICATION':
      return {
        ...state,
        notifications: state.notifications.filter(n => n.id !== action.payload)
      };
    case 'CLEAR_NOTIFICATIONS':
      return {
        ...state,
        notifications: []
      };
    case 'UPDATE_PREFERENCES':
      return {
        ...state,
        userPreferences: {
          ...state.userPreferences,
          ...action.payload
        }
      };
    case 'SET_ONLINE_STATUS':
      return {
        ...state,
        isOnline: action.payload
      };
    case 'SET_CURRENT_STEP':
      return {
        ...state,
        currentStep: action.payload
      };
    case 'CLEAR_DATA':
      return {
        ...state,
        uploadedData: null,
        selectedAnalysis: '',
        analysisResults: null,
        error: null,
        currentStep: 'upload'
      };
    case 'CLEAR_RESULTS':
      return {
        ...state,
        analysisResults: null,
        selectedAnalysis: '',
        error: null,
        currentStep: 'analysis'
      };
    case 'RESET_APP':
      return {
        ...initialState,
        userPreferences: state.userPreferences // Keep user preferences
      };
    default:
      return state;
  }
};

interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  // Helper functions
  setUploadedData: (data: UploadedData) => void;
  setSelectedAnalysis: (analysis: string) => void;
  setAnalysisResults: (results: AnalysisResult) => void;
  addToHistory: (history: AnalysisHistory) => void;
  clearHistory: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  showNotification: (type: NotificationState['type'], title: string, message: string, duration?: number) => void;
  removeNotification: (id: string) => void;
  clearNotifications: () => void;
  updatePreferences: (preferences: Partial<UserPreferences>) => void;
  setLanguage: (language: 'tr' | 'en') => void;
  setTheme: (theme: 'light' | 'dark' | 'system') => void;
  setCurrentStep: (step: 'upload' | 'analysis' | 'results') => void;
  clearData: () => void;
  clearResults: () => void;
  resetApp: () => void;
  // API functions
  performAnalysis: (analysisType: string, data: any[]) => Promise<AnalysisResult>;
  exportResults: (format: 'pdf' | 'excel' | 'json') => Promise<void>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider: React.FC<AppProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(appReducer, initialState);

  // Online status monitoring
  useEffect(() => {
    const handleOnline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: true });
    const handleOffline = () => dispatch({ type: 'SET_ONLINE_STATUS', payload: false });

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  // Auto-remove notifications
  useEffect(() => {
    const timers: NodeJS.Timeout[] = [];
    
    state.notifications.forEach(notification => {
      if (notification.duration && notification.duration > 0) {
        const timer = setTimeout(() => {
          removeNotification(notification.id);
        }, notification.duration);
        timers.push(timer);
      }
    });

    return () => {
      timers.forEach(timer => clearTimeout(timer));
    };
  }, [state.notifications]);

  // Helper functions
  const setUploadedData = (data: UploadedData) => {
    dispatch({ type: 'SET_UPLOADED_DATA', payload: data });
    showNotification('success', 'Başarılı', 'Veri başarıyla yüklendi', 3000);
  };

  const setSelectedAnalysis = (analysis: string) => {
    dispatch({ type: 'SET_SELECTED_ANALYSIS', payload: analysis });
  };

  const setAnalysisResults = (results: AnalysisResult) => {
    dispatch({ type: 'SET_ANALYSIS_RESULTS', payload: results });
    
    // Add to history
    const historyItem: AnalysisHistory = {
      id: results.id,
      fileName: results.fileName,
      analysisType: results.type,
      timestamp: results.timestamp,
      results
    };
    dispatch({ type: 'ADD_TO_HISTORY', payload: historyItem });
    
    showNotification('success', 'Analiz Tamamlandı', 'Analiz sonuçları hazır', 5000);
  };

  const addToHistory = (history: AnalysisHistory) => {
    dispatch({ type: 'ADD_TO_HISTORY', payload: history });
  };

  const clearHistory = () => {
    dispatch({ type: 'CLEAR_HISTORY' });
  };

  const setLoading = (loading: boolean) => {
    dispatch({ type: 'SET_LOADING', payload: loading });
  };

  const setError = (error: string | null) => {
    dispatch({ type: 'SET_ERROR', payload: error });
    if (error) {
      showNotification('error', 'Hata', error, 5000);
    }
  };

  const showNotification = (type: NotificationState['type'], title: string, message: string, duration = 4000) => {
    const notification: NotificationState = {
      id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
      type,
      title,
      message,
      timestamp: new Date(),
      duration
    };
    
    dispatch({ type: 'ADD_NOTIFICATION', payload: notification });
    
    // Also show toast
    switch (type) {
      case 'success':
        toast.success(title, { description: message });
        break;
      case 'error':
        toast.error(title, { description: message });
        break;
      case 'warning':
        toast.warning(title, { description: message });
        break;
      case 'info':
        toast.info(title, { description: message });
        break;
    }
  };

  const removeNotification = (id: string) => {
    dispatch({ type: 'REMOVE_NOTIFICATION', payload: id });
  };

  const clearNotifications = () => {
    dispatch({ type: 'CLEAR_NOTIFICATIONS' });
  };

  const updatePreferences = (preferences: Partial<UserPreferences>) => {
    dispatch({ type: 'UPDATE_PREFERENCES', payload: preferences });
    
    // Save to localStorage
    try {
      const updatedPrefs = { ...state.userPreferences, ...preferences };
      localStorage.setItem('veriVio_preferences', JSON.stringify(updatedPrefs));
    } catch (error) {
      console.warn('Failed to save preferences to localStorage:', error);
    }
  };

  const setLanguage = (language: 'tr' | 'en') => {
    updatePreferences({ language });
  };

  const setTheme = (theme: 'light' | 'dark' | 'system') => {
    updatePreferences({ theme });
  };

  const setCurrentStep = (step: 'upload' | 'analysis' | 'results') => {
    dispatch({ type: 'SET_CURRENT_STEP', payload: step });
  };

  const clearData = () => {
    dispatch({ type: 'CLEAR_DATA' });
    showNotification('info', 'Temizlendi', 'Tüm veriler temizlendi', 2000);
  };

  const clearResults = () => {
    dispatch({ type: 'CLEAR_RESULTS' });
  };

  const resetApp = () => {
    dispatch({ type: 'RESET_APP' });
    showNotification('info', 'Sıfırlandı', 'Uygulama başlangıç durumuna döndürüldü', 2000);
  };

  // API functions
  const performAnalysis = async (analysisType: string, data: any[]): Promise<AnalysisResult> => {
    setLoading(true);
    
    try {
      // Simulate API call - replace with actual API endpoint
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock analysis result
      const result: AnalysisResult = {
        id: Date.now().toString() + Math.random().toString(36).substr(2, 9),
        type: analysisType,
        fileName: state.uploadedData?.fileName || 'unknown',
        data: data.slice(0, 100), // Sample data
        summary: {
          totalRows: data.length,
          totalColumns: Object.keys(data[0] || {}).length,
          analysisType,
          completedAt: new Date().toISOString()
        },
        statistics: {
          mean: '45.67',
          stdDev: '12.34',
          min: '10.00',
          max: '89.50',
          count: data.length
        },
        charts: [
          { type: 'bar', data: data.slice(0, 10) },
          { type: 'line', data: data.slice(0, 15) }
        ],
        timestamp: new Date()
      };
      
      setAnalysisResults(result);
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Analiz sırasında bir hata oluştu';
      setError(errorMessage);
      throw error;
    } finally {
      setLoading(false);
    }
  };

  const exportResults = async (format: 'pdf' | 'excel' | 'json'): Promise<void> => {
    if (!state.analysisResults) {
      showNotification('warning', 'Uyarı', 'Dışa aktarılacak sonuç bulunamadı', 3000);
      return;
    }

    setLoading(true);
    
    try {
      // Simulate export process
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      showNotification('success', 'Başarılı', `Sonuçlar ${format.toUpperCase()} formatında dışa aktarıldı`, 3000);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Dışa aktarma sırasında bir hata oluştu';
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Load preferences from localStorage on mount
  useEffect(() => {
    try {
      const savedPreferences = localStorage.getItem('veriVio_preferences');
      if (savedPreferences) {
        const preferences = JSON.parse(savedPreferences);
        dispatch({ type: 'UPDATE_PREFERENCES', payload: preferences });
      }
    } catch (error) {
      console.warn('Failed to load preferences from localStorage:', error);
    }
  }, []);

  const value: AppContextType = {
    state,
    dispatch,
    setUploadedData,
    setSelectedAnalysis,
    setAnalysisResults,
    addToHistory,
    clearHistory,
    setLoading,
    setError,
    showNotification,
    removeNotification,
    clearNotifications,
    updatePreferences,
    setLanguage,
    setTheme,
    setCurrentStep,
    clearData,
    clearResults,
    resetApp,
    performAnalysis,
    exportResults
  };

  return (
    <AppContext.Provider value={value}>
      {children}
    </AppContext.Provider>
  );
};