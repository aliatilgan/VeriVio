import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

// Virtual scrolling hook for large lists
export const useVirtualScroll = <T>(
  items: T[],
  itemHeight: number,
  containerHeight: number,
  overscan: number = 5
) => {
  const [scrollTop, setScrollTop] = useState(0);

  const visibleRange = useMemo(() => {
    const start = Math.floor(scrollTop / itemHeight);
    const end = Math.min(
      start + Math.ceil(containerHeight / itemHeight) + overscan,
      items.length
    );
    
    return {
      start: Math.max(0, start - overscan),
      end,
    };
  }, [scrollTop, itemHeight, containerHeight, overscan, items.length]);

  const visibleItems = useMemo(() => {
    return items.slice(visibleRange.start, visibleRange.end).map((item, index) => ({
      item,
      index: visibleRange.start + index,
    }));
  }, [items, visibleRange]);

  const totalHeight = items.length * itemHeight;
  const offsetY = visibleRange.start * itemHeight;

  const handleScroll = useCallback((e: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(e.currentTarget.scrollTop);
  }, []);

  return {
    visibleItems,
    totalHeight,
    offsetY,
    handleScroll,
  };
};

// Debounced value hook
export const useDebounce = <T>(value: T, delay: number): T => {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
};

// Throttled callback hook
export const useThrottle = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number
): T => {
  const lastRun = useRef(Date.now());

  return useCallback(
    ((...args) => {
      if (Date.now() - lastRun.current >= delay) {
        callback(...args);
        lastRun.current = Date.now();
      }
    }) as T,
    [callback, delay]
  );
};

// Intersection Observer hook for lazy loading
export const useIntersectionObserver = (
  options: IntersectionObserverInit = {}
) => {
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [entry, setEntry] = useState<IntersectionObserverEntry | null>(null);
  const elementRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const element = elementRef.current;
    if (!element) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsIntersecting(entry.isIntersecting);
        setEntry(entry);
      },
      options
    );

    observer.observe(element);

    return () => {
      observer.unobserve(element);
    };
  }, [options]);

  return { elementRef, isIntersecting, entry };
};

// Memory usage monitoring hook
export const useMemoryMonitor = () => {
  const [memoryInfo, setMemoryInfo] = useState<{
    usedJSHeapSize: number;
    totalJSHeapSize: number;
    jsHeapSizeLimit: number;
  } | null>(null);

  useEffect(() => {
    const updateMemoryInfo = () => {
      if ('memory' in performance) {
        const memory = (performance as any).memory;
        setMemoryInfo({
          usedJSHeapSize: memory.usedJSHeapSize,
          totalJSHeapSize: memory.totalJSHeapSize,
          jsHeapSizeLimit: memory.jsHeapSizeLimit,
        });
      }
    };

    updateMemoryInfo();
    const interval = setInterval(updateMemoryInfo, 5000);

    return () => clearInterval(interval);
  }, []);

  const memoryUsagePercentage = memoryInfo
    ? (memoryInfo.usedJSHeapSize / memoryInfo.jsHeapSizeLimit) * 100
    : 0;

  return {
    memoryInfo,
    memoryUsagePercentage,
    isHighMemoryUsage: memoryUsagePercentage > 80,
  };
};

// Performance timing hook
export const usePerformanceTiming = () => {
  const [timings, setTimings] = useState<{
    [key: string]: number;
  }>({});

  const startTiming = useCallback((key: string) => {
    setTimings(prev => ({
      ...prev,
      [`${key}_start`]: performance.now(),
    }));
  }, []);

  const endTiming = useCallback((key: string) => {
    setTimings(prev => {
      const startTime = prev[`${key}_start`];
      if (startTime) {
        return {
          ...prev,
          [key]: performance.now() - startTime,
        };
      }
      return prev;
    });
  }, []);

  const getTiming = useCallback((key: string) => {
    return timings[key] || 0;
  }, [timings]);

  return {
    startTiming,
    endTiming,
    getTiming,
    timings,
  };
};

// Chunked processing hook for large datasets
export const useChunkedProcessing = <T, R>(
  data: T[],
  processor: (chunk: T[]) => R[],
  chunkSize: number = 1000,
  delay: number = 10
) => {
  const [processedData, setProcessedData] = useState<R[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const processChunks = useCallback(async () => {
    if (data.length === 0) {
      setProcessedData([]);
      return;
    }

    setIsProcessing(true);
    setProgress(0);
    
    const results: R[] = [];
    const totalChunks = Math.ceil(data.length / chunkSize);

    for (let i = 0; i < data.length; i += chunkSize) {
      const chunk = data.slice(i, i + chunkSize);
      const chunkResults = processor(chunk);
      results.push(...chunkResults);

      const currentChunk = Math.floor(i / chunkSize) + 1;
      setProgress((currentChunk / totalChunks) * 100);

      // Allow UI to update
      if (delay > 0) {
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    setProcessedData(results);
    setIsProcessing(false);
  }, [data, processor, chunkSize, delay]);

  useEffect(() => {
    processChunks();
  }, [processChunks]);

  return {
    processedData,
    isProcessing,
    progress,
    reprocess: processChunks,
  };
};

// Image lazy loading hook
export const useLazyImage = (src: string, placeholder?: string) => {
  const [imageSrc, setImageSrc] = useState(placeholder || '');
  const [isLoaded, setIsLoaded] = useState(false);
  const [isError, setIsError] = useState(false);
  const { elementRef, isIntersecting } = useIntersectionObserver({
    threshold: 0.1,
  });

  useEffect(() => {
    if (isIntersecting && src) {
      const img = new Image();
      
      img.onload = () => {
        setImageSrc(src);
        setIsLoaded(true);
      };
      
      img.onerror = () => {
        setIsError(true);
      };
      
      img.src = src;
    }
  }, [isIntersecting, src]);

  return {
    elementRef,
    imageSrc,
    isLoaded,
    isError,
  };
};

// Optimized search hook
export const useOptimizedSearch = <T>(
  items: T[],
  searchFields: (keyof T)[],
  searchTerm: string,
  options: {
    debounceMs?: number;
    caseSensitive?: boolean;
    exactMatch?: boolean;
    maxResults?: number;
  } = {}
) => {
  const {
    debounceMs = 300,
    caseSensitive = false,
    exactMatch = false,
    maxResults = 100,
  } = options;

  const debouncedSearchTerm = useDebounce(searchTerm, debounceMs);

  const searchResults = useMemo(() => {
    if (!debouncedSearchTerm.trim()) {
      return items.slice(0, maxResults);
    }

    const term = caseSensitive 
      ? debouncedSearchTerm 
      : debouncedSearchTerm.toLowerCase();

    const results = items.filter(item => {
      return searchFields.some(field => {
        const fieldValue = String(item[field]);
        const value = caseSensitive ? fieldValue : fieldValue.toLowerCase();
        
        return exactMatch 
          ? value === term 
          : value.includes(term);
      });
    });

    return results.slice(0, maxResults);
  }, [items, searchFields, debouncedSearchTerm, caseSensitive, exactMatch, maxResults]);

  return {
    searchResults,
    isSearching: searchTerm !== debouncedSearchTerm,
    resultCount: searchResults.length,
  };
};

// Bundle size monitoring
export const useBundleSize = () => {
  const [bundleInfo, setBundleInfo] = useState<{
    totalSize: number;
    gzippedSize: number;
    loadTime: number;
  } | null>(null);

  useEffect(() => {
    // Estimate bundle size based on performance timing
    const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    
    if (navigation) {
      const loadTime = navigation.loadEventEnd - navigation.startTime;
      
      // Rough estimation - in a real app, you'd get this from build tools
      setBundleInfo({
        totalSize: 0, // Would be provided by webpack-bundle-analyzer
        gzippedSize: 0, // Would be provided by build tools
        loadTime,
      });
    }
  }, []);

  return bundleInfo;
};

// Resource preloading hook
export const useResourcePreloader = () => {
  const preloadedResources = useRef(new Set<string>());

  const preloadImage = useCallback((src: string) => {
    if (preloadedResources.current.has(src)) return;
    
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'image';
    link.href = src;
    document.head.appendChild(link);
    
    preloadedResources.current.add(src);
  }, []);

  const preloadScript = useCallback((src: string) => {
    if (preloadedResources.current.has(src)) return;
    
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'script';
    link.href = src;
    document.head.appendChild(link);
    
    preloadedResources.current.add(src);
  }, []);

  const preloadStylesheet = useCallback((href: string) => {
    if (preloadedResources.current.has(href)) return;
    
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'style';
    link.href = href;
    document.head.appendChild(link);
    
    preloadedResources.current.add(href);
  }, []);

  return {
    preloadImage,
    preloadScript,
    preloadStylesheet,
  };
};