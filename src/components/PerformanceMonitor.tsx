import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Activity, 
  Zap, 
  Clock, 
  Database, 
  Wifi, 
  AlertTriangle,
  CheckCircle,
  X,
  BarChart3,
  TrendingUp,
  TrendingDown
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { useMemoryMonitor, usePerformanceTiming } from '@/hooks/usePerformance';

interface PerformanceMetric {
  name: string;
  value: number;
  unit: string;
  status: 'good' | 'warning' | 'critical';
  threshold: {
    warning: number;
    critical: number;
  };
}

interface NetworkInfo {
  effectiveType: string;
  downlink: number;
  rtt: number;
}

export const PerformanceMonitor: React.FC = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null);
  const [fps, setFps] = useState(0);
  
  const { memoryInfo, memoryUsagePercentage, isHighMemoryUsage } = useMemoryMonitor();
  const { timings, startTiming, endTiming } = usePerformanceTiming();

  // FPS monitoring
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animationId: number;

    const measureFPS = () => {
      frameCount++;
      const currentTime = performance.now();
      
      if (currentTime - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (currentTime - lastTime)));
        frameCount = 0;
        lastTime = currentTime;
      }
      
      animationId = requestAnimationFrame(measureFPS);
    };

    if (isVisible) {
      animationId = requestAnimationFrame(measureFPS);
    }

    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
    };
  }, [isVisible]);

  // Network information
  useEffect(() => {
    if ('connection' in navigator) {
      const connection = (navigator as any).connection;
      setNetworkInfo({
        effectiveType: connection.effectiveType || 'unknown',
        downlink: connection.downlink || 0,
        rtt: connection.rtt || 0,
      });

      const updateNetworkInfo = () => {
        setNetworkInfo({
          effectiveType: connection.effectiveType || 'unknown',
          downlink: connection.downlink || 0,
          rtt: connection.rtt || 0,
        });
      };

      connection.addEventListener('change', updateNetworkInfo);
      return () => connection.removeEventListener('change', updateNetworkInfo);
    }
  }, []);

  // Update metrics
  useEffect(() => {
    const updateMetrics = () => {
      const newMetrics: PerformanceMetric[] = [
        {
          name: 'FPS',
          value: fps,
          unit: 'fps',
          status: fps >= 55 ? 'good' : fps >= 30 ? 'warning' : 'critical',
          threshold: { warning: 30, critical: 15 }
        },
        {
          name: 'Memory Usage',
          value: memoryUsagePercentage,
          unit: '%',
          status: memoryUsagePercentage < 60 ? 'good' : memoryUsagePercentage < 80 ? 'warning' : 'critical',
          threshold: { warning: 60, critical: 80 }
        }
      ];

      if (networkInfo) {
        newMetrics.push({
          name: 'Network RTT',
          value: networkInfo.rtt,
          unit: 'ms',
          status: networkInfo.rtt < 100 ? 'good' : networkInfo.rtt < 300 ? 'warning' : 'critical',
          threshold: { warning: 100, critical: 300 }
        });
      }

      // Add timing metrics
      Object.entries(timings).forEach(([key, value]) => {
        if (!key.endsWith('_start') && value > 0) {
          newMetrics.push({
            name: key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()),
            value: value,
            unit: 'ms',
            status: value < 100 ? 'good' : value < 500 ? 'warning' : 'critical',
            threshold: { warning: 100, critical: 500 }
          });
        }
      });

      setMetrics(newMetrics);
    };

    if (isVisible) {
      updateMetrics();
      const interval = setInterval(updateMetrics, 1000);
      return () => clearInterval(interval);
    }
  }, [isVisible, fps, memoryUsagePercentage, networkInfo, timings]);

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'good':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-yellow-500" />;
      case 'critical':
        return <AlertTriangle className="w-4 h-4 text-red-500" />;
      default:
        return <Activity className="w-4 h-4 text-gray-500" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'good':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'warning':
        return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'critical':
        return 'bg-red-100 text-red-800 border-red-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case 'good':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'critical':
        return 'bg-red-500';
      default:
        return 'bg-gray-500';
    }
  };

  if (!isVisible) {
    return (
      <Button
        onClick={() => setIsVisible(true)}
        variant="outline"
        size="sm"
        className="fixed bottom-4 left-4 z-40 bg-white shadow-lg"
      >
        <Activity className="w-4 h-4 mr-2" />
        Performance
      </Button>
    );
  }

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, x: -300 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -300 }}
        transition={{ duration: 0.3 }}
        className="fixed bottom-4 left-4 z-40 w-80"
      >
        <Card className="shadow-2xl border-2">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5" />
                Performance Monitor
              </CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsVisible(false)}
                className="p-1"
              >
                <X className="w-4 h-4" />
              </Button>
            </div>
          </CardHeader>
          
          <CardContent className="space-y-4">
            {/* Overall Status */}
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1">
                {isHighMemoryUsage ? (
                  <TrendingDown className="w-4 h-4 text-red-500" />
                ) : (
                  <TrendingUp className="w-4 h-4 text-green-500" />
                )}
                <span className="text-sm font-medium">
                  {isHighMemoryUsage ? 'Performance Issues' : 'Good Performance'}
                </span>
              </div>
            </div>

            {/* Metrics */}
            <div className="space-y-3">
              {metrics.map((metric, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getStatusIcon(metric.status)}
                      <span className="text-sm font-medium">{metric.name}</span>
                    </div>
                    <Badge variant="outline" className={getStatusColor(metric.status)}>
                      {metric.value.toFixed(1)} {metric.unit}
                    </Badge>
                  </div>
                  
                  <div className="space-y-1">
                    <Progress 
                      value={Math.min((metric.value / (metric.threshold.critical * 1.5)) * 100, 100)}
                      className="h-2"
                    />
                    <div className="flex justify-between text-xs text-gray-500">
                      <span>0</span>
                      <span className="text-yellow-600">{metric.threshold.warning}</span>
                      <span className="text-red-600">{metric.threshold.critical}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Network Info */}
            {networkInfo && (
              <div className="pt-3 border-t border-gray-200">
                <div className="flex items-center gap-2 mb-2">
                  <Wifi className="w-4 h-4 text-blue-500" />
                  <span className="text-sm font-medium">Network</span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Type:</span>
                    <span className="ml-1 font-medium">{networkInfo.effectiveType}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Speed:</span>
                    <span className="ml-1 font-medium">{networkInfo.downlink} Mbps</span>
                  </div>
                </div>
              </div>
            )}

            {/* Memory Details */}
            {memoryInfo && (
              <div className="pt-3 border-t border-gray-200">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="w-4 h-4 text-purple-500" />
                  <span className="text-sm font-medium">Memory</span>
                </div>
                <div className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Used:</span>
                    <span className="font-medium">
                      {(memoryInfo.usedJSHeapSize / 1024 / 1024).toFixed(1)} MB
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Total:</span>
                    <span className="font-medium">
                      {(memoryInfo.totalJSHeapSize / 1024 / 1024).toFixed(1)} MB
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Limit:</span>
                    <span className="font-medium">
                      {(memoryInfo.jsHeapSizeLimit / 1024 / 1024).toFixed(1)} MB
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Performance Tips */}
            {isHighMemoryUsage && (
              <div className="pt-3 border-t border-gray-200">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-orange-500" />
                  <span className="text-sm font-medium text-orange-700">Optimization Tips</span>
                </div>
                <ul className="text-xs text-gray-600 space-y-1">
                  <li>• Close unused browser tabs</li>
                  <li>• Reduce data visualization complexity</li>
                  <li>• Clear browser cache</li>
                  <li>• Use smaller datasets for testing</li>
                </ul>
              </div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </AnimatePresence>
  );
};