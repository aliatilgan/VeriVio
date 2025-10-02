import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Menu, 
  X, 
  ChevronLeft, 
  ChevronRight,
  Monitor,
  Tablet,
  Smartphone
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

// Breakpoint definitions
export const breakpoints = {
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  '2xl': 1536,
} as const;

export type Breakpoint = keyof typeof breakpoints;
export type DeviceType = 'mobile' | 'tablet' | 'desktop';

// Hook for responsive behavior
export const useResponsive = () => {
  const [windowSize, setWindowSize] = useState({
    width: typeof window !== 'undefined' ? window.innerWidth : 1024,
    height: typeof window !== 'undefined' ? window.innerHeight : 768,
  });

  const [deviceType, setDeviceType] = useState<DeviceType>('desktop');
  const [currentBreakpoint, setCurrentBreakpoint] = useState<Breakpoint>('lg');

  useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setWindowSize({ width, height });

      // Determine device type
      if (width < breakpoints.md) {
        setDeviceType('mobile');
      } else if (width < breakpoints.lg) {
        setDeviceType('tablet');
      } else {
        setDeviceType('desktop');
      }

      // Determine current breakpoint
      if (width >= breakpoints['2xl']) {
        setCurrentBreakpoint('2xl');
      } else if (width >= breakpoints.xl) {
        setCurrentBreakpoint('xl');
      } else if (width >= breakpoints.lg) {
        setCurrentBreakpoint('lg');
      } else if (width >= breakpoints.md) {
        setCurrentBreakpoint('md');
      } else {
        setCurrentBreakpoint('sm');
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const isMobile = deviceType === 'mobile';
  const isTablet = deviceType === 'tablet';
  const isDesktop = deviceType === 'desktop';

  const isBreakpoint = (bp: Breakpoint) => {
    return windowSize.width >= breakpoints[bp];
  };

  return {
    windowSize,
    deviceType,
    currentBreakpoint,
    isMobile,
    isTablet,
    isDesktop,
    isBreakpoint,
  };
};

interface ResponsiveContainerProps {
  children: React.ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export const ResponsiveContainer: React.FC<ResponsiveContainerProps> = ({
  children,
  className,
  maxWidth = 'xl',
  padding = 'md',
}) => {
  const maxWidthClasses = {
    sm: 'max-w-sm',
    md: 'max-w-md',
    lg: 'max-w-lg',
    xl: 'max-w-7xl',
    '2xl': 'max-w-none',
    full: 'max-w-full',
  };

  const paddingClasses = {
    none: '',
    sm: 'px-4 py-2',
    md: 'px-6 py-4',
    lg: 'px-8 py-6',
  };

  return (
    <div
      className={cn(
        'mx-auto w-full',
        maxWidthClasses[maxWidth],
        paddingClasses[padding],
        className
      )}
    >
      {children}
    </div>
  );
};

interface ResponsiveGridProps {
  children: React.ReactNode;
  className?: string;
  cols?: {
    default?: number;
    sm?: number;
    md?: number;
    lg?: number;
    xl?: number;
    '2xl'?: number;
  };
  gap?: 'sm' | 'md' | 'lg' | 'xl';
}

export const ResponsiveGrid: React.FC<ResponsiveGridProps> = ({
  children,
  className,
  cols = { default: 1, md: 2, lg: 3 },
  gap = 'md',
}) => {
  const gapClasses = {
    sm: 'gap-2',
    md: 'gap-4',
    lg: 'gap-6',
    xl: 'gap-8',
  };

  const getGridCols = () => {
    const classes = [];
    
    if (cols.default) classes.push(`grid-cols-${cols.default}`);
    if (cols.sm) classes.push(`sm:grid-cols-${cols.sm}`);
    if (cols.md) classes.push(`md:grid-cols-${cols.md}`);
    if (cols.lg) classes.push(`lg:grid-cols-${cols.lg}`);
    if (cols.xl) classes.push(`xl:grid-cols-${cols.xl}`);
    if (cols['2xl']) classes.push(`2xl:grid-cols-${cols['2xl']}`);
    
    return classes.join(' ');
  };

  return (
    <div
      className={cn(
        'grid',
        getGridCols(),
        gapClasses[gap],
        className
      )}
    >
      {children}
    </div>
  );
};

interface SidebarLayoutProps {
  children: React.ReactNode;
  sidebar: React.ReactNode;
  sidebarWidth?: 'sm' | 'md' | 'lg';
  collapsible?: boolean;
  defaultCollapsed?: boolean;
  className?: string;
}

export const SidebarLayout: React.FC<SidebarLayoutProps> = ({
  children,
  sidebar,
  sidebarWidth = 'md',
  collapsible = true,
  defaultCollapsed = false,
  className,
}) => {
  const { isMobile, isTablet } = useResponsive();
  const [isCollapsed, setIsCollapsed] = useState(defaultCollapsed);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const sidebarWidthClasses = {
    sm: 'w-64',
    md: 'w-80',
    lg: 'w-96',
  };

  const collapsedWidth = 'w-16';

  useEffect(() => {
    if (isMobile) {
      setIsMobileMenuOpen(false);
    }
  }, [isMobile]);

  const toggleSidebar = () => {
    if (isMobile) {
      setIsMobileMenuOpen(!isMobileMenuOpen);
    } else {
      setIsCollapsed(!isCollapsed);
    }
  };

  return (
    <div className={cn('flex h-screen bg-gray-50', className)}>
      {/* Mobile Menu Button */}
      {isMobile && (
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleSidebar}
          className="fixed top-4 left-4 z-50 bg-white shadow-md"
        >
          <Menu className="w-5 h-5" />
        </Button>
      )}

      {/* Sidebar */}
      <AnimatePresence>
        {((!isMobile && !isTablet) || isMobileMenuOpen) && (
          <>
            {/* Mobile Backdrop */}
            {isMobile && isMobileMenuOpen && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setIsMobileMenuOpen(false)}
                className="fixed inset-0 bg-black/50 z-40"
              />
            )}

            {/* Sidebar Content */}
            <motion.aside
              initial={isMobile ? { x: -320 } : { width: 0 }}
              animate={
                isMobile
                  ? { x: 0 }
                  : { width: isCollapsed ? 64 : sidebarWidthClasses[sidebarWidth].replace('w-', '') }
              }
              exit={isMobile ? { x: -320 } : { width: 0 }}
              transition={{ duration: 0.3, ease: 'easeInOut' }}
              className={cn(
                'bg-white border-r border-gray-200 flex flex-col relative z-50',
                isMobile
                  ? 'fixed left-0 top-0 h-full w-80'
                  : isCollapsed
                  ? collapsedWidth
                  : sidebarWidthClasses[sidebarWidth]
              )}
            >
              {/* Collapse Button */}
              {collapsible && !isMobile && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleSidebar}
                  className="absolute -right-3 top-6 bg-white border border-gray-200 rounded-full p-1 shadow-sm z-10"
                >
                  {isCollapsed ? (
                    <ChevronRight className="w-4 h-4" />
                  ) : (
                    <ChevronLeft className="w-4 h-4" />
                  )}
                </Button>
              )}

              {/* Mobile Close Button */}
              {isMobile && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsMobileMenuOpen(false)}
                  className="absolute right-4 top-4"
                >
                  <X className="w-5 h-5" />
                </Button>
              )}

              {/* Sidebar Content */}
              <div className="flex-1 overflow-hidden">
                {sidebar}
              </div>
            </motion.aside>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        {children}
      </main>
    </div>
  );
};

interface DevicePreviewProps {
  children: React.ReactNode;
  device: DeviceType;
  className?: string;
}

export const DevicePreview: React.FC<DevicePreviewProps> = ({
  children,
  device,
  className,
}) => {
  const getDeviceStyles = () => {
    switch (device) {
      case 'mobile':
        return {
          width: '375px',
          height: '667px',
          icon: <Smartphone className="w-4 h-4" />,
        };
      case 'tablet':
        return {
          width: '768px',
          height: '1024px',
          icon: <Tablet className="w-4 h-4" />,
        };
      case 'desktop':
        return {
          width: '1200px',
          height: '800px',
          icon: <Monitor className="w-4 h-4" />,
        };
    }
  };

  const deviceStyles = getDeviceStyles();

  return (
    <div className={cn('flex flex-col items-center space-y-4', className)}>
      <div className="flex items-center gap-2 text-sm text-gray-600">
        {deviceStyles.icon}
        <span className="capitalize">{device}</span>
        <span className="text-gray-400">
          {deviceStyles.width} × {deviceStyles.height}
        </span>
      </div>
      
      <div
        className="border border-gray-300 rounded-lg overflow-hidden shadow-lg bg-white"
        style={{
          width: deviceStyles.width,
          height: deviceStyles.height,
          maxWidth: '100%',
          maxHeight: '80vh',
        }}
      >
        <div className="w-full h-full overflow-auto">
          {children}
        </div>
      </div>
    </div>
  );
};

// Utility component for responsive text
interface ResponsiveTextProps {
  children: React.ReactNode;
  className?: string;
  size?: {
    default?: string;
    sm?: string;
    md?: string;
    lg?: string;
    xl?: string;
    '2xl'?: string;
  };
}

export const ResponsiveText: React.FC<ResponsiveTextProps> = ({
  children,
  className,
  size = { default: 'text-base', md: 'md:text-lg', lg: 'lg:text-xl' },
}) => {
  const getSizeClasses = () => {
    const classes = [];
    
    if (size.default) classes.push(size.default);
    if (size.sm) classes.push(`sm:${size.sm}`);
    if (size.md) classes.push(`md:${size.md}`);
    if (size.lg) classes.push(`lg:${size.lg}`);
    if (size.xl) classes.push(`xl:${size.xl}`);
    if (size['2xl']) classes.push(`2xl:${size['2xl']}`);
    
    return classes.join(' ');
  };

  return (
    <div className={cn(getSizeClasses(), className)}>
      {children}
    </div>
  );
};