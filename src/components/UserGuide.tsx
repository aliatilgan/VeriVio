import React, { useState, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  X, 
  ChevronLeft, 
  ChevronRight, 
  Play, 
  Pause, 
  RotateCcw,
  BookOpen,
  FileText,
  BarChart3,
  Settings,
  HelpCircle,
  CheckCircle,
  ArrowRight,
  Upload,
  MousePointer,
  Eye
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { useApp } from '@/contexts/AppContext';

interface GuideStep {
  id: string;
  title: string;
  description: string;
  content: string;
  target?: string;
  action?: string;
  duration?: number;
  image?: string;
  video?: string;
}

interface GuideSection {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  steps: GuideStep[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: number;
}

interface UserGuideProps {
  isOpen: boolean;
  onClose: () => void;
  startWithTutorial?: boolean;
}

export const UserGuide: React.FC<UserGuideProps> = ({ 
  isOpen, 
  onClose, 
  startWithTutorial = false 
}) => {
  const { t } = useTranslation();
  const { state, updatePreferences } = useApp();
  const { userPreferences } = state;
  const [currentSection, setCurrentSection] = useState<string>('overview');
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState<string>('');

  const guideSections: GuideSection[] = [
    {
      id: 'overview',
      title: t('guide.overview.title'),
      description: t('guide.overview.description'),
      icon: <BookOpen className="w-5 h-5" />,
      difficulty: 'beginner',
      estimatedTime: 5,
      steps: [
        {
          id: 'welcome',
          title: t('guide.overview.welcome.title'),
          description: t('guide.overview.welcome.description'),
          content: t('guide.overview.welcome.content'),
          duration: 30
        },
        {
          id: 'features',
          title: t('guide.overview.features.title'),
          description: t('guide.overview.features.description'),
          content: t('guide.overview.features.content'),
          duration: 45
        },
        {
          id: 'navigation',
          title: t('guide.overview.navigation.title'),
          description: t('guide.overview.navigation.description'),
          content: t('guide.overview.navigation.content'),
          duration: 30
        }
      ]
    },
    {
      id: 'upload',
      title: t('guide.upload.title'),
      description: t('guide.upload.description'),
      icon: <Upload className="w-5 h-5" />,
      difficulty: 'beginner',
      estimatedTime: 10,
      steps: [
        {
          id: 'file-formats',
          title: t('guide.upload.formats.title'),
          description: t('guide.upload.formats.description'),
          content: t('guide.upload.formats.content'),
          target: '.upload-area',
          duration: 45
        },
        {
          id: 'drag-drop',
          title: t('guide.upload.dragdrop.title'),
          description: t('guide.upload.dragdrop.description'),
          content: t('guide.upload.dragdrop.content'),
          action: 'highlight',
          duration: 30
        },
        {
          id: 'preview',
          title: t('guide.upload.preview.title'),
          description: t('guide.upload.preview.description'),
          content: t('guide.upload.preview.content'),
          duration: 60
        }
      ]
    },
    {
      id: 'analysis',
      title: t('guide.analysis.title'),
      description: t('guide.analysis.description'),
      icon: <BarChart3 className="w-5 h-5" />,
      difficulty: 'intermediate',
      estimatedTime: 15,
      steps: [
        {
          id: 'selection',
          title: t('guide.analysis.selection.title'),
          description: t('guide.analysis.selection.description'),
          content: t('guide.analysis.selection.content'),
          duration: 60
        },
        {
          id: 'configuration',
          title: t('guide.analysis.configuration.title'),
          description: t('guide.analysis.configuration.description'),
          content: t('guide.analysis.configuration.content'),
          duration: 90
        },
        {
          id: 'execution',
          title: t('guide.analysis.execution.title'),
          description: t('guide.analysis.execution.description'),
          content: t('guide.analysis.execution.content'),
          duration: 45
        }
      ]
    },
    {
      id: 'results',
      title: t('guide.results.title'),
      description: t('guide.results.description'),
      icon: <Eye className="w-5 h-5" />,
      difficulty: 'intermediate',
      estimatedTime: 12,
      steps: [
        {
          id: 'interpretation',
          title: t('guide.results.interpretation.title'),
          description: t('guide.results.interpretation.description'),
          content: t('guide.results.interpretation.content'),
          duration: 90
        },
        {
          id: 'charts',
          title: t('guide.results.charts.title'),
          description: t('guide.results.charts.description'),
          content: t('guide.results.charts.content'),
          duration: 75
        },
        {
          id: 'export',
          title: t('guide.results.export.title'),
          description: t('guide.results.export.description'),
          content: t('guide.results.export.content'),
          duration: 45
        }
      ]
    },
    {
      id: 'advanced',
      title: t('guide.advanced.title'),
      description: t('guide.advanced.description'),
      icon: <Settings className="w-5 h-5" />,
      difficulty: 'advanced',
      estimatedTime: 20,
      steps: [
        {
          id: 'customization',
          title: t('guide.advanced.customization.title'),
          description: t('guide.advanced.customization.description'),
          content: t('guide.advanced.customization.content'),
          duration: 120
        },
        {
          id: 'automation',
          title: t('guide.advanced.automation.title'),
          description: t('guide.advanced.automation.description'),
          content: t('guide.advanced.automation.content'),
          duration: 150
        },
        {
          id: 'integration',
          title: t('guide.advanced.integration.title'),
          description: t('guide.advanced.integration.description'),
          content: t('guide.advanced.integration.content'),
          duration: 90
        }
      ]
    }
  ];

  const currentSectionData = guideSections.find(section => section.id === currentSection);
  const totalSteps = currentSectionData?.steps.length || 0;
  const progress = totalSteps > 0 ? ((currentStep + 1) / totalSteps) * 100 : 0;

  useEffect(() => {
    if (startWithTutorial) {
      setCurrentSection('overview');
      setCurrentStep(0);
      setIsPlaying(true);
    }
  }, [startWithTutorial]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isPlaying && currentSectionData) {
      const currentStepData = currentSectionData.steps[currentStep];
      const duration = currentStepData?.duration || 30;
      
      interval = setTimeout(() => {
        handleNextStep();
      }, duration * 1000);
    }
    
    return () => {
      if (interval) clearTimeout(interval);
    };
  }, [isPlaying, currentStep, currentSection]);

  const handleNextStep = () => {
    if (!currentSectionData) return;
    
    const currentStepId = currentSectionData.steps[currentStep]?.id;
    if (currentStepId) {
      setCompletedSteps(prev => new Set([...prev, currentStepId]));
    }
    
    if (currentStep < currentSectionData.steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    } else {
      setIsPlaying(false);
      // Auto-advance to next section if available
      const currentIndex = guideSections.findIndex(s => s.id === currentSection);
      if (currentIndex < guideSections.length - 1) {
        setCurrentSection(guideSections[currentIndex + 1].id);
        setCurrentStep(0);
      }
    }
  };

  const handlePreviousStep = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    } else {
      // Go to previous section
      const currentIndex = guideSections.findIndex(s => s.id === currentSection);
      if (currentIndex > 0) {
        const prevSection = guideSections[currentIndex - 1];
        setCurrentSection(prevSection.id);
        setCurrentStep(prevSection.steps.length - 1);
      }
    }
  };

  const handleSectionChange = (sectionId: string) => {
    setCurrentSection(sectionId);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  const handleTogglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const handleRestart = () => {
    setCurrentStep(0);
    setIsPlaying(false);
    setCompletedSteps(new Set());
  };

  const handleSkipTutorial = () => {
    updatePreferences({ showTutorial: false });
    onClose();
  };

  const filteredSections = guideSections.filter(section =>
    section.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
    section.steps.some(step => 
      step.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      step.description.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
        onClick={(e) => e.target === e.currentTarget && onClose()}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-white rounded-xl shadow-2xl w-full max-w-6xl h-[90vh] flex flex-col overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b bg-gradient-to-r from-blue-50 to-indigo-50">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-100 rounded-lg">
                <HelpCircle className="w-6 h-6 text-blue-600" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900">
                  {t('guide.title')}
                </h2>
                <p className="text-gray-600">
                  {t('guide.subtitle')}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {startWithTutorial && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleSkipTutorial}
                  className="text-gray-600"
                >
                  {t('guide.skipTutorial')}
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
                className="text-gray-500 hover:text-gray-700"
              >
                <X className="w-5 h-5" />
              </Button>
            </div>
          </div>

          <div className="flex flex-1 overflow-hidden">
            {/* Sidebar */}
            <div className="w-80 border-r bg-gray-50 flex flex-col">
              {/* Search */}
              <div className="p-4 border-b">
                <input
                  type="text"
                  placeholder={t('guide.searchPlaceholder')}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>

              {/* Sections List */}
              <div className="flex-1 overflow-y-auto p-4 space-y-2">
                {filteredSections.map((section) => (
                  <motion.div
                    key={section.id}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Card
                      className={`cursor-pointer transition-all ${
                        currentSection === section.id
                          ? 'ring-2 ring-blue-500 bg-blue-50'
                          : 'hover:shadow-md'
                      }`}
                      onClick={() => handleSectionChange(section.id)}
                    >
                      <CardContent className="p-4">
                        <div className="flex items-start gap-3">
                          <div className="p-2 bg-white rounded-lg shadow-sm">
                            {section.icon}
                          </div>
                          <div className="flex-1 min-w-0">
                            <h3 className="font-semibold text-gray-900 truncate">
                              {section.title}
                            </h3>
                            <p className="text-sm text-gray-600 line-clamp-2">
                              {section.description}
                            </p>
                            <div className="flex items-center gap-2 mt-2">
                              <Badge
                                variant="secondary"
                                className={getDifficultyColor(section.difficulty)}
                              >
                                {t(`guide.difficulty.${section.difficulty}`)}
                              </Badge>
                              <span className="text-xs text-gray-500">
                                {section.estimatedTime} {t('guide.minutes')}
                              </span>
                            </div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                ))}
              </div>
            </div>

            {/* Main Content */}
            <div className="flex-1 flex flex-col">
              {currentSectionData && (
                <>
                  {/* Progress Bar */}
                  <div className="p-4 border-b bg-white">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-semibold text-gray-900">
                        {currentSectionData.title}
                      </h3>
                      <span className="text-sm text-gray-500">
                        {currentStep + 1} / {totalSteps}
                      </span>
                    </div>
                    <Progress value={progress} className="h-2" />
                  </div>

                  {/* Step Content */}
                  <div className="flex-1 overflow-y-auto p-6">
                    <AnimatePresence mode="wait">
                      <motion.div
                        key={`${currentSection}-${currentStep}`}
                        initial={{ opacity: 0, x: 20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: -20 }}
                        transition={{ duration: 0.3 }}
                        className="space-y-6"
                      >
                        {currentSectionData.steps[currentStep] && (
                          <>
                            <div>
                              <h4 className="text-xl font-bold text-gray-900 mb-2">
                                {currentSectionData.steps[currentStep].title}
                              </h4>
                              <p className="text-gray-600 mb-4">
                                {currentSectionData.steps[currentStep].description}
                              </p>
                            </div>

                            <Card>
                              <CardContent className="p-6">
                                <div className="prose max-w-none">
                                  <p className="text-gray-700 leading-relaxed">
                                    {currentSectionData.steps[currentStep].content}
                                  </p>
                                </div>
                              </CardContent>
                            </Card>

                            {/* Interactive Elements */}
                            {currentSectionData.steps[currentStep].action && (
                              <Card className="border-blue-200 bg-blue-50">
                                <CardContent className="p-4">
                                  <div className="flex items-center gap-3">
                                    <MousePointer className="w-5 h-5 text-blue-600" />
                                    <span className="text-blue-800 font-medium">
                                      {t('guide.tryItOut')}
                                    </span>
                                  </div>
                                </CardContent>
                              </Card>
                            )}
                          </>
                        )}
                      </motion.div>
                    </AnimatePresence>
                  </div>

                  {/* Controls */}
                  <div className="p-4 border-t bg-gray-50 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handlePreviousStep}
                        disabled={currentStep === 0 && guideSections.findIndex(s => s.id === currentSection) === 0}
                      >
                        <ChevronLeft className="w-4 h-4 mr-1" />
                        {t('common.previous')}
                      </Button>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleTogglePlay}
                      >
                        {isPlaying ? (
                          <Pause className="w-4 h-4 mr-1" />
                        ) : (
                          <Play className="w-4 h-4 mr-1" />
                        )}
                        {isPlaying ? t('guide.pause') : t('guide.play')}
                      </Button>
                      
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleRestart}
                      >
                        <RotateCcw className="w-4 h-4 mr-1" />
                        {t('guide.restart')}
                      </Button>
                    </div>

                    <div className="flex items-center gap-2">
                      {completedSteps.has(currentSectionData.steps[currentStep]?.id) && (
                        <div className="flex items-center gap-1 text-green-600">
                          <CheckCircle className="w-4 h-4" />
                          <span className="text-sm">{t('guide.completed')}</span>
                        </div>
                      )}
                      
                      <Button
                        onClick={handleNextStep}
                        disabled={currentStep === totalSteps - 1 && guideSections.findIndex(s => s.id === currentSection) === guideSections.length - 1}
                      >
                        {t('common.next')}
                        <ChevronRight className="w-4 h-4 ml-1" />
                      </Button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default UserGuide;