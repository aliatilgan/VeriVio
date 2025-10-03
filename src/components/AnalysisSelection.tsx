import { BarChart2, PieChart, Activity, TrendingUp, Brain, Microscope, CheckCircle, Clock, Users } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";
import { useApp } from "@/contexts/AppContext";
import { motion } from "framer-motion";

interface AnalysisSelectionProps {
  onAnalysisSelected: (analysisType: string) => void;
  selectedAnalysis: string;
}

interface AnalysisCategory {
  id: string;
  titleKey: string;
  descriptionKey: string;
  icon: any;
  color: string;
  bgColor: string;
  tagsKey: string[];
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  estimatedTime: string;
  popularity: number;
}

const AnalysisSelection = ({ onAnalysisSelected, selectedAnalysis }: AnalysisSelectionProps) => {
  const { t } = useTranslation();
  const { state } = useApp();

  const analysisCategories: AnalysisCategory[] = [
    {
      id: "descriptive",
      titleKey: "analysisSelection.categories.descriptive.title",
      descriptionKey: "analysisSelection.categories.descriptive.description",
      icon: BarChart2,
      color: "text-blue-500",
      bgColor: "bg-blue-500/10",
      tagsKey: ["analysisSelection.tags.basic", "analysisSelection.tags.fast"],
      difficulty: 'beginner',
      estimatedTime: '2-5 min',
      popularity: 95
    },
    {
      id: "visualization",
      titleKey: "analysisSelection.categories.visualization.title",
      descriptionKey: "analysisSelection.categories.visualization.description",
      icon: PieChart,
      color: "text-purple-500",
      bgColor: "bg-purple-500/10",
      tagsKey: ["analysisSelection.tags.visual", "analysisSelection.tags.reporting"],
      difficulty: 'beginner',
      estimatedTime: '3-8 min',
      popularity: 88
    },
    {
      id: "hypothesis",
      titleKey: "analysisSelection.categories.hypothesis.title",
      descriptionKey: "analysisSelection.categories.hypothesis.description",
      icon: Activity,
      color: "text-green-500",
      bgColor: "bg-green-500/10",
      tagsKey: ["analysisSelection.tags.statistics", "analysisSelection.tags.comparison"],
      difficulty: 'intermediate',
      estimatedTime: '5-15 min',
      popularity: 72
    },
    {
      id: "regression",
      titleKey: "analysisSelection.categories.regression.title",
      descriptionKey: "analysisSelection.categories.regression.description",
      icon: TrendingUp,
      color: "text-orange-500",
      bgColor: "bg-orange-500/10",
      tagsKey: ["analysisSelection.tags.prediction", "analysisSelection.tags.modeling"],
      difficulty: 'intermediate',
      estimatedTime: '10-25 min',
      popularity: 65
    },
    {
      id: "advanced",
      titleKey: "analysisSelection.categories.advanced.title",
      descriptionKey: "analysisSelection.categories.advanced.description",
      icon: Brain,
      color: "text-pink-500",
      bgColor: "bg-pink-500/10",
      tagsKey: ["analysisSelection.tags.advanced", "analysisSelection.tags.complex"],
      difficulty: 'advanced',
      estimatedTime: '15-45 min',
      popularity: 45
    },
    {
      id: "domain",
      titleKey: "analysisSelection.categories.domain.title",
      descriptionKey: "analysisSelection.categories.domain.description",
      icon: Microscope,
      color: "text-cyan-500",
      bgColor: "bg-cyan-500/10",
      tagsKey: ["analysisSelection.tags.specialized", "analysisSelection.tags.sectoral"],
      difficulty: 'advanced',
      estimatedTime: '20-60 min',
      popularity: 38
    },
  ];

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'text-green-600 bg-green-100';
      case 'intermediate': return 'text-yellow-600 bg-yellow-100';
      case 'advanced': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getDifficultyIcon = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return CheckCircle;
      case 'intermediate': return Clock;
      case 'advanced': return Brain;
      default: return CheckCircle;
    }
  };

  return (
    <div className="space-y-6">
      <div className="text-center">
        <h2 className="text-3xl font-bold text-foreground mb-3">
          {t('analysisSelection.title')}
        </h2>
        <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
          {t('analysisSelection.description')}
        </p>
        {state.uploadedData && state.uploadedData.parsedData && state.uploadedData.columns && (
          <div className="mt-4 flex items-center justify-center gap-2 text-sm text-muted-foreground">
            <CheckCircle className="h-4 w-4 text-green-500" />
            {t('analysisSelection.dataReady', { 
              rows: state.uploadedData.parsedData.length,
              columns: state.uploadedData.columns.length 
            })}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {analysisCategories.map((category, index) => {
          const Icon = category.icon;
          const DifficultyIcon = getDifficultyIcon(category.difficulty);
          const isSelected = selectedAnalysis === category.id;
          
          return (
            <motion.div
              key={category.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <Card
                className={`p-6 cursor-pointer transition-all hover:shadow-lg hover:scale-[1.02] relative overflow-hidden ${
                  isSelected ? 'ring-2 ring-primary shadow-glow' : ''
                }`}
                onClick={() => onAnalysisSelected(category.id)}
              >
                {/* Popularity indicator */}
                <div className="absolute top-2 right-2 flex items-center gap-1">
                  <Users className="h-3 w-3 text-muted-foreground" />
                  <span className="text-xs text-muted-foreground">{category.popularity}%</span>
                </div>

                <div className="space-y-4">
                  <div className="flex items-start justify-between">
                    <div className={`p-3 rounded-lg ${category.bgColor}`}>
                      <Icon className={`h-6 w-6 ${category.color}`} />
                    </div>
                    {isSelected && (
                      <Badge variant="default" className="bg-gradient-to-r from-primary to-primary/80">
                        {t('analysisSelection.selected')}
                      </Badge>
                    )}
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-foreground mb-2">
                      {t(category.titleKey)}
                    </h3>
                    <p className="text-sm text-muted-foreground leading-relaxed">
                      {t(category.descriptionKey)}
                    </p>
                  </div>

                  {/* Difficulty and time info */}
                  <div className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-1">
                      <DifficultyIcon className="h-3 w-3" />
                      <Badge 
                        variant="secondary" 
                        className={`${getDifficultyColor(category.difficulty)} border-0`}
                      >
                        {t(`analysisSelection.difficulty.${category.difficulty}`)}
                      </Badge>
                    </div>
                    <div className="flex items-center gap-1 text-muted-foreground">
                      <Clock className="h-3 w-3" />
                      <span>{category.estimatedTime}</span>
                    </div>
                  </div>

                  <div className="flex flex-wrap gap-2">
                    {category.tagsKey.map((tagKey) => (
                      <Badge key={tagKey} variant="outline" className="text-xs">
                        {t(tagKey)}
                      </Badge>
                    ))}
                  </div>

                  {/* Action button for selected analysis */}
                  {isSelected && (
                    <Button 
                      className="w-full mt-2" 
                      size="sm"
                      onClick={(e) => {
                        e.stopPropagation();
                        // Navigate to analysis configuration
                      }}
                    >
                      {t('analysisSelection.configure')}
                    </Button>
                  )}
                </div>
              </Card>
            </motion.div>
          );
        })}
      </div>

      {/* Help section */}
      <div className="text-center mt-8">
        <p className="text-sm text-muted-foreground mb-4">
          {t('analysisSelection.needHelp')}
        </p>
        <Button variant="outline" size="sm">
          {t('analysisSelection.viewGuide')}
        </Button>
      </div>
    </div>
  );
};

export default AnalysisSelection;
