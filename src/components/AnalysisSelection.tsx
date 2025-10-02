import { BarChart2, PieChart, Activity, TrendingUp, Brain, Microscope } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface AnalysisSelectionProps {
  onAnalysisSelected: (analysisType: string) => void;
  selectedAnalysis: string;
}

const analysisCategories = [
  {
    id: "descriptive",
    title: "Betimsel İstatistikler",
    description: "Ortalama, medyan, mod, varyans ve standart sapma hesaplamaları",
    icon: BarChart2,
    color: "text-blue-500",
    bgColor: "bg-blue-500/10",
    tags: ["Temel", "Hızlı"],
  },
  {
    id: "visualization",
    title: "Görselleştirme",
    description: "Histogram, boxplot, scatter plot, heatmap ve daha fazlası",
    icon: PieChart,
    color: "text-purple-500",
    bgColor: "bg-purple-500/10",
    tags: ["Görsel", "Raporlama"],
  },
  {
    id: "hypothesis",
    title: "Hipotez Testleri",
    description: "T-Test, Ki-Kare, Mann-Whitney U, Wilcoxon, ANOVA, ANCOVA",
    icon: Activity,
    color: "text-green-500",
    bgColor: "bg-green-500/10",
    tags: ["İstatistik", "Karşılaştırma"],
  },
  {
    id: "regression",
    title: "Regresyon Analizleri",
    description: "Doğrusal, lojistik ve çoklu regresyon modelleri",
    icon: TrendingUp,
    color: "text-orange-500",
    bgColor: "bg-orange-500/10",
    tags: ["Tahmin", "Modelleme"],
  },
  {
    id: "advanced",
    title: "İleri Düzey Analizler",
    description: "Faktör analizi, kümeleme, MANOVA, SEM, zaman serisi",
    icon: Brain,
    color: "text-pink-500",
    bgColor: "bg-pink-500/10",
    tags: ["İleri", "Karmaşık"],
  },
  {
    id: "domain",
    title: "Alan Uygulamaları",
    description: "Sosyal bilimler, biyoistatistik, eğitim, mühendislik",
    icon: Microscope,
    color: "text-cyan-500",
    bgColor: "bg-cyan-500/10",
    tags: ["Özelleşmiş", "Sektörel"],
  },
];

const AnalysisSelection = ({ onAnalysisSelected, selectedAnalysis }: AnalysisSelectionProps) => {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold text-foreground mb-2">Analiz Türü Seçin</h2>
        <p className="text-muted-foreground">
          Verilerinize uygulamak istediğiniz analiz kategorisini seçin
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {analysisCategories.map((category) => {
          const Icon = category.icon;
          const isSelected = selectedAnalysis === category.id;
          
          return (
            <Card
              key={category.id}
              className={`p-6 cursor-pointer transition-all hover:shadow-lg hover:scale-[1.02] ${
                isSelected ? 'ring-2 ring-primary shadow-glow' : ''
              }`}
              onClick={() => onAnalysisSelected(category.id)}
            >
              <div className="space-y-4">
                <div className="flex items-start justify-between">
                  <div className={`p-3 rounded-lg ${category.bgColor}`}>
                    <Icon className={`h-6 w-6 ${category.color}`} />
                  </div>
                  {isSelected && (
                    <Badge variant="default" className="bg-gradient-primary">
                      Seçili
                    </Badge>
                  )}
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    {category.title}
                  </h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {category.description}
                  </p>
                </div>

                <div className="flex flex-wrap gap-2">
                  {category.tags.map((tag) => (
                    <Badge key={tag} variant="secondary" className="text-xs">
                      {tag}
                    </Badge>
                  ))}
                </div>
              </div>
            </Card>
          );
        })}
      </div>
    </div>
  );
};

export default AnalysisSelection;
