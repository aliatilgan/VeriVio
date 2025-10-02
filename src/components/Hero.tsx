import { Sparkles, TrendingUp, Zap } from "lucide-react";

const Hero = () => {
  return (
    <section className="relative overflow-hidden rounded-2xl bg-gradient-hero p-8 md:p-12 border border-border shadow-lg">
      <div className="relative z-10">
        <div className="flex items-center gap-2 mb-4">
          <Sparkles className="h-5 w-5 text-primary" />
          <span className="text-sm font-medium text-primary">Güçlü Veri Analizi</span>
        </div>
        
        <h2 className="text-3xl md:text-5xl font-bold text-foreground mb-4 leading-tight">
          Verilerinizi Anlamlı
          <br />
          <span className="bg-gradient-primary bg-clip-text text-transparent">
            İçgörülere Dönüştürün
          </span>
        </h2>
        
        <p className="text-lg text-muted-foreground mb-8 max-w-2xl">
          İstatistiksel analizlerden makine öğrenmesine, hipotez testlerinden görselleştirmeye kadar 
          ihtiyacınız olan tüm araçlar tek platformda.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
          <div className="flex items-start gap-3 p-4 rounded-xl bg-card/50 backdrop-blur-sm border border-border">
            <div className="p-2 rounded-lg bg-primary/10">
              <TrendingUp className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-1">40+ Analiz Türü</h3>
              <p className="text-sm text-muted-foreground">
                Temel istatistiklerden ileri düzey analizlere
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-4 rounded-xl bg-card/50 backdrop-blur-sm border border-border">
            <div className="p-2 rounded-lg bg-accent/10">
              <Zap className="h-5 w-5 text-accent" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-1">Hızlı Sonuçlar</h3>
              <p className="text-sm text-muted-foreground">
                Saniyeler içinde profesyonel raporlar
              </p>
            </div>
          </div>

          <div className="flex items-start gap-3 p-4 rounded-xl bg-card/50 backdrop-blur-sm border border-border">
            <div className="p-2 rounded-lg bg-primary/10">
              <Sparkles className="h-5 w-5 text-primary" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground mb-1">Kolay Kullanım</h3>
              <p className="text-sm text-muted-foreground">
                Sezgisel arayüz, teknik bilgi gerektirmez
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Decorative background elements */}
      <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 left-0 w-64 h-64 bg-accent/5 rounded-full blur-3xl" />
    </section>
  );
};

export default Hero;
