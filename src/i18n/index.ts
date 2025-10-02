import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';

const resources = {
  tr: {
    translation: {
      // Header
      "header.title": "VeriVio Analytics",
      "header.home": "Ana Sayfa",
      "header.analyses": "Analizler",
      "header.documentation": "Dokümantasyon",
      "header.support": "Destek",
      "header.login": "Giriş Yap",
      "header.start": "Başla",
      
      // Hero Section
      "hero.title": "Güçlü Veri Analizi Platformu",
      "hero.subtitle": "Verilerinizi yükleyin, analiz edin ve anlamlı sonuçlar elde edin. SPSS benzeri güçlü istatistiksel analizler artık web tarayıcınızda.",
      "hero.uploadButton": "Veri Yükle",
      "hero.learnMore": "Daha Fazla Bilgi",
      
      // Data Upload
      dataUpload: {
        title: "Veri Dosyası Yükleyin",
        description: "CSV, Excel veya JSON formatında dosyanızı sürükleyip bırakın veya seçin",
        selectFile: "Dosya Seç",
        dragDrop: "veya dosyayı buraya sürükleyin",
        maxSize: "Maksimum dosya boyutu: 50MB",
        processing: "İşleniyor",
        success: "Dosya başarıyla yüklendi!",
        successDescription: "{{fileName}} - {{rowCount}} satır veri",
        error: "Dosya yüklenirken hata oluştu",
        invalidFormat: "Geçersiz dosya formatı. Lütfen CSV, Excel veya JSON dosyası yükleyin.",
        fileTooLarge: "Dosya çok büyük. Maksimum 50MB boyutunda dosya yükleyebilirsiniz.",
        processingError: "Dosya işlenirken bir hata oluştu.",
        fileUploaded: "Dosya Yüklendi",
        rows: "{{count}} satır",
        columns: "{{count}} sütun",
        showPreview: "Veri Önizleme",
        hidePreview: "Önizlemeyi Gizle",
        selectDifferent: "Farklı Dosya Seç",
        previewTitle: "Veri Önizleme (İlk 10 Satır)",
        previewRows: "{{count}} satır gösteriliyor",
        moreRows: "{{remaining}} satır daha var (Toplam: {{total}} satır)"
      },
      
      // Analysis Selection
      analysisSelection: {
        title: "Analiz Türü Seçin",
        description: "Verilerinize uygulamak istediğiniz analiz kategorisini seçin",
        dataReady: "{{rows}} satır, {{columns}} sütun veri hazır",
        selected: "Seçili",
        configure: "Yapılandır",
        needHelp: "Hangi analizi seçeceğinizden emin değil misiniz?",
        viewGuide: "Rehberi Görüntüle",
        difficulty: {
          beginner: "Başlangıç",
          intermediate: "Orta",
          advanced: "İleri"
        },
        categories: {
          descriptive: {
            title: "Betimsel İstatistikler",
            description: "Ortalama, medyan, mod, varyans ve standart sapma hesaplamaları"
          },
          visualization: {
            title: "Görselleştirme",
            description: "Histogram, boxplot, scatter plot, heatmap ve daha fazlası"
          },
          hypothesis: {
            title: "Hipotez Testleri",
            description: "T-Test, Ki-Kare, Mann-Whitney U, Wilcoxon, ANOVA, ANCOVA"
          },
          regression: {
            title: "Regresyon Analizleri",
            description: "Doğrusal, lojistik ve çoklu regresyon modelleri"
          },
          advanced: {
            title: "İleri Düzey Analizler",
            description: "Faktör analizi, kümeleme, MANOVA, SEM, zaman serisi"
          },
          domain: {
            title: "Alan Uygulamaları",
            description: "Sosyal bilimler, biyoistatistik, eğitim, mühendislik"
          }
        },
        tags: {
          basic: "Temel",
          fast: "Hızlı",
          visual: "Görsel",
          reporting: "Raporlama",
          statistics: "İstatistik",
          comparison: "Karşılaştırma",
          prediction: "Tahmin",
          modeling: "Modelleme",
          advanced: "İleri",
          complex: "Karmaşık",
          specialized: "Özelleşmiş",
          sectoral: "Sektörel"
        }
      },
      
      // Results
      results: {
        title: "Analiz Sonuçları",
        description: "Veri analizi sonuçlarınızı inceleyin",
        dataReady: "Veri hazır",
        totalRows: "Toplam Satır",
        totalColumns: "Toplam Sütun",
        refreshData: "Verileri Yenile",
        downloadPDF: "PDF İndir",
        downloadExcel: "Excel İndir",
        share: "Paylaş",
        searchPlaceholder: "Verilerde ara...",
        selectChartType: "Grafik türü seçin",
        selectColumn: "Sütun seçin",
        summary: "Özet",
        charts: "Grafikler",
        statistics: "İstatistikler",
        data: "Veri",
        mean: "Ortalama",
        stdDev: "Standart Sapma",
        min: "Minimum",
        max: "Maksimum",
        count: "Sayı",
        range: "Aralık",
        interpretation: "Analiz Yorumu",
        interpretationText: "Bu analiz sonuçları, verilerinizin genel eğilimlerini ve önemli bulgularını göstermektedir. Detaylı inceleme için grafikleri ve istatistikleri kullanabilirsiniz.",
        dataQuality: "Veri Kalitesi",
        completeness: "Tamlık",
        consistency: "Tutarlılık",
        accuracy: "Doğruluk",
        searchResults: "Arama Sonuçları",
        noResults: "Arama kriterlerinize uygun sonuç bulunamadı",
        distributionAnalysis: "Dağılım Analizi",
        statisticalSummary: "İstatistiksel Özet",
        metric: "Metrik",
        value: "Değer",
        countDesc: "Toplam veri noktası sayısı",
        meanDesc: "Sayısal değerlerin aritmetik ortalaması",
        stdDevDesc: "Verilerin ortalamadan sapma ölçüsü",
        minDesc: "En küçük değer",
        maxDesc: "En büyük değer",
        noStatistics: "İstatistik verisi bulunamadı",
        dataTable: "Veri Tablosu",
        showingFiltered: "{{showing}}/{{total}} filtrelenmiş satır gösteriliyor",
        showingRows: "{{showing}}/{{total}} satır gösteriliyor",
        clearFilter: "Filtreyi Temizle",
        emptyValue: "Boş",
        moreRowsAvailable: "Toplam {{total}} satır mevcut",
        noData: "Veri bulunamadı",
        chartTypes: {
          bar: "Çubuk Grafik",
          line: "Çizgi Grafik",
          area: "Alan Grafik",
          pie: "Pasta Grafik"
        },
        noChartData: "Grafik verisi bulunamadı",
        multiColumnComparison: "Çoklu Sütun Karşılaştırması",
        distribution: "Dağılım"
      },
      
      "results.title": "Analiz Sonuçları",
      "results.summary": "Özet",
      "results.charts": "Grafikler",
      "results.statistics": "İstatistikler",
      "results.export": "Dışa Aktar",
      "results.exportPdf": "PDF İndir",
      "results.exportExcel": "Excel İndir",
      
      // Common
      "common.loading": "Yükleniyor...",
      "common.error": "Hata",
      "common.success": "Başarılı",
      "common.cancel": "İptal",
      "common.save": "Kaydet",
      "common.close": "Kapat",
      "common.next": "İleri",
      "common.previous": "Geri",
      "common.finish": "Bitir",
      
      // Tags
      "tags.basic": "Temel",
      "tags.fast": "Hızlı",
      "tags.visual": "Görsel",
      "tags.reporting": "Raporlama",
      "tags.statistics": "İstatistik",
      "tags.comparison": "Karşılaştırma",
      "tags.prediction": "Tahmin",
      "tags.modeling": "Modelleme",
      "tags.advanced": "İleri Düzey",
      "tags.research": "Araştırma",
      "tags.specialized": "Özel",
      "tags.domain": "Alan",

      // User Guide
      guide: {
        title: "Kullanıcı Rehberi",
        subtitle: "VeriVio'yu etkili bir şekilde kullanmayı öğrenin",
        searchPlaceholder: "Rehberde ara...",
        skipTutorial: "Öğreticiyi Atla",
        minutes: "dakika",
        tryItOut: "Deneyin",
        pause: "Duraklat",
        play: "Oynat",
        restart: "Yeniden Başlat",
        completed: "Tamamlandı",
        difficulty: {
          beginner: "Başlangıç",
          intermediate: "Orta",
          advanced: "İleri"
        },
        overview: {
          title: "Genel Bakış",
          description: "VeriVio platformuna giriş ve temel özellikler",
          welcome: {
            title: "VeriVio'ya Hoş Geldiniz",
            description: "Güçlü veri analizi platformuna giriş",
            content: "VeriVio, verilerinizi kolayca analiz etmenizi sağlayan modern bir web platformudur. SPSS benzeri güçlü istatistiksel analizleri web tarayıcınızda gerçekleştirebilirsiniz. Bu rehber, platformun tüm özelliklerini etkili bir şekilde kullanmanıza yardımcı olacaktır."
          },
          features: {
            title: "Ana Özellikler",
            description: "Platform özelliklerine genel bakış",
            content: "VeriVio şu ana özellikleri sunar: Çoklu dosya formatı desteği (CSV, Excel, JSON), İnteraktif veri görselleştirme, Kapsamlı istatistiksel analizler, Gerçek zamanlı sonuç görüntüleme, PDF ve Excel rapor dışa aktarma, Çoklu dil desteği ve Kullanıcı dostu arayüz."
          },
          navigation: {
            title: "Navigasyon",
            description: "Platform içinde nasıl gezineceğinizi öğrenin",
            content: "Platform ana menüsünden farklı bölümlere erişebilirsiniz. Veri yükleme, analiz seçimi, sonuç görüntüleme ve ayarlar bölümleri arasında kolayca geçiş yapabilirsiniz. Her bölümde size rehberlik edecek ipuçları ve yardım butonları bulunmaktadır."
          }
        },
        upload: {
          title: "Veri Yükleme",
          description: "Dosyalarınızı nasıl yükleyeceğinizi öğrenin",
          formats: {
            title: "Desteklenen Dosya Formatları",
            description: "Hangi dosya türlerini yükleyebileceğinizi öğrenin",
            content: "VeriVio şu dosya formatlarını destekler: CSV (Virgülle Ayrılmış Değerler), Excel (.xlsx, .xls), JSON (JavaScript Object Notation). Maksimum dosya boyutu 50MB'dır. Dosyanızın ilk satırının sütun başlıklarını içerdiğinden emin olun."
          },
          dragdrop: {
            title: "Sürükle ve Bırak",
            description: "Dosyaları kolayca yükleme yöntemi",
            content: "Dosyalarınızı doğrudan yükleme alanına sürükleyip bırakabilirsiniz. Alternatif olarak, 'Dosya Seç' butonuna tıklayarak dosya gezgininden seçim yapabilirsiniz. Yükleme işlemi otomatik olarak başlayacaktır."
          },
          preview: {
            title: "Veri Önizleme",
            description: "Yüklenen verileri kontrol etme",
            content: "Dosya yüklendikten sonra, verilerinizin ilk 10 satırını önizleyebilirsiniz. Bu, verilerin doğru şekilde yüklendiğini kontrol etmenize yardımcı olur. Sütun sayısı, satır sayısı ve veri türleri hakkında bilgi alabilirsiniz."
          }
        },
        analysis: {
          title: "Analiz Seçimi",
          description: "Uygun analiz türünü seçme rehberi",
          selection: {
            title: "Analiz Türü Seçimi",
            description: "Verileriniz için en uygun analizi seçin",
            content: "VeriVio farklı analiz kategorileri sunar: Betimsel İstatistikler (ortalama, medyan, standart sapma), Görselleştirme (histogram, kutu grafiği, dağılım grafiği), Hipotez Testleri (t-testi, ki-kare, ANOVA), Regresyon Analizi (doğrusal, lojistik regresyon), İleri Düzey Analizler (faktör analizi, kümeleme)."
          },
          configuration: {
            title: "Analiz Yapılandırması",
            description: "Analiz parametrelerini ayarlama",
            content: "Seçtiğiniz analiz türüne göre, gerekli parametreleri yapılandırabilirsiniz. Değişken seçimi, güven aralığı, anlamlılık düzeyi gibi ayarları ihtiyaçlarınıza göre düzenleyebilirsiniz. Her parametre için açıklayıcı bilgiler mevcuttur."
          },
          execution: {
            title: "Analiz Çalıştırma",
            description: "Analizi başlatma ve izleme",
            content: "Tüm ayarları tamamladıktan sonra, 'Analizi Başlat' butonuna tıklayarak analizi çalıştırabilirsiniz. Analiz süreci gerçek zamanlı olarak izlenebilir. Büyük veri setleri için işlem biraz zaman alabilir."
          }
        },
        results: {
          title: "Sonuç Görüntüleme",
          description: "Analiz sonuçlarını anlama ve yorumlama",
          interpretation: {
            title: "Sonuç Yorumlama",
            description: "İstatistiksel sonuçları anlama",
            content: "Analiz sonuçları dört ana sekmede sunulur: Özet (genel bulgular), Grafikler (görsel sunumlar), İstatistikler (detaylı sayısal sonuçlar), Veri (işlenmiş veri tablosu). Her sekme farklı perspektiflerden sonuçları incelemenize olanak sağlar."
          },
          charts: {
            title: "Grafik Görüntüleme",
            description: "İnteraktif grafiklerle veri keşfi",
            content: "Grafikler sekmesinde verilerinizi farklı grafik türleriyle görselleştirebilirsiniz: Çubuk grafik, Çizgi grafik, Alan grafik, Pasta grafik. Grafik türünü ve görüntülenecek sütunları dinamik olarak değiştirebilirsiniz. Grafikler üzerinde zoom ve pan işlemleri yapabilirsiniz."
          },
          export: {
            title: "Sonuç Dışa Aktarma",
            description: "Sonuçları farklı formatlarda kaydetme",
            content: "Analiz sonuçlarınızı PDF veya Excel formatında dışa aktarabilirsiniz. PDF raporu tüm grafikleri ve istatistikleri içerir. Excel dosyası ham verileri ve hesaplanmış değerleri içerir. Dışa aktarma işlemi birkaç saniye sürer."
          }
        },
        advanced: {
          title: "İleri Düzey Özellikler",
          description: "Gelişmiş kullanım teknikleri",
          customization: {
            title: "Özelleştirme",
            description: "Arayüzü ve ayarları kişiselleştirme",
            content: "VeriVio'yu ihtiyaçlarınıza göre özelleştirebilirsiniz: Tema seçimi (açık/koyu), Dil tercihi, Grafik renk paletleri, Varsayılan analiz ayarları, Otomatik kaydetme seçenekleri. Tüm tercihleriniz tarayıcınızda saklanır."
          },
          automation: {
            title: "Otomasyon",
            description: "Tekrarlayan işlemleri otomatikleştirme",
            content: "Sık kullandığınız analiz türlerini şablon olarak kaydedebilirsiniz. Benzer veri setleri için aynı analizi hızlıca tekrarlayabilirsiniz. Toplu işlem özelliği ile birden fazla dosyayı aynı anda analiz edebilirsiniz."
          },
          integration: {
            title: "Entegrasyon",
            description: "Diğer araçlarla entegrasyon",
            content: "VeriVio sonuçlarını diğer analiz araçlarında kullanabilirsiniz. API desteği ile programatik erişim sağlayabilirsiniz. Sonuçları R, Python, SPSS gibi araçlara aktarabilirsiniz. Veri tabanı bağlantıları kurabilirsiniz."
          }
        }
      }
    }
  },
  en: {
    translation: {
      // Header
      "header.title": "VeriVio Analytics",
      "header.home": "Home",
      "header.analyses": "Analyses",
      "header.documentation": "Documentation",
      "header.support": "Support",
      "header.login": "Login",
      "header.start": "Get Started",
      
      // Hero Section
      "hero.title": "Powerful Data Analysis Platform",
      "hero.subtitle": "Upload your data, analyze it, and get meaningful results. SPSS-like powerful statistical analyses now in your web browser.",
      "hero.uploadButton": "Upload Data",
      "hero.learnMore": "Learn More",
      
      // Data Upload
      dataUpload: {
        title: "Upload Data File",
        description: "Drag and drop or select your CSV, Excel or JSON file",
        selectFile: "Select File",
        dragDrop: "or drag and drop file here",
        maxSize: "Maximum file size: 50MB",
        processing: "Processing",
        success: "File uploaded successfully!",
        successDescription: "{{fileName}} - {{rowCount}} rows of data",
        error: "Error occurred while uploading file",
        invalidFormat: "Invalid file format. Please upload CSV, Excel or JSON file.",
        fileTooLarge: "File too large. Maximum file size is 50MB.",
        processingError: "An error occurred while processing the file.",
        fileUploaded: "File Uploaded",
        rows: "{{count}} rows",
        columns: "{{count}} columns",
        showPreview: "Data Preview",
        hidePreview: "Hide Preview",
        selectDifferent: "Select Different File",
        previewTitle: "Data Preview (First 10 Rows)",
        previewRows: "Showing {{count}} rows",
        moreRows: "{{remaining}} more rows (Total: {{total}} rows)"
      },
      
      // Analysis Selection
      analysisSelection: {
        title: "Select Analysis Type",
        description: "Choose the analysis category you want to apply to your data",
        dataReady: "{{rows}} rows, {{columns}} columns of data ready",
        selected: "Selected",
        configure: "Configure",
        needHelp: "Not sure which analysis to choose?",
        viewGuide: "View Guide",
        difficulty: {
          beginner: "Beginner",
          intermediate: "Intermediate",
          advanced: "Advanced"
        },
        categories: {
          descriptive: {
            title: "Descriptive Statistics",
            description: "Mean, median, mode, variance and standard deviation calculations"
          },
          visualization: {
            title: "Visualization",
            description: "Histogram, boxplot, scatter plot, heatmap and more"
          },
          hypothesis: {
            title: "Hypothesis Tests",
            description: "T-Test, Chi-Square, Mann-Whitney U, Wilcoxon, ANOVA, ANCOVA"
          },
          regression: {
            title: "Regression Analysis",
            description: "Linear, logistic and multiple regression models"
          },
          advanced: {
            title: "Advanced Analysis",
            description: "Factor analysis, clustering, MANOVA, SEM, time series"
          },
          domain: {
            title: "Domain Applications",
            description: "Social sciences, biostatistics, education, engineering"
          }
        },
        tags: {
          basic: "Basic",
          fast: "Fast",
          visual: "Visual",
          reporting: "Reporting",
          statistics: "Statistics",
          comparison: "Comparison",
          prediction: "Prediction",
          modeling: "Modeling",
          advanced: "Advanced",
          complex: "Complex",
          specialized: "Specialized",
          sectoral: "Sectoral"
        }
      },
      
      // Results
      results: {
        title: "Analysis Results",
        description: "Review your data analysis results",
        dataReady: "Data ready",
        totalRows: "Total Rows",
        totalColumns: "Total Columns",
        refreshData: "Refresh Data",
        downloadPDF: "Download PDF",
        downloadExcel: "Download Excel",
        share: "Share",
        searchPlaceholder: "Search in data...",
        selectChartType: "Select chart type",
        selectColumn: "Select column",
        summary: "Summary",
        charts: "Charts",
        statistics: "Statistics",
        data: "Data",
        mean: "Mean",
        stdDev: "Standard Deviation",
        min: "Minimum",
        max: "Maximum",
        count: "Count",
        range: "Range",
        interpretation: "Analysis Interpretation",
        interpretationText: "These analysis results show the general trends and important findings of your data. You can use charts and statistics for detailed examination.",
        dataQuality: "Data Quality",
        completeness: "Completeness",
        consistency: "Consistency",
        accuracy: "Accuracy",
        searchResults: "Search Results",
        noResults: "No results found matching your search criteria",
        distributionAnalysis: "Distribution Analysis",
        statisticalSummary: "Statistical Summary",
        metric: "Metric",
        value: "Value",
        countDesc: "Total number of data points",
        meanDesc: "Arithmetic average of numerical values",
        stdDevDesc: "Measure of data deviation from mean",
        minDesc: "Smallest value",
        maxDesc: "Largest value",
        noStatistics: "No statistical data found",
        dataTable: "Data Table",
        showingFiltered: "Showing {{showing}}/{{total}} filtered rows",
        showingRows: "Showing {{showing}}/{{total}} rows",
        clearFilter: "Clear Filter",
        emptyValue: "Empty",
        moreRowsAvailable: "Total {{total}} rows available",
        noData: "No data found",
        chartTypes: {
          bar: "Bar Chart",
          line: "Line Chart",
          area: "Area Chart",
          pie: "Pie Chart"
        },
        noChartData: "No chart data found",
        multiColumnComparison: "Multi-Column Comparison",
        distribution: "Distribution"
      },
      
      "results.title": "Analysis Results",
      "results.summary": "Summary",
      "results.charts": "Charts",
      "results.statistics": "Statistics",
      "results.export": "Export",
      "results.exportPdf": "Download PDF",
      "results.exportExcel": "Download Excel",
      
      // Common
      "common.loading": "Loading...",
      "common.error": "Error",
      "common.success": "Success",
      "common.cancel": "Cancel",
      "common.save": "Save",
      "common.close": "Close",
      "common.next": "Next",
      "common.previous": "Previous",
      "common.finish": "Finish",
      
      // Tags
      "tags.basic": "Basic",
      "tags.fast": "Fast",
      "tags.visual": "Visual",
      "tags.reporting": "Reporting",
      "tags.statistics": "Statistics",
      "tags.comparison": "Comparison",
      "tags.prediction": "Prediction",
      "tags.modeling": "Modeling",
      "tags.advanced": "Advanced",
      "tags.research": "Research",
      "tags.specialized": "Specialized",
      "tags.domain": "Domain",

      // User Guide
      guide: {
        title: "User Guide",
        subtitle: "Learn how to use VeriVio effectively",
        searchPlaceholder: "Search in guide...",
        skipTutorial: "Skip Tutorial",
        minutes: "minutes",
        tryItOut: "Try it out",
        pause: "Pause",
        play: "Play",
        restart: "Restart",
        completed: "Completed",
        difficulty: {
          beginner: "Beginner",
          intermediate: "Intermediate",
          advanced: "Advanced"
        },
        overview: {
          title: "Overview",
          description: "Introduction to VeriVio platform and basic features",
          welcome: {
            title: "Welcome to VeriVio",
            description: "Introduction to the powerful data analysis platform",
            content: "VeriVio is a modern web platform that allows you to easily analyze your data. You can perform powerful statistical analyses similar to SPSS in your web browser. This guide will help you use all the features of the platform effectively."
          },
          features: {
            title: "Main Features",
            description: "Overview of platform features",
            content: "VeriVio offers these main features: Multi-format file support (CSV, Excel, JSON), Interactive data visualization, Comprehensive statistical analyses, Real-time result display, PDF and Excel report export, Multi-language support, and User-friendly interface."
          },
          navigation: {
            title: "Navigation",
            description: "Learn how to navigate within the platform",
            content: "You can access different sections from the main menu of the platform. You can easily switch between data upload, analysis selection, result display, and settings sections. Each section has tips and help buttons to guide you."
          }
        },
        upload: {
          title: "Data Upload",
          description: "Learn how to upload your files",
          formats: {
            title: "Supported File Formats",
            description: "Learn which file types you can upload",
            content: "VeriVio supports the following file formats: CSV (Comma Separated Values), Excel (.xlsx, .xls), JSON (JavaScript Object Notation). Maximum file size is 50MB. Make sure the first row of your file contains column headers."
          },
          dragdrop: {
            title: "Drag and Drop",
            description: "Easy file upload method",
            content: "You can drag and drop your files directly into the upload area. Alternatively, you can click the 'Select File' button to choose from the file explorer. The upload process will start automatically."
          },
          preview: {
            title: "Data Preview",
            description: "Checking uploaded data",
            content: "After the file is uploaded, you can preview the first 10 rows of your data. This helps you verify that the data has been loaded correctly. You can get information about the number of columns, rows, and data types."
          }
        },
        analysis: {
          title: "Analysis Selection",
          description: "Guide to selecting the appropriate analysis type",
          selection: {
            title: "Analysis Type Selection",
            description: "Choose the most suitable analysis for your data",
            content: "VeriVio offers different analysis categories: Descriptive Statistics (mean, median, standard deviation), Visualization (histogram, box plot, scatter plot), Hypothesis Tests (t-test, chi-square, ANOVA), Regression Analysis (linear, logistic regression), Advanced Analyses (factor analysis, clustering)."
          },
          configuration: {
            title: "Analysis Configuration",
            description: "Setting analysis parameters",
            content: "According to the analysis type you selected, you can configure the necessary parameters. You can adjust settings such as variable selection, confidence interval, significance level according to your needs. Explanatory information is available for each parameter."
          },
          execution: {
            title: "Running Analysis",
            description: "Starting and monitoring the analysis",
            content: "After completing all settings, you can run the analysis by clicking the 'Start Analysis' button. The analysis process can be monitored in real-time. Processing may take some time for large datasets."
          }
        },
        results: {
          title: "Result Display",
          description: "Understanding and interpreting analysis results",
          interpretation: {
            title: "Result Interpretation",
            description: "Understanding statistical results",
            content: "Analysis results are presented in four main tabs: Summary (general findings), Charts (visual presentations), Statistics (detailed numerical results), Data (processed data table). Each tab allows you to examine results from different perspectives."
          },
          charts: {
            title: "Chart Display",
            description: "Data exploration with interactive charts",
            content: "In the Charts tab, you can visualize your data with different chart types: Bar chart, Line chart, Area chart, Pie chart. You can dynamically change the chart type and columns to display. You can zoom and pan on the charts."
          },
          export: {
            title: "Result Export",
            description: "Saving results in different formats",
            content: "You can export your analysis results in PDF or Excel format. The PDF report includes all charts and statistics. The Excel file contains raw data and calculated values. The export process takes a few seconds."
          }
        },
        advanced: {
          title: "Advanced Features",
          description: "Advanced usage techniques",
          customization: {
            title: "Customization",
            description: "Personalizing interface and settings",
            content: "You can customize VeriVio according to your needs: Theme selection (light/dark), Language preference, Chart color palettes, Default analysis settings, Auto-save options. All your preferences are stored in your browser."
          },
          automation: {
            title: "Automation",
            description: "Automating repetitive tasks",
            content: "You can save frequently used analysis types as templates. You can quickly repeat the same analysis for similar datasets. With the batch processing feature, you can analyze multiple files simultaneously."
          },
          integration: {
            title: "Integration",
            description: "Integration with other tools",
            content: "You can use VeriVio results in other analysis tools. You can provide programmatic access with API support. You can export results to tools like R, Python, SPSS. You can establish database connections."
          }
        }
      }
    }
  }
};

i18n
  .use(initReactI18next)
  .init({
    resources,
    lng: 'tr', // default language
    fallbackLng: 'en',
    interpolation: {
      escapeValue: false
    }
  });

export default i18n;