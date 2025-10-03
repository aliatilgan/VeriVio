import Header from "@/components/Header";
import { ResponsiveContainer } from "@/components/ResponsiveLayout";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { 
  MessageCircle, 
  Mail, 
  Phone, 
  Clock, 
  HelpCircle, 
  Send,
  ExternalLink,
  CheckCircle
} from "lucide-react";
import { useTranslation } from 'react-i18next';

const Support = () => {
  const { t } = useTranslation();

  const faqItems = [
    {
      question: "Hangi dosya formatları destekleniyor?",
      answer: "VeriVio CSV, Excel (.xlsx, .xls), JSON ve TSV formatlarını destekler. Dosya boyutu maksimum 100MB olabilir."
    },
    {
      question: "Analiz sonuçları nasıl dışa aktarılır?",
      answer: "Analiz tamamlandıktan sonra sonuçlar panelinde 'Dışa Aktar' butonunu kullanarak PDF, Excel veya CSV formatında indirebilirsiniz."
    },
    {
      question: "Verilerim güvende mi?",
      answer: "Evet, tüm veriler şifrelenmiş olarak saklanır ve analiz tamamlandıktan sonra 24 saat içinde otomatik olarak silinir."
    },
    {
      question: "Büyük veri setleri için sınır var mı?",
      answer: "Ücretsiz hesaplar için 100MB ve 1 milyon satır sınırı vardır. Premium hesaplar için daha yüksek limitler mevcuttur."
    },
    {
      question: "API kullanımı nasıl çalışır?",
      answer: "API anahtarınızı hesap ayarlarından alabilir ve REST API endpoints'lerini kullanarak programatik erişim sağlayabilirsiniz."
    },
    {
      question: "Analiz sonuçları ne kadar süre saklanır?",
      answer: "Analiz sonuçları hesabınızda 30 gün boyunca saklanır. Bu süre sonunda otomatik olarak silinir."
    }
  ];

  const contactMethods = [
    {
      icon: MessageCircle,
      title: "Canlı Destek",
      description: "7/24 canlı destek hizmeti",
      action: "Sohbeti Başlat",
      available: true
    },
    {
      icon: Mail,
      title: "E-posta Desteği",
      description: "support@verivio.com",
      action: "E-posta Gönder",
      available: true
    },
    {
      icon: Phone,
      title: "Telefon Desteği",
      description: "+90 (212) 555-0123",
      action: "Ara",
      available: false
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <ResponsiveContainer maxWidth="xl" padding="md">
        <div className="py-8">
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold bg-gradient-primary bg-clip-text text-transparent mb-4">
              Destek Merkezi
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Size yardımcı olmak için buradayız. Sorularınız için bizimle iletişime geçin.
            </p>
          </div>

          <Tabs defaultValue="faq" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="faq">Sık Sorulan Sorular</TabsTrigger>
              <TabsTrigger value="contact">İletişim</TabsTrigger>
              <TabsTrigger value="ticket">Destek Talebi</TabsTrigger>
            </TabsList>

            <TabsContent value="faq" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <HelpCircle className="w-5 h-5" />
                    Sık Sorulan Sorular
                  </CardTitle>
                  <CardDescription>
                    En çok merak edilen konular ve cevapları
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Accordion type="single" collapsible className="w-full">
                    {faqItems.map((item, index) => (
                      <AccordionItem key={index} value={`item-${index}`}>
                        <AccordionTrigger className="text-left">
                          {item.question}
                        </AccordionTrigger>
                        <AccordionContent className="text-muted-foreground">
                          {item.answer}
                        </AccordionContent>
                      </AccordionItem>
                    ))}
                  </Accordion>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="contact" className="mt-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {contactMethods.map((method, index) => {
                  const IconComponent = method.icon;
                  return (
                    <Card key={index} className={`${!method.available ? 'opacity-60' : ''}`}>
                      <CardHeader className="text-center">
                        <div className="mx-auto w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mb-4">
                          <IconComponent className="w-6 h-6 text-primary" />
                        </div>
                        <CardTitle>{method.title}</CardTitle>
                        <CardDescription>{method.description}</CardDescription>
                      </CardHeader>
                      <CardContent className="text-center">
                        <Button 
                          className="w-full" 
                          disabled={!method.available}
                          variant={method.available ? "default" : "outline"}
                        >
                          {method.action}
                          {method.available && <ExternalLink className="w-4 h-4 ml-2" />}
                        </Button>
                        {!method.available && (
                          <p className="text-xs text-muted-foreground mt-2">
                            Yakında aktif olacak
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  );
                })}
              </div>

              <Card className="mt-6">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="w-5 h-5" />
                    Destek Saatleri
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-semibold mb-2">Canlı Destek</h4>
                      <p className="text-sm text-muted-foreground">7/24 Aktif</p>
                    </div>
                    <div>
                      <h4 className="font-semibold mb-2">E-posta Desteği</h4>
                      <p className="text-sm text-muted-foreground">24 saat içinde yanıt</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="ticket" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Send className="w-5 h-5" />
                    Destek Talebi Oluştur
                  </CardTitle>
                  <CardDescription>
                    Detaylı yardım için destek talebi oluşturun
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="name">Ad Soyad</Label>
                      <Input id="name" placeholder="Adınızı girin" />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="email">E-posta</Label>
                      <Input id="email" type="email" placeholder="E-posta adresinizi girin" />
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="subject">Konu</Label>
                    <Input id="subject" placeholder="Sorun başlığını girin" />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="message">Mesaj</Label>
                    <Textarea 
                      id="message" 
                      placeholder="Sorununuzu detaylı olarak açıklayın..."
                      className="min-h-[120px]"
                    />
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>Talebiniz 24 saat içinde yanıtlanacaktır</span>
                  </div>
                  
                  <Button className="w-full">
                    <Send className="w-4 h-4 mr-2" />
                    Destek Talebi Gönder
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </ResponsiveContainer>
    </div>
  );
};

export default Support;