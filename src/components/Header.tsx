import { BarChart3, Menu, User, LogIn, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "./ThemeToggle";
import { LanguageToggle } from "./LanguageToggle";
import { useTranslation } from 'react-i18next';
import { Link, useLocation } from 'react-router-dom';
import { useState } from 'react';

const Header = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  const isActive = (path: string) => location.pathname === path;

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const closeMobileMenu = () => {
    setIsMobileMenuOpen(false);
  };

  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-md sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <div className="bg-gradient-primary p-2 rounded-lg">
              <BarChart3 className="h-6 w-6 text-white" />
            </div>
            <h1 className="text-xl font-bold bg-gradient-primary bg-clip-text text-transparent">
              DataAnalytics Pro
            </h1>
          </Link>
          
          <nav className="hidden md:flex items-center gap-6">
            <Link 
              to="/" 
              className={`text-sm font-medium transition-colors ${
                isActive('/') 
                  ? 'text-primary font-semibold' 
                  : 'text-muted-foreground hover:text-primary'
              }`}
            >
              Ana Sayfa
            </Link>
            <Link 
              to="/analyses" 
              className={`text-sm font-medium transition-colors ${
                isActive('/analyses') 
                  ? 'text-primary font-semibold' 
                  : 'text-muted-foreground hover:text-primary'
              }`}
            >
              Analizler
            </Link>
            <Link 
              to="/documentation" 
              className={`text-sm font-medium transition-colors ${
                isActive('/documentation') 
                  ? 'text-primary font-semibold' 
                  : 'text-muted-foreground hover:text-primary'
              }`}
            >
              Dokümantasyon
            </Link>
            <Link 
              to="/support" 
              className={`text-sm font-medium transition-colors ${
                isActive('/support') 
                  ? 'text-primary font-semibold' 
                  : 'text-muted-foreground hover:text-primary'
              }`}
            >
              Destek
            </Link>
          </nav>

          <div className="flex items-center gap-2">
            <ThemeToggle />
            <LanguageToggle />
            <Button 
              variant="ghost" 
              size="icon" 
              className="md:hidden"
              onClick={toggleMobileMenu}
            >
              {isMobileMenuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
            <Button variant="outline" className="hidden md:inline-flex">
              <User className="h-4 w-4 mr-2" />
              {t('header.profile')}
            </Button>
            <Button className="bg-gradient-primary hover:opacity-90 transition-opacity">
              <LogIn className="h-4 w-4 mr-2" />
              {t('header.login')}
            </Button>
          </div>
        </div>
        
        {/* Mobile Menu */}
        {isMobileMenuOpen && (
          <div className="md:hidden border-t border-border bg-card/95 backdrop-blur-md">
            <nav className="container mx-auto px-4 py-4 space-y-2">
              <Link 
                to="/" 
                className={`block py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  isActive('/') 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-primary hover:bg-muted'
                }`}
                onClick={closeMobileMenu}
              >
                Ana Sayfa
              </Link>
              <Link 
                to="/analyses" 
                className={`block py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  isActive('/analyses') 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-primary hover:bg-muted'
                }`}
                onClick={closeMobileMenu}
              >
                Analizler
              </Link>
              <Link 
                to="/documentation" 
                className={`block py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  isActive('/documentation') 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-primary hover:bg-muted'
                }`}
                onClick={closeMobileMenu}
              >
                Dokümantasyon
              </Link>
              <Link 
                to="/support" 
                className={`block py-2 px-3 rounded-md text-sm font-medium transition-colors ${
                  isActive('/support') 
                    ? 'bg-primary text-primary-foreground' 
                    : 'text-muted-foreground hover:text-primary hover:bg-muted'
                }`}
                onClick={closeMobileMenu}
              >
                Destek
              </Link>
              <div className="pt-2 border-t border-border">
                <Button variant="ghost" size="sm" className="w-full justify-start">
                  <User className="h-4 w-4 mr-2" />
                  Profil
                </Button>
                <Button size="sm" className="w-full mt-2">
                  <LogIn className="h-4 w-4 mr-2" />
                  Giriş Yap
                </Button>
              </div>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
