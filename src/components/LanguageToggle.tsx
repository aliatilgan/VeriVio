import { Languages } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useTranslation } from 'react-i18next';
import { useApp } from "@/contexts/AppContext";

export function LanguageToggle() {
  const { i18n } = useTranslation();
  const { state, setLanguage } = useApp();

  const changeLanguage = (lng: 'tr' | 'en') => {
    i18n.changeLanguage(lng);
    setLanguage(lng);
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" size="icon">
          <Languages className="h-[1.2rem] w-[1.2rem]" />
          <span className="sr-only">Change language</span>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuItem 
          onClick={() => changeLanguage("tr")}
          className={state.language === "tr" ? "bg-accent" : ""}
        >
          <span className="mr-2">🇹🇷</span>
          <span>Türkçe</span>
        </DropdownMenuItem>
        <DropdownMenuItem 
          onClick={() => changeLanguage("en")}
          className={state.language === "en" ? "bg-accent" : ""}
        >
          <span className="mr-2">🇺🇸</span>
          <span>English</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}