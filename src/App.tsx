import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Upload, Table, BarChart3 } from "lucide-react";

// API Service
async function runAnalysis(file: File, testType: string, params: any) {
  console.log("🔍 Frontend Debug - testType:", testType);
  console.log("🔍 Frontend Debug - params:", params);
  
  // Step 1: Upload file
  const formData = new FormData();
  formData.append("file", file);

  const uploadRes = await fetch("http://localhost:8000/upload", {
    method: "POST",
    body: formData,
  });
  if (!uploadRes.ok) throw new Error("File upload failed");
  const uploadData = await uploadRes.json();
  console.log("✅ File uploaded with ID:", uploadData.file_id);

  // Step 2: Run analysis with file_id
  const analysisPayload = {
    file_id: uploadData.file_id,
    analysis_type: "hypothesis",
    parameters: { test_type: testType, ...params },
  };
  
  console.log("📤 Sending analysis payload:", JSON.stringify(analysisPayload, null, 2));

  const analysisRes = await fetch("http://localhost:8000/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(analysisPayload),
  });
  
  if (!analysisRes.ok) {
    const errorText = await analysisRes.text();
    console.error("❌ Analysis failed:", analysisRes.status, errorText);
    throw new Error(`Analysis failed: ${analysisRes.status} - ${errorText}`);
  }
  
  const result = await analysisRes.json();
  console.log("✅ Analysis result:", result);
  return result;
}

// Sidebar Component
const Sidebar = ({ selected, onSelect }: { selected: string; onSelect: (id: string) => void }) => {
  const tests = [
    { id: "t_test_independent", label: "Bağımsız T-Testi" },
    { id: "t_test_one_sample", label: "Tek Örneklem T-Testi" },
    { id: "t_test_paired", label: "Eşleştirilmiş T-Testi" },
    { id: "anova_one_way", label: "ANOVA (Tek Yönlü)" },
    { id: "anova_two_way", label: "ANOVA (İki Yönlü)" },
    { id: "mixed_anova", label: "Karma ANOVA" },
    { id: "manova", label: "MANOVA" },
    { id: "correlation", label: "Korelasyon" },
    { id: "chi_square", label: "Ki-Kare" },
    { id: "wilcoxon_signed_rank", label: "Wilcoxon İşaretli Sıra" },
    { id: "mann_whitney_u", label: "Mann-Whitney U" },
    { id: "kruskal_wallis", label: "Kruskal-Wallis" },
  ];

  return (
    <aside className="w-64 h-screen bg-gray-100 dark:bg-gray-900 p-4 border-r">
      <h2 className="text-xl font-bold mb-4">Analiz Seçimi</h2>
      <ul className="space-y-2">
        {tests.map((t) => (
          <li key={t.id}>
            <button
              className={`w-full text-left px-3 py-2 rounded-lg ${
                selected === t.id ? "bg-blue-500 text-white" : "hover:bg-gray-200 dark:hover:bg-gray-800"
              }`}
              onClick={() => onSelect(t.id)}
            >
              {t.label}
            </button>
          </li>
        ))}
      </ul>
    </aside>
  );
};

// Upload Form
const UploadForm = ({ onFile }: { onFile: (file: File) => void }) => (
  <Card className="mb-4">
    <CardContent className="p-4">
      <label className="flex items-center gap-2 cursor-pointer">
        <Upload className="w-5 h-5" />
        <input type="file" className="hidden" onChange={(e) => e.target.files && onFile(e.target.files[0])} />
        Veri Yükle (CSV/Excel)
      </label>
    </CardContent>
  </Card>
);

// Parameters Form
const ParametersForm = ({ selectedTest, onRun }: { selectedTest: string; onRun: (params: any) => void }) => {
  const [form, setForm] = useState<any>({});
  const handleChange = (k: string, v: string | string[]) => setForm({ ...form, [k]: v });

  return (
    <Card className="mb-4">
      <CardContent className="p-4 space-y-2">
        <h3 className="font-semibold">Parametreler ({selectedTest})</h3>
        {selectedTest === "t_test_paired" && (
          <>
            <input className="border p-2 rounded w-full" placeholder="paired_col_1 (örn: pre)" onChange={(e) => handleChange("paired_col_1", e.target.value)} />
            <input className="border p-2 rounded w-full" placeholder="paired_col_2 (örn: post)" onChange={(e) => handleChange("paired_col_2", e.target.value)} />
          </>
        )}
        {selectedTest === "anova_one_way" && (
          <>
            <input className="border p-2 rounded w-full" placeholder="dependent_var (örn: score)" onChange={(e) => handleChange("dependent_var", e.target.value)} />
            <input className="border p-2 rounded w-full" placeholder="independent_var (örn: group)" onChange={(e) => handleChange("independent_var", e.target.value)} />
          </>
        )}
        {selectedTest === "manova" && (
          <>
            <input className="border p-2 rounded w-full" placeholder="dependent_columns (örn: y1,y2)" onChange={(e) => handleChange("dependent_columns", e.target.value.split(","))} />
            <input className="border p-2 rounded w-full" placeholder="independent_formula (örn: group)" onChange={(e) => handleChange("independent_formula", e.target.value)} />
          </>
        )}
        {selectedTest === "mixed_anova" && (
          <>
            <input className="border p-2 rounded w-full" placeholder="subject_column (örn: id)" onChange={(e) => handleChange("subject_column", e.target.value)} />
            <input className="border p-2 rounded w-full" placeholder="within_column (örn: time)" onChange={(e) => handleChange("within_column", e.target.value)} />
            <input className="border p-2 rounded w-full" placeholder="between_column (örn: group)" onChange={(e) => handleChange("between_column", e.target.value)} />
            <input className="border p-2 rounded w-full" placeholder="dv_column (örn: score)" onChange={(e) => handleChange("dv_column", e.target.value)} />
          </>
        )}
        <Button className="w-full mt-2" onClick={() => onRun(form)}>Analizi Çalıştır</Button>
      </CardContent>
    </Card>
  );
};

// Results
const Results = ({ data }: { data: any }) => (
  <Card>
    <CardContent className="p-4 space-y-4">
      <h3 className="font-semibold flex items-center gap-2"><Table className="w-5 h-5" /> Sonuçlar</h3>
      {data?.anova_table && <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm">{JSON.stringify(data.anova_table, null, 2)}</pre>}
      {data?.interpretation && <div className="p-2 bg-blue-50 dark:bg-blue-900 rounded">{data.interpretation}</div>}
      {data?.manova_result_summary && <pre className="bg-gray-100 dark:bg-gray-800 p-2 rounded text-sm">{data.manova_result_summary}</pre>}
      <div className="flex items-center gap-2 text-gray-500"><BarChart3 className="w-5 h-5" /> Grafikler buraya gelecek</div>
    </CardContent>
  </Card>
);

// Main App
export default function App() {
  const [selectedTest, setSelectedTest] = useState("t_test_independent");
  const [file, setFile] = useState<File | null>(null);
  const [results, setResults] = useState<any>(null);

  const handleRun = async (params: any) => {
    if (!file) {
      alert("Lütfen önce veri dosyası yükleyin.");
      return;
    }
    try {
      const res = await runAnalysis(file, selectedTest, params);
      setResults(res);
    } catch (e: any) {
      alert("Hata: " + e.message);
    }
  };

  return (
    <div className="flex h-screen">
      <Sidebar selected={selectedTest} onSelect={setSelectedTest} />
      <main className="flex-1 p-6 overflow-y-auto bg-white dark:bg-gray-950">
        <h1 className="text-2xl font-bold mb-6">VeriVio – Akademik Analiz Platformu</h1>
        <UploadForm onFile={setFile} />
        <ParametersForm selectedTest={selectedTest} onRun={handleRun} />
        <Results data={results} />
      </main>
    </div>
  );
}
