import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [fileList, setFileList] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleFiles = (e) => {
    // Convert FileList to Array and merge with existing to allow multiple adds
    const newFiles = Array.from(e.target.files);
    setFileList((prev) => [...prev, ...newFiles]);
  };

  const removeFile = (index) => {
    setFileList(fileList.filter((_, i) => i !== index));
  };

  const processBatch = async () => {
    setLoading(true);
    let batch = [];
    for (const file of fileList) {
      const formData = new FormData();
      formData.append("file", file);
      try {
        const res = await axios.post("http://localhost:8000/predict", formData);
        batch.push({ name: file.name, ...res.data });
      } catch (err) { 
        batch.push({ name: file.name, label: "Error", confidence: 0, tier: "N/A" });
      }
    }
    setResults(batch);
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-[#F4F8F8] p-10 font-sans text-[#003c3c]">
      <div className="max-w-5xl mx-auto bg-white p-10 rounded-[24px] shadow-2xl border-l-[16px] border-[#003c3c]">
        <h1 className="text-5xl font-black mb-2 tracking-tight">DocuClassAI</h1>
        <p className="text-gray-500 font-bold mb-10 text-xl">Statutory Planning Document Triage</p>

        {/* Upload Zone */}
        <label className="block w-full border-4 border-dashed border-[#003c3c]/20 rounded-2xl p-10 bg-[#F4F8F8] text-center cursor-pointer hover:border-[#003c3c] transition-all">
          <input type="file" multiple onChange={handleFiles} className="hidden" />
          <p className="font-bold text-lg">Click to select Planning PDFs/DOCX</p>
        </label>

        {/* File List Grid */}
        <div className="mt-6 grid grid-cols-2 gap-2">
          {fileList.map((f, i) => (
            <div key={i} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200">
              <span className="text-xs truncate font-medium">{f.name}</span>
              <button onClick={() => removeFile(i)} className="text-red-500 hover:text-red-700 font-bold ml-2">✕</button>
            </div>
          ))}
        </div>

        {/* Process Button */}
        <button onClick={processBatch} disabled={loading || fileList.length === 0}
          className="w-full mt-8 bg-[#003c3c] text-white py-4 rounded-2xl font-black text-lg hover:bg-[#015e5e] transition-all disabled:opacity-50">
          {loading ? "PROCESSING BATCH..." : "CLASSIFY DOCUMENT BATCH"}
        </button>

        {/* Results Table */}
        {results.length > 0 && (
          <div className="mt-10 overflow-hidden rounded-2xl border border-[#003c3c]/20">
            <table className="w-full text-left">
              <thead className="bg-[#F4F8F8] uppercase text-xs font-black">
                <tr><th className="p-4">Filename</th><th className="p-4">Class</th><th className="p-4">Confidence</th><th className="p-4">Tier</th></tr>
              </thead>
              <tbody>
                {results.map((r, i) => (
                  <tr key={i} className="border-t">
                    <td className="p-4 text-sm font-medium">{r.name}</td>
                    <td className="p-4 font-bold">{r.label}</td>
                    <td className="p-4 font-mono font-bold">{(r.confidence * 100).toFixed(1)}%</td>
                    <td className={`p-4 font-bold text-xs ${r.tier.includes("1") ? 'text-green-600' : 'text-orange-600'}`}>
                      {r.tier}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;