import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleUpload = async () => {
    const formData = new FormData();
    formData.append("file", file);
    
    // Call FastAPI backend
    const res = await axios.post("http://localhost:8000/predict", formData);
    setResult(res.data);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-10">
      <div className="max-w-2xl mx-auto bg-white p-8 rounded-xl shadow-lg border-t-8 border-[#003c3c]">
        <h1 className="text-3xl font-bold text-[#003c3c] mb-6">DocuClassAI Portal</h1>
        <input type="file" onChange={(e) => setFile(e.target.files[0])} className="mb-4 block w-full text-sm text-gray-500" />
        <button onClick={handleUpload} className="w-full bg-[#003c3c] text-white py-3 rounded-lg font-bold">
          Classify Document
        </button>
        
        {result && (
          <div className="mt-6 p-4 bg-green-50 rounded-lg">
            <p className="text-xl">Result: <span className="font-bold text-[#003c3c]">{result.label}</span></p>
          </div>
        )}
      </div>
    </div>
  );
}
export default App;