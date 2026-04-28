import React, { useState } from 'react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function ConfidenceBadge({ confidence }) {
  const pct = (confidence * 100).toFixed(1);
  if (confidence >= 0.90) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-green-100 text-green-800">
        {pct}%
      </span>
    );
  }
  if (confidence >= 0.80) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-800">
        {pct}%
      </span>
    );
  }
  if (confidence > 0) {
    return (
      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-red-100 text-red-800">
        {pct}%
      </span>
    );
  }
  return <span className="text-xs text-gray-400">—</span>;
}

function TierBadge({ tier }) {
  if (tier === 'Tier 1') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
        <span className="w-1.5 h-1.5 rounded-full bg-blue-500 inline-block" />
        Tier 1
      </span>
    );
  }
  if (tier === 'Tier 2') {
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-bold bg-purple-100 text-purple-800">
        <span className="w-1.5 h-1.5 rounded-full bg-purple-500 inline-block" />
        Tier 2
      </span>
    );
  }
  return (
    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-bold bg-gray-100 text-gray-500">
      {tier}
    </span>
  );
}

function exportCSV(results) {
  const headers = ['Filename', 'Predicted Class', 'Confidence (%)', 'Tier', 'Escalation Reason', 'Fidelity Status', 'Timestamp'];
  const rows = results.map((r) => [
    r.name,
    r.label,
    r.confidence > 0 ? (r.confidence * 100).toFixed(1) : '0.0',
    r.tier,
    r.escalation_reason ?? '',
    r.fidelity_status ?? '',
    r.timestamp,
  ]);

  const csv = [headers, ...rows]
    .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
    .join('\n');

  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `docuclassai_manifest_${new Date().toISOString().slice(0, 10)}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

export default function App() {
  const [fileList, setFileList] = useState([]);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const addFiles = (incoming) => {
    const newFiles = Array.from(incoming).filter(
      (f) => f.name.toLowerCase().endsWith('.pdf') || f.name.toLowerCase().endsWith('.docx')
    );
    setFileList((prev) => [...prev, ...newFiles]);
    setError(null);
  };

  const handleInputChange = (e) => addFiles(e.target.files);

  const handleDrop = (e) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  };

  const removeFile = (index) => setFileList((prev) => prev.filter((_, i) => i !== index));

  const clearAll = () => {
    setFileList([]);
    setResults([]);
    setError(null);
  };

  const processBatch = async () => {
    setLoading(true);
    setError(null);
    const timestamp = new Date().toISOString();
    const batch = [];

    for (const file of fileList) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const res = await axios.post(`${API_BASE}/classify`, formData);
        batch.push({ name: file.name, timestamp, ...res.data });
      } catch (err) {
        if (!err.response) {
          setError(
            `Cannot reach the DocuClassAI backend at ${API_BASE}. ` +
            'Ensure the API server is running: uv run serve'
          );
          setLoading(false);
          return;
        }
        batch.push({
          name: file.name,
          timestamp,
          label: 'Error',
          confidence: 0,
          tier: 'N/A',
          fidelity_status: 'ERROR',
          escalation_reason: err.response?.data?.detail ?? 'Unknown error',
        });
      }
    }

    setResults(batch);
    setLoading(false);
  };

  const tier2Count = results.filter((r) => r.tier === 'Tier 2').length;
  const fidelityFailCount = results.filter((r) => r.fidelity_status === 'FAILED').length;

  return (
    <div className="min-h-screen bg-[#F4F8F8] p-10 font-sans text-[#003c3c]">
      <div className="max-w-5xl mx-auto bg-white p-10 rounded-[24px] shadow-2xl border-l-[16px] border-[#003c3c]">

        {/* Header */}
        <h1 className="text-5xl font-black mb-2 tracking-tight">DocuClassAI</h1>
        <p className="text-gray-500 font-bold mb-10 text-xl">Statutory Planning Document Triage</p>

        {/* Error banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl text-red-800 text-sm font-medium">
            {error}
          </div>
        )}

        {/* Upload zone */}
        <label
          className="block w-full border-4 border-dashed border-[#003c3c]/20 rounded-2xl p-10 bg-[#F4F8F8] text-center cursor-pointer hover:border-[#003c3c] transition-all"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept=".pdf,.docx"
            onChange={handleInputChange}
            className="hidden"
          />
          <p className="font-bold text-lg">Click or drag to add planning documents</p>
          <p className="text-sm text-gray-400 mt-1">PDF and DOCX accepted · multiple files supported</p>
        </label>

        {/* File queue */}
        {fileList.length > 0 && (
          <div className="mt-6">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-bold text-gray-500 uppercase tracking-wide">
                {fileList.length} file{fileList.length > 1 ? 's' : ''} queued
              </p>
              <button
                onClick={clearAll}
                className="text-xs text-red-500 hover:text-red-700 font-bold"
              >
                Clear all
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {fileList.map((f, i) => (
                <div
                  key={i}
                  className="flex justify-between items-center p-3 bg-gray-50 rounded-lg border border-gray-200"
                >
                  <span className="text-xs truncate font-medium">{f.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-red-400 hover:text-red-700 font-bold ml-2 text-sm flex-shrink-0"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Classify button */}
        <button
          onClick={processBatch}
          disabled={loading || fileList.length === 0}
          className="w-full mt-8 bg-[#003c3c] text-white py-4 rounded-2xl font-black text-lg hover:bg-[#015e5e] transition-all disabled:opacity-50 flex items-center justify-center gap-3"
        >
          {loading ? (
            <>
              <svg
                className="animate-spin h-5 w-5 text-white"
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
              >
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
              </svg>
              CLASSIFYING…
            </>
          ) : (
            `CLASSIFY ${fileList.length > 0 ? fileList.length + ' DOCUMENT' + (fileList.length > 1 ? 'S' : '') : 'BATCH'}`
          )}
        </button>

        {/* Results */}
        {results.length > 0 && (
          <div className="mt-10">

            {/* Results header + summary badges */}
            <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
              <div className="flex items-center gap-3">
                <h2 className="font-black text-lg">
                  Results — {results.length} document{results.length > 1 ? 's' : ''}
                </h2>
                {tier2Count > 0 && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-purple-100 text-purple-800">
                    {tier2Count} Tier 2
                  </span>
                )}
                {fidelityFailCount > 0 && (
                  <span className="px-2 py-0.5 rounded-full text-xs font-bold bg-amber-100 text-amber-800">
                    ⚠ {fidelityFailCount} scan{fidelityFailCount > 1 ? 's' : ''} flagged
                  </span>
                )}
              </div>
              <button
                onClick={() => exportCSV(results)}
                className="px-4 py-2 text-sm font-bold bg-[#003c3c] text-white rounded-xl hover:bg-[#015e5e] transition-all"
              >
                Export CSV
              </button>
            </div>

            {/* Fidelity warning banner */}
            {fidelityFailCount > 0 && (
              <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-xl text-amber-800 text-sm font-medium">
                ⚠ {fidelityFailCount} document{fidelityFailCount > 1 ? 's' : ''} could not be extracted — likely image-based scans.
                These require manual classification and have not been auto-filed.
              </div>
            )}

            {/* Results table */}
            <div className="overflow-hidden rounded-2xl border border-[#003c3c]/20">
              <table className="w-full text-left">
                <thead className="bg-[#F4F8F8] uppercase text-xs font-black tracking-wide">
                  <tr>
                    <th className="p-4">Filename</th>
                    <th className="p-4">Class</th>
                    <th className="p-4">Confidence</th>
                    <th className="p-4">Tier</th>
                    <th className="p-4">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr
                      key={i}
                      className={`border-t ${r.fidelity_status === 'FAILED' ? 'bg-amber-50/60' : ''}`}
                    >
                      <td className="p-4 text-sm font-medium max-w-[180px] truncate">{r.name}</td>
                      <td className="p-4 font-bold text-sm">{r.label}</td>
                      <td className="p-4">
                        <ConfidenceBadge confidence={r.confidence} />
                      </td>
                      <td className="p-4">
                        <TierBadge tier={r.tier} />
                      </td>
                      <td className="p-4 text-xs">
                        {r.fidelity_status === 'FAILED' ? (
                          <span className="font-bold text-amber-700">⚠ Scan — human review</span>
                        ) : r.tier === 'Tier 2' && r.escalation_reason ? (
                          <span className="text-purple-600 font-medium">
                            {r.escalation_reason.replace(/_/g, ' ')}
                          </span>
                        ) : (
                          <span className="text-gray-300">—</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Legend */}
            <p className="mt-3 text-xs text-gray-400">
              <span className="font-bold">Tier 1</span>: MiniLM transformer triage ·{' '}
              <span className="font-bold">Tier 2</span>: Mistral-Nemo adjudicator ·{' '}
              Confidence threshold: 80% ·{' '}
              <span className="text-green-700 font-bold">Green</span> ≥90% ·{' '}
              <span className="text-amber-700 font-bold">Amber</span> 80–90% ·{' '}
              <span className="text-red-700 font-bold">Red</span> &lt;80%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
