import { useState } from 'react'
import ReactMarkdown from 'react-markdown'

// Auto-detect API URL: use env var in prod, localhost in dev
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [citations, setCitations] = useState([])
  const [loading, setLoading] = useState(false)
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [showDebug, setShowDebug] = useState(false)

  const handleUpload = async () => {
    if (!file) return alert('Please select a PDF first')
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (data.status === 'success') {
        alert(`✅ Indexed ${data.chunks} chunks from ${data.filename}`)
        setAnswer('')
        setCitations([])
      } else {
        alert(`❌ Upload failed: ${JSON.stringify(data)}`)
      }
    } catch (e) {
      alert(`❌ Error: ${e.message}`)
    }
    setUploading(false)
  }

  const handleAsk = async () => {
    if (!question.trim()) return
    setLoading(true)
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      })
      const data = await res.json()
      setAnswer(data.answer)
      setCitations(data.citations || [])
    } catch (e) {
      setAnswer(`❌ Error connecting to backend: ${e.message}`)
      setCitations([])
    }
    setLoading(false)
  }

  return (
    <div className="container" style={{ 
      maxWidth: 850, margin: '2rem auto', padding: '2rem', fontFamily: 'system-ui, -apple-system, sans-serif',
      background: '#ffffff', borderRadius: 16, boxShadow: '0 4px 20px rgba(0,0,0,0.08)', minHeight: '90vh'
    }}>
      <h1 style={{ fontSize: '1.75rem', fontWeight: 700, marginBottom: '1.5rem', color: '#111' }}>
        📚 RAG Document Assistant
      </h1>
      
      {/* PDF Upload */}
      <div style={{ marginBottom: '1.5rem', padding: '1rem', background: '#f8fafc', borderRadius: 12, border: '1px solid #e2e8f0' }}>
        <label style={{ display: 'block', fontWeight: 600, marginBottom: '0.5rem', color: '#334155' }}>
          1. Upload a PDF:
        </label>
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <input type="file" accept=".pdf" onChange={e => setFile(e.target.files[0])} 
                 style={{ padding: '0.5rem', border: '1px solid #cbd5e1', borderRadius: 8, background: '#fff', flex: 1, minWidth: 200 }} />
          <button onClick={handleUpload} disabled={!file || uploading} style={{
            background: '#0f172a', color: '#fff', border: 'none', padding: '0.6rem 1.2rem', borderRadius: 8, 
            fontWeight: 500, cursor: uploading ? 'not-allowed' : 'pointer', opacity: uploading ? 0.7 : 1
          }}>
            {uploading ? 'Indexing...' : 'Upload'}
          </button>
        </div>
      </div>

      {/* Question Input */}
      <label style={{ display: 'block', fontWeight: 600, marginBottom: '0.5rem', color: '#334155' }}>
        2. Ask a question:
      </label>
      <textarea
        value={question}
        onChange={e => setQuestion(e.target.value)}
        placeholder="e.g., What is the main contribution of this paper?"
        onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleAsk())}
        style={{ width: '100%', minHeight: 100, padding: '0.75rem', border: '1px solid #cbd5e1', borderRadius: 10, 
                 fontSize: '1rem', resize: 'vertical', fontFamily: 'inherit', boxSizing: 'border-box' }}
      />
      <button onClick={handleAsk} disabled={loading || !question.trim()} style={{
        background: '#2563eb', color: '#fff', border: 'none', padding: '0.75rem 1.5rem', borderRadius: 8, 
        fontWeight: 600, fontSize: '1rem', cursor: loading || !question.trim() ? 'not-allowed' : 'pointer',
        marginTop: '0.75rem', transition: 'background 0.2s'
      }}>
        {loading ? 'Thinking...' : 'Ask'}
      </button>

      {/* Answer Display */}
      {answer && (
        <div style={{ marginTop: '2rem', padding: '1.25rem', background: '#fafbfc', borderRadius: 12, border: '1px solid #e2e8f0' }}>
          <div style={{ fontWeight: 700, fontSize: '1.1rem', marginBottom: '1rem', color: '#0f172a' }}>💡 Answer:</div>
          
          <ReactMarkdown
            components={{
              // Prominent, scannable headings
              h1: ({node, ...props}) => <h3 style={{marginTop: '1.25rem', marginBottom: '0.75rem', fontSize: '1.35rem', fontWeight: 700, color: '#0f172a'}} {...props} />,
              h2: ({node, ...props}) => <h4 style={{marginTop: '1.1rem', marginBottom: '0.6rem', fontSize: '1.2rem', fontWeight: 700, color: '#1e293b'}} {...props} />,
              h3: ({node, ...props}) => <h5 style={{marginTop: '0.9rem', marginBottom: '0.5rem', fontSize: '1.05rem', fontWeight: 600, color: '#334155'}} {...props} />,
              // Clean lists
              ul: ({node, ...props}) => <ul style={{paddingLeft: '1.4rem', margin: '0.6rem 0'}} {...props} />,
              li: ({node, ...props}) => <li style={{margin: '0.35rem 0', lineHeight: 1.6}} {...props} />,
              // Readable inline & block code
              code: ({node, inline, ...props}) => 
                inline ? <code style={{background: '#e2e8f0', padding: '2px 6px', borderRadius: 6, fontSize: '0.95em', color: '#0f172a'}} {...props} /> 
                       : <pre style={{background: '#f1f5f9', padding: '0.85rem', borderRadius: 10, overflowX: 'auto', fontSize: '0.9em', border: '1px solid #cbd5e1'}} {...props} />,
              strong: ({node, ...props}) => <strong style={{fontWeight: 700, color: '#0f172a'}} {...props} />,
              p: ({node, ...props}) => <p style={{lineHeight: 1.65, marginBottom: '0.8rem'}} {...props} />,
            }}
          >
            {answer}
          </ReactMarkdown>
          
          {/* Citations with Debug Toggle */}
          {citations.length > 0 && (
            <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0' }}>
              <details style={{ width: '100%' }}>
                <summary style={{ cursor: 'pointer', fontWeight: 600, color: '#0f172a', fontSize: '1rem', marginBottom: '0.5rem' }}>
                  📚 Sources ({citations.length})
                </summary>
                <div style={{ marginTop: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {citations.map((c, i) => (
                    <div key={i} style={{ 
                      padding: '0.7rem', background: '#fff', borderRadius: 8, 
                      borderLeft: '4px solid #3b82f6', boxShadow: '0 2px 6px rgba(0,0,0,0.04)',
                      fontSize: '0.95rem', lineHeight: 1.5
                    }}>
                      <strong style={{ color: '#0f172a' }}>Page {c.page}</strong>: {c.snippet}
                      {showDebug && (
                        <span style={{ color: '#64748b', fontSize: '0.85rem', marginLeft: '0.5rem', background: '#f1f5f9', padding: '2px 6px', borderRadius: 4 }}>
                          (score: {c.score.toFixed(3)})
                        </span>
                      )}
                    </div>
                  ))}
                </div>
              </details>
              
              <div style={{ marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <label style={{ fontSize: '0.85rem', color: '#64748b', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <input 
                    type="checkbox" 
                    checked={showDebug} 
                    onChange={e => setShowDebug(e.target.checked)} 
                    style={{ accentColor: '#3b82f6' }} 
                  />
                  Show retrieval scores (debug)
                </label>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App