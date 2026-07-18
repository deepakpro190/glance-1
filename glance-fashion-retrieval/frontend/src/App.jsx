import { useMemo, useState } from 'react';

const sampleQueries = [
  'white dress for office',
  'red tie and white shirt in a formal office',
  'dress for funeral',
  'blue shirt sitting on a park bench',
];

function App() {
  const [query, setQuery] = useState('white dress for office');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [parsedTerms, setParsedTerms] = useState([]);
  const backendBaseUrl = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8001';

  const search = async (inputQuery) => {
    const q = inputQuery.trim();
    if (!q) return;

    setLoading(true);
    setError('');

    try {
      const response = await fetch(`${backendBaseUrl}/search?q=${encodeURIComponent(q)}`);
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }
      setParsedTerms(data.parsed_terms || []);
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setResults([]);
      setParsedTerms([]);
    } finally {
      setLoading(false);
    }
  };

  const stats = useMemo(() => {
    if (!results.length) return null;
    const top = results[0];
    return {
      topImageId: top.image_id,
      topFinal: top.final_score,
      topEmbedding: top.embedding_score,
      avgFinal: (results.reduce((sum, item) => sum + item.final_score, 0) / results.length).toFixed(2),
      avgEmbedding: (results.reduce((sum, item) => sum + item.embedding_score, 0) / results.length).toFixed(2),
    };
  }, [results]);

  return (
    <div className="app-shell">
      <header>
        <div>
          <p className="eyebrow">Fashion Retrieval</p>
          <h1>Visual fashion search with ranked results</h1>
          <p className="subtitle">Search any clothing prompt and inspect the full result breakdown, scores, and metadata.</p>
        </div>
      </header>

      <section className="panel search-panel">
        <div className="search-row">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && search(query)}
            placeholder="Try: white dress for office"
          />
          <button onClick={() => search(query)} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
        <div className="chip-row">
          {sampleQueries.map((sample) => (
            <button key={sample} className="chip" onClick={() => { setQuery(sample); search(sample); }}>
              {sample}
            </button>
          ))}
        </div>
      </section>

      {error ? <div className="error">{error}</div> : null}

      {results.length ? (
        <>
          <section className="panel stats-panel">
            <h2>Query summary</h2>
            <div className="stats-grid">
              <div>
                <span className="label">Query</span>
                <strong>{query}</strong>
              </div>
              <div>
                <span className="label">Parsed terms</span>
                <strong>{parsedTerms.join(', ') || '—'}</strong>
              </div>
              <div>
                <span className="label">Top result ID</span>
                <strong>{stats?.topImageId}</strong>
              </div>
              <div>
                <span className="label">Top final score</span>
                <strong>{stats?.topFinal}</strong>
              </div>
              <div>
                <span className="label">Top embedding score</span>
                <strong>{stats?.topEmbedding}</strong>
              </div>
              <div>
                <span className="label">Avg final score</span>
                <strong>{stats?.avgFinal}</strong>
              </div>
            </div>
          </section>

          <section className="panel results-panel">
            <h2>Results</h2>
            <div className="results-list">
              {results.map((item, index) => (
                <article key={item.image_id} className="result-card">
                  <div className="result-head">
                    <div>
                      <h3>#{index + 1} · Image {item.image_id}</h3>
                      <p className="muted">Objects: {item.num_objects}</p>
                    </div>
                    <div className="score-box">
                      <span className="score-value">{item.final_score}</span>
                      <span className="score-label">Final</span>
                    </div>
                  </div>
                  {item.image_url ? (
                    <img src={item.image_url} alt={`Result ${item.image_id}`} className="result-image" />
                  ) : (
                    <div className="result-image placeholder">No image available</div>
                  )}
                  <div className="metrics-grid">
                    <div><span>Embedding</span><strong>{item.embedding_score}</strong></div>
                    <div><span>Category</span><strong>{item.category_score}</strong></div>
                    <div><span>Color</span><strong>{item.color_score}</strong></div>
                    <div><span>Style</span><strong>{item.style_score}</strong></div>
                    <div><span>Environment</span><strong>{item.environment_score}</strong></div>
                    <div><span>Object</span><strong>{item.object_score}</strong></div>
                  </div>
                  <div className="details-grid">
                    <div>
                      <h4>Categories</h4>
                      <p>{item.categories.join(', ') || '—'}</p>
                    </div>
                    <div>
                      <h4>Components</h4>
                      <pre>{JSON.stringify(item.component_scores, null, 2)}</pre>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </div>
  );
}

export default App;
