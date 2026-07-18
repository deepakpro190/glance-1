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
  const [parsed, setParsed] = useState({ colors: [], styles: [], environments: [], categories: [] });
  const [retrievalMode, setRetrievalMode] = useState('default');
  const [retrievalMessage, setRetrievalMessage] = useState('');
  const backendBaseUrl = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000';

  const getFallbackBackendUrl = (baseUrl) => {
    if (baseUrl.includes('127.0.0.1:8000')) return 'http://127.0.0.1:8001';
    if (baseUrl.includes('127.0.0.1:8001')) return 'http://127.0.0.1:8000';
    return null;
  };

  const search = async (inputQuery) => {
    const q = inputQuery.trim();
    if (!q) return;

    setLoading(true);
    setError('');

    try {
      let response;
      let data;

      try {
        response = await fetch(`${backendBaseUrl}/search?q=${encodeURIComponent(q)}`);
        data = await response.json();
      } catch (networkErr) {
        const fallbackUrl = getFallbackBackendUrl(backendBaseUrl);
        if (!fallbackUrl) {
          throw networkErr;
        }
        response = await fetch(`${fallbackUrl}/search?q=${encodeURIComponent(q)}`);
        data = await response.json();
      }

      if (!response.ok) {
        throw new Error(data.detail || 'Request failed');
      }
      setParsedTerms(data.parsed_terms || []);
      setParsed(data.parsed || { colors: [], styles: [], environments: [], categories: [] });
      setRetrievalMode(data.retrieval_mode || 'default');
      setRetrievalMessage(data.message || '');
      setResults(data.results || []);
    } catch (err) {
      setError(err.message || 'Something went wrong');
      setResults([]);
      setParsedTerms([]);
      setParsed({ colors: [], styles: [], environments: [], categories: [] });
      setRetrievalMode('default');
      setRetrievalMessage('');
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

  const hasColorConstraint = (parsed.colors || []).length > 0;
  const hasStyleConstraint = (parsed.styles || []).length > 0;
  const hasEnvironmentConstraint = (parsed.environments || []).length > 0;
  const metricText = (enabled, value) => (enabled ? value : 'N/A');

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
      {!error && retrievalMessage ? <div className="error">{retrievalMessage}</div> : null}

      {!error && !loading && !results.length && query.trim() ? (
        <section className="panel stats-panel">
          <h2>No results</h2>
          <p className="subtitle">
            {retrievalMode === 'none'
              ? 'No strict or close match found. Try relaxing one attribute such as color, style, or location.'
              : 'No results found for this query.'}
          </p>
        </section>
      ) : null}

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
                    <div><span>Color</span><strong>{metricText(hasColorConstraint, item.color_score)}</strong></div>
                    <div><span>Style</span><strong>{metricText(hasStyleConstraint, item.style_score)}</strong></div>
                    <div><span>Environment</span><strong>{metricText(hasEnvironmentConstraint, item.environment_score)}</strong></div>
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
