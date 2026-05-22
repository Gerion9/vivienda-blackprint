import { useState, useEffect } from 'react';
import Head from 'next/head';

export default function Login() {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const cookies = document.cookie.split(';');
    const hasToken = cookies.some(c => c.trim().startsWith('auth_token=authenticated'));
    if (hasToken) {
      window.location.href = '/dashboard';
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const res = await fetch('/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ password }),
      });

      if (res.ok) {
        window.location.href = '/dashboard';
      } else {
        const data = await res.json();
        setError(data.message || 'Contraseña incorrecta');
      }
    } catch (err) {
      setError('Ocurrió un error de conexión');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-shell">
      <Head>
        <title>BlackPrint × Orange | Vivienda CDMX</title>
        <meta name="description" content="Acceso al Módulo de Vivienda CDMX — BlackPrint × Orange" />
        <link rel="icon" href="/brand/blackprint-emblem-light.png" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Bai+Jamjuree:wght@200;300;400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Work+Sans:ital,wght@0,300..800;1,300..800&display=swap"
          rel="stylesheet"
        />
      </Head>

      <div className="login-card">
        <div className="login-lockup">
          <img
            src="/brand/blackprint-emblem-light.png"
            alt="BlackPrint"
            className="lockup-bp"
          />
          <span className="lk-cross" aria-hidden="true">×</span>
          <span className="lk-orange">
            <img src="/brand/orange-logo.png" alt="" className="orange-logo" aria-hidden="true" />
            <span className="orange-name">Orange</span>
          </span>
        </div>
        <p className="login-eyebrow">Plataforma de inteligencia territorial</p>
        <h1 className="login-title">
          Módulo de Vivienda <span>CDMX</span>
        </h1>
        <p className="login-desc">
          Una colaboración de BlackPrint para <span className="orange-name">Orange</span>. Ingresa la contraseña de acceso para abrir el reporte editorial.
        </p>

        <form onSubmit={handleSubmit} className="login-form">
          <label htmlFor="password">Contraseña de acceso</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="••••••••••••"
            required
            disabled={loading}
            autoFocus
          />

          {error && <div className="login-error">{error}</div>}

          <button type="submit" className="login-submit" disabled={loading}>
            {loading ? (
              <span className="spinner" aria-hidden="true" />
            ) : (
              <>
                <span>Ingresar al dashboard</span>
                <span className="arrow" aria-hidden="true">→</span>
              </>
            )}
          </button>
        </form>

        <div className="login-foot">
          <span>Conexión segura · Gated Edge Middleware</span>
          <span>© {new Date().getFullYear()} BlackPrint</span>
        </div>
      </div>

      <style jsx global>{`
        :root {
          --depth-9: #2F2F2F;
          --depth-6: #646669;
          --depth-5: #8D9398;
          --depth-3: #B7BCC0;
          --depth-1: #D7DCE0;
          --on-secondary: #231F20;

          --blue-p: #0875E3;
          --blue-h: #52BCF5;
          --coral:  #FF6F61;

          --surface:   #FFFFFF;
          --surface-2: #F7F8F9;
          --ink:       var(--on-secondary);
          --ink-soft:  var(--depth-6);
          --ink-mute:  var(--depth-5);
          --line:      var(--depth-1);

          --font-display: 'Bai Jamjuree', 'Work Sans', sans-serif;
          --font-sans:    'Work Sans', system-ui, sans-serif;
          --font-ui:      'Inter', 'Work Sans', system-ui, sans-serif;
          --font-mono:    'JetBrains Mono', ui-monospace, monospace;
          --font-orange:  'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }

        html, body {
          background: var(--surface-2);
          color: var(--ink);
          font-family: var(--font-sans);
          -webkit-font-smoothing: antialiased;
        }

        body {
          background:
            radial-gradient(rgba(35,31,32,0.025) 1px, transparent 1px) 0 0 / 22px 22px,
            var(--surface-2);
          min-height: 100vh;
        }

        .login-shell {
          min-height: 100vh;
          display: flex;
          align-items: center;
          justify-content: center;
          padding: 24px;
        }

        .login-card {
          width: 100%;
          max-width: 460px;
          background: var(--surface);
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 48px 40px 32px;
          box-shadow:
            0 8px 24px rgba(0, 0, 0, 0.06),
            0 32px 64px rgba(0, 0, 0, 0.08);
          animation: card-in 600ms cubic-bezier(0.16, 1, 0.3, 1);
        }
        @keyframes card-in {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: none; }
        }

        .login-lockup {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 16px;
          margin-bottom: 30px;
          font-family: var(--font-ui);
          font-size: 14px;
        }
        .lockup-bp {
          height: 24px;
          width: auto;
          display: block;
          filter: brightness(0);
          opacity: 0.88;
        }
        .lk-cross {
          color: var(--ink-mute);
          font-size: 20px;
          font-weight: 300;
          line-height: 1;
          user-select: none;
          letter-spacing: 0;
        }
        .lk-orange {
          display: inline-flex;
          align-items: center;
          gap: 7px;
          color: var(--coral);
          padding: 4px 12px 4px 7px;
          border: 1px solid rgba(255, 111, 97, 0.45);
          border-radius: 999px;
          background: rgba(255, 111, 97, 0.10);
          line-height: 1;
        }
        .lk-orange .orange-name {
          font-family: var(--font-orange);
          font-weight: 700;
          font-size: 14px;
          letter-spacing: -0.02em;
          color: var(--coral);
          line-height: 1;
          display: inline-block;
          transform: translateY(1px);
        }
        .orange-logo {
          height: 18px;
          width: auto;
          display: block;
          flex-shrink: 0;
        }
        .orange-name {
          font-family: var(--font-orange);
          font-weight: 700;
          color: var(--coral);
          letter-spacing: -0.02em;
          line-height: 1;
          display: inline-block;
        }

        .login-eyebrow {
          text-align: center;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.22em;
          text-transform: uppercase;
          color: var(--ink-mute);
          margin: 0 0 12px;
        }

        .login-title {
          text-align: center;
          font-family: var(--font-display);
          font-weight: 700;
          font-size: 32px;
          line-height: 1.1;
          letter-spacing: -0.025em;
          color: var(--ink);
          margin: 0 0 12px;
        }
        .login-title span { color: var(--blue-p); }

        .login-desc {
          text-align: center;
          font-family: var(--font-sans);
          font-size: 14px;
          line-height: 1.55;
          color: var(--ink-soft);
          margin: 0 auto 30px;
          max-width: 38ch;
        }

        .login-form { display: flex; flex-direction: column; gap: 0; }
        .login-form label {
          display: block;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.20em;
          text-transform: uppercase;
          color: var(--ink-mute);
          margin-bottom: 8px;
        }

        .login-form input[type="password"] {
          width: 100%;
          height: 48px;
          padding: 0 16px;
          background: var(--surface-2);
          border: 1px solid var(--line);
          border-radius: 12px;
          font-family: var(--font-ui);
          font-size: 15px;
          color: var(--ink);
          outline: none;
          transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }
        .login-form input[type="password"]:focus {
          background: var(--surface);
          border-color: var(--blue-p);
          box-shadow: 0 0 0 4px rgba(8, 117, 227, 0.16);
        }
        .login-form input[type="password"]::placeholder {
          color: var(--ink-mute);
          letter-spacing: 0.2em;
        }

        .login-error {
          margin-top: 14px;
          padding: 10px 14px;
          background: rgba(170, 32, 9, 0.06);
          border: 1px solid rgba(170, 32, 9, 0.2);
          border-radius: 10px;
          color: #AA2009;
          font-size: 13px;
          font-family: var(--font-ui);
        }

        .login-submit {
          margin-top: 22px;
          width: 100%;
          height: 48px;
          background: var(--blue-p);
          color: var(--surface);
          border: none;
          border-radius: 999px;
          font-family: var(--font-ui);
          font-size: 15px;
          font-weight: 500;
          letter-spacing: 0.01em;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          cursor: pointer;
          transition: background 120ms ease, box-shadow 220ms ease, transform 120ms ease;
          box-shadow: 0 4px 14px rgba(8, 117, 227, 0.28);
        }
        .login-submit:hover:not(:disabled) {
          background: var(--blue-h);
          box-shadow: 0 6px 20px rgba(8, 117, 227, 0.32);
        }
        .login-submit:active:not(:disabled) {
          transform: translateY(1px);
        }
        .login-submit:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
        .login-submit .arrow {
          font-size: 16px;
          transition: transform 180ms ease;
        }
        .login-submit:hover:not(:disabled) .arrow {
          transform: translateX(3px);
        }

        .spinner {
          width: 18px;
          height: 18px;
          border: 2px solid rgba(255,255,255,0.35);
          border-top-color: #ffffff;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }
        @keyframes spin { to { transform: rotate(360deg); } }

        .login-foot {
          margin-top: 28px;
          padding-top: 20px;
          border-top: 1px solid var(--line);
          display: flex;
          justify-content: space-between;
          gap: 12px;
          font-family: var(--font-mono);
          font-size: 10px;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--ink-mute);
        }

        @media (max-width: 480px) {
          .login-card { padding: 36px 24px 24px; border-radius: 18px; }
          .login-title { font-size: 28px; }
          .login-foot { flex-direction: column; align-items: flex-start; }
        }
      `}</style>
    </div>
  );
}
