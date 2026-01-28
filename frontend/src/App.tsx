import React, { useState, useEffect } from 'react';
import './App.css';
import Dashboard from './components/Dashboard';
import WebSocketService from './services/WebSocketService';

// Define TypeScript interfaces
interface User {
  username: string;
  role: string;
}

interface LoginCredentials {
  username: string;
  password: string;
}

// Protected Route Component
const ProtectedRoute: React.FC<{ children: React.ReactNode; isAuthenticated: boolean }> = ({ 
  children, 
  isAuthenticated 
}) => {
  if (!isAuthenticated) {
    return <LoginForm />;
  }
  return <>{children}</>;
};

// Login Form Component
const LoginForm: React.FC = () => {
  const [credentials, setCredentials] = useState<LoginCredentials>({ username: '', password: '' });
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      // Make login request to backend
      const response = await fetch('/auth/token', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          username: credentials.username,
          password: credentials.password,
        }),
      });

      if (!response.ok) {
        throw new Error('Invalid credentials');
      }

      const data = await response.json();
      const { access_token, token_type } = data;

      // Store token in localStorage
      localStorage.setItem('access_token', access_token);
      localStorage.setItem('token_type', token_type);

      // Reload the app to trigger authentication check
      window.location.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="login-container">
      <div className="login-form">
        <h2>Login</h2>
        {error && <div className="error-message">{error}</div>}
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="username">Username:</label>
            <input
              type="text"
              id="username"
              value={credentials.username}
              onChange={(e) => setCredentials({...credentials, username: e.target.value})}
              required
            />
          </div>
          <div className="input-group">
            <label htmlFor="password">Password:</label>
            <input
              type="password"
              id="password"
              value={credentials.password}
              onChange={(e) => setCredentials({...credentials, password: e.target.value})}
              required
            />
          </div>
          <button type="submit" disabled={isLoading}>
            {isLoading ? 'Logging in...' : 'Login'}
          </button>
        </form>
      </div>
    </div>
  );
};

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Check authentication status on mount
  useEffect(() => {
    const token = localStorage.getItem('access_token');
    setIsAuthenticated(!!token);
  }, []);

  useEffect(() => {
    // Initialize WebSocket connection
    const wsService = WebSocketService.getInstance();
    
    wsService.on('connected', () => {
      setIsConnected(true);
    });

    wsService.on('system_update', (data: any) => {
      setSystemStatus(data);
    });

    wsService.on('disconnected', () => {
      setIsConnected(false);
    });

    // Connect to WebSocket with token if available
    const token = localStorage.getItem('access_token');
    wsService.connect(token);

    // Cleanup on unmount
    return () => {
      wsService.disconnect();
    };
  }, []);

  return (
    <div className="App">
      <ProtectedRoute isAuthenticated={isAuthenticated}>
        <>
          <header className="App-header">
            <h1>Manufacturing Cyber-Physical System Dashboard</h1>
            <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
              {isConnected ? '● Connected' : '● Disconnected'}
            </div>
          </header>
          
          <main>
            {isConnected ? (
              <Dashboard systemStatus={systemStatus} />
            ) : (
              <div className="connection-pending">
                <p>Connecting to Manufacturing CPS...</p>
              </div>
            )}
          </main>
        </>
      </ProtectedRoute>
    </div>
  );
}

export default App;