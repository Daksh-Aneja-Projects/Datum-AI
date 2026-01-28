import React from 'react';

interface StatusPanelProps {
  systemStatus: any;
}

const StatusPanel: React.FC<StatusPanelProps> = ({ systemStatus }) => {
  const healthMetrics = systemStatus?.health_metrics || {};
  const components = healthMetrics.components || {};

  return (
    <div className="status-panel">
      <h2>System Status Overview</h2>
      
      <div className="status-grid">
        <div className="status-card">
          <h3>Overall System</h3>
          <div className={`status-indicator ${systemStatus?.status?.toLowerCase() || 'unknown'}`}>
            {systemStatus?.status || 'UNKNOWN'}
          </div>
          <p>Last Updated: {systemStatus?.timestamp ? new Date(systemStatus.timestamp).toLocaleString() : 'Never'}</p>
        </div>

        <div className="status-card">
          <h3>Active Components</h3>
          <ul>
            {(systemStatus?.active_components || []).map((component: string, index: number) => (
              <li key={index}>{component}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="components-health">
        <h3>Component Health Status</h3>
        <div className="health-grid">
          {Object.entries(components).map(([name, data]: [string, any]) => (
            <div key={name} className="component-card">
              <h4>{name.replace('_', ' ').toUpperCase()}</h4>
              <div className={`health-status ${data?.status?.toLowerCase() || 'unknown'}`}>
                {data?.status || 'UNKNOWN'}
              </div>
              {data?.last_heartbeat && (
                <p>Last heartbeat: {new Date(data.last_heartbeat * 1000).toLocaleTimeString()}</p>
              )}
              {data?.health_metrics && (
                <div className="metrics">
                  {Object.entries(data.health_metrics).map(([metric, value]: [string, any]) => (
                    <span key={metric} className="metric">
                      {metric}: {typeof value === 'number' ? value.toFixed(2) : value}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {systemStatus?.error_messages && systemStatus.error_messages.length > 0 && (
        <div className="error-messages">
          <h3>Error Messages</h3>
          <ul>
            {systemStatus.error_messages.map((error: string, index: number) => (
              <li key={index} className="error-item">{error}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default StatusPanel;