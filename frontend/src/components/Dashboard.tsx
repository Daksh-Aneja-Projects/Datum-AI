import React, { useState, Component } from 'react';
import SurfaceVisualizer from './SurfaceVisualizer';
import ControlPanel from './ControlPanel';
import StatusPanel from './StatusPanel';
import SetupWizard from './SetupWizard';
import './Dashboard.css';

// Define strict TypeScript interfaces for SystemStatus
interface SystemStatus {
  status: string;
  timestamp: string;
  active_components: string[];
  health_metrics: {
    [key: string]: any;
  };
}

interface DashboardProps {
  systemStatus: SystemStatus | null;
}

// Error Boundary Component
class SystemStatusErrorBoundary extends Component<{ children: React.ReactNode }, { hasError: boolean }> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('SystemStatus rendering error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return <div className="error-boundary">Error rendering system status</div>;
    }

    return this.props.children;
  }
}

const Dashboard: React.FC<DashboardProps> = ({ systemStatus }) => {
  const [selectedTab, setSelectedTab] = useState<'visualization' | 'controls' | 'status' | 'setup'>('visualization');
  const [showSetupWizard, setShowSetupWizard] = useState(false);

  const handleShowSetupWizard = () => {
    setShowSetupWizard(true);
  };

  const handleSetupComplete = (config: any) => {
    console.log('Setup completed:', config);
    setShowSetupWizard(false);
    alert('Setup completed successfully!');
  };

  const handleSetupCancel = () => {
    setShowSetupWizard(false);
  };

  return (
    <div className="dashboard">
      <nav className="dashboard-nav">
        <button 
          className={selectedTab === 'visualization' ? 'active' : ''}
          onClick={() => setSelectedTab('visualization')}
        >
          3D Visualization
        </button>
        <button 
          className={selectedTab === 'controls' ? 'active' : ''}
          onClick={() => setSelectedTab('controls')}
        >
          Controls
        </button>
        <button 
          className={selectedTab === 'status' ? 'active' : ''}
          onClick={() => setSelectedTab('status')}
        >
          System Status
        </button>
        <button 
          className={selectedTab === 'setup' ? 'active' : ''}
          onClick={handleShowSetupWizard}
        >
          Setup Wizard
        </button>
      </nav>

      <div className="dashboard-content">
        {selectedTab === 'visualization' && (
          <SystemStatusErrorBoundary>
            <SurfaceVisualizer surfaceData={systemStatus?.surface_quality} />
          </SystemStatusErrorBoundary>
        )}
        
        {selectedTab === 'controls' && (
          <SystemStatusErrorBoundary>
            <ControlPanel systemStatus={systemStatus} />
          </SystemStatusErrorBoundary>
        )}
        
        {selectedTab === 'status' && (
          <SystemStatusErrorBoundary>
            <StatusPanel systemStatus={systemStatus} />
          </SystemStatusErrorBoundary>
        )}
      </div>

      {showSetupWizard && (
        <SetupWizard 
          onComplete={handleSetupComplete} 
          onCancel={handleSetupCancel} 
        />
      )}
    </div>
  );
};

export default Dashboard;