import React, { useState } from 'react';
import axios from 'axios';
import { toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

interface ControlPanelProps {
  systemStatus: any;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ systemStatus }) => {
  const [forceTarget, setForceTarget] = useState<number>(45);
  const [isEmergencyStopActive, setIsEmergencyStopActive] = useState<boolean>(false);
  const [jobId, setJobId] = useState<string>('JOB_' + Date.now());
  const [beginnerMode, setBeginnerMode] = useState<boolean>(true);  // Default to beginner mode

  const handleStartJob = async () => {
    try {
      const jobSpec = {
        job_id: jobId,
        target_quality: 'DIN_876_GRADE_00',
        cycles: 10,
        force_target: forceTarget
      };

      const response = await axios.post('http://localhost:8000/jobs/start', jobSpec);
      console.log('Job started:', response.data);
      toast.success('Job started successfully!');
    } catch (error) {
      console.error('Failed to start job:', error);
      toast.error('Failed to start job');
    }
  };

  const handleEmergencyStop = async () => {
    try {
      const response = await axios.post('http://localhost:8000/control/emergency-stop');
      console.log('Emergency stop triggered:', response.data);
      setIsEmergencyStopActive(true);
      setTimeout(() => setIsEmergencyStopActive(false), 3000);
    } catch (error) {
      console.error('Failed to trigger emergency stop:', error);
    }
  };

  const handleForceUpdate = async () => {
    try {
      const response = await axios.post(`http://localhost:8000/control/force-update?force_target=${forceTarget}`);
      console.log('Force target updated:', response.data);
      toast.success(`Force target updated to ${forceTarget}N`);
    } catch (error) {
      console.error('Failed to update force target:', error);
      toast.error('Failed to update force target');
    }
  };

  return (
    <div className="control-panel">
      <h2>System Controls</h2>
      
      {/* Beginner Mode Toggle */}
      <div className="control-section">
        <div className="control-group">
          <label>
            <input
              type="checkbox"
              checked={beginnerMode}
              onChange={(e) => setBeginnerMode(e.target.checked)}
              style={{ marginRight: '8px' }}
            />
            Beginner Mode
          </label>
          <p style={{ fontSize: '12px', color: '#666', margin: '5px 0 0 0' }}>
            {beginnerMode 
              ? 'Beginner mode: Advanced controls hidden for safety' 
              : 'Expert mode: All controls visible'}
          </p>
        </div>
      </div>
      
      <div className="control-section">
        <h3>Process Control</h3>
        <div className="control-group">
          <label htmlFor="force-target">Target Force (N):</label>
          <input
            id="force-target"
            type="number"
            min="10"
            max="100"
            step="1"
            value={forceTarget}
            onChange={(e) => setForceTarget(Number(e.target.value))}
          />
          <button onClick={handleForceUpdate}>Update Force</button>
        </div>
        
        <div className="control-group">
          <label htmlFor="job-id">Job ID:</label>
          <input
            id="job-id"
            type="text"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
          />
          <button onClick={handleStartJob}>Start Job</button>
        </div>
        
        {/* Advanced controls - only show in Expert Mode */}
        {!beginnerMode && (
          <div className="advanced-controls">
            <h4>Advanced Settings</h4>
            <div className="control-group">
              <label htmlFor="spindle-speed">Spindle Speed (RPM):</label>
              <input
                id="spindle-speed"
                type="number"
                min="1000"
                max="6000"
                step="100"
                defaultValue="3000"
              />
            </div>
            <div className="control-group">
              <label htmlFor="feed-rate">Feed Rate (mm/s):</label>
              <input
                id="feed-rate"
                type="number"
                min="0.1"
                max="10.0"
                step="0.1"
                defaultValue="2.0"
              />
            </div>
            <div className="control-group">
              <label htmlFor="dwell-time">Dwell Time (s):</label>
              <input
                id="dwell-time"
                type="number"
                min="0.1"
                max="10.0"
                step="0.1"
                defaultValue="1.0"
              />
            </div>
          </div>
        )}
      </div>

      <div className="control-section">
        <h3>Safety Controls</h3>
        <button 
          className={`emergency-stop ${isEmergencyStopActive ? 'active' : ''}`}
          onClick={handleEmergencyStop}
        >
          EMERGENCY STOP
        </button>
      </div>

      <div className="control-section">
        <h3>Current Status</h3>
        <div className="status-info">
          <p><strong>System Status:</strong> {systemStatus?.status || 'Unknown'}</p>
          <p><strong>Active Components:</strong> {systemStatus?.active_components?.join(', ') || 'None'}</p>
          <p><strong>Last Update:</strong> {systemStatus?.timestamp ? new Date(systemStatus.timestamp).toLocaleString() : 'Never'}</p>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;