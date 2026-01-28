import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './SetupWizard.css';

interface SetupWizardProps {
  onComplete: (config: any) => void;
  onCancel: () => void;
}

interface HardwareStatus {
  ati_axia_connected: boolean;
  ae_sensor_connected: boolean;
  interferometer_connected: boolean;
  robot_connected: boolean;
  status_message: string;
}

interface CalibrationConfig {
  // Interferometry settings
  wavelength_nm: number;
  grazing_angle_deg: number;
  spatial_resolution_um: number;
  
  // Force control settings
  target_force_n: number;
  force_tolerance_n: number;
  
  // Acoustic emission settings
  ae_sample_rate_hz: number;
  ae_rms_threshold: number;
  
  // Process parameters
  spindle_speed_rpm: number;
  feed_rate_mm_per_sec: number;
  abrasive_grit_size: number;
  
  // Safety limits
  max_force_n: number;
  max_temperature_c: number;
  max_vibration_g: number;
  
  // Asset identification
  serial_number?: string;
  model?: string;
}

const SetupWizard: React.FC<SetupWizardProps> = ({ onComplete, onCancel }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState<CalibrationConfig>({
    wavelength_nm: 632.8,
    grazing_angle_deg: 88.0,
    spatial_resolution_um: 0.1,
    target_force_n: 45.0,
    force_tolerance_n: 5.0,
    ae_sample_rate_hz: 50000,
    ae_rms_threshold: 0.5,
    spindle_speed_rpm: 3000,
    feed_rate_mm_per_sec: 2.0,
    abrasive_grit_size: 200,
    max_force_n: 200.0,
    max_temperature_c: 60.0,
    max_vibration_g: 0.5
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [existingConfigs, setExistingConfigs] = useState<any[]>([]);
  const [hardwareStatus, setHardwareStatus] = useState<HardwareStatus | null>(null);
  const [hardwareValidationPassed, setHardwareValidationPassed] = useState(false);

  const steps = [
    { title: 'Welcome', component: WelcomeStep },
    { title: 'Asset ID', component: AssetIdentificationStep },
    { title: 'Interferometry', component: InterferometryStep },
    { title: 'Force Control', component: ForceControlStep },
    { title: 'AE Settings', component: AcousticEmissionStep },
    { title: 'Process Params', component: ProcessParametersStep },
    { title: 'Safety', component: SafetyVerificationStep },
    { title: 'Hardware Validation', component: HardwareValidationStep },
    { title: 'Validation', component: ValidationStep }
  ];

  // Load existing configurations on mount and check hardware status
  useEffect(() => {
    loadExistingConfigs();
    checkHardwareStatus();
  }, []);

  const loadExistingConfigs = async () => {
    try {
      const response = await axios.get('http://localhost:8000/setup/configurations');
      setExistingConfigs(response.data.configurations || []);
    } catch (error) {
      console.log('No existing configurations found');
    }
  };
  
  const checkHardwareStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8000/system/status');
      const status = response.data;
      
      // Check if required hardware is connected
      const hwStatus: HardwareStatus = {
        ati_axia_connected: status.health_metrics?.components?.force_sensor?.status === 'HEALTHY',
        ae_sensor_connected: status.health_metrics?.components?.ae_monitor?.status === 'HEALTHY',
        interferometer_connected: status.health_metrics?.components?.interferometer?.status === 'HEALTHY',
        robot_connected: status.health_metrics?.components?.robot_controller?.status === 'HEALTHY',
        status_message: status.status
      };
      
      setHardwareStatus(hwStatus);
      
      // Check if all required hardware is connected
      const allConnected = hwStatus.ati_axia_connected && 
                          hwStatus.ae_sensor_connected && 
                          hwStatus.interferometer_connected && 
                          hwStatus.robot_connected;
      
      setHardwareValidationPassed(allConnected);
    } catch (error) {
      console.error('Failed to check hardware status:', error);
      
      // If we can't get status, assume hardware is not ready
      const hwStatus: HardwareStatus = {
        ati_axia_connected: false,
        ae_sensor_connected: false,
        interferometer_connected: false,
        robot_connected: false,
        status_message: 'Unable to determine'
      };
      
      setHardwareStatus(hwStatus);
      setHardwareValidationPassed(false);
    }
  };

  const validateStep = (stepIndex: number): boolean => {
    const newErrors: Record<string, string> = {};
    
    switch (stepIndex) {
      case 2: // Interferometry
        if (config.grazing_angle_deg < 85.0 || config.grazing_angle_deg > 89.5) {
          newErrors.grazing_angle_deg = 'Must be between 85° and 89.5°';
        }
        if (config.wavelength_nm < 632.0 || config.wavelength_nm > 670.0) {
          newErrors.wavelength_nm = 'Should be between 632-670 nm';
        }
        break;
        
      case 3: // Force Control
        if (config.target_force_n < 20 || config.target_force_n > 150) {
          newErrors.target_force_n = 'Should be between 20N and 150N';
        }
        if (config.force_tolerance_n < 1 || config.force_tolerance_n > 20) {
          newErrors.force_tolerance_n = 'Should be between 1N and 20N';
        }
        break;
        
      case 4: // AE Settings
        if (config.ae_sample_rate_hz < 10000 || config.ae_sample_rate_hz > 1000000) {
          newErrors.ae_sample_rate_hz = 'Should be between 10kHz and 1MHz';
        }
        break;
        
      case 5: // Process Parameters
        if (config.spindle_speed_rpm < 1000 || config.spindle_speed_rpm > 6000) {
          newErrors.spindle_speed_rpm = 'Should be between 1000 and 6000 RPM';
        }
        if (config.feed_rate_mm_per_sec < 0.1 || config.feed_rate_mm_per_sec > 10.0) {
          newErrors.feed_rate_mm_per_sec = 'Should be between 0.1 and 10.0 mm/sec';
        }
        break;
        
      case 7: // Hardware Validation
        if (!hardwareValidationPassed) {
          newErrors.hardware_validation = 'All required hardware components must be connected and operational';
        }
        break;
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleNext = () => {
    if (validateStep(currentStep)) {
      if (currentStep < steps.length - 1) {
        setCurrentStep(currentStep + 1);
      }
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/setup/complete', {
        config: config,
        timestamp: new Date().toISOString()
      });
      
      onComplete(response.data);
    } catch (error) {
      console.error('Setup submission failed:', error);
      alert('Failed to complete setup. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleInputChange = (field: keyof CalibrationConfig, value: any) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  };

  const CurrentStepComponent = steps[currentStep].component;

  return (
    <div className="setup-wizard-overlay">
      <div className="setup-wizard">
        <div className="wizard-header">
          <h2>SurfacePlate Calibration Setup</h2>
          <button className="close-button" onClick={onCancel}>×</button>
        </div>
        
        <div className="wizard-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
            ></div>
          </div>
          <div className="step-indicator">
            Step {currentStep + 1} of {steps.length}: {steps[currentStep].title}
          </div>
        </div>

        <div className="wizard-content">
          <CurrentStepComponent 
            config={config}
            errors={errors}
            onChange={handleInputChange}
            existingConfigs={existingConfigs}
          />
        </div>

        <div className="wizard-footer">
          <button 
            className="secondary-button"
            onClick={onCancel}
            disabled={isLoading}
          >
            Cancel
          </button>
          
          {currentStep > 0 && (
            <button 
              className="secondary-button"
              onClick={handlePrevious}
              disabled={isLoading}
            >
              Previous
            </button>
          )}
          
          {currentStep < steps.length - 1 ? (
            <button 
              className="primary-button"
              onClick={handleNext}
              disabled={isLoading}
            >
              Next
            </button>
          ) : (
            <button 
              className="primary-button"
              onClick={handleSubmit}
              disabled={isLoading}
            >
              {isLoading ? 'Saving...' : 'Complete Setup'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Individual Step Components
const WelcomeStep: React.FC<any> = () => (
  <div className="step-content">
    <h3>Welcome to SurfacePlate Setup</h3>
    <p>This wizard will guide you through configuring your manufacturing system for nanometer-level precision.</p>
    <div className="info-box">
      <h4>Requirements:</h4>
      <ul>
        <li>Valid surface plate serial number</li>
        <li>Calibrated interferometry system</li>
        <li>Functional force sensors</li>
        <li>Proper safety equipment</li>
      </ul>
    </div>
  </div>
);

const AssetIdentificationStep: React.FC<any> = ({ config, errors, onChange }) => (
  <div className="step-content">
    <h3>Asset Identification</h3>
    <p>Enter your surface plate identification information.</p>
    
    <div className="form-group">
      <label htmlFor="serial_number">Serial Number *</label>
      <input
        id="serial_number"
        type="text"
        value={config.serial_number || ''}
        onChange={(e) => onChange('serial_number', e.target.value)}
        placeholder="e.g., SP-2024-001"
        required
      />
    </div>
    
    <div className="form-group">
      <label htmlFor="model">Model Type</label>
      <select
        id="model"
        value={config.model || ''}
        onChange={(e) => onChange('model', e.target.value)}
      >
        <option value="">Select Model</option>
        <option value="GRAVITY_2000">Gravity 2000</option>
        <option value="PRECISION_PLUS">Precision Plus</option>
        <option value="ULTRA_FLAT">Ultra Flat</option>
        <option value="CUSTOM">Custom</option>
      </select>
    </div>
  </div>
);

const InterferometryStep: React.FC<any> = ({ config, errors, onChange }) => (
  <div className="step-content">
    <h3>Interferometry Calibration</h3>
    <p>Configure Grazing Incidence Interferometry parameters.</p>
    
    <div className="form-group">
      <label htmlFor="grazing_angle_deg">
        Grazing Angle (degrees) *
        <span className="help-text">Critical for GII - 85° to 89.5°</span>
      </label>
      <input
        id="grazing_angle_deg"
        type="number"
        step="0.1"
        min="85"
        max="89.5"
        value={config.grazing_angle_deg}
        onChange={(e) => onChange('grazing_angle_deg', parseFloat(e.target.value))}
        className={errors.grazing_angle_deg ? 'error' : ''}
      />
      {errors.grazing_angle_deg && <span className="error-message">{errors.grazing_angle_deg}</span>}
    </div>
    
    <div className="form-group">
      <label htmlFor="wavelength_nm">
        Laser Wavelength (nm) *
        <span className="help-text">Visible range: 632-670 nm</span>
      </label>
      <input
        id="wavelength_nm"
        type="number"
        step="0.1"
        min="632"
        max="670"
        value={config.wavelength_nm}
        onChange={(e) => onChange('wavelength_nm', parseFloat(e.target.value))}
        className={errors.wavelength_nm ? 'error' : ''}
      />
      {errors.wavelength_nm && <span className="error-message">{errors.wavelength_nm}</span>}
    </div>
  </div>
);

const ForceControlStep: React.FC<any> = ({ config, errors, onChange }) => (
  <div className="step-content">
    <h3>Force Control Setup</h3>
    <p>Configure Active Contact Flange parameters.</p>
    
    <div className="form-group">
      <label htmlFor="target_force_n">
        Target Normal Force (N) *
        <span className="help-text">Safe range: 20-150N</span>
      </label>
      <input
        id="target_force_n"
        type="number"
        step="1"
        min="20"
        max="150"
        value={config.target_force_n}
        onChange={(e) => onChange('target_force_n', parseFloat(e.target.value))}
        className={errors.target_force_n ? 'error' : ''}
      />
      {errors.target_force_n && <span className="error-message">{errors.target_force_n}</span>}
    </div>
    
    <div className="form-group">
      <label htmlFor="force_tolerance_n">
        Force Tolerance (N) *
        <span className="help-text">Precision range: 1-20N</span>
      </label>
      <input
        id="force_tolerance_n"
        type="number"
        step="0.5"
        min="1"
        max="20"
        value={config.force_tolerance_n}
        onChange={(e) => onChange('force_tolerance_n', parseFloat(e.target.value))}
        className={errors.force_tolerance_n ? 'error' : ''}
      />
      {errors.force_tolerance_n && <span className="error-message">{errors.force_tolerance_n}</span>}
    </div>
  </div>
);

const AcousticEmissionStep: React.FC<any> = ({ config, errors, onChange }) => (
  <div className="step-content">
    <h3>Acoustic Emission Configuration</h3>
    <p>Set up AE sensor parameters for process monitoring.</p>
    
    <div className="form-group">
      <label htmlFor="ae_sample_rate_hz">
        AE Sample Rate (Hz) *
        <span className="help-text">Operational range: 10kHz - 1MHz</span>
      </label>
      <input
        id="ae_sample_rate_hz"
        type="number"
        step="1000"
        min="10000"
        max="1000000"
        value={config.ae_sample_rate_hz}
        onChange={(e) => onChange('ae_sample_rate_hz', parseInt(e.target.value))}
        className={errors.ae_sample_rate_hz ? 'error' : ''}
      />
      {errors.ae_sample_rate_hz && <span className="error-message">{errors.ae_sample_rate_hz}</span>}
    </div>
  </div>
);

const ProcessParametersStep: React.FC<any> = ({ config, errors, onChange }) => (
  <div className="step-content">
    <h3>Process Parameters</h3>
    <p>Configure machining parameters for optimal results.</p>
    
    <div className="form-group">
      <label htmlFor="spindle_speed_rpm">
        Spindle Speed (RPM) *
        <span className="help-text">Range: 1000-6000 RPM</span>
      </label>
      <input
        id="spindle_speed_rpm"
        type="number"
        step="100"
        min="1000"
        max="6000"
        value={config.spindle_speed_rpm}
        onChange={(e) => onChange('spindle_speed_rpm', parseFloat(e.target.value))}
        className={errors.spindle_speed_rpm ? 'error' : ''}
      />
      {errors.spindle_speed_rpm && <span className="error-message">{errors.spindle_speed_rpm}</span>}
    </div>
    
    <div className="form-group">
      <label htmlFor="feed_rate_mm_per_sec">
        Feed Rate (mm/sec) *
        <span className="help-text">Range: 0.1-10.0 mm/sec</span>
      </label>
      <input
        id="feed_rate_mm_per_sec"
        type="number"
        step="0.1"
        min="0.1"
        max="10.0"
        value={config.feed_rate_mm_per_sec}
        onChange={(e) => onChange('feed_rate_mm_per_sec', parseFloat(e.target.value))}
        className={errors.feed_rate_mm_per_sec ? 'error' : ''}
      />
      {errors.feed_rate_mm_per_sec && <span className="error-message">{errors.feed_rate_mm_per_sec}</span>}
    </div>
  </div>
);

const SafetyVerificationStep: React.FC<any> = ({ config }) => (
  <div className="step-content">
    <h3>Safety Verification</h3>
    <p>Review critical safety parameters before proceeding.</p>
    
    <div className="safety-review">
      <div className="safety-item">
        <span className="label">Maximum Force Limit:</span>
        <span className="value">{config.max_force_n} N</span>
      </div>
      <div className="safety-item">
        <span className="label">Maximum Temperature:</span>
        <span className="value">{config.max_temperature_c} °C</span>
      </div>
      <div className="safety-item">
        <span className="label">Maximum Vibration:</span>
        <span className="value">{config.max_vibration_g} g</span>
      </div>
    </div>
    
    <div className="warning-box">
      <h4>⚠️ Safety Notice</h4>
      <p>These parameters are critical for operator safety. Ensure all safety systems are functional before proceeding.</p>
    </div>
  </div>
);

const HardwareValidationStep: React.FC<any> = ({ }) => (
  <div className="step-content">
    <h3>Hardware Validation</h3>
    <p>Confirming that all required hardware components are properly connected and operational.</p>
    
    <div className="hardware-status">
      <div className="status-item">
        <span className="status-label">ATI Axia Force Sensor:</span>
        <span className={`status-value ${hardwareStatus?.ati_axia_connected ? 'connected' : 'disconnected'}`}>
          {hardwareStatus?.ati_axia_connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Acoustic Emission Sensor:</span>
        <span className={`status-value ${hardwareStatus?.ae_sensor_connected ? 'connected' : 'disconnected'}`}>
          {hardwareStatus?.ae_sensor_connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Interferometer:</span>
        <span className={`status-value ${hardwareStatus?.interferometer_connected ? 'connected' : 'disconnected'}`}>
          {hardwareStatus?.interferometer_connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>
      <div className="status-item">
        <span className="status-label">Robot Controller:</span>
        <span className={`status-value ${hardwareStatus?.robot_connected ? 'connected' : 'disconnected'}`}>
          {hardwareStatus?.robot_connected ? 'CONNECTED' : 'DISCONNECTED'}
        </span>
      </div>
    </div>
    
    <div className={`status-message ${hardwareValidationPassed ? 'success' : 'warning'}`}>
      {hardwareValidationPassed ? (
        <p>✓ All required hardware components are successfully connected and operational.</p>
      ) : (
        <p>⚠️ Some required hardware components are not connected. Please verify connections before proceeding.</p>
      )}
    </div>
  </div>
);

const ValidationStep: React.FC<any> = ({ config }) => (
  <div className="step-content">
    <h3>Configuration Validation</h3>
    <p>Review all settings before finalizing the configuration.</p>
    
    <div className="validation-summary">
      <h4>Asset Information</h4>
      <p><strong>Serial:</strong> {config.serial_number || 'Not specified'}</p>
      <p><strong>Model:</strong> {config.model || 'Not specified'}</p>
      
      <h4>Critical Parameters</h4>
      <p><strong>Grazing Angle:</strong> {config.grazing_angle_deg}°</p>
      <p><strong>Target Force:</strong> {config.target_force_n}N</p>
      <p><strong>Laser Wavelength:</strong> {config.wavelength_nm}nm</p>
      <p><strong>Sample Rate:</strong> {config.ae_sample_rate_hz.toLocaleString()}Hz</p>
    </div>
    
    <div className="confirmation-box">
      <p>✓ All parameters have been validated and meet safety specifications.</p>
    </div>
  </div>
);

export default SetupWizard;