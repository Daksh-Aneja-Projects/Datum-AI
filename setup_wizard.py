"""
Setup Wizard for SurfacePlate Calibration
Enforces a "valid configuration tunnel" before allowing the main control loop to start
"""

import asyncio
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import hashlib
from datetime import datetime


class SetupStateMachine:
    """State machine for setup wizard with checkpoints and rollback capability"""
    
    def __init__(self):
        self.state = 'START'
        self.states = [
            'START',
            'WELCOME',
            'ASSET_IDENTIFICATION',
            'INTERFEROMETRY_CALIBRATION',
            'FORCE_CONTROL_SETUP',
            'ACOUSTIC_EMISSION_CONFIG',
            'PROCESS_PARAMETERS',
            'SAFETY_VERIFICATION',
            'VALIDATION',
            'COMPLETED',
            'ERROR'
        ]
        self.checkpoints = {}  # Store state snapshots
        self.rollback_stack = []  # Track states for potential rollback
        
    def transition_to(self, new_state: str):
        """Transition to a new state"""
        if new_state in self.states:
            self.rollback_stack.append(self.state)  # Save current state for potential rollback
            self.state = new_state
            return True
        return False
    
    def get_current_state(self):
        """Get current state"""
        return self.state
    
    def can_rollback(self):
        """Check if rollback is possible"""
        return len(self.rollback_stack) > 0
    
    def rollback(self):
        """Rollback to previous state"""
        if self.can_rollback():
            previous_state = self.rollback_stack.pop()
            self.state = previous_state
            return previous_state
        return None
    
    def save_checkpoint(self, checkpoint_id: str, data: Dict):
        """Save current state data as a checkpoint"""
        self.checkpoints[checkpoint_id] = {
            'state': self.state,
            'data': data,
            'timestamp': datetime.now().isoformat()
        }
    
    def restore_checkpoint(self, checkpoint_id: str):
        """Restore state from a checkpoint"""
        if checkpoint_id in self.checkpoints:
            checkpoint = self.checkpoints[checkpoint_id]
            self.state = checkpoint['state']
            return checkpoint['data']
        return None


@dataclass
class CalibrationConfig:
    """Configuration for surface plate calibration"""
    # Interferometry settings
    wavelength_nm: float = 632.8
    pixel_size_um: float = 3.45
    magnification: float = 10.0
    acquisition_rate_fps: int = 30
    phase_shift_steps: int = 4
    spatial_resolution_um: float = 0.1
    grazing_angle_deg: float = 88.0  # Critical for GII
    
    # Force control settings
    target_force_n: float = 45.0
    force_tolerance_n: float = 5.0
    compliance_bandwidth_hz: float = 100.0
    
    # Acoustic emission settings
    ae_sample_rate_hz: int = 50000
    ae_rms_threshold: float = 0.5
    ae_kurtosis_threshold: float = 5.0
    
    # Process parameters
    spindle_speed_rpm: float = 3000
    feed_rate_mm_per_sec: float = 2.0
    abrasive_grit_size: int = 200
    coolant_flow_rate: float = 5.0
    
    # Safety limits
    max_force_n: float = 200.0
    max_temperature_c: float = 60.0
    max_vibration_g: float = 0.5


class SetupWizard:
    """Interactive setup wizard for SurfacePlate calibration"""
    
    def __init__(self, config_file: str = "calibration_config.json"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.config = CalibrationConfig()
        self.setup_complete = False
        self.state_machine = SetupStateMachine()
        self.setup_data = {}  # Store intermediate setup data
        
    async def run_setup_wizard(self) -> bool:
        """Run the complete setup wizard"""
        print("=" * 60)
        print("SurfacePlate Calibration Setup Wizard")
        print("=" * 60)
        
        try:
            # Check if config already exists
            if self._config_exists():
                print(f"Configuration found: {self.config_file}")
                if self._confirm_use_existing_config():
                    self.config = self._load_config()
                    return True
            
            # Step-by-step configuration
            await self._step_welcome()
            await self._step_asset_identification()
            await self._step_interferometry_calibration()
            await self._step_force_control_setup()
            await self._step_acoustic_emission_config()
            await self._step_process_parameters()
            await self._step_safety_verification()
            await self._step_validation()
            
            # Save configuration
            self._save_config()
            
            print("\n✓ Setup complete! Configuration saved.")
            self.setup_complete = True
            return True
            
        except Exception as e:
            self.logger.error(f"Setup wizard failed: {e}")
            print(f"\n✗ Setup failed: {e}")
            return False
    
    async def _step_welcome(self):
        """Welcome message and overview"""
        print("\n📋 Welcome to SurfacePlate Setup")
        print("This wizard will guide you through the configuration process.")
        print("Each step is critical for achieving nanometer-level precision.")
        input("\nPress Enter to continue...")
    
    async def _step_asset_identification(self):
        """Asset identification and serial number"""
        print("\n🏷️  Asset Identification")
        print("Enter the serial number and other identifying information.")
        
        serial_number = input("Surface Plate Serial Number: ").strip()
        model = input("Model Type (e.g., GRAVITY_2000): ").strip()
        calibration_date = datetime.now().isoformat()
        
        # Add these to the config dictionary when saving
        self.config_dict = getattr(self, 'config_dict', {})
        self.config_dict['serial_number'] = serial_number
        self.config_dict['model'] = model
        self.config_dict['calibration_date'] = calibration_date
        
        print(f"Asset configured: {model} #{serial_number}")
    
    async def _step_interferometry_calibration(self):
        """Configure interferometry settings"""
        print("\n🔍 Interferometry Calibration")
        print("Setting up Grazing Incidence Interferometry parameters...")
        
        # Validate grazing angle (critical for GII)
        while True:
            try:
                grazing_angle = float(input(f"Grazing Angle (deg) [{self.config.grazing_angle_deg}]: ") or self.config.grazing_angle_deg)
                if 85.0 <= grazing_angle <= 89.5:
                    self.config.grazing_angle_deg = grazing_angle
                    break
                else:
                    print("⚠️  Grazing angle must be between 85° and 89.5° for optimal GII performance")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        # Validate wavelength
        while True:
            try:
                wavelength = float(input(f"Laser Wavelength (nm) [{self.config.wavelength_nm}]: ") or self.config.wavelength_nm)
                if 632.0 <= wavelength <= 670.0:
                    self.config.wavelength_nm = wavelength
                    break
                else:
                    print("⚠️  Wavelength should be in visible range (632-670 nm)")
            except ValueError:
                print("⚠️  Please enter a valid number")
    
    async def _step_force_control_setup(self):
        """Configure force control parameters"""
        print("\n⚖️  Force Control Setup")
        print("Configuring Active Contact Flange (ACF) parameters...")
        
        while True:
            try:
                target_force = float(input(f"Target Normal Force (N) [{self.config.target_force_n}]: ") or self.config.target_force_n)
                if 20 <= target_force <= 150:
                    self.config.target_force_n = target_force
                    break
                else:
                    print("⚠️  Target force should be between 20N and 150N")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        while True:
            try:
                tolerance = float(input(f"Force Tolerance (N) [{self.config.force_tolerance_n}]: ") or self.config.force_tolerance_n)
                if 1 <= tolerance <= 20:
                    self.config.force_tolerance_n = tolerance
                    break
                else:
                    print("⚠️  Tolerance should be between 1N and 20N")
            except ValueError:
                print("⚠️  Please enter a valid number")
    
    async def _step_acoustic_emission_config(self):
        """Configure acoustic emission monitoring"""
        print("\n🔊 Acoustic Emission Configuration")
        print("Setting up AE sensor parameters for process monitoring...")
        
        while True:
            try:
                sample_rate = int(input(f"AE Sample Rate (Hz) [{self.config.ae_sample_rate_hz}]: ") or self.config.ae_sample_rate_hz)
                if 10000 <= sample_rate <= 1000000:
                    self.config.ae_sample_rate_hz = sample_rate
                    break
                else:
                    print("⚠️  Sample rate should be between 10kHz and 1MHz")
            except ValueError:
                print("⚠️  Please enter a valid number")
    
    async def _step_process_parameters(self):
        """Configure process parameters"""
        print("\n⚙️  Process Parameter Configuration")
        print("Setting up machining parameters...")
        
        while True:
            try:
                rpm = float(input(f"Spindle Speed (RPM) [{self.config.spindle_speed_rpm}]: ") or self.config.spindle_speed_rpm)
                if 1000 <= rpm <= 6000:
                    self.config.spindle_speed_rpm = rpm
                    break
                else:
                    print("⚠️  Spindle speed should be between 1000 and 6000 RPM")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        while True:
            try:
                feed_rate = float(input(f"Feed Rate (mm/sec) [{self.config.feed_rate_mm_per_sec}]: ") or self.config.feed_rate_mm_per_sec)
                if 0.1 <= feed_rate <= 10.0:
                    self.config.feed_rate_mm_per_sec = feed_rate
                    break
                else:
                    print("⚠️  Feed rate should be between 0.1 and 10.0 mm/sec")
            except ValueError:
                print("⚠️  Please enter a valid number")
    
    async def _step_safety_verification(self):
        """Verify safety parameters"""
        print("\n🛡️  Safety Parameter Verification")
        print("Reviewing critical safety limits...")
        
        print(f"Max Force Limit: {self.config.max_force_n} N")
        print(f"Max Temperature: {self.config.max_temperature_c} °C")
        print(f"Max Vibration: {self.config.max_vibration_g} g")
        
        confirm = input("\nAre these safety limits acceptable? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Please restart the wizard with appropriate safety limits.")
            raise ValueError("Safety limits not confirmed")
    
    async def _step_validation(self):
        """Validate the entire configuration"""
        print("\n✅ Configuration Validation")
        print("Verifying all parameters meet specifications...")
        
        # Validate critical parameters
        issues = []
        
        if not (85.0 <= self.config.grazing_angle_deg <= 89.5):
            issues.append("Grazing angle outside optimal range (85-89.5°)")
        
        if not (632.0 <= self.config.wavelength_nm <= 670.0):
            issues.append("Wavelength outside visible range")
        
        if not (20 <= self.config.target_force_n <= 150):
            issues.append("Target force outside safe range")
        
        if not (10000 <= self.config.ae_sample_rate_hz <= 1000000):
            issues.append("AE sample rate outside operational range")
        
        if issues:
            print("\n❌ Configuration validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            raise ValueError("Configuration validation failed")
        
        print("✓ All parameters validated successfully")
    
    def _config_exists(self) -> bool:
        """Check if config file already exists"""
        return os.path.exists(self.config_file)
    
    def _confirm_use_existing_config(self) -> bool:
        """Ask user if they want to use existing config"""
        response = input(f"Use existing config {self.config_file}? (Y/n): ").lower().strip()
        return response != 'n'
    
    def _save_config(self):
        """Save the current configuration to file"""
        config_dict = asdict(self.config)
        
        # Add additional fields that were collected during setup
        if hasattr(self, 'config_dict'):
            config_dict.update(self.config_dict)
        
        config_dict['config_hash'] = self._generate_config_hash(config_dict)
        config_dict['setup_timestamp'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def _load_config(self) -> CalibrationConfig:
        """Load configuration from file"""
        with open(self.config_file, 'r') as f:
            config_dict = json.load(f)
        
        # Store additional fields separately
        self.config_dict = {}
        additional_fields = ['serial_number', 'model', 'calibration_date']
        for field in additional_fields:
            if field in config_dict:
                self.config_dict[field] = config_dict.pop(field, None)
        
        # Remove metadata fields not in CalibrationConfig
        metadata_keys = ['config_hash', 'setup_timestamp']
        for key in metadata_keys:
            config_dict.pop(key, None)
        
        # Create new config with loaded values
        config = CalibrationConfig(**config_dict)
        return config
    
    def _generate_config_hash(self, config_dict: Dict) -> str:
        """Generate hash of configuration for integrity verification"""
        # Remove timestamp from hash calculation
        hash_dict = {k: v for k, v in config_dict.items() if k != 'setup_timestamp'}
        config_str = json.dumps(hash_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def get_validated_config(self) -> Optional[CalibrationConfig]:
        """Get the validated configuration if setup is complete"""
        if self.setup_complete:
            return self.config
        elif self._config_exists():
            try:
                config = self._load_config()
                return config
            except:
                return None
        return None


# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)
    wizard = SetupWizard()
    
    success = await wizard.run_setup_wizard()
    if success:
        config = wizard.get_validated_config()
        if config:
            # Access additional fields from config_dict
            model = getattr(wizard, 'config_dict', {}).get('model', 'Unknown')
            serial_number = getattr(wizard, 'config_dict', {}).get('serial_number', 'Unknown')
            print(f"\nConfiguration ready: {model} #{serial_number}")
            print(f"Target force: {config.target_force_n}N")
            print(f"Grazing angle: {config.grazing_angle_deg}°")


if __name__ == "__main__":
    asyncio.run(main())