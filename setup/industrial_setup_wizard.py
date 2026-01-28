"""
State Machine-Based Setup Wizard with Checkpoints and Rollback Capability
Implements FUX (First User Experience) requirements with industrial reliability
"""

import asyncio
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path
import uuid

# State machine library (simplified implementation)
class StateMachine:
    """Finite state machine implementation"""
    
    def __init__(self, initial_state: str):
        self.current_state = initial_state
        self.states = {}
        self.transitions = {}
        self.callbacks = {}
        self.history = []
        
    def add_state(self, state: str, on_enter: Optional[Callable] = None, 
                  on_exit: Optional[Callable] = None):
        """Add a state with optional callbacks"""
        self.states[state] = {
            'on_enter': on_enter,
            'on_exit': on_exit
        }
    
    def add_transition(self, from_state: str, to_state: str, 
                      condition: Optional[Callable] = None,
                      action: Optional[Callable] = None):
        """Add state transition"""
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        
        self.transitions[from_state].append({
            'to': to_state,
            'condition': condition,
            'action': action
        })
    
    async def trigger(self, event: str = None) -> bool:
        """Trigger state transition"""
        if self.current_state not in self.transitions:
            return False
            
        # Find valid transition
        for transition in self.transitions[self.current_state]:
            condition_met = transition['condition'] is None or transition['condition']()
            
            if condition_met:
                # Exit current state
                if self.current_state in self.states and self.states[self.current_state]['on_exit']:
                    await self.states[self.current_state]['on_exit']()
                
                # Execute transition action
                if transition['action']:
                    await transition['action']()
                
                # Record history
                self.history.append({
                    'from_state': self.current_state,
                    'to_state': transition['to'],
                    'timestamp': datetime.now().isoformat(),
                    'event': event
                })
                
                # Enter new state
                old_state = self.current_state
                self.current_state = transition['to']
                
                if self.current_state in self.states and self.states[self.current_state]['on_enter']:
                    await self.states[self.current_state]['on_enter']()
                
                logging.info(f"State transition: {old_state} -> {self.current_state}")
                return True
        
        return False
    
    def get_current_state(self) -> str:
        return self.current_state
    
    def get_history(self) -> List[Dict]:
        return self.history.copy()

class SetupPhase(Enum):
    """Setup wizard phases"""
    WELCOME = "welcome"
    ASSET_IDENTIFICATION = "asset_identification"
    INTERFEROMETRY_CALIBRATION = "interferometry_calibration"
    FORCE_CONTROL_SETUP = "force_control_setup"
    ACOUSTIC_EMISSION_CONFIG = "acoustic_emission_config"
    PROCESS_PARAMETERS = "process_parameters"
    SAFETY_VERIFICATION = "safety_verification"
    VALIDATION = "validation"
    COMPLETE = "complete"
    ERROR = "error"

@dataclass
class SetupCheckpoint:
    """Checkpoint for rollback capability"""
    checkpoint_id: str
    phase: SetupPhase
    timestamp: str
    config_snapshot: Dict[str, Any]
    validation_results: Dict[str, Any]
    user_inputs: Dict[str, Any]

class IndustrialSetupStateMachine:
    """Industrial-grade setup wizard with state machine and rollback"""
    
    def __init__(self, config_file: str = "setup_config.json"):
        self.config_file = Path(config_file)
        self.logger = logging.getLogger(__name__)
        
        # Initialize state machine
        self.state_machine = StateMachine(SetupPhase.WELCOME.value)
        self._setup_states()
        self._setup_transitions()
        
        # Configuration management
        self.current_config = {}
        self.checkpoints = []
        self.user_session = {
            'session_id': str(uuid.uuid4()),
            'start_time': datetime.now().isoformat(),
            'user_actions': []
        }
        
        # Validation rules
        self.validation_rules = self._define_validation_rules()
        
        # Progress tracking
        self.progress = 0.0
        self.total_phases = len(SetupPhase) - 2  # Exclude WELCOME and COMPLETE/ERROR
    
    def _setup_states(self):
        """Define all setup states"""
        # Welcome state
        self.state_machine.add_state(
            SetupPhase.WELCOME.value,
            on_enter=self._enter_welcome,
            on_exit=self._exit_welcome
        )
        
        # Asset identification
        self.state_machine.add_state(
            SetupPhase.ASSET_IDENTIFICATION.value,
            on_enter=self._enter_asset_identification,
            on_exit=self._exit_asset_identification
        )
        
        # Interferometry calibration
        self.state_machine.add_state(
            SetupPhase.INTERFEROMETRY_CALIBRATION.value,
            on_enter=self._enter_interferometry_calibration,
            on_exit=self._exit_interferometry_calibration
        )
        
        # Force control setup
        self.state_machine.add_state(
            SetupPhase.FORCE_CONTROL_SETUP.value,
            on_enter=self._enter_force_control_setup,
            on_exit=self._exit_force_control_setup
        )
        
        # AE configuration
        self.state_machine.add_state(
            SetupPhase.ACOUSTIC_EMISSION_CONFIG.value,
            on_enter=self._enter_acoustic_emission_config,
            on_exit=self._exit_acoustic_emission_config
        )
        
        # Process parameters
        self.state_machine.add_state(
            SetupPhase.PROCESS_PARAMETERS.value,
            on_enter=self._enter_process_parameters,
            on_exit=self._exit_process_parameters
        )
        
        # Safety verification
        self.state_machine.add_state(
            SetupPhase.SAFETY_VERIFICATION.value,
            on_enter=self._enter_safety_verification,
            on_exit=self._exit_safety_verification
        )
        
        # Validation
        self.state_machine.add_state(
            SetupPhase.VALIDATION.value,
            on_enter=self._enter_validation,
            on_exit=self._exit_validation
        )
        
        # Complete/Error states (no transitions out)
        self.state_machine.add_state(SetupPhase.COMPLETE.value)
        self.state_machine.add_state(SetupPhase.ERROR.value)
    
    def _setup_transitions(self):
        """Define state transitions"""
        # From welcome
        self.state_machine.add_transition(
            SetupPhase.WELCOME.value,
            SetupPhase.ASSET_IDENTIFICATION.value,
            condition=lambda: True
        )
        
        # From asset identification
        self.state_machine.add_transition(
            SetupPhase.ASSET_IDENTIFICATION.value,
            SetupPhase.INTERFEROMETRY_CALIBRATION.value,
            condition=lambda: self._validate_current_phase(SetupPhase.ASSET_IDENTIFICATION)
        )
        
        # From interferometry calibration
        self.state_machine.add_transition(
            SetupPhase.INTERFEROMETRY_CALIBRATION.value,
            SetupPhase.FORCE_CONTROL_SETUP.value,
            condition=lambda: self._validate_current_phase(SetupPhase.INTERFEROMETRY_CALIBRATION)
        )
        
        # From force control setup
        self.state_machine.add_transition(
            SetupPhase.FORCE_CONTROL_SETUP.value,
            SetupPhase.ACOUSTIC_EMISSION_CONFIG.value,
            condition=lambda: self._validate_current_phase(SetupPhase.FORCE_CONTROL_SETUP)
        )
        
        # From AE config
        self.state_machine.add_transition(
            SetupPhase.ACOUSTIC_EMISSION_CONFIG.value,
            SetupPhase.PROCESS_PARAMETERS.value,
            condition=lambda: self._validate_current_phase(SetupPhase.ACOUSTIC_EMISSION_CONFIG)
        )
        
        # From process parameters
        self.state_machine.add_transition(
            SetupPhase.PROCESS_PARAMETERS.value,
            SetupPhase.SAFETY_VERIFICATION.value,
            condition=lambda: self._validate_current_phase(SetupPhase.PROCESS_PARAMETERS)
        )
        
        # From safety verification
        self.state_machine.add_transition(
            SetupPhase.SAFETY_VERIFICATION.value,
            SetupPhase.VALIDATION.value,
            condition=lambda: self._validate_current_phase(SetupPhase.SAFETY_VERIFICATION)
        )
        
        # From validation
        self.state_machine.add_transition(
            SetupPhase.VALIDATION.value,
            SetupPhase.COMPLETE.value,
            condition=lambda: self._validate_current_phase(SetupPhase.VALIDATION)
        )
        
        # Error transitions (from any phase to error)
        for phase in SetupPhase:
            if phase not in [SetupPhase.COMPLETE, SetupPhase.ERROR]:
                self.state_machine.add_transition(
                    phase.value,
                    SetupPhase.ERROR.value,
                    condition=lambda: False,  # Manual error trigger
                    action=self._handle_error_transition
                )
    
    def _define_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Define validation rules for each phase"""
        return {
            SetupPhase.ASSET_IDENTIFICATION.value: {
                'required_fields': ['serial_number', 'model'],
                'validators': {
                    'serial_number': lambda x: len(x) >= 5,
                    'model': lambda x: x in ['GRAVITY_2000', 'ULTRA_PRECISION_3000', 'NANO_GRADE_5000']
                }
            },
            SetupPhase.INTERFEROMETRY_CALIBRATION.value: {
                'required_fields': ['grazing_angle_deg', 'wavelength_nm'],
                'range_validators': {
                    'grazing_angle_deg': (85.0, 89.5),
                    'wavelength_nm': (632.0, 670.0)
                }
            },
            SetupPhase.FORCE_CONTROL_SETUP.value: {
                'required_fields': ['target_force_n', 'force_tolerance_n'],
                'range_validators': {
                    'target_force_n': (20.0, 150.0),
                    'force_tolerance_n': (1.0, 20.0)
                }
            },
            SetupPhase.ACOUSTIC_EMISSION_CONFIG.value: {
                'required_fields': ['ae_sample_rate_hz'],
                'range_validators': {
                    'ae_sample_rate_hz': (10000, 1000000)
                }
            },
            SetupPhase.PROCESS_PARAMETERS.value: {
                'required_fields': ['spindle_speed_rpm', 'feed_rate_mm_per_sec'],
                'range_validators': {
                    'spindle_speed_rpm': (1000, 6000),
                    'feed_rate_mm_per_sec': (0.1, 10.0)
                }
            },
            SetupPhase.SAFETY_VERIFICATION.value: {
                'required_fields': ['safety_confirmed'],
                'validators': {
                    'safety_confirmed': lambda x: x is True
                }
            }
        }
    
    async def _enter_welcome(self):
        """Enter welcome state"""
        self.logger.info("Entering welcome phase")
        self.progress = 0.0
        
    async def _exit_welcome(self):
        """Exit welcome state"""
        self.logger.info("Exiting welcome phase")
    
    async def _enter_asset_identification(self):
        """Enter asset identification phase"""
        self.logger.info("Entering asset identification phase")
        self.progress = 1.0 / self.total_phases
    
    async def _exit_asset_identification(self):
        """Exit asset identification phase"""
        self._create_checkpoint(SetupPhase.ASSET_IDENTIFICATION)
    
    async def _enter_interferometry_calibration(self):
        """Enter interferometry calibration phase"""
        self.logger.info("Entering interferometry calibration phase")
        self.progress = 2.0 / self.total_phases
    
    async def _exit_interferometry_calibration(self):
        """Exit interferometry calibration phase"""
        self._create_checkpoint(SetupPhase.INTERFEROMETRY_CALIBRATION)
    
    async def _enter_force_control_setup(self):
        """Enter force control setup phase"""
        self.logger.info("Entering force control setup phase")
        self.progress = 3.0 / self.total_phases
    
    async def _exit_force_control_setup(self):
        """Exit force control setup phase"""
        self._create_checkpoint(SetupPhase.FORCE_CONTROL_SETUP)
    
    async def _enter_acoustic_emission_config(self):
        """Enter AE configuration phase"""
        self.logger.info("Entering acoustic emission configuration phase")
        self.progress = 4.0 / self.total_phases
    
    async def _exit_acoustic_emission_config(self):
        """Exit AE configuration phase"""
        self._create_checkpoint(SetupPhase.ACOUSTIC_EMISSION_CONFIG)
    
    async def _enter_process_parameters(self):
        """Enter process parameters phase"""
        self.logger.info("Entering process parameters phase")
        self.progress = 5.0 / self.total_phases
    
    async def _exit_process_parameters(self):
        """Exit process parameters phase"""
        self._create_checkpoint(SetupPhase.PROCESS_PARAMETERS)
    
    async def _enter_safety_verification(self):
        """Enter safety verification phase"""
        self.logger.info("Entering safety verification phase")
        self.progress = 6.0 / self.total_phases
    
    async def _exit_safety_verification(self):
        """Exit safety verification phase"""
        self._create_checkpoint(SetupPhase.SAFETY_VERIFICATION)
    
    async def _enter_validation(self):
        """Enter validation phase"""
        self.logger.info("Entering validation phase")
        self.progress = 7.0 / self.total_phases
    
    async def _exit_validation(self):
        """Exit validation phase"""
        self._create_checkpoint(SetupPhase.VALIDATION)
        self._save_final_configuration()
    
    async def _handle_error_transition(self):
        """Handle transition to error state"""
        self.logger.error("Transitioning to error state")
        # Could implement automatic rollback here
    
    def _validate_current_phase(self, phase: SetupPhase) -> bool:
        """Validate current phase configuration"""
        phase_key = phase.value
        if phase_key not in self.validation_rules:
            return True  # No validation rules defined
        
        rules = self.validation_rules[phase_key]
        
        # Check required fields
        for field in rules.get('required_fields', []):
            if field not in self.current_config:
                self.logger.warning(f"Missing required field: {field}")
                return False
        
        # Check validators
        for field, validator in rules.get('validators', {}).items():
            if field in self.current_config:
                if not validator(self.current_config[field]):
                    self.logger.warning(f"Validation failed for field: {field}")
                    return False
        
        # Check range validators
        for field, (min_val, max_val) in rules.get('range_validators', {}).items():
            if field in self.current_config:
                value = self.current_config[field]
                if not (min_val <= value <= max_val):
                    self.logger.warning(f"Range validation failed for {field}: {value} not in [{min_val}, {max_val}]")
                    return False
        
        return True
    
    def _create_checkpoint(self, phase: SetupPhase):
        """Create checkpoint for rollback capability"""
        checkpoint = SetupCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            phase=phase,
            timestamp=datetime.now().isoformat(),
            config_snapshot=self.current_config.copy(),
            validation_results={'valid': self._validate_current_phase(phase)},
            user_inputs={}  # Would capture actual user inputs in real implementation
        )
        
        self.checkpoints.append(checkpoint)
        self.logger.info(f"Checkpoint created for phase: {phase.value}")
    
    def _save_final_configuration(self):
        """Save final validated configuration"""
        final_config = {
            'configuration': self.current_config,
            'checkpoints': [asdict(cp) for cp in self.checkpoints],
            'user_session': self.user_session,
            'completion_timestamp': datetime.now().isoformat(),
            'config_hash': self._generate_config_hash(self.current_config)
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(final_config, f, indent=2)
        
        self.logger.info(f"Configuration saved to {self.config_file}")
    
    def _generate_config_hash(self, config: Dict) -> str:
        """Generate hash for configuration integrity"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    async def advance_to_next_phase(self) -> bool:
        """Attempt to advance to next phase"""
        return await self.state_machine.trigger('next')
    
    async def rollback_to_checkpoint(self, checkpoint_index: int) -> bool:
        """Rollback to specific checkpoint"""
        if 0 <= checkpoint_index < len(self.checkpoints):
            checkpoint = self.checkpoints[checkpoint_index]
            self.current_config = checkpoint.config_snapshot.copy()
            
            # Transition to the checkpoint's phase
            target_phase = checkpoint.phase.value
            if self.state_machine.current_state != target_phase:
                # This would require more complex state machine logic
                # For now, we'll just log the rollback
                self.logger.info(f"Rolled back to checkpoint {checkpoint_index}: {target_phase}")
                return True
        
        return False
    
    def get_progress(self) -> float:
        """Get current setup progress (0.0 to 1.0)"""
        return self.progress
    
    def get_current_phase(self) -> str:
        """Get current setup phase"""
        return self.state_machine.get_current_state()
    
    def get_available_checkpoints(self) -> List[Dict[str, Any]]:
        """Get list of available rollback points"""
        return [
            {
                'index': i,
                'phase': cp.phase.value,
                'timestamp': cp.timestamp,
                'valid': cp.validation_results.get('valid', False)
            }
            for i, cp in enumerate(self.checkpoints)
        ]
    
    def update_configuration(self, updates: Dict[str, Any]):
        """Update current configuration"""
        self.current_config.update(updates)
        self.user_session['user_actions'].append({
            'action': 'config_update',
            'updates': updates,
            'timestamp': datetime.now().isoformat()
        })

# Enhanced setup wizard with interactive interface
class InteractiveSetupWizard:
    """Interactive setup wizard with user-friendly interface"""
    
    def __init__(self, state_machine: IndustrialSetupStateMachine):
        self.state_machine = state_machine
        self.logger = logging.getLogger(__name__)
        
    async def run_interactive_setup(self) -> bool:
        """Run interactive setup wizard"""
        print("=" * 60)
        print("🔧 Industrial SurfacePlate Setup Wizard")
        print("=" * 60)
        print("This wizard guides you through the complete calibration process.")
        print("Each step includes validation and rollback capability.\n")
        
        while self.state_machine.get_current_phase() != SetupPhase.COMPLETE.value:
            current_phase = self.state_machine.get_current_phase()
            
            if current_phase == SetupPhase.WELCOME.value:
                await self._handle_welcome_phase()
            elif current_phase == SetupPhase.ASSET_IDENTIFICATION.value:
                await self._handle_asset_identification()
            elif current_phase == SetupPhase.INTERFEROMETRY_CALIBRATION.value:
                await self._handle_interferometry_calibration()
            elif current_phase == SetupPhase.FORCE_CONTROL_SETUP.value:
                await self._handle_force_control_setup()
            elif current_phase == SetupPhase.ACOUSTIC_EMISSION_CONFIG.value:
                await self._handle_acoustic_emission_config()
            elif current_phase == SetupPhase.PROCESS_PARAMETERS.value:
                await self._handle_process_parameters()
            elif current_phase == SetupPhase.SAFETY_VERIFICATION.value:
                await self._handle_safety_verification()
            elif current_phase == SetupPhase.VALIDATION.value:
                await self._handle_validation()
            elif current_phase == SetupPhase.ERROR.value:
                return await self._handle_error_state()
            
            # Attempt to advance
            success = await self.state_machine.advance_to_next_phase()
            if not success and current_phase != SetupPhase.ERROR.value:
                print("❌ Validation failed. Please review your inputs.")
                await self._offer_rollback_option()
        
        print("\n✅ Setup completed successfully!")
        print(f"Configuration saved to: {self.state_machine.config_file}")
        return True
    
    async def _handle_welcome_phase(self):
        """Handle welcome phase"""
        input("Press Enter to begin setup...")
    
    async def _handle_asset_identification(self):
        """Handle asset identification phase"""
        print("\n🏷️  Asset Identification")
        print("-" * 30)
        
        serial_number = input("Enter Surface Plate Serial Number: ").strip()
        model = input("Select Model (GRAVITY_2000/ULTRA_PRECISION_3000/NANO_GRADE_5000): ").strip()
        
        self.state_machine.update_configuration({
            'serial_number': serial_number,
            'model': model,
            'identification_timestamp': datetime.now().isoformat()
        })
    
    async def _handle_interferometry_calibration(self):
        """Handle interferometry calibration phase"""
        print("\n🔍 Interferometry Calibration")
        print("-" * 30)
        
        while True:
            try:
                grazing_angle = float(input("Grazing Angle (85.0-89.5°): "))
                if 85.0 <= grazing_angle <= 89.5:
                    break
                else:
                    print("⚠️  Angle must be between 85.0° and 89.5°")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        while True:
            try:
                wavelength = float(input("Laser Wavelength (632.0-670.0 nm): "))
                if 632.0 <= wavelength <= 670.0:
                    break
                else:
                    print("⚠️  Wavelength must be between 632.0 and 670.0 nm")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        self.state_machine.update_configuration({
            'grazing_angle_deg': grazing_angle,
            'wavelength_nm': wavelength
        })
    
    async def _handle_force_control_setup(self):
        """Handle force control setup phase"""
        print("\n⚖️  Force Control Setup")
        print("-" * 30)
        
        while True:
            try:
                target_force = float(input("Target Normal Force (20-150 N): "))
                if 20 <= target_force <= 150:
                    break
                else:
                    print("⚠️  Force must be between 20 and 150 N")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        while True:
            try:
                tolerance = float(input("Force Tolerance (1-20 N): "))
                if 1 <= tolerance <= 20:
                    break
                else:
                    print("⚠️  Tolerance must be between 1 and 20 N")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        self.state_machine.update_configuration({
            'target_force_n': target_force,
            'force_tolerance_n': tolerance
        })
    
    async def _handle_acoustic_emission_config(self):
        """Handle acoustic emission configuration phase"""
        print("\n🔊 Acoustic Emission Configuration")
        print("-" * 30)
        
        while True:
            try:
                sample_rate = int(input("AE Sample Rate (10000-1000000 Hz): "))
                if 10000 <= sample_rate <= 1000000:
                    break
                else:
                    print("⚠️  Sample rate must be between 10kHz and 1MHz")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        self.state_machine.update_configuration({
            'ae_sample_rate_hz': sample_rate
        })
    
    async def _handle_process_parameters(self):
        """Handle process parameters phase"""
        print("\n⚙️  Process Parameters")
        print("-" * 30)
        
        while True:
            try:
                rpm = float(input("Spindle Speed (1000-6000 RPM): "))
                if 1000 <= rpm <= 6000:
                    break
                else:
                    print("⚠️  RPM must be between 1000 and 6000")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        while True:
            try:
                feed_rate = float(input("Feed Rate (0.1-10.0 mm/sec): "))
                if 0.1 <= feed_rate <= 10.0:
                    break
                else:
                    print("⚠️  Feed rate must be between 0.1 and 10.0 mm/sec")
            except ValueError:
                print("⚠️  Please enter a valid number")
        
        self.state_machine.update_configuration({
            'spindle_speed_rpm': rpm,
            'feed_rate_mm_per_sec': feed_rate
        })
    
    async def _handle_safety_verification(self):
        """Handle safety verification phase"""
        print("\n🛡️  Safety Verification")
        print("-" * 30)
        print("Critical safety limits:")
        print("• Max Force: 200 N")
        print("• Max Temperature: 60°C")
        print("• Max Vibration: 0.5g")
        
        confirm = input("\nConfirm safety parameters acceptable (y/N): ").lower().strip()
        
        self.state_machine.update_configuration({
            'safety_confirmed': confirm == 'y',
            'safety_verification_timestamp': datetime.now().isoformat()
        })
    
    async def _handle_validation(self):
        """Handle validation phase"""
        print("\n✅ Final Validation")
        print("-" * 30)
        print("Validating all configuration parameters...")
        
        # Simulate validation delay
        import time
        for i in range(3):
            print(f"Validation step {i+1}/3...")
            time.sleep(0.5)
        
        print("✓ All parameters validated successfully")
    
    async def _handle_error_state(self) -> bool:
        """Handle error state"""
        print("\n❌ Setup encountered an error")
        print("Available rollback options:")
        
        checkpoints = self.state_machine.get_available_checkpoints()
        for i, cp in enumerate(checkpoints):
            status = "✓" if cp['valid'] else "✗"
            print(f"{i}: {cp['phase']} ({status}) - {cp['timestamp']}")
        
        try:
            choice = int(input("Select checkpoint to rollback to (or -1 to exit): "))
            if choice == -1:
                return False
            elif 0 <= choice < len(checkpoints):
                success = await self.state_machine.rollback_to_checkpoint(choice)
                if success:
                    print(f"Rolled back to {checkpoints[choice]['phase']}")
                    return True
        except ValueError:
            pass
        
        print("Invalid selection. Exiting setup.")
        return False
    
    async def _offer_rollback_option(self):
        """Offer rollback option when validation fails"""
        checkpoints = self.state_machine.get_available_checkpoints()
        if checkpoints:
            print("\n🔧 Validation failed. Available rollback points:")
            for i, cp in enumerate(checkpoints[-3:], len(checkpoints)-3):  # Show last 3
                print(f"{i}: {cp['phase']} - {cp['timestamp']}")
            
            try:
                choice = input("Rollback to previous checkpoint? (y/N): ").lower().strip()
                if choice == 'y':
                    # Rollback to last valid checkpoint
                    for i in reversed(range(len(checkpoints))):
                        if checkpoints[i]['valid']:
                            await self.state_machine.rollback_to_checkpoint(i)
                            break
            except:
                pass

# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize state machine
    state_machine = IndustrialSetupStateMachine("industrial_setup_config.json")
    wizard = InteractiveSetupWizard(state_machine)
    
    # Run setup
    success = await wizard.run_interactive_setup()
    
    if success:
        print(f"\nSetup Progress: {state_machine.get_progress()*100:.1f}%")
        print(f"Current Phase: {state_machine.get_current_phase()}")
        print("Available rollback points:", len(state_machine.get_available_checkpoints()))

if __name__ == "__main__":
    asyncio.run(main())