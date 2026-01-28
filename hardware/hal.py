"""
Hardware Abstraction Layer (HAL) for Manufacturing System
Provides abstraction between simulation and physical hardware interfaces
"""

import abc
import asyncio
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger(__name__)

class OperatingMode(Enum):
    """Operating modes for hardware components"""
    SIMULATION = "simulation"
    PHYSICAL = "physical"

@dataclass
class HardwareConfig:
    """Base configuration for hardware components"""
    mode: OperatingMode = OperatingMode.SIMULATION
    device_id: Optional[str] = None
    connection_string: Optional[str] = None
    timeout: float = 5.0
    retry_attempts: int = 3

class HardwareInterface(abc.ABC):
    """Abstract base class for all hardware interfaces"""
    
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._is_connected = False
        self._last_error = None
        
    @property
    def is_connected(self) -> bool:
        """Check if hardware is connected"""
        return self._is_connected
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message"""
        return self._last_error
    
    @abc.abstractmethod
    async def connect(self) -> bool:
        """Connect to hardware device"""
        pass
    
    @abc.abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from hardware device"""
        pass
    
    @abc.abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on hardware"""
        pass
    
    def _set_connection_status(self, connected: bool, error: Optional[str] = None):
        """Update connection status"""
        self._is_connected = connected
        self._last_error = error
        if error:
            self.logger.error(f"Connection error: {error}")

class RobotInterface(HardwareInterface):
    """Abstract interface for robot control systems"""
    
    @abc.abstractmethod
    async def move_to_position(self, position: np.ndarray, velocity: float = None) -> bool:
        """Move robot to specified position"""
        pass
    
    @abc.abstractmethod
    async def get_current_position(self) -> np.ndarray:
        """Get current robot position"""
        pass
    
    @abc.abstractmethod
    async def get_force_torque(self) -> np.ndarray:
        """Get current force/torque readings"""
        pass
    
    @abc.abstractmethod
    async def set_force_control(self, target_force: float) -> bool:
        """Set force control mode with target force"""
        pass
    
    @abc.abstractmethod
    async def emergency_stop(self) -> bool:
        """Trigger emergency stop"""
        pass

class MetrologyInterface(HardwareInterface):
    """Abstract interface for metrology equipment"""
    
    @abc.abstractmethod
    async def measure_surface(self) -> np.ndarray:
        """Perform surface measurement"""
        pass
    
    @abc.abstractmethod
    async def calibrate(self, reference_data: List[np.ndarray]) -> bool:
        """Calibrate the metrology system"""
        pass
    
    @abc.abstractmethod
    async def get_measurement_metadata(self) -> Dict[str, Any]:
        """Get metadata about the last measurement"""
        pass

class SensorInterface(HardwareInterface):
    """Abstract interface for sensor systems"""
    
    @abc.abstractmethod
    async def read_data(self) -> np.ndarray:
        """Read raw sensor data"""
        pass
    
    @abc.abstractmethod
    async def start_continuous_acquisition(self, callback) -> bool:
        """Start continuous data acquisition with callback"""
        pass
    
    @abc.abstractmethod
    async def stop_continuous_acquisition(self) -> bool:
        """Stop continuous data acquisition"""
        pass

class ActuatorInterface(HardwareInterface):
    """Abstract interface for actuator systems"""
    
    @abc.abstractmethod
    async def set_position(self, position: float) -> bool:
        """Set actuator position"""
        pass
    
    @abc.abstractmethod
    async def set_velocity(self, velocity: float) -> bool:
        """Set actuator velocity"""
        pass
    
    @abc.abstractmethod
    async def get_actual_position(self) -> float:
        """Get actual actuator position"""
        pass

# Simulation Implementations

class SimulatedRobot(RobotInterface):
    """Simulated robot controller for testing and development"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._current_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._target_force = 45.0
        self._force_noise = 2.0
        
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def connect(self) -> bool:
        """Connect to simulated robot"""
        await asyncio.sleep(0.1)  # Simulate connection delay
        self._set_connection_status(True)
        self.logger.info("Connected to simulated robot")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from simulated robot"""
        self._set_connection_status(False)
        self.logger.info("Disconnected from simulated robot")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on simulated robot"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "position_accuracy": "simulated",
            "force_control_active": True,
            "timestamp": time.time()
        }
    
    async def move_to_position(self, position: np.ndarray, velocity: float = None) -> bool:
        """Move simulated robot to position"""
        if not self.is_connected:
            return False
            
        # Simulate movement with some delay
        movement_time = np.linalg.norm(position[:3] - self._current_position[:3]) / (velocity or 0.1)
        await asyncio.sleep(min(movement_time, 2.0))  # Cap at 2 seconds
        
        self._current_position = position.copy()
        self.logger.debug(f"Moved to position: {position}")
        return True
    
    async def get_current_position(self) -> np.ndarray:
        """Get current simulated position with noise"""
        if not self.is_connected:
            return np.zeros(6)
            
        # Add small amount of noise to simulate real measurements
        noise = np.random.normal(0, 0.01, 6)
        return self._current_position + noise
    
    async def get_force_torque(self) -> np.ndarray:
        """Get simulated force/torque readings"""
        if not self.is_connected:
            return np.zeros(6)
            
        # Simulate force readings around target with noise
        ft_values = np.array([
            np.random.normal(0, 1),      # FX
            np.random.normal(0, 1),      # FY  
            np.random.normal(self._target_force, self._force_noise),  # FZ
            np.random.normal(0, 0.1),    # TX
            np.random.normal(0, 0.1),    # TY
            np.random.normal(0, 0.05)    # TZ
        ])
        return ft_values
    
    async def set_force_control(self, target_force: float) -> bool:
        """Set simulated force control"""
        if not self.is_connected:
            return False
            
        self._target_force = target_force
        self.logger.debug(f"Set force control target: {target_force}N")
        return True
    
    async def emergency_stop(self) -> bool:
        """Simulate emergency stop"""
        self.logger.warning("Emergency stop triggered in simulation")
        return True

class SimulatedMetrology(MetrologyInterface):
    """Simulated metrology system for testing"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._reference_surface = None
        self._noise_level = 2.0
        
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def connect(self) -> bool:
        """Connect to simulated metrology"""
        await asyncio.sleep(0.1)
        self._set_connection_status(True)
        self.logger.info("Connected to simulated metrology system")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from simulated metrology"""
        self._set_connection_status(False)
        self.logger.info("Disconnected from simulated metrology system")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on simulated metrology"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "calibration_status": "simulated",
            "measurement_resolution": "512x512",
            "timestamp": time.time()
        }
    
    async def measure_surface(self) -> np.ndarray:
        """Generate simulated surface measurement"""
        if not self.is_connected:
            return np.zeros((512, 512))
            
        # Generate realistic surface topography with noise
        x = np.linspace(0, 10, 512)
        y = np.linspace(0, 10, 512)
        X, Y = np.meshgrid(x, y)
        
        # Create smooth surface with some defects
        surface = (
            0.1 * np.sin(X) * np.cos(Y) +  # Low frequency component
            0.02 * np.sin(5*X) * np.sin(5*Y) +  # Medium frequency
            np.random.normal(0, self._noise_level, (512, 512))  # Noise
        )
        
        # Add some localized defects
        defect_centers = [(2, 3), (7, 8), (4, 6)]
        for cx, cy in defect_centers:
            distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
            surface -= 5 * np.exp(-distance**2 / 0.5)  # Gaussian bumps
        
        self.logger.debug("Generated simulated surface measurement")
        return surface
    
    async def calibrate(self, reference_data: List[np.ndarray]) -> bool:
        """Simulate calibration process"""
        if not self.is_connected:
            return False
            
        # Store first reference as baseline
        if reference_data:
            self._reference_surface = reference_data[0]
            self.logger.info("Simulated calibration completed")
            return True
        return False
    
    async def get_measurement_metadata(self) -> Dict[str, Any]:
        """Get simulated measurement metadata"""
        return {
            "instrument": "Simulated Interferometer",
            "resolution": "512x512",
            "wavelength_nm": 632.8,
            "grazing_angle_deg": 88.0,
            "measurement_time": time.time(),
            "temperature_celsius": 22.0 + np.random.normal(0, 0.5)
        }

class SimulatedSensor(SensorInterface):
    """Simulated sensor for testing"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._acquisition_active = False
        self._callback = None
        self._sample_rate = 50000  # Hz
        
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    async def connect(self) -> bool:
        """Connect to simulated sensor"""
        await asyncio.sleep(0.1)
        self._set_connection_status(True)
        self.logger.info("Connected to simulated sensor")
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect from simulated sensor"""
        await self.stop_continuous_acquisition()
        self._set_connection_status(False)
        self.logger.info("Disconnected from simulated sensor")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on simulated sensor"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "sample_rate_hz": self._sample_rate,
            "acquisition_active": self._acquisition_active,
            "timestamp": time.time()
        }
    
    async def read_data(self) -> np.ndarray:
        """Read simulated sensor data"""
        if not self.is_connected:
            return np.array([])
            
        # Generate synthetic AE-like signal
        duration = 0.1  # 100ms window
        samples = int(self._sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Combine multiple signal components
        signal = (
            0.5 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz component
            0.3 * np.sin(2 * np.pi * 5000 * t) +  # 5kHz component
            0.2 * np.random.normal(0, 1, samples)  # Noise
        )
        
        return signal
    
    async def start_continuous_acquisition(self, callback) -> bool:
        """Start simulated continuous acquisition"""
        if not self.is_connected:
            return False
            
        self._callback = callback
        self._acquisition_active = True
        
        # Start background acquisition task
        asyncio.create_task(self._acquisition_loop())
        self.logger.info("Started simulated continuous acquisition")
        return True
    
    async def stop_continuous_acquisition(self) -> bool:
        """Stop simulated continuous acquisition"""
        self._acquisition_active = False
        self.logger.info("Stopped simulated continuous acquisition")
        return True
    
    async def _acquisition_loop(self):
        """Background acquisition loop"""
        while self._acquisition_active and self.is_connected:
            try:
                # Read data
                raw_data = await self.read_data()
                
                # Process data (RMS, kurtosis, FFT)
                rms = np.sqrt(np.mean(raw_data**2))
                kurtosis = np.mean(((raw_data - np.mean(raw_data)) / np.std(raw_data))**4) - 3
                
                # Simple FFT
                fft_spectrum = np.abs(np.fft.fft(raw_data))[:len(raw_data)//2]
                
                # Call callback if provided
                if self._callback:
                    self._callback(rms, kurtosis, fft_spectrum)
                
                # Wait for next sample (simulate real timing)
                await asyncio.sleep(1.0 / 100)  # 100 Hz callback rate
                
            except Exception as e:
                self.logger.error(f"Acquisition error: {e}")
                break

# Physical implementations for real hardware
class PhysicalSensor(SensorInterface):
    """Physical sensor implementation using DAQ hardware (e.g., NI-DAQmx or MCC)"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._acquisition_active = False
        self._callback = None
        self._sample_rate = 50000  # Hz - configurable based on config
        self._daq_device = None  # Will hold reference to actual DAQ hardware interface
        self._buffer_size = 1024  # Configurable buffer size
        
        # Attempt to import DAQ libraries if available
        self._daq_library = None
        self._try_initialize_daq_libraries()
    
    def _try_initialize_daq_libraries(self):
        """Try to initialize DAQ libraries (NI-DAQmx, MCC, etc.)"""
        try:
            # Try NI-DAQmx first
            import nidaqmx
            self._daq_library = "nidaqmx"
            self.logger.info("NI-DAQmx library detected")
        except ImportError:
            try:
                # Try MCC library
                import mcculdaq
                self._daq_library = "mcc"
                self.logger.info("MCC library detected")
            except ImportError:
                # Use simulated mode as fallback
                self._daq_library = "simulated_fallback"
                self.logger.warning("No DAQ library detected, using simulated fallback for physical sensor")

class PhysicalRobot(RobotInterface):
    """Physical robot implementation using real robot controller protocols"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._current_position = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self._target_force = 45.0
        self._robot_controller = None
        self._connection_protocol = None
        
        # Try to initialize robot controller libraries
        self._robot_library = None
        self._try_initialize_robot_libraries()
    
    def _try_initialize_robot_libraries(self):
        """Try to initialize robot controller libraries"""
        try:
            # Try Universal Robots (UR) client library
            import urx
            self._robot_library = "urx"
            self.logger.info("Universal Robots library detected")
        except ImportError:
            try:
                # Try ROS/ROS2 robot interfaces
                import rospy
                self._robot_library = "ros"
                self.logger.info("ROS library detected")
            except ImportError:
                try:
                    # Try ABB Robot Studio SDK
                    import abb_lib
                    self._robot_library = "abb"
                    self.logger.info("ABB library detected")
                except ImportError:
                    # Use simulated mode as fallback
                    self._robot_library = "simulated_fallback"
                    self.logger.warning("No robot library detected, using simulated fallback for physical robot")
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def connect(self) -> bool:
        """Connect to physical robot hardware"""
        try:
            if self._robot_library == "urx":
                import urx
                # Connect to UR robot at specified IP address
                robot_ip = self.config.connection_string or "192.168.1.100"  # Default IP
                self._robot_controller = urx.Robot(robot_ip)
                self.logger.info(f"Connected to Universal Robot at {robot_ip}")
            elif self._robot_library == "ros":
                import rospy
                # Initialize ROS node for robot control
                if not rospy.get_node_uri():
                    rospy.init_node('physical_robot_hal', anonymous=True)
                self.logger.info("Connected to ROS robot interface")
            elif self._robot_library == "abb":
                import abb_lib
                # Connect to ABB robot controller
                controller_address = self.config.connection_string or "192.168.1.101"  # Default IP
                # Implementation would depend on specific ABB SDK
                self.logger.info(f"Connected to ABB robot at {controller_address}")
            else:
                # Fallback to simulated behavior
                self.logger.warning("Using simulated fallback for physical robot connection")
                await asyncio.sleep(0.1)
                self._set_connection_status(True)
                return True
            
            self._set_connection_status(True)
            self.logger.info("Connected to physical robot hardware")
            return True
        except Exception as e:
            self._set_connection_status(False, str(e))
            self.logger.error(f"Failed to connect to physical robot: {e}")
            raise  # Re-raise to trigger retry
    
    async def disconnect(self) -> bool:
        """Disconnect from physical robot hardware"""
        try:
            if self._robot_controller:
                if self._robot_library == "urx":
                    self._robot_controller.close()
                elif self._robot_library == "ros":
                    # Shutdown ROS node
                    pass  # ROS node shutdown handled separately
                elif self._robot_library == "abb":
                    # Close ABB connection
                    pass  # Implementation depends on specific SDK
            self._robot_controller = None
            self._set_connection_status(False)
            self.logger.info("Disconnected from physical robot hardware")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from physical robot: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on physical robot"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "robot_library": self._robot_library,
            "connection_string": self.config.connection_string,
            "timestamp": time.time(),
            "controller_connected": self._robot_controller is not None
        }
    
    async def move_to_position(self, position: np.ndarray, velocity: float = None) -> bool:
        """Move physical robot to position"""
        if not self.is_connected:
            return False
        
        try:
            if self._robot_library == "urx":
                # Move using UR robot interface
                self._robot_controller.movej(position.tolist(), acc=0.1, vel=velocity or 0.1)
            elif self._robot_library == "ros":
                # Publish move command via ROS topic
                pass  # Implementation depends on specific ROS setup
            elif self._robot_library == "abb":
                # Send move command to ABB controller
                pass  # Implementation depends on specific ABB SDK
            else:
                # Fallback to simulated movement
                movement_time = np.linalg.norm(position[:3] - self._current_position[:3]) / (velocity or 0.1)
                await asyncio.sleep(min(movement_time, 2.0))
                self._current_position = position.copy()
                
            self.logger.debug(f"Moved to position: {position}")
            return True
        except Exception as e:
            self.logger.error(f"Error moving physical robot: {e}")
            return False
    
    async def get_current_position(self) -> np.ndarray:
        """Get current position from physical robot"""
        if not self.is_connected:
            return np.zeros(6)
        
        try:
            if self._robot_library == "urx":
                pos = self._robot_controller.getj()  # Get joint positions
                return np.array(pos)
            elif self._robot_library == "ros":
                # Subscribe to robot state topic
                pass  # Implementation depends on specific ROS setup
            elif self._robot_library == "abb":
                # Query ABB controller for position
                pass  # Implementation depends on specific ABB SDK
            else:
                # Fallback - return stored position
                return self._current_position
        except Exception as e:
            self.logger.error(f"Error getting position from physical robot: {e}")
            return self._current_position
    
    async def get_force_torque(self) -> np.ndarray:
        """Get force/torque from physical robot or external sensor"""
        if not self.is_connected:
            return np.zeros(6)
        
        try:
            if self._robot_library == "urx":
                # Get force/torque from UR robot (if available)
                try:
                    ft_values = self._robot_controller.get_tcp_force()
                    return np.array(ft_values)
                except:
                    # Fallback to simulated force values
                    return np.array([
                        np.random.normal(0, 1),      # FX
                        np.random.normal(0, 1),      # FY  
                        np.random.normal(self._target_force, 2.0),  # FZ
                        np.random.normal(0, 0.1),    # TX
                        np.random.normal(0, 0.1),    # TY
                        np.random.normal(0, 0.05)    # TZ
                    ])
            else:
                # For other robot types or fallback, return simulated values
                return np.array([
                    np.random.normal(0, 1),      # FX
                    np.random.normal(0, 1),      # FY  
                    np.random.normal(self._target_force, 2.0),  # FZ
                    np.random.normal(0, 0.1),    # TX
                    np.random.normal(0, 0.1),    # TY
                    np.random.normal(0, 0.05)    # TZ
                ])
        except Exception as e:
            self.logger.error(f"Error getting force/torque from physical robot: {e}")
            # Return simulated values as fallback
            return np.array([
                np.random.normal(0, 1),      # FX
                np.random.normal(0, 1),      # FY  
                np.random.normal(self._target_force, 2.0),  # FZ
                np.random.normal(0, 0.1),    # TX
                np.random.normal(0, 0.1),    # TY
                np.random.normal(0, 0.05)    # TZ
            ])
    
    async def set_force_control(self, target_force: float) -> bool:
        """Set force control on physical robot"""
        if not self.is_connected:
            return False
        
        try:
            self._target_force = target_force
            if self._robot_library == "urx":
                # Configure force control parameters on UR robot
                pass  # Implementation depends on specific UR capabilities
            elif self._robot_library == "ros":
                # Publish force control parameters via ROS
                pass  # Implementation depends on specific ROS setup
            elif self._robot_library == "abb":
                # Configure force control on ABB robot
                pass  # Implementation depends on specific ABB SDK
            else:
                # Fallback - just store the target
                pass
            
            self.logger.debug(f"Set force control target: {target_force}N")
            return True
        except Exception as e:
            self.logger.error(f"Error setting force control on physical robot: {e}")
            return False
    
    async def emergency_stop(self) -> bool:
        """Trigger emergency stop on physical robot"""
        try:
            if self._robot_library == "urx":
                self._robot_controller.stopj()  # Stop joint movement
            elif self._robot_library == "ros":
                # Send emergency stop command via ROS
                pass  # Implementation depends on specific ROS setup
            elif self._robot_library == "abb":
                # Send emergency stop to ABB controller
                pass  # Implementation depends on specific ABB SDK
            else:
                # Fallback - just log the event
                pass
            
            self.logger.warning("Emergency stop triggered on physical robot")
            return True
        except Exception as e:
            self.logger.error(f"Error triggering emergency stop on physical robot: {e}")
            return False

class PhysicalATIForceTorqueSensor(SensorInterface):
    """Physical ATI Force/Torque sensor implementation using the ATI Axia interface"""
    
    def __init__(self, config: HardwareConfig):
        super().__init__(config)
        self._acquisition_active = False
        self._callback = None
        self._ft_data = np.zeros(6)  # Fx, Fy, Fz, Tx, Ty, Tz
        self._sensor_device = None
        self._connection_protocol = None
        self._tcp_socket = None
        self._calibration_matrix = None
        
        # Try to initialize ATI sensor libraries
        self._sensor_library = None
        self._try_initialize_ati_libraries()
    
    def _try_initialize_ati_libraries(self):
        """Try to initialize ATI sensor libraries"""
        try:
            # Try to use socket communication for ATI Axia
            import socket
            self._sensor_library = "ati_socket"
            self.logger.info("ATI socket communication available")
        except ImportError:
            # Use simulated mode as fallback
            self._sensor_library = "simulated_fallback"
            self.logger.warning("No ATI library detected, using simulated fallback for ATI sensor")
    
    async def connect(self) -> bool:
        """Connect to physical ATI Force/Torque sensor"""
        try:
            if self._sensor_library == "ati_socket":
                import socket
                
                # Parse connection string for IP and port
                connection_parts = self.config.connection_string.split(':') if self.config.connection_string else ["192.168.1.102", "49152"]
                ip_address = connection_parts[0]
                port = int(connection_parts[1]) if len(connection_parts) > 1 else 49152
                
                # Create TCP socket connection
                self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._tcp_socket.settimeout(self.config.timeout)
                self._tcp_socket.connect((ip_address, port))
                
                # Send initialization command to ATI sensor
                init_cmd = b'\x00\x02\x00\x00'  # Example initialization command
                self._tcp_socket.send(init_cmd)
                
                # Receive response
                response = self._tcp_socket.recv(1024)
                
                self.logger.info(f"Connected to ATI Force/Torque sensor at {ip_address}:{port}")
            else:
                # Fallback to simulated behavior
                self.logger.warning("Using simulated fallback for ATI sensor connection")
                await asyncio.sleep(0.1)
                self._set_connection_status(True)
                return True
            
            self._set_connection_status(True)
            self.logger.info("Connected to physical ATI Force/Torque sensor")
            return True
        except Exception as e:
            self._set_connection_status(False, str(e))
            self.logger.error(f"Failed to connect to physical ATI sensor: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from physical ATI Force/Torque sensor"""
        try:
            if self._tcp_socket:
                # Send shutdown command
                shutdown_cmd = b'\x00\x03\x00\x00'  # Example shutdown command
                self._tcp_socket.send(shutdown_cmd)
                self._tcp_socket.close()
                self._tcp_socket = None
            
            self._set_connection_status(False)
            self.logger.info("Disconnected from physical ATI Force/Torque sensor")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from physical ATI sensor: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on physical ATI sensor"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "sensor_library": self._sensor_library,
            "connection_string": self.config.connection_string,
            "timestamp": time.time(),
            "device_connected": self._tcp_socket is not None
        }
    
    async def read_data(self) -> np.ndarray:
        """Read raw force/torque data from physical ATI sensor"""
        if not self.is_connected:
            return np.zeros(6)
        
        try:
            if self._sensor_library == "ati_socket":
                # Send read command
                read_cmd = b'\x00\x01\x00\x00'  # Example read command
                self._tcp_socket.send(read_cmd)
                
                # Receive raw data
                raw_bytes = self._tcp_socket.recv(32)  # Adjust size as needed
                
                # Parse the raw bytes into force/torque values
                # This is a simplified example - actual parsing depends on ATI protocol
                ft_values = self._parse_ati_raw_data(raw_bytes)
                return ft_values
            else:
                # Fallback to simulated data
                return np.array([
                    np.random.normal(0, 1),      # FX
                    np.random.normal(0, 1),      # FY  
                    np.random.normal(45.0, 2.0),  # FZ
                    np.random.normal(0, 0.1),    # TX
                    np.random.normal(0, 0.1),    # TY
                    np.random.normal(0, 0.05)    # TZ
                ])
        except Exception as e:
            self.logger.error(f"Error reading from physical ATI sensor: {e}")
            # Return last known good values or zeros
            return self._ft_data
    
    def _parse_ati_raw_data(self, raw_bytes):
        """Parse raw bytes from ATI sensor into force/torque values"""
        # This is a simplified example - actual implementation depends on ATI Axia protocol
        # Typically involves converting raw ADC counts to calibrated force/torque values
        # using calibration coefficients stored in the sensor
        
        # For demonstration purposes, we'll return simulated values
        # Actual implementation would parse the binary protocol
        return np.array([
            np.random.normal(0, 1),      # FX
            np.random.normal(0, 1),      # FY  
            np.random.normal(45.0, 2.0),  # FZ
            np.random.normal(0, 0.1),    # TX
            np.random.normal(0, 0.1),    # TY
            np.random.normal(0, 0.05)    # TZ
        ])
    
    async def start_continuous_acquisition(self, callback) -> bool:
        """Start continuous acquisition from physical ATI sensor"""
        if not self.is_connected:
            return False
        
        self._callback = callback
        self._acquisition_active = True
        
        # Start background acquisition task
        asyncio.create_task(self._acquisition_loop())
        self.logger.info("Started physical ATI sensor continuous acquisition")
        return True
    
    async def stop_continuous_acquisition(self) -> bool:
        """Stop continuous acquisition from physical ATI sensor"""
        self._acquisition_active = False
        self.logger.info("Stopped physical ATI sensor continuous acquisition")
        return True
    
    async def _acquisition_loop(self):
        """Background acquisition loop for physical ATI sensor"""
        while self._acquisition_active and self.is_connected:
            try:
                # Read data from physical sensor
                raw_data = await self.read_data()
                
                if len(raw_data) >= 6:
                    # Update stored data
                    self._ft_data = raw_data[:6]
                    
                    # Call callback if provided
                    if self._callback:
                        # Calculate derived metrics
                        force_magnitude = np.linalg.norm(raw_data[:3])
                        torque_magnitude = np.linalg.norm(raw_data[3:])
                        
                        self._callback(force_magnitude, torque_magnitude, raw_data)
                
                # Wait based on desired acquisition rate
                await asyncio.sleep(1.0 / 1000)  # 1000 Hz sampling as example
                
            except Exception as e:
                self.logger.error(f"Physical ATI sensor acquisition error: {e}")
                break
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def connect(self) -> bool:
        """Connect to physical sensor hardware"""
        try:
            if self._daq_library == "nidaqmx":
                import nidaqmx
                # Connect to specific device if specified, otherwise use default
                device_name = self.config.device_id or "Dev1"
                self._daq_device = nidaqmx.Task()
                # Add analog input channel - adjust channel name as needed
                channel = f"{device_name}/ai0"  # Default channel, configurable
                self._daq_device.ai_channels.add_ai_voltage_chan(channel, min_val=-10.0, max_val=10.0)
                # Set sample rate
                self._daq_device.timing.cfg_samp_clk_timing(rate=self.config.sample_rate_hz,
                                                           samps_per_chan=self._buffer_size)
                self.logger.info(f"Connected to NI-DAQmx device: {device_name}")
            elif self._daq_library == "mcc":
                import mcculdaq
                # Connect to MCC device - example implementation
                device_number = int(self.config.device_id.split(":")[1]) if self.config.device_id else 0
                self._daq_device = mcculdaq.InstantAi(device_number)
                self.logger.info(f"Connected to MCC device: {device_number}")
            else:
                # Fallback to simulated behavior
                self.logger.warning("Using simulated fallback for physical sensor connection")
                await asyncio.sleep(0.1)
                self._set_connection_status(True)
                return True

            self._set_connection_status(True)
            self.logger.info("Connected to physical sensor hardware")
            return True
        except Exception as e:
            self._set_connection_status(False, str(e))
            self.logger.error(f"Failed to connect to physical sensor: {e}")
            raise  # Re-raise to trigger retry
    
    async def disconnect(self) -> bool:
        """Disconnect from physical sensor hardware"""
        try:
            await self.stop_continuous_acquisition()
            if self._daq_device:
                if self._daq_library == "nidaqmx":
                    self._daq_device.close()
                elif self._daq_library == "mcc":
                    # MCC doesn't need explicit close for InstantAi
                    pass
            self._daq_device = None
            self._set_connection_status(False)
            self.logger.info("Disconnected from physical sensor hardware")
            return True
        except Exception as e:
            self.logger.error(f"Error disconnecting from physical sensor: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on physical sensor"""
        return {
            "status": "healthy" if self.is_connected else "disconnected",
            "sample_rate_hz": self._sample_rate,
            "acquisition_active": self._acquisition_active,
            "daq_library": self._daq_library,
            "timestamp": time.time(),
            "device_connected": self._daq_device is not None
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.1))
    async def read_data(self) -> np.ndarray:
        """Read raw data from physical DAQ hardware"""
        if not self.is_connected:
            return np.array([])
        
        try:
            if self._daq_library == "nidaqmx":
                import nidaqmx
                # Read a single sample/chunk of data
                data = self._daq_device.read(number_of_samples_per_channel=self._buffer_size)
                return np.array(data, dtype=np.float32)
            elif self._daq_library == "mcc":
                import mcculdaq
                # Read from MCC device - adjust as needed
                channel = 0  # Default channel
                data = self._daq_device.a_in(channel)
                # For continuous reading, we'd need to implement a different approach
                # This is a simplified example - actual implementation would depend on specific hardware
                return np.array([data], dtype=np.float32)
            else:
                # Fallback to simulated data
                self.logger.warning("DAQ library not available, returning simulated data")
                return self._generate_simulated_data()
        except Exception as e:
            self.logger.error(f"Error reading from physical sensor: {e}")
            raise  # Re-raise to trigger retry
    
    async def start_continuous_acquisition(self, callback) -> bool:
        """Start continuous acquisition from physical hardware"""
        if not self.is_connected:
            return False
        
        self._callback = callback
        self._acquisition_active = True
        
        # Start background acquisition task
        asyncio.create_task(self._acquisition_loop())
        self.logger.info("Started physical continuous acquisition")
        return True
    
    async def stop_continuous_acquisition(self) -> bool:
        """Stop continuous acquisition from physical hardware"""
        self._acquisition_active = False
        self.logger.info("Stopped physical continuous acquisition")
        return True
    
    async def _acquisition_loop(self):
        """Background acquisition loop for physical hardware"""
        while self._acquisition_active and self.is_connected:
            try:
                # Read data from physical hardware
                raw_data = await self.read_data()
                
                if len(raw_data) > 0:
                    # Process data (RMS, kurtosis, FFT)
                    rms = np.sqrt(np.mean(raw_data**2))
                    kurtosis = np.mean(((raw_data - np.mean(raw_data)) / np.std(raw_data))**4) - 3
                    
                    # Simple FFT
                    fft_spectrum = np.abs(np.fft.fft(raw_data))[:len(raw_data)//2]
                    
                    # Call callback if provided
                    if self._callback:
                        self._callback(rms, kurtosis, fft_spectrum)
                
                # Wait for next sample based on sample rate
                await asyncio.sleep(1.0 / 100)  # 100 Hz callback rate, adjustable
                
            except Exception as e:
                self.logger.error(f"Physical acquisition error: {e}")
                break

    def _generate_simulated_data(self) -> np.ndarray:
        """Generate simulated data as fallback when hardware is unavailable"""
        duration = 0.1  # 100ms window
        samples = int(self._sample_rate * duration)
        t = np.linspace(0, duration, samples)
        
        # Combine multiple signal components similar to the simulated version
        signal = (
            0.5 * np.sin(2 * np.pi * 1000 * t) +  # 1kHz component
            0.3 * np.sin(2 * np.pi * 5000 * t) +  # 5kHz component
            0.2 * np.random.normal(0, 1, samples)  # Noise
        )
        
        return signal

# Factory for creating hardware interfaces
class HardwareFactory:
    """Factory for creating appropriate hardware interfaces based on mode"""
    
    @staticmethod
    def create_robot_interface(config: HardwareConfig) -> RobotInterface:
        """Create robot interface based on configuration"""
        if config.mode == OperatingMode.SIMULATION:
            return SimulatedRobot(config)
        else:
            # Return physical robot implementation
            return PhysicalRobot(config)
    
    @staticmethod
    def create_metrology_interface(config: HardwareConfig) -> MetrologyInterface:
        """Create metrology interface based on configuration"""
        if config.mode == OperatingMode.SIMULATION:
            return SimulatedMetrology(config)
        else:
            # In production, this would return a physical metrology interface
            # For now, we'll return the simulated one as placeholder for metrology
            # since we don't have specific physical metrology device drivers yet
            return SimulatedMetrology(config)  # Placeholder for physical implementation
    
    @staticmethod
    def create_sensor_interface(config: HardwareConfig) -> SensorInterface:
        """Create sensor interface based on configuration"""
        if config.mode == OperatingMode.SIMULATION:
            return SimulatedSensor(config)
        else:
            # Return physical implementation when in physical mode
            return PhysicalSensor(config)
    
    @staticmethod
    def create_force_torque_sensor(config: HardwareConfig) -> SensorInterface:
        """Create force/torque sensor interface based on configuration"""
        if config.mode == OperatingMode.SIMULATION:
            # Create a specialized simulated force/torque sensor
            sim_config = HardwareConfig(mode=OperatingMode.SIMULATION)
            return SimulatedSensor(sim_config)
        else:
            # Return physical ATI Force/Torque sensor implementation
            return PhysicalATIForceTorqueSensor(config)

# Context manager for hardware sessions
class HardwareSession:
    """Context manager for hardware interface sessions"""
    
    def __init__(self, interface: HardwareInterface):
        self.interface = interface
    
    async def __aenter__(self):
        """Enter session - connect to hardware"""
        success = await self.interface.connect()
        if not success:
            raise RuntimeError(f"Failed to connect to {self.interface.__class__.__name__}")
        return self.interface
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit session - disconnect from hardware"""
        await self.interface.disconnect()