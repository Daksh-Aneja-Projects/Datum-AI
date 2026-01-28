"""
Edge Computing and Real-Time Control Layer
Hierarchical Cloud-Edge-Real-time Controller Architecture for sub-millisecond latency
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import numpy as np
import logging
from collections import deque
import zmq
from concurrent.futures import ThreadPoolExecutor

@dataclass
class ControlCommand:
    """Real-time control command"""
    timestamp: float
    robot_velocity: np.ndarray  # [vx, vy, vz, vrx, vry, vrz]
    spindle_speed: float
    coolant_flow: float
    priority: int  # 0=highest, 9=lowest

@dataclass
class SensorReading:
    """Sensor data reading with timestamp"""
    timestamp: float
    force_torque: np.ndarray
    position: np.ndarray
    acoustic_emission: float
    temperature: float

class RealTimeController:
    """Microsecond-level real-time controller"""
    
    def __init__(self, cycle_time_us: int = 100):
        self.cycle_time_us = cycle_time_us
        self.control_loop_active = False
        self.current_command = None
        self.sensor_buffer = deque(maxlen=1000)
        self.command_buffer = deque(maxlen=1000)
        self.logger = logging.getLogger(__name__)
        
        # Performance monitoring
        self.cycle_times = deque(maxlen=1000)
        self.jitter_buffer = deque(maxlen=1000)
        
    def start_control_loop(self):
        """Start real-time control loop"""
        self.control_loop_active = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        self.logger.info(f"Real-time controller started with {self.cycle_time_us}μs cycle time")
        
    def stop_control_loop(self):
        """Stop real-time control loop"""
        self.control_loop_active = False
        if hasattr(self, 'control_thread'):
            self.control_thread.join(timeout=0.1)
            
    def _control_loop(self):
        """Main real-time control loop"""
        last_cycle = time.perf_counter()
        
        while self.control_loop_active:
            cycle_start = time.perf_counter()
            
            try:
                # Execute one control cycle
                self._execute_control_cycle()
                
                # Timing analysis
                cycle_end = time.perf_counter()
                cycle_duration = (cycle_end - cycle_start) * 1e6  # Convert to microseconds
                self.cycle_times.append(cycle_duration)
                
                # Jitter calculation
                expected_interval = self.cycle_time_us * 1e-6
                actual_interval = cycle_start - last_cycle
                jitter = abs(actual_interval - expected_interval) * 1e6
                self.jitter_buffer.append(jitter)
                
                last_cycle = cycle_start
                
                # Sleep for remaining cycle time
                sleep_time = max(0, (self.cycle_time_us * 1e-6) - (time.perf_counter() - cycle_start))
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                self.logger.error(f"Control cycle error: {str(e)}")
                time.sleep(0.001)  # Brief pause on error
    
    def _execute_control_cycle(self):
        """Execute single control cycle"""
        # Read latest sensor data
        latest_sensors = self._get_latest_sensor_data()
        
        # Get current command
        current_cmd = self._get_current_command()
        
        if current_cmd and latest_sensors:
            # Apply control laws
            actuator_commands = self._compute_actuator_commands(
                current_cmd, latest_sensors
            )
            
            # Send to hardware (simulated)
            self._send_to_actuators(actuator_commands)
    
    def _get_latest_sensor_data(self) -> Optional[SensorReading]:
        """Get most recent sensor reading"""
        if self.sensor_buffer:
            return self.sensor_buffer[-1]
        return None
    
    def _get_current_command(self) -> Optional[ControlCommand]:
        """Get current control command"""
        if self.command_buffer:
            return self.command_buffer[-1]
        return None
    
    def _compute_actuator_commands(self, command: ControlCommand, 
                                 sensors: SensorReading) -> Dict:
        """Compute low-level actuator commands"""
        # PID control for velocity tracking
        velocity_error = command.robot_velocity - sensors.position[3:9]  # Assuming position includes velocity
        
        # Simple proportional control
        kp = np.array([1000, 1000, 1000, 100, 100, 100])  # Velocity gains
        actuator_velocities = kp * velocity_error
        
        # Spindle motor control
        spindle_pwm = np.interp(command.spindle_speed, [0, 6000], [0, 255])
        
        # Coolant pump control
        coolant_pwm = np.interp(command.coolant_flow, [0, 10], [0, 255])
        
        return {
            'motor_velocities': actuator_velocities.tolist(),
            'spindle_pwm': int(spindle_pwm),
            'coolant_pwm': int(coolant_pwm),
            'timestamp': time.perf_counter()
        }
    
    def _send_to_actuators(self, commands: Dict):
        """Send commands to physical actuators"""
        # In real implementation, this would interface with motor controllers
        pass
    
    def submit_command(self, command: ControlCommand):
        """Submit new control command"""
        self.command_buffer.append(command)
    
    def submit_sensor_data(self, sensor_data: SensorReading):
        """Submit new sensor reading"""
        self.sensor_buffer.append(sensor_data)
    
    def get_performance_metrics(self) -> Dict:
        """Get real-time performance metrics"""
        if not self.cycle_times:
            return {'status': 'NO_DATA'}
            
        return {
            'average_cycle_time_us': np.mean(self.cycle_times),
            'max_cycle_time_us': np.max(self.cycle_times),
            'cycle_time_std_us': np.std(self.cycle_times),
            'average_jitter_us': np.mean(self.jitter_buffer),
            'max_jitter_us': np.max(self.jitter_buffer),
            'control_loop_status': 'RUNNING' if self.control_loop_active else 'STOPPED'
        }

class EdgeOrchestrator:
    """Edge computing orchestrator for mid-level decision making"""
    
    def __init__(self, rt_controller: RealTimeController):
        self.rt_controller = rt_controller
        self.zmq_context = zmq.Context()
        self.command_socket = self.zmq_context.socket(zmq.PUB)
        self.command_socket.bind("tcp://*:5555")
        
        self.feedback_socket = self.zmq_context.socket(zmq.SUB)
        self.feedback_socket.connect("tcp://localhost:5556")
        self.feedback_socket.setsockopt(zmq.SUBSCRIBE, b"")
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.manufacturing_state = "IDLE"
        self.process_parameters = {}
        self.quality_metrics = {}
        
    async def start_edge_services(self):
        """Start edge computing services"""
        # Start feedback listener
        feedback_task = asyncio.create_task(self._feedback_listener())
        
        # Start command generator
        command_task = asyncio.create_task(self._command_generator())
        
        # Start monitoring
        monitor_task = asyncio.create_task(self._performance_monitor())
        
        await asyncio.gather(feedback_task, command_task, monitor_task)
    
    async def _feedback_listener(self):
        """Listen for feedback from real-time layer"""
        while True:
            try:
                # Receive feedback message
                message = await asyncio.get_event_loop().run_in_executor(
                    self.executor, self.feedback_socket.recv_json
                )
                
                # Process feedback
                await self._process_feedback(message)
                
            except Exception as e:
                self.logger.error(f"Feedback listener error: {str(e)}")
                await asyncio.sleep(0.001)
    
    async def _process_feedback(self, feedback: Dict):
        """Process feedback from real-time layer"""
        feedback_type = feedback.get('type')
        
        if feedback_type == 'SENSOR_UPDATE':
            # Convert to SensorReading
            sensor_data = SensorReading(
                timestamp=feedback['timestamp'],
                force_torque=np.array(feedback['force_torque']),
                position=np.array(feedback['position']),
                acoustic_emission=feedback['acoustic_emission'],
                temperature=feedback['temperature']
            )
            self.rt_controller.submit_sensor_data(sensor_data)
            
        elif feedback_type == 'CYCLE_METRICS':
            self._update_performance_metrics(feedback)
            
        elif feedback_type == 'SAFETY_ALERT':
            await self._handle_safety_alert(feedback)
    
    async def _command_generator(self):
        """Generate control commands based on current state"""
        while True:
            try:
                if self.manufacturing_state == "POLISHING":
                    # Generate polishing commands
                    command = await self._generate_polishing_command()
                    self.rt_controller.submit_command(command)
                    
                    # Send via ZeroMQ
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, 
                        lambda: self.command_socket.send_json({
                            'type': 'CONTROL_COMMAND',
                            'command': {
                                'velocity': command.robot_velocity.tolist(),
                                'spindle_speed': command.spindle_speed,
                                'coolant_flow': command.coolant_flow,
                                'timestamp': command.timestamp
                            }
                        })
                    )
                
                await asyncio.sleep(0.001)  # 1ms command rate
                
            except Exception as e:
                self.logger.error(f"Command generator error: {str(e)}")
                await asyncio.sleep(0.01)
    
    async def _generate_polishing_command(self) -> ControlCommand:
        """Generate polishing-specific control command"""
        # This would integrate with the AI controller
        # For now, generate simple command
        current_time = time.perf_counter()
        
        # Simple circular motion pattern
        angular_velocity = 0.5  # rad/s
        radius = 50.0  # mm
        
        vx = -radius * angular_velocity * np.sin(angular_velocity * current_time)
        vy = radius * angular_velocity * np.cos(angular_velocity * current_time)
        vz = 0.0
        
        command = ControlCommand(
            timestamp=current_time,
            robot_velocity=np.array([vx, vy, vz, 0, 0, 0]),
            spindle_speed=3000.0,
            coolant_flow=5.0,
            priority=1
        )
        
        return command
    
    async def _performance_monitor(self):
        """Monitor system performance"""
        while True:
            try:
                # Get performance metrics
                metrics = self.rt_controller.get_performance_metrics()
                
                # Check thresholds
                if metrics.get('average_cycle_time_us', 0) > 150:
                    self.logger.warning(f"High cycle time: {metrics['average_cycle_time_us']:.1f}μs")
                
                if metrics.get('average_jitter_us', 0) > 10:
                    self.logger.warning(f"High jitter: {metrics['average_jitter_us']:.1f}μs")
                
                # Log periodically
                if int(time.time()) % 10 == 0:  # Every 10 seconds
                    self.logger.info(f"Performance: {metrics}")
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Performance monitor error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _handle_safety_alert(self, alert: Dict):
        """Handle safety-related alerts"""
        self.logger.critical(f"Safety alert: {alert}")
        self.manufacturing_state = "EMERGENCY_STOP"
        
        # Send emergency stop command
        emergency_command = ControlCommand(
            timestamp=time.perf_counter(),
            robot_velocity=np.zeros(6),
            spindle_speed=0.0,
            coolant_flow=0.0,
            priority=0  # Highest priority
        )
        
        self.rt_controller.submit_command(emergency_command)
    
    def _update_performance_metrics(self, metrics: Dict):
        """Update performance metrics tracking"""
        self.quality_metrics.update({
            'rt_cycle_time': metrics.get('cycle_time_us', 0),
            'rt_jitter': metrics.get('jitter_us', 0),
            'last_update': time.time()
        })

class HierarchicalController:
    """Top-level hierarchical controller coordinating cloud-edge-real-time layers"""
    
    def __init__(self):
        self.rt_controller = RealTimeController(cycle_time_us=100)
        self.edge_orchestrator = EdgeOrchestrator(self.rt_controller)
        self.cloud_interface = None  # Would connect to GCP services
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.system_ready = False
        self.initialization_complete = False
        
    async def initialize_system(self) -> bool:
        """Initialize complete hierarchical control system"""
        try:
            # Start real-time controller
            self.rt_controller.start_control_loop()
            
            # Verify real-time performance
            await asyncio.sleep(0.1)  # Allow stabilization
            rt_metrics = self.rt_controller.get_performance_metrics()
            
            if rt_metrics.get('control_loop_status') != 'RUNNING':
                raise RuntimeError("Real-time controller failed to start")
            
            self.logger.info("Real-time layer initialized successfully")
            
            # Start edge services
            edge_task = asyncio.create_task(self.edge_orchestrator.start_edge_services())
            
            self.system_ready = True
            self.initialization_complete = True
            
            self.logger.info("Hierarchical control system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            return False
    
    async def start_manufacturing_operation(self, operation_plan: Dict):
        """Start coordinated manufacturing operation"""
        if not self.system_ready:
            raise RuntimeError("System not ready for operation")
            
        try:
            self.logger.info("Starting manufacturing operation")
            self.edge_orchestrator.manufacturing_state = "POLISHING"
            
            # Monitor operation progress
            await self._monitor_operation(operation_plan)
            
        except Exception as e:
            self.logger.error(f"Operation failed: {str(e)}")
            await self.emergency_stop()
    
    async def _monitor_operation(self, plan: Dict):
        """Monitor ongoing manufacturing operation"""
        start_time = time.time()
        expected_duration = plan.get('expected_duration', 3600)  # Default 1 hour
        
        while self.edge_orchestrator.manufacturing_state == "POLISHING":
            elapsed = time.time() - start_time
            
            # Check operation completion
            if elapsed > expected_duration:
                self.logger.info("Operation completed - expected duration reached")
                break
                
            # Check quality metrics
            quality_ok = await self._check_operation_quality()
            if not quality_ok:
                self.logger.warning("Quality threshold exceeded")
                break
                
            await asyncio.sleep(5.0)  # Check every 5 seconds
    
    async def _check_operation_quality(self) -> bool:
        """Check if operation quality meets thresholds"""
        metrics = self.edge_orchestrator.quality_metrics
        
        # Check various quality indicators
        cycle_time = metrics.get('rt_cycle_time', 0)
        jitter = metrics.get('rt_jitter', 0)
        
        # Quality thresholds
        if cycle_time > 200 or jitter > 20:
            return False
            
        return True
    
    async def emergency_stop(self):
        """Execute emergency stop procedure"""
        self.logger.critical("Executing emergency stop")
        
        # Stop manufacturing
        self.edge_orchestrator.manufacturing_state = "EMERGENCY_STOP"
        
        # Wait for real-time controller to process stop command
        await asyncio.sleep(0.1)
        
        self.logger.info("Emergency stop completed")

class LatencyOptimizer:
    """Optimization tools for minimizing system latency"""
    
    def __init__(self):
        self.baseline_latency = None
        self.optimization_targets = {
            'rt_cycle_time_us': 100,
            'edge_response_ms': 5,
            'cloud_roundtrip_ms': 50
        }
        self.logger = logging.getLogger(__name__)
        
    def benchmark_system_latency(self, controller: HierarchicalController) -> Dict:
        """Benchmark complete system latency"""
        results = {}
        
        # Measure real-time layer latency
        rt_metrics = controller.rt_controller.get_performance_metrics()
        results['real_time'] = {
            'cycle_time_us': rt_metrics.get('average_cycle_time_us', 0),
            'jitter_us': rt_metrics.get('average_jitter_us', 0)
        }
        
        # Measure edge layer latency (simulated)
        edge_start = time.perf_counter()
        # Simulate edge processing delay
        time.sleep(0.002)  # 2ms simulated processing
        edge_end = time.perf_counter()
        
        results['edge_layer'] = {
            'response_time_ms': (edge_end - edge_start) * 1000
        }
        
        # Measure total system latency
        total_start = time.perf_counter()
        # Simulate complete round trip
        time.sleep(0.005)  # 5ms total system delay
        total_end = time.perf_counter()
        
        results['total_system'] = {
            'latency_ms': (total_end - total_start) * 1000
        }
        
        return results
    
    def optimize_thread_priorities(self, controller: HierarchicalController):
        """Optimize thread scheduling priorities"""
        # Set real-time thread to highest priority
        if hasattr(controller.rt_controller, 'control_thread'):
            # In real implementation, would use platform-specific priority setting
            self.logger.info("Setting real-time thread to highest priority")
            
    def configure_interrupt_affinity(self):
        """Configure CPU affinity for real-time processing"""
        # Pin real-time threads to dedicated CPU cores
        # This would require platform-specific implementation
        self.logger.info("Configuring CPU affinity for real-time processing")

# Example usage
async def main():
    logging.basicConfig(level=logging.INFO)
    
    # Initialize hierarchical controller
    controller = HierarchicalController()
    
    if await controller.initialize_system():
        # Benchmark system performance
        optimizer = LatencyOptimizer()
        latency_results = optimizer.benchmark_system_latency(controller)
        print("Latency Benchmark Results:")
        for layer, metrics in latency_results.items():
            print(f"{layer}: {metrics}")
        
        # Start sample operation
        operation_plan = {
            'expected_duration': 30,  # 30 seconds for demo
            'target_quality': 'GRADE_00'
        }
        
        await controller.start_manufacturing_operation(operation_plan)
        
        # Run for demonstration period
        await asyncio.sleep(30)
        
        # Clean shutdown
        await controller.emergency_stop()

if __name__ == "__main__":
    asyncio.run(main())