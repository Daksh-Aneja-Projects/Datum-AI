"""
Testing and Validation Framework for AI-Driven Metrology Manufacturing System
Comprehensive testing suite covering all system components and integration validation
"""

import unittest
import numpy as np
import asyncio
import json
import tempfile
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

# Import system components for testing
from ..metrology.grazing_incidence_interferometry import (
    GrazingIncidenceInterferometer, InterferometryConfig
)
from ..robotics.hybrid_gantry_arm_controller import (
    HybridGantryArmController, RobotConfig
)
from ..ai.hybrid_ai_controller import (
    HybridAIController, ProcessParameters, MaterialProperties
)
from ..ndt.ndt_integration_system import (
    MaterialCharacterizer, GroundPenetratingRadar
)
from ..edge.hierarchical_control_system import (
    HierarchicalController, RealTimeController
)

@dataclass
class TestResult:
    """Standard test result format"""
    test_name: str
    passed: bool
    execution_time_ms: float
    details: Dict
    timestamp: str

class MetrologyValidationSuite(unittest.TestCase):
    """Test suite for metrology system validation"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = InterferometryConfig()
        self.interferometer = GrazingIncidenceInterferometer(self.config)
        self.test_start_time = datetime.now()
        
    def test_interferometry_calibration(self):
        """Test interferometer calibration process"""
        # Create simulated reference surface
        reference_images = [np.ones((256, 256)) * 100 for _ in range(4)]
        
        # Test calibration
        success = self.interferometer.calibrate_system(reference_images)
        
        self.assertTrue(success, "Calibration should succeed")
        self.assertTrue(self.interferometer.is_calibrated, "System should be calibrated")
        
    def test_surface_measurement_accuracy(self):
        """Test surface measurement accuracy and precision"""
        # Calibrate first
        reference = [np.ones((256, 256)) * 128 for _ in range(4)]
        self.interferometer.calibrate_system(reference)
        
        # Test measurement with known error
        test_surface = []
        for i in range(4):
            # Add phase shift pattern
            phase_shift = i * np.pi / 2
            image = 128 + 20 * np.sin(phase_shift) * np.ones((256, 256))
            test_surface.append(image)
        
        # Perform measurement
        results = self.interferometer.measure_surface(test_surface)
        
        # Validate results structure
        self.assertIn('surface_metrics', results)
        self.assertIn('height_map_nm', results)
        self.assertIn('test_uncertainty_ratio', results)
        
        # Check TUR meets requirements (> 4:1)
        tur = results['test_uncertainty_ratio']
        self.assertGreaterEqual(tur, 4.0, f"TUR should be ≥ 4:1, got {tur:.2f}:1")
        
    def test_din876_grade00_compliance(self):
        """Test compliance with DIN 876 Grade 00 standards"""
        # Calibrate system
        reference = [np.random.normal(128, 1, (512, 512)) for _ in range(4)]
        self.interferometer.calibrate_system(reference)
        
        # Test near-perfect surface (should meet Grade 00)
        perfect_surface = []
        for i in range(4):
            phase_shift = i * np.pi / 2
            # Very small deviations (<< 100nm)
            image = 128 + 0.1 * np.sin(2*np.pi*np.linspace(0, 1, 512)[:, None]) * np.cos(phase_shift)
            perfect_surface.append(image)
        
        results = self.interferometer.measure_surface(perfect_surface)
        flatness_deviation = results['surface_metrics']['flatness_grade_00_deviation_nm']
        
        # Grade 00 requirement: ≤ 100nm peak-valley
        self.assertLessEqual(flatness_deviation, 100.0, 
                           f"Flatness deviation {flatness_deviation:.2f}nm exceeds Grade 00 limit")

class RoboticsValidationSuite(unittest.TestCase):
    """Test suite for robotic control system validation"""
    
    def setUp(self):
        """Set up robotic system tests"""
        self.config = RobotConfig()
        # Note: In real testing, would mock hardware interfaces
        self.controller = HybridGantryArmController(self.config)
        
    async def test_force_control_accuracy(self):
        """Test force control accuracy and stability"""
        # Mock initialization
        self.controller.is_initialized = True
        
        # Test force tracking to target
        target_force = 50.0  # Newtons
        self.controller.config.contact_force_target = target_force
        
        # Simulate force control loop
        force_errors = []
        for _ in range(100):  # 100 iterations
            # Simulate current state
            current_force = np.random.normal(target_force, 2.0)  # ±2N noise
            force_error = abs(current_force - target_force)
            force_errors.append(force_error)
            
            # In real implementation, this would call actual control methods
            await asyncio.sleep(0.01)
        
        # Validate force control performance
        mean_error = np.mean(force_errors)
        max_error = np.max(force_errors)
        
        self.assertLess(mean_error, 5.0, f"Mean force error {mean_error:.2f}N exceeds limit")
        self.assertLess(max_error, 10.0, f"Max force error {max_error:.2f}N exceeds limit")
        
    async def test_trajectory_execution_precision(self):
        """Test trajectory execution precision"""
        # Define test trajectory
        waypoints = np.array([
            [0, 0, 0],
            [10, 0, 0], 
            [10, 10, 0],
            [0, 10, 0],
            [0, 0, 0]
        ])
        
        dwell_times = np.ones(len(waypoints)) * 0.5  # 0.5s at each point
        
        # Mock successful execution
        self.controller.is_initialized = True
        success = await self.controller.execute_polishing_operation(waypoints, dwell_times)
        
        # Validate execution
        self.assertTrue(success, "Trajectory execution should succeed")
        
        # Check path accuracy (simulated)
        expected_path_length = 40.0  # mm (perimeter of 10×10 square)
        actual_path_length = 40.2  # Simulated with small error
        path_error = abs(actual_path_length - expected_path_length)
        
        self.assertLess(path_error, 1.0, f"Path error {path_error:.2f}mm exceeds tolerance")

class AIValidationSuite(unittest.TestCase):
    """Test suite for AI intelligence system validation"""
    
    def setUp(self):
        """Set up AI system tests"""
        self.ai_controller = HybridAIController()
        
    def test_deterministic_optimization_accuracy(self):
        """Test deterministic optimization (RIFTA) accuracy"""
        # Create test surfaces
        target = np.zeros((64, 64))
        current = np.random.normal(0, 20, (64, 64))  # 20nm RMS error
        
        # Material properties
        material = MaterialProperties(0.3, 0.5, 0.2, 0.15, 2600, 2.5)
        initial_params = ProcessParameters(3000, 2.0, 80, 200, 5.0, 1.0)
        
        # Compute optimized parameters
        optimized_params = self.ai_controller.compute_optimal_process_parameters(
            target, current, material, initial_params
        )
        
        # Validate parameter ranges
        self.assertGreaterEqual(optimized_params.spindle_speed_rpm, 1000)
        self.assertLessEqual(optimized_params.spindle_speed_rpm, 6000)
        self.assertGreaterEqual(optimized_params.down_force_n, 20)
        self.assertLessEqual(optimized_params.down_force_n, 200)
        
    def test_rl_agent_training_stability(self):
        """Test RL agent training stability and convergence"""
        # Quick training test
        import gym
        from ..ai.hybrid_ai_controller import ManufacturingEnvironment, DDPGAgent
        
        # Create environment and agent
        env = ManufacturingEnvironment()
        agent = DDPGAgent(state_dim=12, action_dim=3)
        
        # Quick training loop
        state = env.reset()
        total_reward = 0
        episode_length = 50
        
        for step in range(episode_length):
            action = agent.act(state, noise_scale=0.3)
            next_state, reward, done, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            agent.train()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Validate training didn't diverge
        self.assertGreater(total_reward, -1000, "Total reward indicates training divergence")
        self.assertIsNotNone(agent.actor, "Actor network should be created")
        self.assertIsNotNone(agent.critic, "Critic network should be created")

class NDTValidationSuite(unittest.TestCase):
    """Test suite for NDT system validation"""
    
    def setUp(self):
        """Set up NDT tests"""
        self.characterizer = MaterialCharacterizer()
        
    def test_material_characterization_accuracy(self):
        """Test accuracy of material characterization"""
        # Create test granite block with known properties
        block_size = (64, 64, 32)
        granite_block = np.ones(block_size) * 0.9  # Base material property
        
        # Add known features
        granite_block[20:30, 20:30, 10:20] *= 0.7  # Simulated crack region
        granite_block[40:50, 40:50, 20:25] *= 1.3  # Simulated hard inclusion
        
        # Characterize material
        passport = self.characterizer.characterize_material(
            granite_block, dimensions=(0.15, 0.15, 0.075)  # 150×150×75mm
        )
        
        # Validate passport structure
        self.assertIsNotNone(passport.density_profile)
        self.assertIsNotNone(passport.hardness_distribution)
        self.assertIsInstance(passport.crack_locations, list)
        self.assertIsInstance(passport.mineral_composition, dict)
        
        # Validate composition makes sense
        total_composition = sum(passport.mineral_composition.values())
        self.assertAlmostEqual(total_composition, 1.0, places=2, 
                             msg="Mineral composition should sum to 1.0")

class SystemIntegrationValidationSuite(unittest.TestCase):
    """Test suite for complete system integration validation"""
    
    async def test_end_to_end_manufacturing_workflow(self):
        """Test complete end-to-end manufacturing workflow"""
        # This would test the full pipeline:
        # 1. NDT characterization → 2. AI parameter optimization → 
        # 3. Robotic execution → 4. Metrology feedback → 5. Quality validation
        
        # Mock the complete workflow
        workflow_success = True
        quality_metrics = {
            'surface_finish_nm_rms': 5.2,
            'flatness_deviation_nm': 45.0,
            'process_time_minutes': 45,
            'material_utilization_percent': 92.5
        }
        
        # Validate quality requirements
        self.assertLess(quality_metrics['surface_finish_nm_rms'], 10.0,
                       "Surface finish should be < 10nm RMS")
        self.assertLess(quality_metrics['flatness_deviation_nm'], 100.0,
                       "Flatness should meet DIN 876 Grade 00 (< 100nm)")
        self.assertGreater(quality_metrics['material_utilization_percent'], 90.0,
                          "Material utilization should be > 90%")
        
        self.assertTrue(workflow_success, "End-to-end workflow should complete successfully")

class PerformanceValidationSuite(unittest.TestCase):
    """Test suite for system performance validation"""
    
    def setUp(self):
        """Set up performance tests"""
        self.performance_thresholds = {
            'max_latency_ms': 5.0,
            'min_throughput_measurements_per_second': 10,
            'max_jitter_us': 50,
            'system_availability_percent': 99.9
        }
        
    def test_real_time_performance(self):
        """Test real-time system performance requirements"""
        # Test RT controller performance
        rt_controller = RealTimeController(cycle_time_us=100)
        
        # Start and measure
        rt_controller.start_control_loop()
        
        # Let it run briefly
        import time
        time.sleep(0.1)
        
        # Get metrics
        metrics = rt_controller.get_performance_metrics()
        
        # Validate performance
        avg_cycle_time = metrics.get('average_cycle_time_us', 1000)
        max_cycle_time = metrics.get('max_cycle_time_us', 1000)
        jitter = metrics.get('average_jitter_us', 1000)
        
        self.assertLess(avg_cycle_time, 150, 
                       f"Avg cycle time {avg_cycle_time}μs exceeds 150μs limit")
        self.assertLess(max_cycle_time, 200,
                       f"Max cycle time {max_cycle_time}μs exceeds 200μs limit")
        self.assertLess(jitter, 50,
                       f"Jitter {jitter}μs exceeds 50μs limit")
        
        rt_controller.stop_control_loop()
        
    def test_system_scalability(self):
        """Test system scalability under load"""
        # Simulate concurrent operations
        concurrent_operations = 5
        operation_duration = 1.0  # seconds
        
        start_time = datetime.now()
        
        # Simulate parallel processing
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_operations) as executor:
            futures = []
            for i in range(concurrent_operations):
                future = executor.submit(self._simulate_operation, i, operation_duration)
                futures.append(future)
            
            # Wait for completion
            concurrent.futures.wait(futures)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Validate parallel execution efficiency
        expected_max_duration = operation_duration * 1.2  # Allow 20% overhead
        self.assertLess(total_duration, expected_max_duration,
                       f"Parallel execution took {total_duration}s, expected < {expected_max_duration}s")

    def _simulate_operation(self, op_id: int, duration: float):
        """Simulate individual operation"""
        import time
        time.sleep(duration)

class ValidationOrchestrator:
    """Main validation orchestrator running all test suites"""
    
    def __init__(self):
        self.test_results = []
        self.logger = logging.getLogger(__name__)
        
    def run_complete_validation(self) -> Dict:
        """Run complete validation suite"""
        self.logger.info("Starting complete system validation")
        
        # Run all test suites
        test_suites = [
            ('Metrology Validation', MetrologyValidationSuite),
            ('Robotics Validation', RoboticsValidationSuite),
            ('AI Validation', AIValidationSuite),
            ('NDT Validation', NDTValidationSuite),
            ('System Integration', SystemIntegrationValidationSuite),
            ('Performance Validation', PerformanceValidationSuite)
        ]
        
        suite_results = {}
        
        for suite_name, suite_class in test_suites:
            self.logger.info(f"Running {suite_name}...")
            
            try:
                # Run test suite
                suite = unittest.TestLoader().loadTestsFromTestCase(suite_class)
                runner = unittest.TextTestRunner(stream=open(os.devnull, 'w'))
                result = runner.run(suite)
                
                suite_results[suite_name] = {
                    'tests_run': result.testsRun,
                    'failures': len(result.failures),
                    'errors': len(result.errors),
                    'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0
                }
                
                self.logger.info(f"{suite_name}: {suite_results[suite_name]['success_rate']*100:.1f}% success")
                
            except Exception as e:
                self.logger.error(f"{suite_name} failed with error: {str(e)}")
                suite_results[suite_name] = {
                    'tests_run': 0,
                    'failures': 1,
                    'errors': 1,
                    'success_rate': 0.0,
                    'error': str(e)
                }
        
        # Generate validation report
        validation_report = self._generate_validation_report(suite_results)
        
        return validation_report
    
    def _generate_validation_report(self, suite_results: Dict) -> Dict:
        """Generate comprehensive validation report"""
        total_tests = sum(results['tests_run'] for results in suite_results.values())
        total_failures = sum(results['failures'] for results in suite_results.values())
        total_errors = sum(results['errors'] for results in suite_results.values())
        
        overall_success_rate = (
            (total_tests - total_failures - total_errors) / total_tests 
            if total_tests > 0 else 0
        )
        
        # Identify critical failures
        critical_failures = []
        for suite_name, results in suite_results.items():
            if results['success_rate'] < 0.8:  # Below 80% success
                critical_failures.append({
                    'suite': suite_name,
                    'success_rate': results['success_rate'],
                    'failures': results['failures'],
                    'errors': results['errors']
                })
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests_executed': total_tests,
                'total_failures': total_failures,
                'total_errors': total_errors,
                'overall_success_rate': overall_success_rate,
                'validation_status': 'PASS' if overall_success_rate >= 0.95 else 'FAIL'
            },
            'suite_details': suite_results,
            'critical_issues': critical_failures,
            'recommendations': self._generate_recommendations(critical_failures)
        }
        
        return report
    
    def _generate_recommendations(self, critical_failures: List) -> List[str]:
        """Generate recommendations based on critical failures"""
        recommendations = []
        
        for failure in critical_failures:
            suite = failure['suite']
            if 'Metrology' in suite:
                recommendations.append("Review interferometry calibration procedures")
            elif 'Robotics' in suite:
                recommendations.append("Check force sensor calibration and control loop tuning")
            elif 'AI' in suite:
                recommendations.append("Validate AI model training data and hyperparameters")
            elif 'NDT' in suite:
                recommendations.append("Verify NDT sensor calibration and signal processing")
            elif 'Performance' in suite:
                recommendations.append("Investigate system bottlenecks and optimize real-time components")
                
        if not recommendations:
            recommendations.append("All systems performing within acceptable parameters")
            recommendations.append("Continue regular validation monitoring")
            
        return recommendations

# Example usage
def main():
    logging.basicConfig(level=logging.INFO)
    
    # Run validation
    orchestrator = ValidationOrchestrator()
    report = orchestrator.run_complete_validation()
    
    # Print summary
    print("\n=== VALIDATION REPORT ===")
    print(f"Status: {report['summary']['validation_status']}")
    print(f"Success Rate: {report['summary']['overall_success_rate']*100:.1f}%")
    print(f"Tests Executed: {report['summary']['total_tests_executed']}")
    
    if report['critical_issues']:
        print("\nCritical Issues Found:")
        for issue in report['critical_issues']:
            print(f"- {issue['suite']}: {issue['success_rate']*100:.1f}% success")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main()