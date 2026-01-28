"""
Main Orchestration System for AI-Driven Metrology Manufacturing
Central coordinator for all system components and workflows
Uses pydantic-settings for configuration management
"""

import asyncio
import logging
import json
import signal
import sys
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
from pydantic_settings import BaseSettings
from pydantic import Field
import threading
import time

# Import system components
from manufacturing_cps import ManufacturingCPS
from utils.audit_trail import (
    log_audit_event,
    AuditEventType
)

class Settings(BaseSettings):
    """Application settings using pydantic-settings"""
    
    # Application settings
    app_name: str = Field(default="ManufacturingAI", description="Name of the application")
    environment: str = Field(default="production", description="Environment (dev/staging/production)")
    log_level: str = Field(default="INFO", description="Logging level")
    asset_id: str = Field(default="GRANITE_BLOCK_DEFAULT", description="Asset ID for the manufacturing system")
    
    # System settings
    max_cycles: int = Field(default=100, description="Maximum number of manufacturing cycles")
    shutdown_timeout: float = Field(default=30.0, description="Timeout for graceful shutdown in seconds")
    heartbeat_interval: float = Field(default=10.0, description="Interval for heartbeat monitoring in seconds")
    
    # MQTT settings
    mqtt_broker_host: str = Field(default="localhost", description="MQTT broker host")
    mqtt_broker_port: int = Field(default=1883, description="MQTT broker port")
    
    # Performance settings
    control_loop_interval: float = Field(default=0.1, description="Interval for control loop in seconds")
    
    class Config:
        env_prefix = 'MANUFACTURING_'  # All environment variables will start with MANUFACTURING_
        case_sensitive = False


@dataclass
class SystemState:
    """Overall system state"""
    status: str  # INITIALIZING, READY, OPERATING, ERROR, SHUTDOWN
    timestamp: str
    active_components: List[str]
    error_messages: List[str]
    performance_metrics: Dict

class ManufacturingOrchestrator:
    """Main system orchestrator"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = self.setup_logging()
        self.system_state = SystemState(
            status="INITIALIZING",
            timestamp=datetime.now().isoformat(),
            active_components=[],
            error_messages=[],
            performance_metrics={}
        )
        
        # Manufacturing CPS system
        self.manufacturing_cps = None
        
        # Shutdown handling
        self.shutdown_requested = False
        self.setup_signal_handlers()
        
        # Heartbeat monitoring
        self.heartbeat_task = None
        
        # Watchdog timer for safety
        self.watchdog_enabled = True
        self.last_keep_alive = time.time()
        self.watchdog_thread = None
        self.watchdog_interval = 5.0  # seconds
        self.watchdog_timeout = 10.0   # seconds
        self.watchdog_active = False
        
        # API server thread (decoupled from hardware)
        self.api_server_thread = None
        
    def setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('manufacturing_system.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        asyncio.create_task(self.graceful_shutdown())
    
    async def initialize_system(self) -> bool:
        """Initialize the manufacturing CPS system"""
        self.logger.info("Initializing Manufacturing Cyber-Physical System")
        
        try:
            # Log system initialization start
            log_audit_event(
                AuditEventType.SYSTEM_STARTUP,
                asset_id="GRANITE_BLOCK_CPS_S/N_001",
                source_component="ManufacturingOrchestrator",
                details={
                    "initialization_attempt": datetime.now().isoformat(),
                    "environment": self.settings.environment
                }
            )
            
            # Initialize the complete CPS
            self.manufacturing_cps = ManufacturingCPS(asset_id="GRANITE_BLOCK_CPS_S/N_001")
            
            # Initialize all CPS components
            await self.manufacturing_cps.initialize_system()
            
            self.system_state.active_components = ["ManufacturingCPS"]
            
            # Final system validation
            if not await self._validate_system_integrity():
                self.system_state.status = "ERROR"
                
                # Log validation failure
                log_audit_event(
                    AuditEventType.SYSTEM_STARTUP,
                    asset_id="GRANITE_BLOCK_CPS_S/N_001",
                    source_component="ManufacturingOrchestrator",
                    severity="ERROR",
                    details={
                        "validation_result": "FAILED",
                        "reason": "system_integrity_check_failed"
                    }
                )
                
                return False
            
            self.system_state.status = "READY"
            self.system_state.timestamp = datetime.now().isoformat()
            
            self.logger.info("Manufacturing CPS initialization completed successfully")
            
            # Log successful initialization
            log_audit_event(
                AuditEventType.SYSTEM_STARTUP,
                asset_id="GRANITE_BLOCK_CPS_S/N_001",
                source_component="ManufacturingOrchestrator",
                severity="INFO",
                details={
                    "result": "SUCCESS",
                    "active_components": self.system_state.active_components,
                    "ready_time": datetime.now().isoformat()
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {str(e)}")
            self.system_state.error_messages.append(str(e))
            self.system_state.status = "ERROR"
            
            # Log initialization failure
            log_audit_event(
                AuditEventType.SYSTEM_STARTUP,
                asset_id="GRANITE_BLOCK_CPS_S/N_001",
                source_component="ManufacturingOrchestrator",
                severity="CRITICAL",
                details={
                    "result": "FAILED",
                    "error": str(e),
                    "traceback": str(type(e).__name__)
                }
            )
            
            return False
    
    # The Manufacturing CPS handles all initialization internally
    # So we don't need separate initialization methods for individual components
    
    async def _validate_system_integrity(self) -> bool:
        """Validate complete system integrity"""
        try:
            # Check if CPS is functioning properly
            if self.manufacturing_cps is None:
                return False
            
            # Perform basic health check
            quality_report = self.manufacturing_cps.get_surface_quality()
            
            self.logger.info("System integrity validation passed")
            return True
            
        except Exception as e:
            self.logger.error(f"System validation failed: {str(e)}")
            return False
    
    async def execute_manufacturing_job(self, job_spec: Dict) -> Dict:
        """Execute complete manufacturing job using CPS"""
        if self.system_state.status != "READY":
            raise RuntimeError(f"System not ready for operation (status: {self.system_state.status})")
        
        self.system_state.status = "OPERATING"
        self.logger.info(f"Starting manufacturing job: {job_spec.get('job_id', 'unknown')}")
        
        # Log job start event
        log_audit_event(
            AuditEventType.JOB_STARTED,
            asset_id="GRANITE_BLOCK_CPS_S/N_001",
            source_component="ManufacturingOrchestrator",
            details={
                'job_id': job_spec.get('job_id'),
                'cycles': job_spec.get('cycles', 10),
                'start_time': datetime.now().isoformat()
            }
        )
        
        try:
            # Execute continuous operation using CPS
            max_cycles = job_spec.get('cycles', 10)
            await self.manufacturing_cps.run_continuous_operation(max_cycles=max_cycles)
            
            # Get final quality report
            final_quality = self.manufacturing_cps.get_surface_quality()
            
            job_results = {
                'job_id': job_spec.get('job_id'),
                'start_time': datetime.now().isoformat(),
                'final_status': 'COMPLETED',
                'end_time': datetime.now().isoformat(),
                'quality_results': {
                    'acceptable': final_quality['is_acceptable'],
                    'score': final_quality['quality_score'],
                    'metrics': final_quality['metrics']
                },
                'telemetry_summary': 'Published to MDE via MQTT'
            }
            
            self.logger.info(f"Manufacturing job completed with status: {job_results['final_status']}")
            
            # Log job completion event
            log_audit_event(
                AuditEventType.JOB_COMPLETED,
                asset_id="GRANITE_BLOCK_CPS_S/N_001",
                source_component="ManufacturingOrchestrator",
                details={
                    'job_id': job_spec.get('job_id'),
                    'final_status': 'COMPLETED',
                    'quality_results': {
                        'acceptable': final_quality['is_acceptable'],
                        'score': final_quality['quality_score']
                    },
                    'end_time': datetime.now().isoformat()
                }
            )
            
            return job_results
            
        except Exception as e:
            self.logger.error(f"Manufacturing job execution failed: {str(e)}")
            self.system_state.status = "ERROR"
            
            # Log job failure event
            log_audit_event(
                AuditEventType.JOB_FAILED,
                asset_id="GRANITE_BLOCK_CPS_S/N_001",
                source_component="ManufacturingOrchestrator",
                severity="ERROR",
                details={
                    'job_id': job_spec.get('job_id'),
                    'error': str(e),
                    'failure_time': datetime.now().isoformat()
                }
            )
            
            raise
        finally:
            if not self.shutdown_requested:
                self.system_state.status = "READY"
    
# The CPS handles all phases internally, so these methods are no longer needed
    
    async def graceful_shutdown(self):
        """Execute graceful system shutdown"""
        self.logger.info("Initiating graceful shutdown...")
        
        # Log shutdown initiation
        log_audit_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            asset_id="GRANITE_BLOCK_CPS_S/N_001",
            source_component="ManufacturingOrchestrator",
            details={
                'shutdown_initiated': datetime.now().isoformat(),
                'reason': 'graceful_shutdown'
            }
        )
        
        # Stop ongoing operations
        self.system_state.status = "SHUTDOWN"
        
        # Shutdown the CPS
        if self.manufacturing_cps:
            self.manufacturing_cps.shutdown()
            self.logger.info("Manufacturing CPS shut down")
        
        self.logger.info("Graceful shutdown completed")
        
        # Stop watchdog thread
        self.stop_watchdog()
        
        # Log completion of shutdown
        log_audit_event(
            AuditEventType.SYSTEM_SHUTDOWN,
            asset_id="GRANITE_BLOCK_CPS_S/N_001",
            source_component="ManufacturingOrchestrator",
            severity="INFO",
            details={
                'shutdown_completed': datetime.now().isoformat(),
                'result': 'completed_successfully'
            }
        )
    
    def start_watchdog(self):
        """Start the watchdog timer thread for safety"""
        if not self.watchdog_enabled or self.watchdog_active:
            return
            
        self.watchdog_active = True
        self.watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self.watchdog_thread.start()
        self.logger.info("Watchdog timer started")
    
    def stop_watchdog(self):
        """Stop the watchdog timer thread"""
        self.watchdog_active = False
        if self.watchdog_thread:
            self.watchdog_thread.join(timeout=1.0)
        self.logger.info("Watchdog timer stopped")
    
    def _watchdog_loop(self):
        """Watchdog timer loop to ensure system responsiveness"""
        while self.watchdog_active:
            current_time = time.time()
            time_since_keep_alive = current_time - self.last_keep_alive
            
            if time_since_keep_alive > self.watchdog_timeout:
                self.logger.critical("Watchdog timeout! No keep-alive received for {:.2f}s".format(time_since_keep_alive))
                # Trigger emergency stop
                self._trigger_emergency_stop()
                break
            
            time.sleep(self.watchdog_interval)
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop procedure"""
        self.logger.critical("Emergency stop triggered by watchdog timer!")
        # Log emergency stop event
        log_audit_event(
            AuditEventType.EMERGENCY_STOP,
            asset_id="GRANITE_BLOCK_CPS_S/N_001",
            source_component="WatchdogTimer",
            severity="CRITICAL",
            details={
                'reason': 'watchdog_timeout',
                'timeout_seconds': self.watchdog_timeout,
                'time_since_keepalive': time.time() - self.last_keep_alive
            }
        )
        # Try to stop manufacturing CPS if available
        if self.manufacturing_cps:
            try:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.manufacturing_cps.emergency_stop())
                loop.close()
            except Exception as e:
                self.logger.error("Failed to trigger emergency stop on CPS: {}".format(e))
    
    def update_keep_alive(self):
        """Update the keep alive timestamp"""
        self.last_keep_alive = time.time()
    
    def start_api_server_in_background(self):
        """Start API server in a separate thread to decouple from hardware operations"""
        try:
            from api.server import start_server
            self.api_server_thread = threading.Thread(target=start_server, daemon=True)
            self.api_server_thread.start()
            self.logger.info("API server started in background thread")
        except ImportError:
            self.logger.warning("API server module not found, skipping background API server start")

# Global system instance
orchestrator: Optional[ManufacturingOrchestrator] = None

async def main():
    """Main entry point"""
    global orchestrator
    
    # Initialize orchestrator
    orchestrator = ManufacturingOrchestrator()
    
    # Initialize system
    if not await orchestrator.initialize_system():
        print("System initialization failed!")
        sys.exit(1)
    
    print("Manufacturing Cyber-Physical System Ready")
    print("System Status:", orchestrator.system_state.status)
    print("Active Components:", orchestrator.system_state.active_components)
    
    # Example job execution
    sample_job = {
        'job_id': 'CPS_SAMPLE_JOB_001',
        'target_quality': 'DIN_876_GRADE_00',
        'cycles': 5,
        'previous_results': {}  # Will be populated during execution
    }
    
    try:
        results = await orchestrator.execute_manufacturing_job(sample_job)
        print("\nJob Results:")
        print(json.dumps(results, indent=2))
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    finally:
        if orchestrator:
            await orchestrator.graceful_shutdown()

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down...")
        if orchestrator:
            asyncio.run(orchestrator.graceful_shutdown())