"""
Advanced Health Monitoring System for ManufacturingAI
Implements continuous heartbeat monitoring and health reporting
"""
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Callable, Optional
from enum import Enum
import time
import threading
from dataclasses import dataclass, field
from .schemas import validate_system_health, SystemHealthSchema


class ComponentStatus(Enum):
    """Enum for component status"""
    INITIALIZING = "INITIALIZING"
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class ComponentInfo:
    """Information about a system component"""
    name: str
    status: ComponentStatus
    last_heartbeat: float
    version: str = "1.0.0"
    health_metrics: Dict = field(default_factory=dict)
    heartbeat_interval: float = 10.0  # seconds between heartbeats


class HealthMonitor:
    """Advanced health monitoring system with continuous heartbeat tracking"""
    
    def __init__(self):
        self.components: Dict[str, ComponentInfo] = {}
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.monitoring_task = None
        self.heartbeat_callbacks: Dict[str, Callable] = {}
        self.stale_component_threshold = 30.0  # seconds before marking as stale
        self.critical_component_threshold = 60.0  # seconds before marking as critical
        self.health_report_listeners: list[Callable] = []
    
    def register_component(self, name: str, version: str = "1.0.0", 
                          heartbeat_interval: float = 10.0) -> bool:
        """Register a component with the health monitor"""
        if name in self.components:
            self.logger.warning(f"Component {name} already registered, updating...")
        
        self.components[name] = ComponentInfo(
            name=name,
            status=ComponentStatus.INITIALIZING,
            last_heartbeat=time.time(),
            version=version,
            heartbeat_interval=heartbeat_interval
        )
        self.logger.info(f"Registered component: {name} (v{version})")
        return True
    
    def unregister_component(self, name: str) -> bool:
        """Unregister a component from monitoring"""
        if name in self.components:
            del self.components[name]
            if name in self.heartbeat_callbacks:
                del self.heartbeat_callbacks[name]
            self.logger.info(f"Unregistered component: {name}")
            return True
        return False
    
    def update_component_status(self, name: str, status: ComponentStatus) -> bool:
        """Update component status"""
        if name in self.components:
            self.components[name].status = status
            self.components[name].last_heartbeat = time.time()
            self.logger.info(f"Updated status for {name}: {status.value}")
            return True
        else:
            self.logger.warning(f"Attempted to update status for unregistered component: {name}")
            return False
    
    def heartbeat(self, name: str) -> bool:
        """Record heartbeat from a component"""
        if name in self.components:
            self.components[name].last_heartbeat = time.time()
            if self.components[name].status != ComponentStatus.ERROR:
                self.components[name].status = ComponentStatus.HEALTHY
            return True
        else:
            self.logger.warning(f"Heartbeat from unregistered component: {name}")
            return False
    
    def register_heartbeat_callback(self, name: str, callback: Callable):
        """Register a callback to be called when a component should send a heartbeat"""
        self.heartbeat_callbacks[name] = callback
    
    def register_health_report_listener(self, listener: Callable[[Dict], None]):
        """Register a listener to receive health reports"""
        self.health_report_listeners.append(listener)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        now = time.time()
        health_report = {
            "timestamp": now,
            "components": {},
            "overall_status": ComponentStatus.HEALTHY.value,
            "summary": {
                "total_components": len(self.components),
                "healthy_components": 0,
                "warning_components": 0,
                "error_components": 0,
                "stale_components": 0
            }
        }
        
        for name, info in self.components.items():
            age = now - info.last_heartbeat
            component_age_status = "FRESH"
            
            if age > self.critical_component_threshold:
                component_age_status = "CRITICAL"
                component_status = ComponentStatus.ERROR.value
            elif age > self.stale_component_threshold:
                component_age_status = "STALE"
                component_status = ComponentStatus.WARNING.value
            else:
                component_status = info.status.value
            
            health_report["components"][name] = {
                "status": component_status,
                "age_status": component_age_status,
                "last_heartbeat": info.last_heartbeat,
                "age_seconds": age,
                "version": info.version,
                "health_metrics": info.health_metrics,
                "heartbeat_interval": info.heartbeat_interval
            }
            
            # Update summary counts
            if component_status == ComponentStatus.ERROR.value:
                health_report["summary"]["error_components"] += 1
            elif component_status == ComponentStatus.WARNING.value:
                health_report["summary"]["warning_components"] += 1
            elif age <= info.heartbeat_interval * 2:  # Consider healthy if recent heartbeat
                health_report["summary"]["healthy_components"] += 1
            else:
                health_report["summary"]["stale_components"] += 1
        
        # Determine overall status based on worst component and counts
        error_count = health_report["summary"]["error_components"]
        warning_count = health_report["summary"]["warning_components"]
        stale_count = health_report["summary"]["stale_components"]
        
        if error_count > 0:
            health_report["overall_status"] = ComponentStatus.ERROR.value
        elif warning_count > 0 or stale_count > len(self.components) * 0.5:  # More than half stale
            health_report["overall_status"] = ComponentStatus.WARNING.value
        
        # Validate the health report using schema
        try:
            validated_report = validate_system_health(health_report)
            return validated_report.dict()
        except Exception as e:
            self.logger.error(f"Health report validation failed: {e}")
            return health_report  # Return unvalidated report as fallback
    
    async def start_monitoring(self, report_interval: float = 10.0):
        """Start continuous monitoring loop"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.logger.info("Starting health monitoring...")
        
        while self.monitoring_active:
            try:
                # Send heartbeat requests to registered callbacks
                for name, callback in self.heartbeat_callbacks.items():
                    if name in self.components:
                        age = time.time() - self.components[name].last_heartbeat
                        # Request heartbeat if it's time and component is expected to be alive
                        if age >= self.components[name].heartbeat_interval and self.components[name].status != ComponentStatus.SHUTDOWN:
                            try:
                                # Execute heartbeat callback (non-blocking if it's async)
                                if asyncio.iscoroutinefunction(callback):
                                    await callback()
                                else:
                                    callback()
                            except Exception as e:
                                self.logger.error(f"Heartbeat callback for {name} failed: {e}")
                
                # Get and broadcast health report
                health_report = self.get_system_health()
                
                # Notify listeners
                for listener in self.health_report_listeners:
                    try:
                        listener(health_report)
                    except Exception as e:
                        self.logger.error(f"Health report listener failed: {e}")
                
                # Log overall status periodically
                self.logger.info(f"System health: {health_report['overall_status']}, "
                               f"Healthy: {health_report['summary']['healthy_components']}, "
                               f"Warning: {health_report['summary']['warning_components']}, "
                               f"Error: {health_report['summary']['error_components']}")
                
                await asyncio.sleep(report_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1)  # Brief pause before continuing
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.logger.info("Stopping health monitoring...")
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
        self.logger.info("Health monitoring stopped")


class ComponentHeartbeatManager:
    """Helper class to manage heartbeats for a single component"""
    
    def __init__(self, health_monitor: HealthMonitor, component_name: str, 
                 heartbeat_interval: float = 10.0):
        self.health_monitor = health_monitor
        self.component_name = component_name
        self.heartbeat_interval = heartbeat_interval
        self.active = False
        self.task = None
    
    async def start_heartbeating(self):
        """Start sending periodic heartbeats"""
        if self.active:
            return
        
        self.active = True
        self.health_monitor.register_component(
            self.component_name, 
            heartbeat_interval=self.heartbeat_interval
        )
        
        while self.active:
            try:
                self.health_monitor.heartbeat(self.component_name)
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                self.health_monitor.logger.error(f"Error in heartbeat for {self.component_name}: {e}")
                await asyncio.sleep(min(1.0, self.heartbeat_interval))  # Brief pause on error
    
    def stop_heartbeating(self):
        """Stop sending heartbeats"""
        self.active = False
        self.health_monitor.unregister_component(self.component_name)


# Global health monitor instance
global_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    return global_health_monitor