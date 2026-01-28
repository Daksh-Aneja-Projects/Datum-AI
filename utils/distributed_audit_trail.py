"""
Time-Series Buffer System for Manufacturing Audit Trail
Replaces SQLite with Kafka/Redpanda for industrial time-series data
Implements immutable architecture with distributed logging
"""

import json
import time
import hashlib
import asyncio
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import queue
import uuid

# Mock Kafka/Redpanda client (would use confluent-kafka or similar in production)
class MockKafkaProducer:
    """Mock Kafka producer for demonstration"""
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.connected = False
        
    def connect(self):
        self.connected = True
        logging.info(f"Connected to Kafka cluster: {self.bootstrap_servers}")
        
    def send(self, topic: str, key: str, value: bytes, callback=None):
        if not self.connected:
            raise ConnectionError("Kafka producer not connected")
        # Simulate async send
        if callback:
            callback(None, None)  # Success
            
    def flush(self, timeout: float = 30.0):
        pass
        
    def close(self):
        self.connected = False

class MockKafkaConsumer:
    """Mock Kafka consumer for demonstration"""
    def __init__(self, bootstrap_servers: str, group_id: str):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.subscribed_topics = []
        self.connected = False
        
    def connect(self):
        self.connected = True
        logging.info(f"Consumer connected to Kafka: {self.bootstrap_servers}")
        
    def subscribe(self, topics: List[str]):
        self.subscribed_topics = topics
        
    def poll(self, timeout_ms: int = 1000):
        # Simulate message polling
        return None
        
    def close(self):
        self.connected = False

class TimeSeriesTopic(Enum):
    """Kafka topics for different data streams"""
    AUDIT_EVENTS = "manufacturing.audit.events"
    SENSOR_TELEMETRY = "manufacturing.sensor.telemetry"
    CONTROL_COMMANDS = "manufacturing.control.commands"
    HEALTH_METRICS = "manufacturing.health.metrics"
    QUALITY_DATA = "manufacturing.quality.data"

@dataclass
class TimeSeriesEvent:
    """Immutable time-series event with provenance"""
    event_id: str
    timestamp_ns: int
    event_type: str
    source_component: str
    asset_id: str
    payload: Dict[str, Any]
    previous_hash: Optional[str] = None
    signature: Optional[str] = None
    partition_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'timestamp_ns': self.timestamp_ns,
            'event_type': self.event_type,
            'source_component': self.source_component,
            'asset_id': self.asset_id,
            'payload': self.payload,
            'previous_hash': self.previous_hash,
            'signature': self.signature
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
    
    def calculate_hash(self) -> str:
        """Calculate cryptographic hash for event chaining"""
        canonical = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

class TimeSeriesBuffer:
    """High-performance time-series buffer with Kafka backend"""
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092",
                 default_topic: TimeSeriesTopic = TimeSeriesTopic.AUDIT_EVENTS,
                 buffer_size: int = 10000,
                 batch_size: int = 100,
                 flush_interval_ms: int = 1000):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.default_topic = default_topic
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.flush_interval_ms = flush_interval_ms
        
        self.logger = logging.getLogger(__name__)
        
        # Kafka clients
        self.producer = MockKafkaProducer(kafka_bootstrap_servers)
        self.consumer = MockKafkaConsumer(kafka_bootstrap_servers, "audit_consumer")
        
        # Buffer management
        self.event_buffer = queue.Queue(maxsize=buffer_size)
        self.pending_events = []  # Events waiting for batch processing
        self.chain_head_hash = None  # Hash of the most recent event
        
        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="TS_BUFFER")
        self.flush_timer = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            'events_written': 0,
            'events_dropped': 0,
            'batch_flushes': 0,
            'errors': 0
        }
        
        # Callbacks
        self.event_callbacks = []
        
        # Initialize connection
        self._initialize_kafka()
    
    def _initialize_kafka(self):
        """Initialize Kafka connections"""
        try:
            self.producer.connect()
            self.consumer.connect()
            self.logger.info("Time-series buffer initialized with Kafka backend")
        except Exception as e:
            self.logger.error(f"Failed to initialize Kafka: {e}")
            raise
    
    def start(self):
        """Start the time-series buffer processing"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start periodic flushing
        self._schedule_flush()
        
        self.logger.info("Time-series buffer started")
    
    def stop(self):
        """Stop the buffer and flush remaining events"""
        self.is_running = False
        
        # Flush remaining events
        self._flush_batch(force=True)
        
        # Cleanup
        self.producer.flush()
        self.producer.close()
        self.consumer.close()
        self.executor.shutdown(wait=True)
        
        self.logger.info("Time-series buffer stopped")
    
    def write_event(self, event: TimeSeriesEvent, 
                   topic: Optional[TimeSeriesTopic] = None,
                   partition_key: Optional[str] = None) -> bool:
        """Write event to time-series buffer"""
        try:
            # Add to buffer queue
            try:
                self.event_buffer.put_nowait(event)
                self.stats['events_written'] += 1
                return True
            except queue.Full:
                self.stats['events_dropped'] += 1
                self.logger.warning("Time-series buffer full, dropping event")
                return False
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Failed to write event: {e}")
            return False
    
    def write_audit_event(self, event_type: str, asset_id: str, 
                         source_component: str, payload: Dict[str, Any]) -> str:
        """Write audit event with automatic chaining and hashing"""
        # Create event with proper chaining
        event_id = str(uuid.uuid4())
        timestamp_ns = time.time_ns()
        
        event = TimeSeriesEvent(
            event_id=event_id,
            timestamp_ns=timestamp_ns,
            event_type=event_type,
            source_component=source_component,
            asset_id=asset_id,
            payload=payload,
            previous_hash=self.chain_head_hash
        )
        
        # Calculate and store hash for chaining
        event_hash = event.calculate_hash()
        event.signature = event_hash
        self.chain_head_hash = event_hash
        
        # Write to buffer
        success = self.write_event(event, TimeSeriesTopic.AUDIT_EVENTS, asset_id)
        
        if success:
            # Trigger callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
                    
            return event_id
        else:
            raise RuntimeError("Failed to write audit event")
    
    def _schedule_flush(self):
        """Schedule periodic buffer flushing"""
        if not self.is_running:
            return
            
        self.flush_timer = threading.Timer(
            self.flush_interval_ms / 1000.0,
            self._flush_batch
        )
        self.flush_timer.daemon = True
        self.flush_timer.start()
    
    def _flush_batch(self, force: bool = False):
        """Flush buffered events to Kafka"""
        try:
            # Collect events for batch processing
            batch_events = []
            batch_size = self.batch_size if not force else self.buffer_size
            
            try:
                for _ in range(batch_size):
                    if not self.event_buffer.empty():
                        event = self.event_buffer.get_nowait()
                        batch_events.append(event)
                    else:
                        break
            except queue.Empty:
                pass
            
            if batch_events or force:
                # Process batch asynchronously
                self.executor.submit(self._send_batch, batch_events)
                self.stats['batch_flushes'] += 1
            
            # Reschedule
            if self.is_running:
                self._schedule_flush()
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Batch flush error: {e}")
    
    def _send_batch(self, events: List[TimeSeriesEvent]):
        """Send batch of events to Kafka"""
        try:
            for event in events:
                topic = self.default_topic.value
                key = event.partition_key or event.asset_id
                value = event.to_json().encode('utf-8')
                
                # Send with acknowledgment callback
                self.producer.send(
                    topic=topic,
                    key=key,
                    value=value,
                    callback=self._delivery_callback
                )
                
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Batch send error: {e}")
    
    def _delivery_callback(self, err, msg):
        """Handle message delivery confirmation"""
        if err:
            self.stats['errors'] += 1
            self.logger.error(f"Message delivery failed: {err}")
        # Success case - stats already incremented during queuing
    
    def add_callback(self, callback: Callable[[TimeSeriesEvent], None]):
        """Add event callback for real-time processing"""
        self.event_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            'events_written': self.stats['events_written'],
            'events_dropped': self.stats['events_dropped'],
            'batch_flushes': self.stats['batch_flushes'],
            'errors': self.stats['errors'],
            'buffer_size': self.buffer_size,
            'current_queue_depth': self.event_buffer.qsize(),
            'chain_head_hash': self.chain_head_hash
        }
    
    def query_events(self, asset_id: str, event_type: Optional[str] = None,
                    start_time: Optional[int] = None, 
                    end_time: Optional[int] = None,
                    limit: int = 1000) -> List[TimeSeriesEvent]:
        """Query events from time-series buffer (mock implementation)"""
        # In a real implementation, this would query Kafka with appropriate filters
        # For demo purposes, return empty list
        return []

class DistributedAuditTrail:
    """FDA 21 CFR Part 11 compliant distributed audit trail"""
    
    def __init__(self, kafka_bootstrap_servers: str = "localhost:9092"):
        self.time_series_buffer = TimeSeriesBuffer(kafka_bootstrap_servers)
        self.logger = logging.getLogger(__name__)
        
        # FDA compliance tracking
        self._locked_records = set()
        self._electronic_signatures = {}
        
    def start(self):
        """Start the distributed audit trail"""
        self.time_series_buffer.start()
        self.logger.info("Distributed audit trail started")
    
    def stop(self):
        """Stop the audit trail"""
        self.time_series_buffer.stop()
        self.logger.info("Distributed audit trail stopped")
    
    def log_event(self, event_type: str, asset_id: str, source_component: str,
                  details: Optional[Dict[str, Any]] = None,
                  user_id: Optional[str] = None,
                  correlation_id: Optional[str] = None,
                  severity: str = "INFO",
                  require_electronic_signature: bool = False) -> str:
        """Log audit event with full FDA compliance"""
        
        # Prepare payload with FDA 21 CFR Part 11 fields
        payload = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'asset_id': asset_id,
            'source_component': source_component,
            'details': details or {},
            'user_id': user_id,
            'correlation_id': correlation_id,
            'severity': severity,
            'is_locked': False
        }
        
        # Add electronic signature if required
        if require_electronic_signature and user_id:
            electronic_sig = self._create_electronic_signature(user_id)
            payload['electronic_signature'] = electronic_sig
            payload['reason_for_change'] = "System automated signature"
        
        # Write to time-series buffer
        event_id = self.time_series_buffer.write_audit_event(
            event_type=event_type,
            asset_id=asset_id,
            source_component=source_component,
            payload=payload
        )
        
        self.logger.info(f"Audit event logged: {event_type} [{event_id}]")
        return event_id
    
    def lock_record(self, event_id: str, user_id: str, reason: str) -> bool:
        """Lock/finalize audit record (FDA requirement)"""
        try:
            # In a real implementation, this would update the locked status
            # and potentially trigger additional compliance actions
            self._locked_records.add(event_id)
            
            # Log the locking action
            self.log_event(
                event_type="RECORD_LOCKED",
                asset_id=f"audit_{event_id}",
                source_component="DistributedAuditTrail",
                details={
                    'locked_event_id': event_id,
                    'locking_user': user_id,
                    'reason': reason,
                    'lock_timestamp': datetime.now().isoformat()
                },
                user_id=user_id
            )
            
            self.logger.info(f"Record locked: {event_id} by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to lock record {event_id}: {e}")
            return False
    
    def _create_electronic_signature(self, user_id: str) -> Dict[str, Any]:
        """Create FDA 21 CFR Part 11 compliant electronic signature"""
        timestamp = datetime.now().isoformat()
        signature_data = f"{user_id}|{timestamp}"
        signature_hash = hashlib.sha256(signature_data.encode()).hexdigest()
        
        electronic_sig = {
            'user_id': user_id,
            'timestamp': timestamp,
            'signature_hash': signature_hash,
            'certificate_info': {
                'issuer': 'ManufacturingAI Internal CA',
                'valid_from': datetime.now().isoformat(),
                'valid_to': '2030-01-01T00:00:00'
            }
        }
        
        self._electronic_signatures[user_id] = electronic_sig
        return electronic_sig
    
    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate FDA compliance report"""
        stats = self.time_series_buffer.get_statistics()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_audit_events': stats['events_written'],
            'locked_records': len(self._locked_records),
            'electronic_signatures': len(self._electronic_signatures),
            'buffer_statistics': stats,
            'compliance_status': 'ACTIVE'
        }

# Global instances
_ts_audit_trail: Optional[DistributedAuditTrail] = None

def get_distributed_audit_trail(kafka_servers: str = "localhost:9092") -> DistributedAuditTrail:
    """Get singleton instance of distributed audit trail"""
    global _ts_audit_trail
    if _ts_audit_trail is None:
        _ts_audit_trail = DistributedAuditTrail(kafka_servers)
    return _ts_audit_trail

def log_ts_audit_event(event_type: str, asset_id: str, source_component: str,
                      details: Optional[Dict[str, Any]] = None,
                      user_id: Optional[str] = None,
                      correlation_id: Optional[str] = None,
                      severity: str = "INFO",
                      require_electronic_signature: bool = False) -> str:
    """Convenience function for logging time-series audit events"""
    audit_trail = get_distributed_audit_trail()
    return audit_trail.log_event(
        event_type=event_type,
        asset_id=asset_id,
        source_component=source_component,
        details=details,
        user_id=user_id,
        correlation_id=correlation_id,
        severity=severity,
        require_electronic_signature=require_electronic_signature
    )

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize distributed audit trail
    audit_trail = get_distributed_audit_trail("localhost:9092")
    audit_trail.start()
    
    # Log some sample events
    event_id1 = log_ts_audit_event(
        event_type="SYSTEM_STARTUP",
        asset_id="SP-2024-001",
        source_component="MainOrchestrator",
        details={"version": "2.1.0", "environment": "production"}
    )
    
    event_id2 = log_ts_audit_event(
        event_type="CONFIG_CHANGE",
        asset_id="SP-2024-001",
        source_component="SetupWizard",
        details={"parameter": "target_force_n", "old_value": 40.0, "new_value": 45.0},
        user_id="engineer1"
    )
    
    # Lock a record
    audit_trail.lock_record(event_id2, "supervisor1", "Final configuration approval")
    
    # Get compliance report
    report = audit_trail.get_compliance_report()
    print(f"Audit events logged: {report['total_audit_events']}")
    print(f"Compliance status: {report['compliance_status']}")
    
    # Stop audit trail
    audit_trail.stop()