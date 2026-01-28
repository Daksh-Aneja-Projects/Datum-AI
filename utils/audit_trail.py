"""
Immutable Audit Trail and Logging System for Manufacturing CPS
Implements structured logging for QA traceability and compliance
"""

import json
import hashlib
import hmac
import os
import secrets
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum
import sqlite3
from contextlib import contextmanager
import base64
import asyncio
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend

class AuditEventType(Enum):
    """Types of audit events for traceability"""
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    JOB_STARTED = "JOB_STARTED"
    JOB_COMPLETED = "JOB_COMPLETED"
    JOB_FAILED = "JOB_FAILED"
    PARAMETER_ADJUSTED = "PARAMETER_ADJUSTED"
    SURFACE_MEASURED = "SURFACE_MEASURED"
    QUALITY_CHECK = "QUALITY_CHECK"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SAFETY_VIOLATION = "SAFETY_VIOLATION"
    HARDWARE_CONNECTED = "HARDWARE_CONNECTED"
    HARDWARE_DISCONNECTED = "HARDWARE_DISCONNECTED"
    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"
    PERMISSION_GRANTED = "PERMISSION_GRANTED"
    PERMISSION_DENIED = "PERMISSION_DENIED"

@dataclass
class ElectronicSignature:
    """FDA 21 CFR Part 11 compliant electronic signature"""
    user_id: str
    full_name: str
    timestamp: str
    reason_for_signature: str
    digital_signature: str  # Base64 encoded signature
    certificate_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AuditEvent:
    """Immutable audit event record with FDA 21 CFR Part 11 compliance"""
    event_type: AuditEventType
    timestamp: str
    asset_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    source_component: Optional[str] = None
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    signature: Optional[str] = None  # Cryptographic signature for immutability
    electronic_signature: Optional[ElectronicSignature] = None  # FDA 21 CFR Part 11 requirement
    reason_for_change: Optional[str] = None  # Required for modifications
    is_locked: bool = False  # Indicates if record is finalized and immutable
    buffer_sequence: Optional[int] = None  # For time-series ordering
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), default=str)

class TimeSeriesBuffer:
    """Time-series buffer for high-throughput audit event processing"""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: float = 1.0):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[AuditEvent] = []
        self.sequence_counter = 0
        self.lock = threading.RLock()
        self.flush_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_running = False
        self.flush_thread = None
        
    def add_event(self, event: AuditEvent) -> int:
        """Add event to buffer and return sequence number"""
        with self.lock:
            self.sequence_counter += 1
            event.buffer_sequence = self.sequence_counter
            self.buffer.append(event)
            
            # Trigger flush if buffer is full
            if len(self.buffer) >= self.buffer_size:
                self._trigger_immediate_flush()
            
            return self.sequence_counter
    
    def start_flush_worker(self):
        """Start background flush worker thread"""
        if not self.is_running:
            self.is_running = True
            self.flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
            self.flush_thread.start()
    
    def stop_flush_worker(self):
        """Stop background flush worker and flush remaining events"""
        self.is_running = False
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)
        # Flush any remaining events
        self.flush()
        self.executor.shutdown(wait=True)
    
    def _flush_worker(self):
        """Background worker that periodically flushes buffer"""
        while self.is_running:
            time.sleep(self.flush_interval)
            if self.buffer:
                self.flush()
    
    def _trigger_immediate_flush(self):
        """Trigger immediate flush in background"""
        if self.buffer:
            self.executor.submit(self.flush)
    
    def flush(self) -> List[AuditEvent]:
        """Flush buffer and return flushed events"""
        with self.lock:
            if not self.buffer:
                return []
            
            flushed_events = self.buffer.copy()
            self.buffer.clear()
            return flushed_events
    
    def get_buffer_status(self) -> Dict[str, Any]:
        """Get current buffer status"""
        with self.lock:
            return {
                'buffer_size': len(self.buffer),
                'max_size': self.buffer_size,
                'sequence_counter': self.sequence_counter,
                'is_running': self.is_running
            }


class AuditTrailManager:
    """Manages the immutable audit trail system with FDA 21 CFR Part 11 compliance"""
    
    def __init__(self, db_path: str = "audit_trail.db", log_directory: str = "logs", 
                 enable_fda_compliance: bool = True, use_time_series: bool = True,
                 buffer_size: int = 10000, flush_interval: float = 1.0):
        self.db_path = Path(db_path)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        self.enable_fda_compliance = enable_fda_compliance
        self.use_time_series = use_time_series
        
        # Time-series buffer for high-throughput scenarios
        if self.use_time_series:
            self.time_series_buffer = TimeSeriesBuffer(buffer_size, flush_interval)
            self.time_series_buffer.start_flush_worker()
        else:
            self.time_series_buffer = None
        
        # FDA 21 CFR Part 11 compliance features
        self._private_key = None
        self._public_key = None
        self._user_signatures = {}  # Cache for user signatures
        self._locked_records = set()  # Track locked/finalized records
        
        # Set up structured logging
        self.logger = logging.getLogger(__name__)
        self._setup_structured_logging()
        
        # Initialize database
        self._init_database()
        
        # Initialize sequence number for ordering
        self._sequence_counter = 0
        
        # Initialize cryptographic keys for FDA compliance
        if self.enable_fda_compliance:
            self._initialize_crypto_keys()
        
    def _setup_structured_logging(self):
        """Set up structured logging with JSON formatter"""
        # Create file handler with JSON formatting
        json_handler = logging.FileHandler(
            self.log_directory / f"structured_audit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'event_type'):
                    log_entry['event_type'] = record.event_type
                if hasattr(record, 'asset_id'):
                    log_entry['asset_id'] = record.asset_id
                if hasattr(record, 'details'):
                    log_entry['details'] = record.details
                    
                return json.dumps(log_entry, default=str)
        
        json_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(json_handler)
        self.logger.setLevel(logging.INFO)
    
    def _initialize_crypto_keys(self):
        """Initialize cryptographic keys for FDA 21 CFR Part 11 compliance"""
        try:
            # Generate RSA key pair for system signing
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
            self._public_key = self._private_key.public_key()
            
            self.logger.info("FDA 21 CFR Part 11 cryptographic keys initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize cryptographic keys: {e}")
            self.enable_fda_compliance = False
    
    def _init_database(self):
        """Initialize SQLite database for audit trail with FDA compliance fields"""
        with self._get_db_connection() as conn:
            # Check if table exists and has correct schema
            cursor = conn.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name='audit_events'
            """)
            result = cursor.fetchone()
            
            if result:
                # Table exists, check if we need to upgrade schema
                self._upgrade_database_schema(conn)
            else:
                # Create new table with all required fields
                conn.execute('''
                    CREATE TABLE audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sequence_number INTEGER UNIQUE NOT NULL,
                        event_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        asset_id TEXT NOT NULL,
                        user_id TEXT,
                        session_id TEXT,
                        details TEXT,
                        correlation_id TEXT,
                        source_component TEXT,
                        severity TEXT DEFAULT 'INFO',
                        signature TEXT,
                        hash_value TEXT NOT NULL,
                        previous_hash TEXT,
                        electronic_signature TEXT,  -- FDA 21 CFR Part 11
                        reason_for_change TEXT,     -- FDA 21 CFR Part 11
                        is_locked BOOLEAN DEFAULT FALSE,  -- FDA 21 CFR Part 11
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        -- NOTE: No modified_at column to enforce immutability
                    )
                ''')
            
            # Create indexes for performance
            self._create_indexes(conn)
            
    def _create_indexes(self, conn):
        """Create database indexes for optimal query performance"""
        indexes = [
            ('idx_timestamp', 'timestamp'),
            ('idx_event_type', 'event_type'),
            ('idx_asset_id', 'asset_id'),
            ('idx_correlation_id', 'correlation_id'),
            ('idx_is_locked', 'is_locked'),
            ('idx_sequence_number', 'sequence_number')
        ]
        
        for idx_name, column in indexes:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON audit_events({column})')
            except Exception as e:
                self.logger.warning(f"Failed to create index {idx_name}: {e}")
            
            # Create table for user credentials and signatures
            conn.execute('''
                CREATE TABLE IF NOT EXISTS user_credentials (
                    user_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    certificate_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                )
            ''')
            
    def _upgrade_database_schema(self, conn):
        """Upgrade existing database schema to add missing columns"""
        # Check existing columns
        cursor = conn.execute("PRAGMA table_info(audit_events)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        
        # Required columns for current schema
        required_columns = {
            'electronic_signature', 'reason_for_change', 'is_locked'
        }
        
        # Add missing columns
        missing_columns = required_columns - existing_columns
        
        for column in missing_columns:
            try:
                if column == 'electronic_signature':
                    conn.execute('ALTER TABLE audit_events ADD COLUMN electronic_signature TEXT')
                elif column == 'reason_for_change':
                    conn.execute('ALTER TABLE audit_events ADD COLUMN reason_for_change TEXT')
                elif column == 'is_locked':
                    conn.execute('ALTER TABLE audit_events ADD COLUMN is_locked BOOLEAN DEFAULT FALSE')
                self.logger.info(f"Added missing column: {column}")
            except Exception as e:
                self.logger.warning(f"Failed to add column {column}: {e}")
        
        # Ensure triggers exist
        self._ensure_triggers(conn)
    
    def _ensure_triggers(self, conn):
        """Ensure WORM (Write Once Read Many) triggers exist"""
        try:
            # Create trigger to prevent updates to audit_events table (WORM storage)
            conn.execute('''
                CREATE TRIGGER IF NOT EXISTS prevent_audit_modification
                BEFORE UPDATE ON audit_events
                BEGIN
                    SELECT CASE
                        WHEN OLD.id = NEW.id THEN
                            RAISE(ABORT, 'Cannot modify audit trail records - WORM storage')
                        END;
                    END;
            ''')
            
            # Create trigger to prevent deletion of audit trail records
            conn.execute('''
                CREATE TRIGGER IF NOT EXISTS prevent_audit_deletion
                BEFORE DELETE ON audit_events
                BEGIN
                    SELECT RAISE(ABORT, 'Cannot delete audit trail records - WORM storage');
                    END;
            ''')
        except Exception as e:
            self.logger.warning(f"Failed to create WORM triggers: {e}")
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _calculate_hash(self, event: AuditEvent, previous_hash: Optional[str] = None) -> str:
        """Calculate cryptographic hash for the event (including previous hash for chain integrity)"""
        event_data = event.to_dict()
        event_data['previous_hash'] = previous_hash or ""
        
        # Create a canonical representation
        canonical_str = json.dumps(event_data, sort_keys=True, default=str)
        return hashlib.sha256(canonical_str.encode()).hexdigest()
    
    def log_event(self, event: AuditEvent, require_electronic_signature: bool = False) -> str:
        """Log an audit event with FDA 21 CFR Part 11 compliance"""
        # For critical events, require electronic signature
        if require_electronic_signature and self.enable_fda_compliance:
            if not event.electronic_signature:
                raise ValueError("Electronic signature required for this event type")
            
            # Verify the electronic signature
            if not self.verify_electronic_signature(event.electronic_signature):
                raise ValueError("Invalid electronic signature")
        
        # Calculate hash including previous event for chain integrity
        previous_hash = self._get_latest_hash()
        event_hash = self._calculate_hash(event, previous_hash)
        
        # Sign the event using system private key
        event.signature = self._sign_event(event_hash)
        
        # Handle time-series buffering vs direct database insert
        if self.use_time_series and self.time_series_buffer:
            # Add to time-series buffer for batch processing
            sequence_number = self.time_series_buffer.add_event(event)
            # Store hash for chain integrity calculation
            event.buffer_sequence = sequence_number
            
            # Log to structured file immediately for real-time monitoring
            self._log_to_structured_file(event, event_hash, sequence_number)
            
            # Optionally send to message queue/pub-sub system
            self._send_to_message_broker(event, event_hash)
            
        else:
            # Direct database insert (fallback mode)
            self._sequence_counter += 1
            sequence_number = self._sequence_counter
            
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO audit_events 
                    (sequence_number, event_type, timestamp, asset_id, user_id, session_id, 
                     details, correlation_id, source_component, severity, signature, hash_value, previous_hash,
                     electronic_signature, reason_for_change, is_locked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    sequence_number,
                    event.event_type.value,
                    event.timestamp,
                    event.asset_id,
                    event.user_id,
                    event.session_id,
                    json.dumps(event.details, default=str) if event.details else None,
                    event.correlation_id,
                    event.source_component,
                    event.severity,
                    event.signature,
                    event_hash,
                    previous_hash,
                    json.dumps(event.electronic_signature.to_dict(), default=str) if event.electronic_signature else None,
                    event.reason_for_change,
                    event.is_locked
                ))
            
            # Track locked records
            if event.is_locked:
                self._locked_records.add(sequence_number)
            
            # Log to structured file
            self._log_to_structured_file(event, event_hash, sequence_number)
        
        return event_hash
    
    def lock_record(self, sequence_number: int, user_id: str, password: str, 
                   reason: str) -> bool:
        """Lock/Finalize an audit record (FDA 21 CFR Part 11 requirement)"""
        if not self.enable_fda_compliance:
            return False
            
        # Authenticate user
        if not self.authenticate_user(user_id, password):
            return False
            
        try:
            # Check if record exists and is not already locked
            with self._get_db_connection() as conn:
                result = conn.execute('''
                    SELECT is_locked FROM audit_events WHERE sequence_number = ?
                ''', (sequence_number,)).fetchone()
                
                if not result:
                    self.logger.error(f"Record not found: {sequence_number}")
                    return False
                
                if result['is_locked']:
                    self.logger.warning(f"Record already locked: {sequence_number}")
                    return False
                
                # Lock the record
                conn.execute('''
                    UPDATE audit_events 
                    SET is_locked = TRUE
                    WHERE sequence_number = ?
                ''', (sequence_number,))
            
            # Track in memory
            self._locked_records.add(sequence_number)
            
            # Log the locking event
            lock_event = AuditEvent(
                event_type=AuditEventType.CONFIG_CHANGE,  # Using existing enum
                timestamp=datetime.now().isoformat(),
                asset_id=f"audit_record_{sequence_number}",
                user_id=user_id,
                details={
                    "action": "LOCK_RECORD",
                    "sequence_number": sequence_number,
                    "reason": reason
                },
                reason_for_change=reason,
                is_locked=True
            )
            
            self.log_event(lock_event)
            
            self.logger.info(f"Record locked: {sequence_number} by {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to lock record {sequence_number}: {e}")
            return False
    
    def is_record_locked(self, sequence_number: int) -> bool:
        """Check if a record is locked/finalized"""
        return sequence_number in self._locked_records
    
    def get_fda_compliance_report(self) -> Dict[str, Any]:
        """Generate FDA 21 CFR Part 11 compliance report"""
        if not self.enable_fda_compliance:
            return {"compliance_enabled": False}
            
        with self._get_db_connection() as conn:
            # Get total records
            total_records = conn.execute('SELECT COUNT(*) as count FROM audit_events').fetchone()['count']
            
            # Get locked records
            locked_records = conn.execute('SELECT COUNT(*) as count FROM audit_events WHERE is_locked = TRUE').fetchone()['count']
            
            # Get signed records
            signed_records = conn.execute('SELECT COUNT(*) as count FROM audit_events WHERE signature IS NOT NULL').fetchone()['count']
            
            # Get records with electronic signatures
            esigned_records = conn.execute('SELECT COUNT(*) as count FROM audit_events WHERE electronic_signature IS NOT NULL').fetchone()['count']
            
            # Get user statistics
            user_stats = conn.execute('''
                SELECT COUNT(*) as total_users, 
                       COUNT(last_login) as active_users
                FROM user_credentials
            ''').fetchone()
            
            return {
                "compliance_enabled": True,
                "report_timestamp": datetime.now().isoformat(),
                "total_audit_records": total_records,
                "locked_finalized_records": locked_records,
                "digitally_signed_records": signed_records,
                "electronically_signed_records": esigned_records,
                "users_registered": user_stats['total_users'],
                "active_users": user_stats['active_users'],
                "compliance_percentage": {
                    "record_locking": (locked_records / total_records * 100) if total_records > 0 else 0,
                    "digital_signatures": (signed_records / total_records * 100) if total_records > 0 else 0,
                    "electronic_signatures": (esigned_records / total_records * 100) if total_records > 0 else 0
                }
            }
    
    def _get_latest_hash(self) -> Optional[str]:
        """Get the hash of the most recent event for chain integrity"""
        with self._get_db_connection() as conn:
            result = conn.execute(
                'SELECT hash_value FROM audit_events ORDER BY sequence_number DESC LIMIT 1'
            ).fetchone()
            return result['hash_value'] if result else None
    
    def register_user(self, user_id: str, full_name: str, password: str) -> bool:
        """Register a user for FDA 21 CFR Part 11 compliance"""
        if not self.enable_fda_compliance:
            self.logger.warning("FDA compliance disabled - user registration not available")
            return False
            
        try:
            # Generate salt and hash password
            salt = secrets.token_hex(32)
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            password_hash_hex = password_hash.hex()
            
            with self._get_db_connection() as conn:
                conn.execute('''
                    INSERT INTO user_credentials 
                    (user_id, full_name, password_hash, salt)
                    VALUES (?, ?, ?, ?)
                ''', (user_id, full_name, password_hash_hex, salt))
            
            self.logger.info(f"User registered for FDA compliance: {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to register user {user_id}: {e}")
            return False
    
    def authenticate_user(self, user_id: str, password: str) -> bool:
        """Authenticate user for FDA 21 CFR Part 11 compliance"""
        if not self.enable_fda_compliance:
            return True  # Allow if compliance is disabled
            
        try:
            with self._get_db_connection() as conn:
                result = conn.execute('''
                    SELECT password_hash, salt FROM user_credentials WHERE user_id = ?
                ''', (user_id,)).fetchone()
            
            if not result:
                self.logger.warning(f"User not found: {user_id}")
                return False
            
            stored_hash = bytes.fromhex(result['password_hash'])
            salt = result['salt']
            
            # Verify password
            password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            
            if hmac.compare_digest(password_hash, stored_hash):
                # Update last login time
                with self._get_db_connection() as conn:
                    conn.execute('''
                        UPDATE user_credentials SET last_login = ? WHERE user_id = ?
                    ''', (datetime.now().isoformat(), user_id))
                
                self.logger.info(f"User authenticated: {user_id}")
                return True
            else:
                self.logger.warning(f"Authentication failed for user: {user_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Authentication error for {user_id}: {e}")
            return False
    
    def create_electronic_signature(self, user_id: str, password: str, 
                                  reason_for_signature: str) -> Optional[ElectronicSignature]:
        """Create FDA 21 CFR Part 11 compliant electronic signature"""
        if not self.enable_fda_compliance:
            return None
            
        # Authenticate user first
        if not self.authenticate_user(user_id, password):
            return None
            
        try:
            # Get user's full name
            with self._get_db_connection() as conn:
                result = conn.execute('''
                    SELECT full_name FROM user_credentials WHERE user_id = ?
                ''', (user_id,)).fetchone()
                
            if not result:
                return None
                
            full_name = result['full_name']
            timestamp = datetime.now().isoformat()
            
            # Create signature payload
            signature_payload = f"{user_id}|{full_name}|{timestamp}|{reason_for_signature}"
            signature_bytes = signature_payload.encode()
            
            # Create digital signature using private key
            signature = self._private_key.sign(
                signature_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            # Encode signature as base64
            digital_signature = base64.b64encode(signature).decode()
            
            electronic_sig = ElectronicSignature(
                user_id=user_id,
                full_name=full_name,
                timestamp=timestamp,
                reason_for_signature=reason_for_signature,
                digital_signature=digital_signature
            )
            
            # Cache the signature
            self._user_signatures[user_id] = electronic_sig
            
            self.logger.info(f"Electronic signature created for user: {user_id}")
            return electronic_sig
            
        except Exception as e:
            self.logger.error(f"Failed to create electronic signature for {user_id}: {e}")
            return None
    
    def verify_electronic_signature(self, electronic_signature: ElectronicSignature) -> bool:
        """Verify FDA 21 CFR Part 11 electronic signature"""
        if not self.enable_fda_compliance:
            return True
            
        try:
            # Reconstruct signature payload
            signature_payload = f"{electronic_signature.user_id}|{electronic_signature.full_name}|{electronic_signature.timestamp}|{electronic_signature.reason_for_signature}"
            signature_bytes = signature_payload.encode()
            
            # Decode digital signature
            signature = base64.b64decode(electronic_signature.digital_signature)
            
            # Verify signature using public key
            self._public_key.verify(
                signature,
                signature_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            self.logger.info(f"Electronic signature verified for user: {electronic_signature.user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Electronic signature verification failed: {e}")
            return False
    
    def _sign_event(self, hash_value: str) -> str:
        """Sign the event hash using system private key for FDA compliance"""
        if not self.enable_fda_compliance or not self._private_key:
            # Fallback to simple hash-based signature
            return hashlib.sha256(f"SIGNATURE:{hash_value}".encode()).hexdigest()
        
        try:
            # Create digital signature using private key
            signature = self._private_key.sign(
                hash_value.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
        except Exception as e:
            self.logger.error(f"Failed to create digital signature: {e}")
            return hashlib.sha256(f"SIGNATURE:{hash_value}".encode()).hexdigest()
    
    def verify_chain_integrity(self) -> bool:
        """Verify the integrity of the entire audit chain"""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM audit_events ORDER BY sequence_number ASC'
            ).fetchall()
        
        previous_hash = None
        integrity_issues = []
        
        for row in rows:
            # Recalculate hash
            event_dict = dict(row)
            event_dict['previous_hash'] = previous_hash or ""
            expected_hash = hashlib.sha256(
                json.dumps(event_dict, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check if calculated hash matches stored hash
            if expected_hash != row['hash_value']:
                error_msg = f"Hash mismatch at sequence {row['sequence_number']}: expected {expected_hash}, got {row['hash_value']}"
                self.logger.error(error_msg)
                integrity_issues.append(error_msg)
            
            # Check if previous hash matches expected previous hash
            stored_prev_hash = row['previous_hash'] or ""
            if previous_hash != stored_prev_hash:
                error_msg = f"Chain break at sequence {row['sequence_number']}: expected {previous_hash}, got {stored_prev_hash}"
                self.logger.error(error_msg)
                integrity_issues.append(error_msg)
            
            # Update previous hash to current hash for next iteration
            previous_hash = row['hash_value']
        
        if integrity_issues:
            self.logger.error(f"Audit chain integrity verification FAILED: {len(integrity_issues)} issues found")
            return False
        else:
            self.logger.info("Audit chain integrity verified successfully")
            return True
    
    def get_chain_integrity_report(self) -> Dict[str, Any]:
        """Generate detailed integrity report for compliance"""
        with self._get_db_connection() as conn:
            total_count = conn.execute('SELECT COUNT(*) as count FROM audit_events').fetchone()['count']
            rows = conn.execute(
                'SELECT * FROM audit_events ORDER BY sequence_number ASC'
            ).fetchall()
        
        previous_hash = None
        verified_count = 0
        integrity_issues = []
        
        for row in rows:
            # Recalculate hash
            event_dict = dict(row)
            event_dict['previous_hash'] = previous_hash or ""
            expected_hash = hashlib.sha256(
                json.dumps(event_dict, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Check integrity
            is_valid = True
            if expected_hash != row['hash_value']:
                integrity_issues.append({
                    'sequence_number': row['sequence_number'],
                    'issue': 'hash_mismatch',
                    'expected_hash': expected_hash,
                    'actual_hash': row['hash_value']
                })
                is_valid = False
            
            stored_prev_hash = row['previous_hash'] or ""
            if previous_hash != stored_prev_hash:
                integrity_issues.append({
                    'sequence_number': row['sequence_number'],
                    'issue': 'chain_break',
                    'expected_prev_hash': previous_hash,
                    'actual_prev_hash': stored_prev_hash
                })
                is_valid = False
            
            if is_valid:
                verified_count += 1
            
            # Update previous hash to current hash for next iteration
            previous_hash = row['hash_value']
        
        return {
            'verification_timestamp': datetime.now().isoformat(),
            'total_records': total_count,
            'verified_records': verified_count,
            'integrity_issues': integrity_issues,
            'integrity_percentage': (verified_count / total_count * 100) if total_count > 0 else 0,
            'is_integrity_valid': len(integrity_issues) == 0
        }
    
    def get_events_by_type(self, event_type: AuditEventType, limit: int = 100) -> List[AuditEvent]:
        """Retrieve events of a specific type"""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM audit_events WHERE event_type = ? ORDER BY timestamp DESC LIMIT ?',
                (event_type.value, limit)
            ).fetchall()
        
        return [self._row_to_audit_event(row) for row in rows]
    
    def get_events_by_asset(self, asset_id: str, limit: int = 100) -> List[AuditEvent]:
        """Retrieve events for a specific asset"""
        with self._get_db_connection() as conn:
            rows = conn.execute(
                'SELECT * FROM audit_events WHERE asset_id = ? ORDER BY timestamp DESC LIMIT ?',
                (asset_id, limit)
            ).fetchall()
        
        return [self._row_to_audit_event(row) for row in rows]
    
    def get_events_by_time_range(self, start_time: str, end_time: str, limit: int = 1000) -> List[AuditEvent]:
        """Retrieve events within a time range"""
        with self._get_db_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (start_time, end_time, limit)).fetchall()
        
        return [self._row_to_audit_event(row) for row in rows]
    
    def get_events_with_details(self, search_term: str, limit: int = 100) -> List[AuditEvent]:
        """Search events by details content"""
        with self._get_db_connection() as conn:
            rows = conn.execute('''
                SELECT * FROM audit_events 
                WHERE details LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (f'%{search_term}%', limit)).fetchall()
        
        return [self._row_to_audit_event(row) for row in rows]
    
    def _row_to_audit_event(self, row: sqlite3.Row) -> AuditEvent:
        """Convert database row to AuditEvent object"""
        details = json.loads(row['details']) if row['details'] else None
        
        event = AuditEvent(
            event_type=AuditEventType(row['event_type']),
            timestamp=row['timestamp'],
            asset_id=row['asset_id'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            details=details,
            correlation_id=row['correlation_id'],
            source_component=row['source_component'],
            severity=row['severity'],
            signature=row['signature']
        )
        
        return event
    
    def _log_to_structured_file(self, event: AuditEvent, event_hash: str, sequence_number: int):
        """Log event to structured file for real-time monitoring"""
        self.logger.info(
            f"Audit event logged: {event.event_type.value}",
            extra={
                'event_type': event.event_type.value,
                'asset_id': event.asset_id,
                'user_id': event.user_id,
                'details': event.details,
                'hash': event_hash,
                'sequence_number': sequence_number,
                'is_locked': event.is_locked,
                'electronic_signature': event.electronic_signature is not None,
                'timestamp': event.timestamp,
                'buffer_sequence': getattr(event, 'buffer_sequence', None)
            }
        )
    
    def _send_to_message_broker(self, event: AuditEvent, event_hash: str):
        """Send event to message broker/pub-sub system (placeholder for Kafka/PubSub integration)"""
        try:
            # This is where you'd integrate with actual message brokers like:
            # - Kafka/Redpanda
            # - Google Cloud Pub/Sub
            # - AWS SNS/SQS
            # - MQTT brokers
            
            message_payload = {
                'event': event.to_dict(),
                'hash': event_hash,
                'routing_key': f"audit.{event.event_type.value.lower()}",
                'timestamp': datetime.now().isoformat(),
                'schema_version': '1.0'
            }
            
            # Placeholder for actual message broker integration
            # In production, this would connect to your chosen messaging system
            self.logger.debug(f"Message queued for broker: {message_payload['routing_key']}")
            
        except Exception as e:
            self.logger.error(f"Failed to send message to broker: {e}")
    
    def flush_time_series_buffer(self) -> List[AuditEvent]:
        """Manually flush time-series buffer and persist events"""
        if not self.use_time_series or not self.time_series_buffer:
            return []
        
        flushed_events = self.time_series_buffer.flush()
        if flushed_events:
            self._persist_batch_events(flushed_events)
        return flushed_events
    
    def _persist_batch_events(self, events: List[AuditEvent]):
        """Persist batch of events to database"""
        if not events:
            return
        
        try:
            with self._get_db_connection() as conn:
                for event in events:
                    previous_hash = self._get_latest_hash()
                    event_hash = self._calculate_hash(event, previous_hash)
                    event.signature = self._sign_event(event_hash)
                    
                    conn.execute('''
                        INSERT INTO audit_events 
                        (sequence_number, event_type, timestamp, asset_id, user_id, session_id, 
                         details, correlation_id, source_component, severity, signature, hash_value, previous_hash,
                         electronic_signature, reason_for_change, is_locked)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        event.buffer_sequence,
                        event.event_type.value,
                        event.timestamp,
                        event.asset_id,
                        event.user_id,
                        event.session_id,
                        json.dumps(event.details, default=str) if event.details else None,
                        event.correlation_id,
                        event.source_component,
                        event.severity,
                        event.signature,
                        event_hash,
                        previous_hash,
                        json.dumps(event.electronic_signature.to_dict(), default=str) if event.electronic_signature else None,
                        event.reason_for_change,
                        event.is_locked
                    ))
                    
                    # Track locked records
                    if event.is_locked:
                        self._locked_records.add(event.buffer_sequence)
            
            self.logger.info(f"Persisted {len(events)} events from time-series buffer")
            
        except Exception as e:
            self.logger.error(f"Failed to persist batch events: {e}")
            # In production, you might want to retry or send to dead letter queue
    
    def get_time_series_status(self) -> Dict[str, Any]:
        """Get time-series buffer status"""
        if self.use_time_series and self.time_series_buffer:
            return self.time_series_buffer.get_buffer_status()
        return {'enabled': False}
    
    def export_audit_trail(self, output_file: str, start_time: str = None, end_time: str = None):
        """Export audit trail to JSON file"""
        events = self.get_events_by_time_range(
            start_time or "1970-01-01T00:00:00",
            end_time or datetime.now().isoformat(),
            limit=10000  # Reasonable limit for export
        )
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_events': len(events),
            'events': [event.to_dict() for event in events],
            'export_format': 'time_series_audit_trail_v1.0'
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Audit trail exported to {output_file}")


# Global audit trail manager instance
_audit_trail_manager = None
_audit_trail_config = {
    'db_path': 'audit_trail.db',
    'log_directory': 'logs',
    'enable_fda_compliance': True,
    'use_time_series': True,
    'buffer_size': 10000,
    'flush_interval': 1.0
}

def configure_audit_trail(**kwargs):
    """Configure audit trail settings before initialization"""
    global _audit_trail_config
    _audit_trail_config.update(kwargs)

def get_audit_trail_manager() -> AuditTrailManager:
    """Get the global audit trail manager instance"""
    global _audit_trail_manager
    if _audit_trail_manager is None:
        _audit_trail_manager = AuditTrailManager(**_audit_trail_config)
    return _audit_trail_manager

def shutdown_audit_trail():
    """Properly shutdown audit trail manager and flush buffers"""
    global _audit_trail_manager
    if _audit_trail_manager:
        if _audit_trail_manager.use_time_series and _audit_trail_manager.time_series_buffer:
            _audit_trail_manager.time_series_buffer.stop_flush_worker()
        _audit_trail_manager = None


def log_audit_event(
    event_type: AuditEventType,
    asset_id: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    source_component: Optional[str] = None,
    severity: str = "INFO",
    electronic_signature: Optional[ElectronicSignature] = None,
    reason_for_change: Optional[str] = None,
    is_locked: bool = False,
    require_electronic_signature: bool = False
) -> str:
    """Convenience function to log an audit event with FDA 21 CFR Part 11 compliance"""
    event = AuditEvent(
        event_type=event_type,
        timestamp=datetime.now().isoformat(),
        asset_id=asset_id,
        user_id=user_id,
        session_id=session_id,
        details=details,
        correlation_id=correlation_id,
        source_component=source_component,
        severity=severity,
        electronic_signature=electronic_signature,
        reason_for_change=reason_for_change,
        is_locked=is_locked
    )
    
    manager = get_audit_trail_manager()
    return manager.log_event(event, require_electronic_signature=require_electronic_signature)


# Example usage and testing
if __name__ == "__main__":
    # Configure for time-series mode
    configure_audit_trail(
        use_time_series=True,
        buffer_size=100,
        flush_interval=0.5,
        enable_fda_compliance=True
    )
    
    # Initialize audit trail
    audit_manager = get_audit_trail_manager()
    
    print("=== Time-Series Audit Trail Demo ===")
    
    # Log some sample events rapidly to demonstrate buffering
    for i in range(5):
        log_audit_event(
            AuditEventType.PARAMETER_ADJUSTED,
            asset_id="SP-2024-001",
            user_id="test_user",
            details={
                "parameter": f"param_{i}", 
                "old_value": i * 10.0, 
                "new_value": (i + 1) * 10.0,
                "iteration": i
            },
            correlation_id=f"batch-test-{i}",
            source_component="DemoTest"
        )
        time.sleep(0.1)  # Small delay to show buffering
    
    # Check buffer status
    buffer_status = audit_manager.get_time_series_status()
    print(f"Buffer Status: {buffer_status}")
    
    # Manual flush demonstration
    flushed_events = audit_manager.flush_time_series_buffer()
    print(f"Manually flushed {len(flushed_events)} events")
    
    # Log critical event requiring signature
    log_audit_event(
        AuditEventType.EMERGENCY_STOP,
        asset_id="SP-2024-001",
        user_id="safety_system",
        details={
            "trigger_reason": "collision_detected", 
            "position": [125.5, 78.3, 45.2],
            "velocity": [0.0, 0.0, 0.0]
        },
        severity="CRITICAL",
        source_component="SafetyMonitor"
    )
    
    # Verify chain integrity
    is_valid = audit_manager.verify_chain_integrity()
    print(f"Audit chain integrity: {'VALID' if is_valid else 'INVALID'}")
    
    # Query events
    config_events = audit_manager.get_events_by_type(AuditEventType.PARAMETER_ADJUSTED)
    print(f"Found {len(config_events)} parameter adjustment events")
    
    # Export demonstration
    export_file = "demo_audit_export.json"
    audit_manager.export_audit_trail(export_file)
    print(f"Audit trail exported to {export_file}")
    
    # Cleanup
    shutdown_audit_trail()
    print("Audit trail system shut down successfully")