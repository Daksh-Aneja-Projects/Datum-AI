"""
FastAPI Backend Server for Manufacturing Cyber-Physical System
Provides REST API and WebSocket endpoints for frontend integration
"""

import asyncio
import logging
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Process, Queue
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List
import uvicorn
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
import secrets

# Import system components
from manufacturing_cps import ManufacturingCPS
from main import ManufacturingOrchestrator, Settings
from utils.health_monitor import get_health_monitor
from utils.audit_trail import (
    get_audit_trail_manager,
    log_audit_event,
    AuditEventType,
    AuditEvent
)

# Import setup wizard
from setup_wizard import SetupWizard, CalibrationConfig

import os
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global system instances
orchestrator: Optional[ManufacturingOrchestrator] = None
cps_instance: Optional[ManufacturingCPS] = None
websocket_connections: List[WebSocket] = []
settings = Settings()
setup_wizard: Optional[SetupWizard] = None

# Thread pool for CPU-intensive tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

# Authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    # Load from secure secrets management
    secret_file = os.getenv("SECRET_FILE_PATH", "secrets/secret.key")
    secrets_dir = os.path.dirname(secret_file)
    if secrets_dir and not os.path.exists(secrets_dir):
        os.makedirs(secrets_dir, mode=0o700)  # Secure directory permissions
    
    if os.path.exists(secret_file):
        # Ensure the secret file has secure permissions
        os.chmod(secret_file, 0o600)  # Read/write for owner only
        with open(secret_file, "r") as f:
            SECRET_KEY = f.read().strip()
    else:
        SECRET_KEY = secrets.token_urlsafe(32)
        # Save for persistence across restarts with secure permissions
        with open(secret_file, "w", encoding="utf-8") as f:
            f.write(SECRET_KEY)
        os.chmod(secret_file, 0o600)  # Read/write for owner only
        
        # Log warning about generating new secret
        logger.warning(f"Generated new SECRET_KEY and saved to {secret_file}. For production, set SECRET_KEY environment variable.")
        
# Additional secrets for database and other services
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_USER = os.getenv("DB_USER", "manufacturing_user")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# Encryption key for sensitive data
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
if not ENCRYPTION_KEY and os.getenv("ENCRYPTION_KEY_FILE"):
    with open(os.getenv("ENCRYPTION_KEY_FILE"), "r") as f:
        ENCRYPTION_KEY = f.read().strip()

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# API Key authentication
API_KEYS_FILE = os.getenv("API_KEYS_FILE", "secrets/api_keys.json")

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Define API key authentication
async def get_api_key(request: Request) -> str:
    api_key_header = request.headers.get("X-API-Key")
    if not api_key_header:
        raise HTTPException(status_code=401, detail="API Key required")
    
    # Load API keys from secure file
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, "r") as f:
            api_keys = json.load(f)
        if api_key_header in api_keys.get("valid_keys", []):
            return api_key_header
    
    raise HTTPException(status_code=401, detail="Invalid API Key")

# Role-based access control
USER_ROLES = {
    "admin": ["read", "write", "execute", "admin"],
    "operator": ["read", "write"],
    "guest": ["read"]
}

# Database connection setup
DATABASE_PATH = os.getenv("DATABASE_PATH", "users.db")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    # For SQLite, use the path directly
    if DATABASE_PATH.endswith('.db'):
        DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
    else:
        # For other databases, construct URL from components
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/manufacturing_cps" if DB_PASSWORD else f"sqlite:///{DATABASE_PATH}"

def init_db():
    """Initialize the database with users table if it doesn't exist"""
    # Use the configured database URL
    if DATABASE_URL.startswith("sqlite://"):
        db_path = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL
            )
        ''')
        
        # Insert default users if they don't exist
        default_users = [
            ("admin", pwd_context.hash("admin123"), "admin"),
            ("operator", pwd_context.hash("operator123"), "operator")
        ]
        
        for username, hashed_password, role in default_users:
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role)
            )
        
        conn.commit()
        conn.close()
    else:
        # For other database types (PostgreSQL, MySQL, etc.), would use SQLAlchemy
        # This is a simplified example for the purpose of this implementation
        import sqlite3
        db_path = DATABASE_PATH
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create users table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                role TEXT NOT NULL
            )
        ''')
        
        # Insert default users if they don't exist
        default_users = [
            ("admin", pwd_context.hash("admin123"), "admin"),
            ("operator", pwd_context.hash("operator123"), "operator")
        ]
        
        for username, hashed_password, role in default_users:
            cursor.execute(
                "INSERT OR IGNORE INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role)
            )
        
        conn.commit()
        conn.close()

# Initialize the database
init_db()


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
    role: Optional[str] = None


class User(BaseModel):
    username: str
    role: str
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_user(username: str, password: str, role: str):
    """Create a new user in the database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            (username, get_password_hash(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Username already exists
        return False
    finally:
        conn.close()


def get_user(username: str) -> Optional[UserInDB]:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT username, hashed_password, role FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    
    conn.close()
    
    if row:
        username, hashed_password, role = row
        return UserInDB(username=username, hashed_password=hashed_password, role=role)
    return None


def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = get_user(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def check_permission(required_permission: str, role: str) -> bool:
    """Check if role has required permission"""
    if role in USER_ROLES:
        return required_permission in USER_ROLES[role]
    return False


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=role)
    except JWTError:
        raise credentials_exception
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class JobSpec(BaseModel):
    """Job specification for manufacturing operations"""
    job_id: str = Field(..., description="Unique job identifier")
    target_quality: str = Field(default="DIN_876_GRADE_00", description="Target quality standard")
    cycles: int = Field(default=10, ge=1, le=1000, description="Number of manufacturing cycles")
    force_target: float = Field(default=45.0, ge=10.0, le=100.0, description="Target contact force in Newtons")
    material_properties: Optional[Dict[str, Any]] = Field(default=None, description="Material properties override")


class ForceTargetRequest(BaseModel):
    """Schema for validating force_target parameter"""
    force_target: float = Field(..., ge=10.0, le=100.0, description="Target contact force in Newtons, between 10.0 and 100.0")


class ProcessParameterUpdateRequest(BaseModel):
    """Schema for validating process parameter updates"""
    spindle_speed_rpm: float = Field(default=3000, ge=1000, le=6000, description="Spindle speed in RPM, between 1000 and 6000")
    feed_rate_mm_per_sec: float = Field(default=2.0, ge=0.1, le=10.0, description="Feed rate in mm/sec, between 0.1 and 10.0")
    down_force_n: float = Field(default=45.0, ge=20.0, le=200.0, description="Down force in Newtons, between 20.0 and 200.0")
    coolant_flow_rate: float = Field(default=5.0, ge=1.0, le=20.0, description="Coolant flow rate in L/min, between 1.0 and 20.0")
    abrasive_grit_size: int = Field(default=200, ge=100, le=400, description="Abrasive grit size, between 100 and 400")
    dwell_time_sec: float = Field(default=1.0, ge=0.1, le=10.0, description="Dwell time in seconds, between 0.1 and 10.0")


class SurfacePoint(BaseModel):
    """Schema for validating individual surface points"""
    x: float = Field(..., ge=-1000.0, le=1000.0, description="X coordinate in mm")
    y: float = Field(..., ge=-1000.0, le=1000.0, description="Y coordinate in mm")
    z: float = Field(..., ge=-10.0, le=10.0, description="Z coordinate in mm")


class SurfacePointArray(BaseModel):
    """Schema for validating surface point arrays"""
    surface_points: List[SurfacePoint] = Field(..., min_items=1, max_items=10000, description="Array of surface points")
    dwell_times: List[float] = Field(..., min_items=1, max_items=10000, description="Array of dwell times")


class AEDataSchema(BaseModel):
    """Schema for validating Acoustic Emission data"""
    rms: float = Field(..., ge=0.0, le=10.0, description="RMS value between 0.0 and 10.0")
    kurtosis: float = Field(..., ge=0.0, le=100.0, description="Kurtosis value between 0.0 and 100.0")
    frequency_peak: float = Field(..., ge=0.0, le=1000000.0, description="Frequency peak in Hz")

class JobResponse(BaseModel):
    """Response model for job execution"""
    job_id: str
    status: str
    start_time: str
    end_time: Optional[str] = None
    quality_results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class CalibrationConfig(BaseModel):
    """Configuration for surface plate calibration"""
    # Interferometry settings
    wavelength_nm: float = Field(default=632.8, ge=632.0, le=670.0, description="Laser wavelength in nanometers")
    grazing_angle_deg: float = Field(default=88.0, ge=85.0, le=89.5, description="Grazing incidence angle in degrees")
    spatial_resolution_um: float = Field(default=0.1, ge=0.01, le=1.0, description="Spatial resolution in micrometers")
    
    # Force control settings
    target_force_n: float = Field(default=45.0, ge=20.0, le=150.0, description="Target normal force in Newtons")
    force_tolerance_n: float = Field(default=5.0, ge=1.0, le=20.0, description="Force tolerance in Newtons")
    
    # Acoustic emission settings
    ae_sample_rate_hz: int = Field(default=50000, ge=10000, le=1000000, description="AE sample rate in Hz")
    ae_rms_threshold: float = Field(default=0.5, ge=0.0, le=2.0, description="AE RMS threshold")
    
    # Process parameters
    spindle_speed_rpm: float = Field(default=3000, ge=1000, le=6000, description="Spindle speed in RPM")
    feed_rate_mm_per_sec: float = Field(default=2.0, ge=0.1, le=10.0, description="Feed rate in mm/sec")
    abrasive_grit_size: int = Field(default=200, ge=100, le=400, description="Abrasive grit size")
    
    # Safety limits
    max_force_n: float = Field(default=200.0, ge=100.0, le=500.0, description="Maximum force limit in Newtons")
    max_temperature_c: float = Field(default=60.0, ge=20.0, le=100.0, description="Maximum temperature in Celsius")
    max_vibration_g: float = Field(default=0.5, ge=0.1, le=2.0, description="Maximum vibration in g")
    
    # Asset identification
    serial_number: Optional[str] = Field(default=None, description="Surface plate serial number")
    model: Optional[str] = Field(default=None, description="Model type")
    calibration_date: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat, description="Calibration date")
    
    # Additional metadata
    config_hash: Optional[str] = Field(default=None, description="Configuration hash for integrity")
    setup_timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat, description="Setup timestamp")


class SystemStatus(BaseModel):
    """System status response"""
    status: str
    timestamp: str
    active_components: List[str]
    health_metrics: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global orchestrator, cps_instance, setup_wizard
    
    logger.info("Starting Manufacturing CPS API Server...")
    
    # Initialize setup wizard
    setup_wizard = SetupWizard()
    
    # Initialize orchestrator and CPS
    orchestrator = ManufacturingOrchestrator(settings)
    
    if await orchestrator.initialize_system():
        cps_instance = orchestrator.manufacturing_cps
        logger.info("Manufacturing CPS initialized successfully")
    else:
        logger.error("Failed to initialize Manufacturing CPS")
        raise RuntimeError("System initialization failed")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Manufacturing CPS API Server...")
    if orchestrator:
        await orchestrator.graceful_shutdown()

# Create FastAPI app
app = FastAPI(
    title="Manufacturing Cyber-Physical System API",
    description="REST API and WebSocket interface for AI-Driven Metrology Manufacturing",
    version="1.0.0",
    lifespan=lifespan
)

# Add security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Add security headers
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

app.add_middleware(SecurityHeadersMiddleware)

# Add CORS middleware with restricted origins
allowed_origins = []

# Add development origins only if in development mode
if os.getenv("ENVIRONMENT", "production") == "development":
    allowed_origins.extend([
        "http://localhost:3000",
        "http://localhost:8080",
        "https://localhost:3000",
        "https://localhost:8080",
    ])

# Add production origins from environment variable
prod_origins = os.getenv("ALLOWED_ORIGINS", "")
if prod_origins:
    allowed_origins.extend([origin.strip() for origin in prod_origins.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["X-Requested-With", "Content-Type", "Accept", "Authorization", "X-API-Key"],  # More restrictive than ["*"]
    # Additional security options
    allow_origin_regex=None,
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that sanitizes error messages and returns standard HTTP codes.
    This prevents sensitive internal error details from being exposed to clients.
    """
    logger.error(f"Unhandled exception occurred: {exc}\nTraceback: {traceback.format_exc()}")
    
    # Log the audit event for security purposes
    try:
        audit_manager = get_audit_trail_manager()
        log_audit_event(
            audit_manager,
            event_type=AuditEventType.SYSTEM_ERROR,
            asset_id="api-server",
            event_details={
                "endpoint": str(request.url),
                "method": request.method,
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:200]  # Truncate to prevent large payloads
            }
        )
    except Exception as audit_error:
        logger.error(f"Failed to log audit event for exception: {audit_error}")
    
    # Return sanitized error response
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please contact system administrator."
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with sanitized messages.
    """
    logger.warning(f"HTTP exception occurred: {exc.status_code} - {exc.detail}")
    
    # Log the audit event for security purposes
    try:
        audit_manager = get_audit_trail_manager()
        log_audit_event(
            audit_manager,
            event_type=AuditEventType.API_ACCESS,
            asset_id="api-server",
            event_details={
                "endpoint": str(request.url),
                "method": request.method,
                "status_code": exc.status_code,
                "error_detail": str(exc.detail)[:200]  # Truncate to prevent large payloads
            }
        )
    except Exception as audit_error:
        logger.error(f"Failed to log audit event for HTTP exception: {audit_error}")
    
    # Sanitize the detail message for certain status codes
    sanitized_detail = "Bad request" if exc.status_code in [400, 422] else "Unauthorized" if exc.status_code == 401 else "Access denied" if exc.status_code == 403 else "Resource not found" if exc.status_code == 404 else "Request timeout" if exc.status_code == 408 else "Internal server error"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": sanitized_detail,
            "message": f"Request failed with status {exc.status_code}. Please check your request and try again."
        }
    )

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message, default=str))
            except Exception as e:
                logger.error(f"Failed to send WebSocket message: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

connection_manager = ConnectionManager()

@app.post("/auth/token")
async def login_for_access_token(username: str, password: str):
    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/")
async def root():
    """Root endpoint - system health check"""
    return {
        "message": "Manufacturing Cyber-Physical System API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status and health metrics"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    health_monitor = get_health_monitor()
    health_data = health_monitor.get_system_health()
    
    return SystemStatus(
        status=orchestrator.system_state.status,
        timestamp=orchestrator.system_state.timestamp,
        active_components=orchestrator.system_state.active_components,
        health_metrics=health_data
    )


@app.get("/setup/status")
async def get_setup_status():
    """Get current setup wizard status"""
    global setup_wizard
    if setup_wizard:
        validated_config = setup_wizard.get_validated_config()
        return {
            "setup_complete": setup_wizard.setup_complete,
            "config_exists": setup_wizard._config_exists(),
            "validated_config": validated_config.__dict__ if validated_config else None
        }
    else:
        raise HTTPException(status_code=503, detail="Setup wizard not initialized")

@app.post("/jobs/start", response_model=JobResponse)
async def start_job(
    job_spec: JobSpec, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Start a manufacturing job asynchronously with separation of heavy computation"""
    # Check if user has execute permission
    if not check_permission("execute", current_user.role):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if not orchestrator or orchestrator.system_state.status != "READY":
        raise HTTPException(status_code=503, detail="System not ready for operations")
    
    logger.info(f"Starting job: {job_spec.job_id} by user: {current_user.username}")
    
    try:
        # For heavy computation (metrology), use background task
        # The actual job execution will be handled separately
        job_id = job_spec.job_id
        
        # Schedule the job execution as a background task
        background_tasks.add_task(_execute_job_in_background, job_spec.dict(), current_user.username)
        
        return JobResponse(
            job_id=job_id,
            status="started",
            start_time=datetime.now().isoformat(),
            end_time=None,
            quality_results=None
        )
        
    except Exception as e:
        logger.error(f"Job start failed: {e}")
        return JobResponse(
            job_id=job_spec.job_id,
            status="failed",
            start_time=datetime.now().isoformat(),
            error_message=str(e)
        )

async def _execute_job_in_background(job_spec_dict: dict, username: str):
    """Execute the actual job in the background to separate heavy computation from API responses"""
    try:
        job_results = await orchestrator.execute_manufacturing_job(job_spec_dict)
        logger.info(f"Background job completed: {job_spec_dict['job_id']} by {username}")
        
        # Broadcast completion to WebSocket clients
        await connection_manager.broadcast({
            "event": "job_completed",
            "job_id": job_spec_dict['job_id'],
            "results": job_results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Background job failed: {job_spec_dict['job_id']}, error: {e}")
        
        # Broadcast failure to WebSocket clients
        await connection_manager.broadcast({
            "event": "job_failed",
            "job_id": job_spec_dict['job_id'],
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.post("/control/emergency-stop")
async def emergency_stop(current_user: User = Depends(get_current_active_user)):
    """Trigger emergency stop"""
    # Check if user has execute permission
    if not check_permission("execute", current_user.role):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    if cps_instance:
        cps_instance.emergency_stop()
        logger.warning("Emergency stop triggered via API by user: %s", current_user.username)
        
        # Broadcast emergency stop notification
        await connection_manager.broadcast({
            "event": "emergency_stop",
            "timestamp": datetime.now().isoformat(),
            "message": f"Emergency stop activated by {current_user.username}",
            "triggered_by": current_user.username
        })
        
        return {"status": "success", "message": "Emergency stop activated", "triggered_by": current_user.username}
    else:
        raise HTTPException(status_code=503, detail="CPS not available")

@app.post("/control/force-update")
async def update_force_target(
    request: ForceTargetRequest, 
    current_user: User = Depends(get_current_active_user)
):
    """Update the target force for Active Contact Flange with strict validation"""
    # Check if user has write permission
    if not check_permission("write", current_user.role):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    # Validate force target is within safe limits
    if request.force_target < 10.0 or request.force_target > 200.0:
        raise HTTPException(status_code=400, detail="Force target must be between 10.0N and 200.0N")
    
    if not cps_instance:
        raise HTTPException(status_code=503, detail="CPS not available")
    
    try:
        cps_instance.update_control_loop(request.force_target)
        logger.info(f"Force target updated to {request.force_target}N by user: {current_user.username}")
        
        return {
            "status": "success",
            "force_target": request.force_target,
            "timestamp": datetime.now().isoformat(),
            "updated_by": current_user.username
        }
    except Exception as e:
        logger.error(f"Failed to update force target: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/setup/configurations")
async def get_existing_configurations():
    """Get list of existing configurations"""
    import os
    import json
    from pathlib import Path
    
    configs = []
    config_dir = Path("configs")
    
    if config_dir.exists():
        for config_file in config_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    config_data['filename'] = config_file.name
                    configs.append(config_data)
            except Exception as e:
                logger.error(f"Failed to load config {config_file}: {e}")
    
    return {
        "configurations": configs,
        "count": len(configs)
    }

@app.post("/setup/complete")
async def complete_setup(config: CalibrationConfig):
    """Complete the setup wizard and save configuration"""
    import hashlib
    import json
    from pathlib import Path
    
    try:
        # Generate configuration hash for integrity
        config_dict = config.dict()
        config_str = json.dumps({k: v for k, v in config_dict.items() if k not in ['config_hash', 'setup_timestamp']}, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
        config_dict['config_hash'] = config_hash
        
        # Create config directory if it doesn't exist
        config_dir = Path("configs")
        config_dir.mkdir(exist_ok=True)
        
        # Save configuration
        filename = f"calibration_config_{config.serial_number or 'default'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        config_path = config_dir / filename
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {config_path}")
        
        # Optionally, broadcast the configuration update to connected clients
        await connection_manager.broadcast({
            "event": "configuration_updated",
            "timestamp": datetime.now().isoformat(),
            "config_filename": filename
        })
        
        return {
            "status": "success",
            "message": "Configuration saved successfully",
            "config_filename": filename,
            "config_hash": config_hash
        }
        
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")


@app.post("/setup/run-wizard")
async def run_setup_wizard(current_user: User = Depends(get_current_active_user)):
    """Run the setup wizard programmatically"""
    # Check if user has admin permission
    if not check_permission("admin", current_user.role):
        raise HTTPException(status_code=403, detail="Admin permission required")
    
    global setup_wizard
    if setup_wizard:
        try:
            # This would typically run the setup wizard in headless mode
            # For now, we'll just return the current status
            validated_config = setup_wizard.get_validated_config()
            
            return {
                "status": "success",
                "setup_complete": setup_wizard.setup_complete,
                "config_exists": setup_wizard._config_exists(),
                "config": validated_config.__dict__ if validated_config else None,
                "executed_by": current_user.username
            }
        except Exception as e:
            logger.error(f"Failed to run setup wizard: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to run setup wizard: {str(e)}")
    else:
        raise HTTPException(status_code=503, detail="Setup wizard not initialized")

@app.get("/audit/events")
async def get_audit_events(
    event_type: Optional[str] = None,
    asset_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
):
    """Get audit trail events"""
    manager = get_audit_trail_manager()
    
    try:
        if event_type:
            # Get events by specific type
            try:
                audit_event_type = AuditEventType(event_type)
                events = manager.get_events_by_type(audit_event_type, limit)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid event type")
        elif asset_id:
            # Get events by specific asset
            events = manager.get_events_by_asset(asset_id, limit)
        elif start_time and end_time:
            # Get events by time range
            events = manager.get_events_by_time_range(start_time, end_time, limit)
        else:
            # Get latest events
            events = manager.get_events_by_time_range(
                start_time or "1970-01-01T00:00:00",
                end_time or datetime.now().isoformat(),
                limit
            )
        
        # Convert to dict format for JSON response
        events_dict = [event.to_dict() for event in events]
        
        return {
            "events": events_dict,
            "count": len(events_dict),
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"Failed to retrieve audit events: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve audit events")

@app.get("/audit/integrity")
async def verify_audit_integrity():
    """Verify the integrity of the audit trail chain"""
    manager = get_audit_trail_manager()
    
    try:
        is_valid = manager.verify_chain_integrity()
        
        return {
            "integrity_valid": is_valid,
            "verification_time": datetime.now().isoformat(),
            "message": "Audit chain integrity verified" if is_valid else "Audit chain integrity compromised"
        }
    
    except Exception as e:
        logger.error(f"Failed to verify audit integrity: {e}")
        raise HTTPException(status_code=500, detail="Failed to verify audit integrity")

@app.post("/audit/export")
async def export_audit_trail(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    filename: str = "audit_export.json"
):
    """Export audit trail to a file"""
    manager = get_audit_trail_manager()
    
    try:
        export_path = f"exports/{filename}"
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        
        manager.export_audit_trail(export_path, start_time, end_time)
        
        return {
            "status": "success",
            "export_file": export_path,
            "export_time": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to export audit trail: {e}")
        raise HTTPException(status_code=500, detail="Failed to export audit trail")

@app.get("/surface/current-quality")
async def get_current_surface_quality():
    """Get current surface quality metrics"""
    if not cps_instance:
        raise HTTPException(status_code=503, detail="CPS not available")
    
    quality_data = cps_instance.get_surface_quality()
    return {
        "quality_data": quality_data,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/telemetry")
async def websocket_telemetry(websocket: WebSocket):
    """WebSocket endpoint for real-time telemetry streaming"""
    await connection_manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_text(json.dumps({
            "event": "connected",
            "timestamp": datetime.now().isoformat(),
            "message": "Connected to Manufacturing CPS telemetry stream"
        }))
        
        # Keep connection alive and broadcast system updates
        while True:
            try:
                if cps_instance:
                    # Get current system state
                    system_state = {
                        "event": "system_update",
                        "timestamp": datetime.now().isoformat(),
                        "status": orchestrator.system_state.status if orchestrator else "unknown",
                        "surface_quality": cps_instance.get_surface_quality(),
                        "process_parameters": asdict(cps_instance.process_parameters) if cps_instance.process_parameters else None,
                        "health_metrics": cps_instance.get_system_health()
                    }
                    
                    await websocket.send_text(json.dumps(system_state, default=str))
                
                # Send updates every 1 second
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error sending telemetry update: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        connection_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    if not orchestrator:
        return {"status": "unhealthy", "reason": "Orchestrator not initialized"}
    
    health_monitor = get_health_monitor()
    health_data = health_monitor.get_system_health()
    
    # Check if all critical components are healthy
    critical_components = ["manufacturing_cps", "robot_controller", "interferometer", "ai_controller"]
    unhealthy_components = [
        comp for comp in critical_components 
        if comp in health_data.get("components", {}) and 
        health_data["components"][comp]["status"] != "HEALTHY"
    ]
    
    if unhealthy_components:
        return {
            "status": "degraded",
            "unhealthy_components": unhealthy_components,
            "details": health_data
        }
    else:
        return {
            "status": "healthy",
            "details": health_data
        }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    )