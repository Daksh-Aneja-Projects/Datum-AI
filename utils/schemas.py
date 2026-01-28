"""
Schema validation using Pydantic for data integrity in the ManufacturingAI system
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import numpy as np
from datetime import datetime
import json


class ProcessParametersSchema(BaseModel):
    """Schema for manufacturing process parameters"""
    spindle_speed_rpm: float = Field(..., ge=0, le=10000, description="Spindle speed in RPM")
    feed_rate_mm_per_sec: float = Field(..., ge=0, le=100, description="Feed rate in mm/sec")
    down_force_n: float = Field(..., ge=0, le=500, description="Down force in Newtons")
    abrasive_grit_size: int = Field(..., ge=50, le=600, description="Abrasive grit size")
    coolant_flow_rate: float = Field(..., ge=0, le=20, description="Coolant flow rate in L/min")
    dwell_time_sec: float = Field(..., ge=0, le=60, description="Dwell time in seconds")
    
    class Config:
        extra = "forbid"


class MaterialPropertiesSchema(BaseModel):
    """Schema for granite material properties"""
    quartz_content: float = Field(..., ge=0, le=1, description="Quartz content fraction (0-1)")
    feldspar_content: float = Field(..., ge=0, le=1, description="Feldspar content fraction (0-1)")
    mica_content: float = Field(..., ge=0, le=1, description="Mica content fraction (0-1)")
    hardness_variation: float = Field(..., ge=0, le=1, description="Hardness variation coefficient")
    density_kg_per_m3: float = Field(..., ge=2000, le=3000, description="Density in kg/m³")
    thermal_conductivity: float = Field(..., ge=1, le=5, description="Thermal conductivity in W/(m·K)")
    
    @validator('quartz_content', 'feldspar_content', 'mica_content')
    def validate_composition_sum(cls, v):
        # Note: This validator can't access other fields, so we'll validate in a custom function
        return v
    
    class Config:
        extra = "forbid"


class ForceTorqueDataSchema(BaseModel):
    """Schema for force/torque sensor data"""
    fx: float = Field(..., description="Force X component in Newtons")
    fy: float = Field(..., description="Force Y component in Newtons")
    fz: float = Field(..., description="Force Z component in Newtons")
    tx: float = Field(..., description="Torque X component in Nm")
    ty: float = Field(..., description="Torque Y component in Nm")
    tz: float = Field(..., description="Torque Z component in Nm")
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    
    class Config:
        extra = "forbid"


class SurfaceHeightMapSchema(BaseModel):
    """Schema for surface height map data"""
    height_map: List[List[float]] = Field(..., description="2D array of height values in nanometers")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the measurement")
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    
    @validator('height_map')
    def validate_height_map(cls, v):
        if not v or not v[0]:
            raise ValueError('Height map cannot be empty')
        # Validate that it's a rectangular array
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError('Height map must be rectangular')
        return v
    
    class Config:
        extra = "forbid"


class AcousticEmissionDataSchema(BaseModel):
    """Schema for acoustic emission sensor data"""
    rms: float = Field(..., ge=0, description="Root Mean Square of AE signal")
    kurtosis: float = Field(..., ge=0, description="Kurtosis of AE signal")
    spectral_peaks_count: int = Field(..., ge=0, description="Number of spectral peaks")
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    asset_id: str = Field(..., description="Asset identifier")
    
    class Config:
        extra = "forbid"


class ManufacturingJobSchema(BaseModel):
    """Schema for manufacturing job specifications"""
    job_id: str = Field(..., description="Unique job identifier")
    target_quality: str = Field(..., description="Target quality standard (e.g., DIN_876_GRADE_00)")
    cycles: int = Field(..., ge=1, le=1000, description="Maximum number of cycles")
    process_parameters: ProcessParametersSchema = Field(..., description="Process parameters for the job")
    material_properties: MaterialPropertiesSchema = Field(..., description="Material properties")
    surface_grid: List[List[float]] = Field(..., description="Grid of surface points to process")
    dwell_times: List[float] = Field(..., description="Dwell times for each point")
    
    @validator('surface_grid')
    def validate_surface_grid(cls, v):
        if not v:
            raise ValueError('Surface grid cannot be empty')
        for point in v:
            if len(point) != 3:  # x, y, z coordinates
                raise ValueError('Each point must have 3 coordinates (x, y, z)')
        return v
    
    @validator('dwell_times')
    def validate_dwell_times(cls, v, values):
        if 'surface_grid' in values and len(v) != len(values['surface_grid']):
            raise ValueError('Dwell times must match the number of surface points')
        return v
    
    class Config:
        extra = "forbid"


class SystemHealthSchema(BaseModel):
    """Schema for system health monitoring"""
    timestamp: datetime = Field(..., description="Timestamp of health check")
    components: Dict[str, Dict[str, Any]] = Field(..., description="Status of system components")
    overall_status: str = Field(..., description="Overall system status")
    asset_id: str = Field(..., description="Asset identifier")
    
    class Config:
        extra = "forbid"


def validate_process_parameters(data: Dict[str, Any]) -> ProcessParametersSchema:
    """Validate process parameters using schema"""
    return ProcessParametersSchema(**data)


def validate_material_properties(data: Dict[str, Any]) -> MaterialPropertiesSchema:
    """Validate material properties using schema"""
    return MaterialPropertiesSchema(**data)


def validate_force_torque_data(data: Dict[str, Any]) -> ForceTorqueDataSchema:
    """Validate force/torque data using schema"""
    return ForceTorqueDataSchema(**data)


def validate_surface_height_map(data: Dict[str, Any]) -> SurfaceHeightMapSchema:
    """Validate surface height map using schema"""
    return SurfaceHeightMapSchema(**data)


def validate_acoustic_emission_data(data: Dict[str, Any]) -> AcousticEmissionDataSchema:
    """Validate acoustic emission data using schema"""
    return AcousticEmissionDataSchema(**data)


def validate_manufacturing_job(data: Dict[str, Any]) -> ManufacturingJobSchema:
    """Validate manufacturing job using schema"""
    return ManufacturingJobSchema(**data)


def validate_system_health(data: Dict[str, Any]) -> SystemHealthSchema:
    """Validate system health data using schema"""
    return SystemHealthSchema(**data)


class InterferometryReadingSchema(BaseModel):
    """Schema for interferometry measurements with validation for ultra-precision manufacturing"""
    # Enforce array size for a 100x100mm surface map with 1024 data points
    surface_map: List[float] = Field(..., min_items=1024, max_items=1048576, description="Surface height map data (flattened 2D array)")
    
    # Statistical measures with physical constraints
    flatness_rms_nm: float = Field(..., gt=0, lt=5000, description="RMS flatness in nanometers (must be positive and < 5 microns)")
    flatness_pv_nm: float = Field(..., gt=0, lt=10000, description="Peak-to-valley flatness in nanometers (must be positive and < 10 microns)")
    
    # Wavelength and optical parameters
    wavelength_nm: float = Field(..., ge=400, le=1000, description="Laser wavelength in nanometers")
    grazing_angle_deg: float = Field(..., ge=85, le=89.9, description="Grazing incidence angle in degrees")
    
    # Measurement metadata
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    measurement_id: str = Field(..., description="Unique measurement identifier")
    measurement_duration_ms: float = Field(..., ge=0, description="Duration of measurement in milliseconds")
    
    # Quality metrics
    interference_contrast: float = Field(..., ge=0, le=1, description="Contrast ratio of interference fringes")
    phase_precision_rad: float = Field(..., gt=0, lt=0.1, description="Phase measurement precision in radians")
    
    # Environmental conditions
    temperature_celsius: float = Field(..., ge=-10, le=50, description="Ambient temperature in Celsius")
    humidity_percent: float = Field(..., ge=0, le=100, description="Relative humidity percentage")
    atmospheric_pressure_kpa: float = Field(..., ge=80, le=120, description="Atmospheric pressure in kPa")
    
    # Equipment parameters
    objective_magnification: float = Field(..., ge=1, le=100, description="Objective lens magnification")
    numerical_aperture: float = Field(..., ge=0.1, le=1.0, description="Numerical aperture of objective")
    
    @validator('surface_map')
    def validate_surface_map_values(cls, v):
        """Validate that surface map values are within reasonable physical bounds"""
        if v:
            # Check for extreme outliers (more than 100 microns deviation)
            for value in v:
                if abs(value) > 100000:  # 100 microns in nm
                    raise ValueError(f'Surface map value {value}nm exceeds physical bounds (±100 microns)')
        return v
    
    @validator('flatness_rms_nm', 'flatness_pv_nm')
    def validate_flatness_bounds(cls, v):
        """Ensure flatness values are physically reasonable"""
        if v <= 0:
            raise ValueError(f'Flatness must be positive, got {v}')
        return v
    
    class Config:
        extra = "forbid"


class GrazingIncidenceInterferometrySchema(BaseModel):
    """Schema for grazing incidence interferometry specific measurements"""
    measurement_type: str = Field("grazing_incidence", const=True, description="Type of interferometry measurement")
    
    # Core interferometry data
    interferogram: List[List[float]] = Field(..., min_items=100, max_items=2048, description="Raw interferogram data")
    phase_map: List[List[float]] = Field(..., min_items=100, max_items=2048, description="Extracted phase map")
    amplitude_map: List[List[float]] = Field(..., min_items=100, max_items=2048, description="Amplitude map")
    
    # Interferometry-specific parameters
    incident_angle_rad: float = Field(..., ge=1.5, le=1.57, description="Incident angle in radians (close to 90° for grazing incidence)")
    polarization_state: str = Field(..., description="Polarization state (s, p, or mixed)")
    coherence_length_um: float = Field(..., ge=1, le=100, description="Coherence length in micrometers")
    
    # Calibration data
    calibration_matrix: List[List[float]] = Field(..., min_items=3, max_items=10, description="Calibration matrix")
    reference_surface: List[List[float]] = Field(..., min_items=100, max_items=2048, description="Reference surface map")
    
    # Analysis results
    surface_profile: List[List[float]] = Field(..., min_items=100, max_items=2048, description="Reconstructed surface profile")
    surface_roughness_nm_rms: float = Field(..., ge=0, le=100, description="RMS surface roughness in nanometers")
    surface_roughness_nm_ra: float = Field(..., ge=0, le=100, description="Ra surface roughness in nanometers")
    
    # Quality assessment
    fringe_density_avg: float = Field(..., ge=0, description="Average fringe density")
    fringe_visibility: float = Field(..., ge=0, le=1, description="Average fringe visibility")
    reconstruction_quality: float = Field(..., ge=0, le=1, description="Quality metric for surface reconstruction")
    
    timestamp: datetime = Field(..., description="Timestamp of measurement")
    measurement_id: str = Field(..., description="Unique measurement identifier")
    
    @validator('interferogram', 'phase_map', 'amplitude_map', 'surface_profile')
    def validate_2d_maps_shape(cls, v):
        """Validate that 2D maps have consistent dimensions"""
        if not v or not v[0]:
            raise ValueError('2D maps cannot be empty')
        # Validate that it's a rectangular array
        row_len = len(v[0])
        for row in v:
            if len(row) != row_len:
                raise ValueError('2D maps must be rectangular')
        return v
    
    @validator('incident_angle_rad')
    def validate_grazing_incidence_angle(cls, v):
        """Validate that the angle is appropriate for grazing incidence"""
        if v < 1.5 or v > 1.57:  # ~85-90 degrees in radians
            raise ValueError(f'Grazing incidence angle {v}rad is outside valid range (1.5-1.57 rad)')
        return v
    
    class Config:
        extra = "forbid"


class AERealtimeMetricsSchema(BaseModel):
    """Schema for real-time acoustic emission metrics"""
    timestamp_ns: int = Field(..., gt=0, description="Timestamp in nanoseconds")
    rms_value: float = Field(..., ge=0, description="RMS value of AE signal")
    kurtosis_value: float = Field(..., ge=0, description="Kurtosis of AE signal")
    spectral_peaks: List[Dict[str, float]] = Field(..., description="List of spectral peaks")
    dominant_frequency: float = Field(..., ge=0, description="Dominant frequency in Hz")
    energy_band_power: Dict[str, float] = Field(..., description="Energy distribution across frequency bands")
    data_overruns: int = Field(0, ge=0, description="Count of data overruns")
    processing_latency_us: int = Field(0, ge=0, description="Processing latency in microseconds")
    is_valid: bool = Field(True, description="Validity flag")
    
    @validator('spectral_peaks')
    def validate_spectral_peaks(cls, v):
        """Validate spectral peaks structure"""
        for peak in v:
            if 'frequency' not in peak or 'power' not in peak:
                raise ValueError('Each spectral peak must have frequency and power keys')
            if peak['frequency'] < 0:
                raise ValueError(f'Spectral peak frequency must be non-negative: {peak["frequency"]}')
            if peak['power'] < 0:
                raise ValueError(f'Spectral peak power must be non-negative: {peak["power"]}')
        return v
    
    @validator('energy_band_power')
    def validate_energy_bands(cls, v):
        """Validate energy band structure"""
        required_bands = ['low', 'mid', 'high']
        for band in required_bands:
            if band not in v:
                raise ValueError(f'Missing required energy band: {band}')
            if v[band] < 0:
                raise ValueError(f'Energy band {band} must be non-negative: {v[band]}')
        return v
    
    class Config:
        extra = "forbid"


def validate_interferometry_reading(data: Dict[str, Any]) -> InterferometryReadingSchema:
    """Validate interferometry reading using schema"""
    return InterferometryReadingSchema(**data)


def validate_grazing_incidence_interferometry(data: Dict[str, Any]) -> GrazingIncidenceInterferometrySchema:
    """Validate grazing incidence interferometry data using schema"""
    return GrazingIncidenceInterferometrySchema(**data)


def validate_ae_realtime_metrics(data: Dict[str, Any]) -> AERealtimeMetricsSchema:
    """Validate AE real-time metrics using schema"""
    return AERealtimeMetricsSchema(**data)