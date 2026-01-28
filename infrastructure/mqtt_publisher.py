"""
MQTT Publisher for Manufacturing Data Engine (MDE)
Publishes telemetry data with asset context to EMQX broker
"""

import paho.mqtt.client as mqtt
import json
import time
import threading
from typing import Dict, Any
import logging
from datetime import datetime
import uuid


class MDEPublisher:
    """
    Manufacturing Data Engine publisher for contextual telemetry
    Publishes force data with asset ID (Block Serial #) context
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883, 
                 username: str = None, password: str = None):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.username = username
        self.password = password
        
        self.client = mqtt.Client()
        if username and password:
            self.client.username_pw_set(username, password)
        
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.connected = False
        self.logger = logging.getLogger(__name__)
        
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        else:
            self.logger.error(f"Failed to connect to MQTT broker, return code: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        self.connected = False
        self.logger.warning("Disconnected from MQTT broker")
    
    def _on_publish(self, client, userdata, mid):
        pass  # Successfully published
    
    def connect(self):
        """Connect to the MQTT broker"""
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10  # 10 seconds timeout
            start_time = time.time()
            while not self.connected and time.time() - start_time < timeout:
                time.sleep(0.1)
                
            if not self.connected:
                raise Exception("Could not connect to MQTT broker within timeout")
                
        except Exception as e:
            self.logger.error(f"Error connecting to MQTT broker: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the MQTT broker"""
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False
        self.logger.info("Disconnected from MQTT broker")
    
    def publish_force_telemetry(self, asset_id: str, force_data: Dict[str, Any]):
        """
        Publish force telemetry data with asset context
        
        Args:
            asset_id: Block Serial Number or Asset ID
            force_data: Dictionary containing force/torque data
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker, skipping telemetry publish")
            return False
        
        # Create telemetry payload with asset context
        telemetry_payload = {
            "asset_id": asset_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "telemetry_type": "force_torque",
            "data": force_data,
            "metadata": {
                "source": "ati_axia80",
                "unit": "newton_meter",
                "sample_rate": 7000  # 7kHz as per ATI Axia80 spec
            }
        }
        
        # Publish to MDE topic with asset context
        topic = f"mde/assets/{asset_id}/telemetry/force_torque"
        payload_json = json.dumps(telemetry_payload)
        
        try:
            result = self.client.publish(topic, payload_json, qos=1)
            self.logger.debug(f"Published force telemetry to {topic}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            self.logger.error(f"Error publishing force telemetry: {e}")
            return False
    
    def publish_surface_metrics(self, asset_id: str, metrics: Dict[str, Any]):
        """
        Publish surface quality metrics
        
        Args:
            asset_id: Block Serial Number or Asset ID
            metrics: Dictionary containing surface quality metrics
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker, skipping metrics publish")
            return False
        
        # Create metrics payload with asset context
        metrics_payload = {
            "asset_id": asset_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "telemetry_type": "surface_metrics",
            "data": metrics,
            "metadata": {
                "source": "grazing_incidence_interferometer",
                "unit": "nanometer",
                "measurement_quality": metrics.get("measurement_quality", "unknown")
            }
        }
        
        # Publish to MDE topic
        topic = f"mde/assets/{asset_id}/metrics/surface_quality"
        payload_json = json.dumps(metrics_payload)
        
        try:
            result = self.client.publish(topic, payload_json, qos=1)
            self.logger.debug(f"Published surface metrics to {topic}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            self.logger.error(f"Error publishing surface metrics: {e}")
            return False
    
    def publish_process_parameters(self, asset_id: str, params: Dict[str, Any]):
        """
        Publish process parameters
        
        Args:
            asset_id: Block Serial Number or Asset ID
            params: Dictionary containing process parameters
        """
        if not self.connected:
            self.logger.warning("Not connected to MQTT broker, skipping parameters publish")
            return False
        
        # Create parameters payload with asset context
        params_payload = {
            "asset_id": asset_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "telemetry_type": "process_parameters",
            "data": params,
            "metadata": {
                "source": "hybrid_ai_controller",
                "parameter_source": "realtime_optimization"
            }
        }
        
        # Publish to MDE topic
        topic = f"mde/assets/{asset_id}/parameters/process"
        payload_json = json.dumps(params_payload)
        
        try:
            result = self.client.publish(topic, payload_json, qos=1)
            self.logger.debug(f"Published process parameters to {topic}")
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            self.logger.error(f"Error publishing process parameters: {e}")
            return False


class MDETelemetryAggregator:
    """Aggregates and contextualizes telemetry data before publishing"""
    
    def __init__(self, publisher: MDEPublisher, asset_id: str):
        self.publisher = publisher
        self.asset_id = asset_id
        self.logger = logging.getLogger(__name__)
        
    def send_force_telemetry(self, fx: float, fy: float, fz: float, 
                           tx: float, ty: float, tz: float):
        """Send 6-axis force/torque data"""
        force_data = {
            "fx_newtons": fx,
            "fy_newtons": fy,
            "fz_newtons": fz,
            "tx_newton_meters": tx,
            "ty_newton_meters": ty,
            "tz_newton_meters": tz,
            "force_magnitude_newtons": (fx**2 + fy**2 + fz**2)**0.5,
            "torque_magnitude_nm": (tx**2 + ty**2 + tz**2)**0.5
        }
        
        return self.publisher.publish_force_telemetry(self.asset_id, force_data)
    
    def send_surface_metrics(self, metrics: Dict[str, Any]):
        """Send surface quality metrics"""
        return self.publisher.publish_surface_metrics(self.asset_id, metrics)
    
    def send_process_parameters(self, params: Dict[str, Any]):
        """Send process parameters"""
        return self.publisher.publish_process_parameters(self.asset_id, params)


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize MQTT publisher
    publisher = MDEPublisher(broker_host="localhost", broker_port=1883)
    
    try:
        # Connect to broker
        publisher.connect()
        
        # Create aggregator for specific asset
        asset_id = "GRANITE_BLOCK_S/N_001"
        aggregator = MDETelemetryAggregator(publisher, asset_id)
        
        # Simulate sending telemetry
        print("Sending sample telemetry data...")
        
        # Send force telemetry
        success = aggregator.send_force_telemetry(
            fx=0.5, fy=-0.3, fz=45.2,
            tx=0.01, ty=-0.02, tz=0.005
        )
        print(f"Force telemetry sent: {success}")
        
        # Send sample surface metrics
        sample_metrics = {
            "mean_height_nm": 0.2,
            "std_deviation_nm": 2.1,
            "peak_valley_nm": 8.5,
            "rms_roughness_nm": 1.8,
            "flatness_grade_00_deviation_nm": 4.2,
            "measurement_quality": "Good",
            "test_uncertainty_ratio": 6.2
        }
        
        success = aggregator.send_surface_metrics(sample_metrics)
        print(f"Surface metrics sent: {success}")
        
        # Send sample process parameters
        sample_params = {
            "spindle_speed_rpm": 3200,
            "feed_rate_mm_per_sec": 2.1,
            "down_force_n": 45.0,
            "abrasive_grit_size": 200,
            "coolant_flow_rate": 5.2,
            "dwell_time_sec": 1.5
        }
        
        success = aggregator.send_process_parameters(sample_params)
        print(f"Process parameters sent: {success}")
        
        # Keep alive for a bit to allow messages to be sent
        time.sleep(2)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Disconnect
        publisher.disconnect()