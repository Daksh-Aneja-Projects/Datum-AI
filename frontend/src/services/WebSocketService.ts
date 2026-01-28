import { io, Socket } from 'socket.io-client';

// Define TypeScript interfaces for the WebSocket payloads
interface TelemetryData {
  event: string;
  timestamp: string;
  status?: string;
  surface_quality?: any;
  process_parameters?: any;
  health_metrics?: any;
}

interface SystemUpdateData {
  event: 'system_update';
  timestamp: string;
  status: string;
  surface_quality: any;
  process_parameters: any;
  health_metrics: any;
}

interface ConnectedData {
  event: 'connected';
  timestamp: string;
  message: string;
}

interface DisconnectedData {
  event: 'disconnected';
  reason: string;
}

interface EmergencyStopData {
  event: 'emergency_stop';
  timestamp: string;
  message: string;
  triggered_by?: string;
}

interface ConfigurationUpdateData {
  event: 'configuration_updated';
  timestamp: string;
  config_filename: string;
}

class WebSocketService {
  private static instance: WebSocketService;
  private socket: Socket | null = null;
  private listeners: Map<string, Function[]> = new Map();
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 10;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private reconnectDelay: number = 1000; // Initial delay in ms
  private maxReconnectDelay: number = 30000; // Maximum delay in ms
  private reconnectJitter: number = 1000; // Jitter in ms
  
  // Heartbeat mechanism
  private heartbeatInterval: ReturnType<typeof setInterval> | null = null;
  private heartbeatTimeout: ReturnType<typeof setTimeout> | null = null;
  private lastPongReceived: number = Date.now();
  private heartbeatIntervalMs: number = 5000; // 5 seconds
  private heartbeatTimeoutMs: number = 10000; // 10 seconds timeout
  private pingPayload: any = { event: 'ping', timestamp: new Date().toISOString() };

  private constructor() {}

  public static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }

  public connect(token?: string): void {
    // Get API URL from environment or default to localhost
    const apiUrl = (window as any)._env_?.REACT_APP_API_URL || 'http://localhost:8000';
    
    // Build connection options with authentication token if provided
    const connectOptions: any = {
      transports: ['websocket'],
      path: '/ws/telemetry',
      // Reconnection options with exponential backoff
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
      reconnectionDelayMax: this.maxReconnectDelay,
      randomizationFactor: 0.5, // Jitter factor for reconnection delays
      timeout: 20000
    };
    
    // Add auth token to connection if provided
    if (token) {
      connectOptions.auth = { token };
    }
    
    // Connect to the FastAPI WebSocket endpoint
    this.socket = io(apiUrl, connectOptions);

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0; // Reset attempts on successful connection
      this.lastPongReceived = Date.now();
      this.startHeartbeat();
      this.emit('connected', {});
    });

    this.socket.on('disconnect', (reason: string) => {
      console.log('WebSocket disconnected:', reason);
      this.stopHeartbeat();
      this.emit('disconnected', { reason });
      
      // Attempt to reconnect with exponential backoff if not manually disconnected
      if (reason !== 'io client disconnect') {
        this.scheduleReconnect();
      }
    });

    // Listen for pong responses (heartbeat)
    this.socket.on('pong', () => {
      this.lastPongReceived = Date.now();
      console.debug('Pong received - connection alive');
      this.clearHeartbeatTimeout();
    });
    
    // Listen for telemetry updates
    this.socket.on('message', (data: string) => {
      try {
        const parsedData = JSON.parse(data);
        this.emit(parsedData.event || 'data', parsedData);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    });

    this.socket.on('error', (error: any) => {
      console.error('WebSocket error:', error);
    });
    
    // Handle reconnection events
    this.socket.on('reconnect_attempt', (attemptNumber: number) => {
      console.log(`Reconnection attempt #${attemptNumber}`);
    });
    
    this.socket.on('reconnect_failed', () => {
      console.error('Reconnection failed after max attempts');
      this.emit('reconnect_failed', {});
    });
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('reconnect_failed', {});
      return;
    }

    // Calculate delay with exponential backoff and jitter
    const baseDelay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts), this.maxReconnectDelay);
    // Add jitter to prevent thundering herd
    const jitter = Math.random() * this.reconnectJitter;
    const delay = baseDelay + jitter;

    console.log(`Scheduling reconnection attempt ${this.reconnectAttempts + 1} in ${delay}ms`);
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    
    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++;
      this.connect();
    }, delay);
  }

  private startHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
    }
    
    this.heartbeatInterval = setInterval(() => {
      if (this.socket && this.socket.connected) {
        // Send ping
        this.socket.emit('message', JSON.stringify(this.pingPayload));
        console.debug('Ping sent');
        
        // Set timeout for pong response
        this.setHeartbeatTimeout();
      }
    }, this.heartbeatIntervalMs);
  }
  
  private handleConnectionFailure(error: any): void {
    console.error('WebSocket connection failed:', error);
    this.emit('connection_failed', { error: error });
    
    // Reset connection state
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    
    // Attempt reconnection
    this.scheduleReconnect();
  }
  
  public getConnectionStatus(): { connected: boolean; reconnectAttempts: number; lastPongTime: number; latency: number } {
    return {
      connected: this.isConnected(),
      reconnectAttempts: this.reconnectAttempts,
      lastPongTime: this.getLastPongTime(),
      latency: this.getConnectionLatency()
    };
  }
  
  public forceReconnect(): void {
    console.log('Forcing reconnection...');
    
    // Clear any scheduled reconnection
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    // Reset reconnect attempts
    this.resetReconnectAttempts();
    
    // Disconnect current socket if exists
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    
    // Start fresh connection
    this.connect();
  }
  
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    this.clearHeartbeatTimeout();
  }
  
  private setHeartbeatTimeout(): void {
    this.clearHeartbeatTimeout();
    this.heartbeatTimeout = setTimeout(() => {
      const timeSinceLastPong = Date.now() - this.lastPongReceived;
      if (timeSinceLastPong > this.heartbeatTimeoutMs) {
        console.warn(`Heartbeat timeout - no pong received for ${timeSinceLastPong}ms`);
        this.handleHeartbeatFailure();
      }
    }, this.heartbeatTimeoutMs);
  }
  
  private clearHeartbeatTimeout(): void {
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  }
  
  private handleHeartbeatFailure(): void {
    console.error('Heartbeat failure detected - forcing disconnection');
    this.emit('heartbeat_timeout', {
      lastPongReceived: this.lastPongReceived,
      currentTime: Date.now()
    });
    
    // Force disconnection to trigger reconnection logic
    if (this.socket) {
      this.socket.disconnect();
    }
  }
  
  public disconnect(): void {
    // Stop heartbeat
    this.stopHeartbeat();
    
    // Clear any scheduled reconnection
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  public on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event)?.push(callback);
  }

  public off(event: string, callback: Function): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  public isConnected(): boolean {
    return this.socket?.connected || false;
  }
  
  public getReconnectAttempts(): number {
    return this.reconnectAttempts;
  }
  
  public resetReconnectAttempts(): void {
    this.reconnectAttempts = 0;
  }
  
  public getLastPongTime(): number {
    return this.lastPongReceived;
  }
  
  public getConnectionLatency(): number {
    return Date.now() - this.lastPongReceived;
  }
}

export default WebSocketService;