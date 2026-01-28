import React, { useRef, useEffect } from 'react';
import * as THREE from 'three';

interface SurfaceVisualizerProps {
  surfaceData: any;
}

const SurfaceVisualizer: React.FC<SurfaceVisualizerProps> = ({ surfaceData }) => {
  const mountRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!mountRef.current) return;

    // Scene setup
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    
    renderer.setSize(window.innerWidth * 0.8, window.innerHeight * 0.6);
    renderer.setClearColor(0x1a1a1a);
    mountRef.current.appendChild(renderer.domElement);

    // Create surface geometry
    const createSurfaceGeometry = () => {
      if (!surfaceData?.metrics) return null;

      const width = 50;
      const height = 50;
      const segments = 32;
      
      const geometry = new THREE.PlaneGeometry(width, height, segments, segments);
      const vertices = geometry.attributes.position.array as Float32Array;
      
      // Modify z-coordinates based on surface data
      const errorMap = generateSurfaceErrorMap(segments + 1);
      for (let i = 0; i <= segments; i++) {
        for (let j = 0; j <= segments; j++) {
          const idx = (i * (segments + 1) + j) * 3;
          const errorValue = errorMap[i][j];
          vertices[idx + 2] = errorValue * 5; // Scale for visualization
        }
      }
      
      geometry.attributes.position.needsUpdate = true;
      geometry.computeVertexNormals();
      
      return geometry;
    };

    const generateSurfaceErrorMap = (size: number) => {
      // Generate simulated surface error data in micrometers
      const map = [];
      for (let i = 0; i < size; i++) {
        const row = [];
        for (let j = 0; j < size; j++) {
          // Create realistic surface error pattern in micrometers
          const x = (i / size) * 10 - 5;
          const y = (j / size) * 10 - 5;
          
          // Combine multiple wave patterns to simulate realistic granite surface variations
          // Some areas will be within tolerance (<= 3.0 µm), others will exceed it
          const error = Math.sin(x) * Math.cos(y) * 1.5 +  // Base variation
                       Math.sin(x * 2) * Math.sin(y * 2) * 0.8 +  // Secondary pattern
                       (Math.random() - 0.5) * 1.0 +  // Random noise
                       Math.sin(x * 0.5) * Math.cos(y * 0.5) * 1.2;  // Large-scale variation
          
          row.push(error);
        }
        map.push(row);
      }
      return map;
    };

    // Create mesh
    const geometry = createSurfaceGeometry();
    if (geometry) {
      // Create vertex colors based on error values for heatmap visualization
      const errorMap = generateSurfaceErrorMap(33); // 32 segments + 1
      const colors = [];
      
      // DIN 876 Grade 00 tolerance is 3.0 µm (micrometers)
      const din876Grade00Tolerance = 3.0; // micrometers
      
      for (let i = 0; i <= 32; i++) {
        for (let j = 0; j <= 32; j++) {
          const errorValue = Math.abs(errorMap[i][j]);
          
          // Normalize error value relative to DIN 876 Grade 00 tolerance
          // Values within tolerance (blue/green), values exceeding tolerance (red/yellow)
          const normalizedError = Math.min(errorValue / din876Grade00Tolerance, 1.0);
          
          // Create heatmap colors: Blue -> Green -> Yellow -> Red
          let r, g, b;
          
          if (normalizedError <= 0.5) {
            // Blue to Green transition
            r = 0;
            g = Math.floor(255 * (normalizedError * 2));
            b = Math.floor(255 * (1 - normalizedError * 2));
          } else {
            // Green to Red transition
            r = Math.floor(255 * ((normalizedError - 0.5) * 2));
            g = Math.floor(255 * (1 - (normalizedError - 0.5) * 2));
            b = 0;
          }
          
          colors.push(r / 255, g / 255, b / 255);
        }
      }
      
      // Apply colors to geometry
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
      
      const material = new THREE.MeshStandardMaterial({
        vertexColors: true,
        wireframe: false,
        roughness: 0.3,
        metalness: 0.7
      });
      
      const mesh = new THREE.Mesh(geometry, material);
      scene.add(mesh);

      // Add lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
      scene.add(ambientLight);

      const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
      directionalLight.position.set(5, 10, 7);
      scene.add(directionalLight);

      // Position camera
      camera.position.z = 40;
      camera.position.y = 20;
      camera.lookAt(0, 0, 0);

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate);
        
        // Rotate surface slowly
        if (mesh) {
          mesh.rotation.x = Date.now() * 0.0001;
          mesh.rotation.z = Date.now() * 0.0002;
        }
        
        renderer.render(scene, camera);
      };

      animate();

      // Handle window resize
      const handleResize = () => {
        if (!mountRef.current) return;
        camera.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
      };

      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => {
        window.removeEventListener('resize', handleResize);
        if (mountRef.current && renderer.domElement) {
          mountRef.current.removeChild(renderer.domElement);
        }
        geometry.dispose();
        material.dispose();
        // Dispose of color attributes
        if (geometry.getAttribute('color')) {
          geometry.getAttribute('color').dispose();
        }
      };
    }
  }, [surfaceData]);

  return (
    <div className="surface-visualizer">
      <h2>Surface Topography Visualization</h2>
      <div ref={mountRef} className="three-container"></div>
      {surfaceData && (
        <div className="surface-metrics">
          <h3>Surface Metrics</h3>
          <p>PV Error: {surfaceData.metrics?.pv_error_nm?.toFixed(2)} nm</p>
          <p>Local Slope: {surfaceData.metrics?.avg_gradient?.toFixed(4)}</p>
          <p>Quality Score: {surfaceData.quality_score?.toFixed(2)}</p>
          <p>Status: {surfaceData.is_acceptable ? 'Acceptable' : 'Needs Improvement'}</p>
        </div>
      )}
      <div className="heatmap-legend">
        <h4>DIN 876 Grade 00 Tolerance: ≤3.0 µm</h4>
        <div className="legend-colors">
          <span style={{ color: 'blue' }}>≤1.5µm</span>
          <span style={{ color: 'green' }}>≤2.25µm</span>
          <span style={{ color: 'yellow' }}>≤3.0µm</span>
          <span style={{ color: 'red' }}>&gt;3.0µm</span>
        </div>
      </div>
    </div>
  );
};

export default SurfaceVisualizer;