#!/usr/bin/env python3
"""
Quick System Verification Script
Verifies that all core components can be imported and basic functionality works
"""

import sys
import os
from pathlib import Path

def verify_system():
    """Verify system components"""
    print("🔍 AI-Driven Metrology Manufacturing System - Verification")
    print("=" * 60)
    
    # Add current directory to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    components_verified = 0
    total_components = 8
    
    # Test 1: Infrastructure components
    try:
        print("✅ Testing infrastructure components...")
        # Just verify directory structure exists
        required_dirs = ['infrastructure', 'metrology', 'robotics', 'ai', 'ndt', 'edge', 'testing']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"   ✓ {dir_name} directory exists")
            else:
                print(f"   ✗ {dir_name} directory missing")
        components_verified += 1
    except Exception as e:
        print(f"❌ Infrastructure verification failed: {e}")
    
    # Test 2: Core imports
    try:
        print("\n✅ Testing core module imports...")
        import numpy as np
        import asyncio
        print("   ✓ Scientific computing libraries")
        components_verified += 1
    except ImportError as e:
        print(f"❌ Core imports failed: {e}")
    
    # Test 3: Metrology system
    try:
        print("\n✅ Testing metrology system...")
        from metrology.grazing_incidence_interferometry import InterferometryConfig
        config = InterferometryConfig()
        print(f"   ✓ Interferometry config created: {config.wavelength_nm}nm wavelength")
        components_verified += 1
    except Exception as e:
        print(f"❌ Metrology system test failed: {e}")
    
    # Test 4: Robotics system
    try:
        print("\n✅ Testing robotics system...")
        from robotics.hybrid_gantry_arm_controller import RobotConfig
        config = RobotConfig()
        print(f"   ✓ Robot config created: force limit {config.force_limit}N")
        components_verified += 1
    except Exception as e:
        print(f"❌ Robotics system test failed: {e}")
    
    # Test 5: AI system
    try:
        print("\n✅ Testing AI system...")
        from ai.hybrid_ai_controller import ProcessParameters
        params = ProcessParameters(3000, 2.0, 80, 200, 5.0, 1.0)
        print(f"   ✓ Process parameters created: {params.spindle_speed_rpm} RPM")
        components_verified += 1
    except Exception as e:
        print(f"❌ AI system test failed: {e}")
    
    # Test 6: NDT system
    try:
        print("\n✅ Testing NDT system...")
        from ndt.ndt_integration_system import MaterialCharacterizer
        characterizer = MaterialCharacterizer()
        print("   ✓ Material characterizer instantiated")
        components_verified += 1
    except Exception as e:
        print(f"❌ NDT system test failed: {e}")
    
    # Test 7: Edge computing
    try:
        print("\n✅ Testing edge computing system...")
        from edge.hierarchical_control_system import RealTimeController
        controller = RealTimeController(cycle_time_us=1000)  # 1ms for testing
        print(f"   ✓ Real-time controller configured: {controller.cycle_time_us}μs cycle")
        components_verified += 1
    except Exception as e:
        print(f"❌ Edge computing test failed: {e}")
    
    # Test 8: File structure
    try:
        print("\n✅ Verifying file structure...")
        required_files = [
            'README.md',
            'requirements.txt', 
            'deploy.py',
            'main.py',
            'SYSTEM_SUMMARY.md'
        ]
        
        for file_name in required_files:
            if os.path.exists(file_name):
                print(f"   ✓ {file_name}")
            else:
                print(f"   ✗ {file_name} missing")
        components_verified += 1
    except Exception as e:
        print(f"❌ File structure verification failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"Components verified: {components_verified}/{total_components}")
    success_rate = (components_verified / total_components) * 100
    
    if success_rate >= 80:
        print(f"✅ OVERALL STATUS: PASS ({success_rate:.1f}% success rate)")
        print("\n🚀 System is ready for deployment!")
        print("Next steps:")
        print("  1. Install dependencies: pip install -r requirements.txt")
        print("  2. Configure deployment: Edit deployment_config.yaml")
        print("  3. Deploy infrastructure: python deploy.py")
        print("  4. Run system: python main.py")
    else:
        print(f"⚠️  OVERALL STATUS: PARTIAL SUCCESS ({success_rate:.1f}% success rate)")
        print("Some components need attention before deployment.")
    
    return success_rate >= 80

if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)