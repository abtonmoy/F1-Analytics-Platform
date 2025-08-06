import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class F1SystemManager:
    def __init__(self):
        self.processes = {}
        self.running = True
        
    def start_database_init(self):
        """Initialize database"""
        logger.info("Initializing database...")
        result = subprocess.run([
            sys.executable, "-m", "src.database.schema"
        ], cwd=Path(__file__).parent.parent)
        
        if result.returncode == 0:
            logger.info("Database initialized successfully")
        else:
            logger.error("Database initialization failed")
            return False
        return True
    
    def start_etl_pipeline(self):
        """Start ETL pipeline for sample data"""
        logger.info("Running ETL pipeline for sample data...")
        
        # This would run a sample ETL job
        script = """
from src.data_pipeline.etl import F1ETLPipeline
pipeline = F1ETLPipeline(max_workers=1)
results = pipeline.run_incremental_update(2024, 'Monaco Grand Prix', ['R'])
print(f'ETL Results: {len([r for r in results if r.success])} successful jobs')
"""
        
        result = subprocess.run([
            sys.executable, "-c", script
        ], cwd=Path(__file__).parent.parent)
        
        return result.returncode == 0
    
    def start_api_server(self):
        """Start FastAPI server"""
        logger.info("Starting API server...")
        
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], cwd=Path(__file__).parent.parent)
        
        self.processes['api'] = process
        return process
    
    def start_dashboard(self):
        """Start Streamlit dashboard"""
        logger.info("Starting dashboard...")
        
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "src/dashboard/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], cwd=Path(__file__).parent.parent)
        
        self.processes['dashboard'] = process
        return process
    
    def check_health(self):
        """Check system health"""
        import requests
        
        try:
            # Check API
            response = requests.get("http://localhost:8000/health", timeout=5)
            api_healthy = response.status_code == 200
        except:
            api_healthy = False
        
        try:
            # Check dashboard (simplified)
            response = requests.get("http://localhost:8501", timeout=5)
            dashboard_healthy = response.status_code == 200
        except:
            dashboard_healthy = False
        
        return {
            'api': api_healthy,
            'dashboard': dashboard_healthy
        }
    
    def stop_all(self):
        """Stop all processes"""
        logger.info("Stopping all processes...")
        
        for name, process in self.processes.items():
            logger.info(f"Stopping {name}...")
            process.terminate()
            
            try:
                process.wait(timeout=10)
                logger.info(f"{name} stopped")
            except subprocess.TimeoutExpired:
                logger.warning(f"Force killing {name}...")
                process.kill()
                process.wait()
        
        self.processes.clear()
        self.running = False
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        logger.info("Received interrupt signal...")
        self.stop_all()
        sys.exit(0)
    
    def run_complete_system(self):
        """Run the complete F1 analytics system"""
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("🏎️  F1 Analytics Platform - Complete System")
        print("=" * 60)
        
        try:
            # Step 1: Initialize database
            if not self.start_database_init():
                logger.error("Failed to initialize database")
                return 1
            
            # Step 2: Load sample data (optional)
            print("\n📊 Loading sample data...")
            self.start_etl_pipeline()
            
            # Step 3: Start services
            print("\n🚀 Starting services...")
            
            # Start API server
            self.start_api_server()
            time.sleep(5)  # Wait for API to start
            
            # Start dashboard
            self.start_dashboard()
            time.sleep(5)  # Wait for dashboard to start
            
            # Step 4: Health check
            print("\n🔍 Checking system health...")
            health = self.check_health()
            
            print(f"API Server: {'✅' if health['api'] else '❌'}")
            print(f"Dashboard: {'✅' if health['dashboard'] else '❌'}")
            
            if all(health.values()):
                print("\n✅ All services are running!")
                print("🌐 Access Points:")
                print("   📊 Dashboard: http://localhost:8501")
                print("   🚀 API: http://localhost:8000")
                print("   📚 API Docs: http://localhost:8000/docs")
                
                print("\n🔧 System Commands:")
                print("   Press Ctrl+C to stop all services")
                print("   Check logs in the 'logs/' directory")
                
                # Keep running until interrupted
                while self.running:
                    time.sleep(10)
                    
                    # Periodic health check
                    health = self.check_health()
                    if not all(health.values()):
                        logger.warning("Some services are unhealthy")
                        for service, healthy in health.items():
                            if not healthy:
                                logger.warning(f"{service} is not responding")
            else:
                logger.error("Some services failed to start properly")
                self.stop_all()
                return 1
                
        except Exception as e:
            logger.error(f"System error: {e}")
            self.stop_all()
            return 1
        
        return 0

def main():
    """Main entry point"""
    manager = F1SystemManager()
    return manager.run_complete_system()

if __name__ == "__main__":
    sys.exit(main())
