import subprocess
import sys
import time
import requests
from pathlib import Path
from config.settings import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def check_api_health(url: str = "http://localhost:8000", max_retries: int = 30) -> bool:
    """Check if API is running and healthy"""
    for i in range(max_retries):
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("API is healthy!")
                return True
        except requests.exceptions.ConnectionError:
            if i == 0:
                logger.info("Waiting for API to start...")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        
        time.sleep(2)
    
    return False

def launch_api():
    """Launch the FastAPI server"""
    logger.info("Starting FastAPI server...")
    
    api_script = Path(__file__).parent.parent / "src" / "api" / "main.py"
    
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "src.api.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ], cwd=Path(__file__).parent.parent)
    
    return process

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    logger.info("Starting Streamlit dashboard...")
    
    dashboard_script = Path(__file__).parent.parent / "src" / "dashboard" / "app.py"
    
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_script),
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])
    
    return process

def main():
    """Main launcher function"""
    print("🏎️  F1 Analytics Platform Launcher")
    print("=" * 50)
    
    try:
        # Launch API server
        api_process = launch_api()
        
        # Wait for API to be healthy
        if not check_api_health():
            logger.error("API failed to start properly")
            api_process.terminate()
            return
        
        # Launch dashboard
        dashboard_process = launch_dashboard()
        
        print("\n✅ Services started successfully!")
        print("📊 Dashboard: http://localhost:8501")
        print("🚀 API: http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop all services...")
        
        # Wait for user interrupt
        try:
            api_process.wait()
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            
            # Terminate processes
            dashboard_process.terminate()
            api_process.terminate()
            
            # Wait for graceful shutdown
            dashboard_process.wait(timeout=10)
            api_process.wait(timeout=10)
            
            print("✅ All services stopped")
    
    except Exception as e:
        logger.error(f"Error launching services: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
