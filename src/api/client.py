import requests
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
from config.settings import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class APIResponse:
    """Wrapper for API responses"""
    success: bool
    data: Any
    status_code: int
    message: Optional[str] = None

class F1AnalyticsClient:
    """Client for F1 Analytics API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.logger = setup_logger(self.__class__.__name__)
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make HTTP request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, **kwargs)
            
            if response.status_code == 200:
                return APIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    status_code=response.status_code,
                    message=response.text
                )
                
        except Exception as e:
            self.logger.error(f"API request failed: {e}")
            return APIResponse(
                success=False,
                data=None,
                status_code=0,
                message=str(e)
            )
    
    def health_check(self) -> APIResponse:
        """Check API health"""
        return self._make_request("GET", "/health")
    
    def get_sessions(self) -> APIResponse:
        """Get list of available sessions"""
        return self._make_request("GET", "/api/v1/sessions")
    
    def get_session_summary(self, session_id: str) -> APIResponse:
        """Get session summary"""
        return self._make_request("GET", f"/api/v1/sessions/{session_id}/summary")
    
    def get_lap_times(self, session_id: str, driver_code: Optional[str] = None,
                     min_lap: Optional[int] = None, max_lap: Optional[int] = None,
                     limit: int = 1000) -> APIResponse:
        """Get lap times with filters"""
        params = {"limit": limit}
        if driver_code:
            params["driver_code"] = driver_code
        if min_lap is not None:
            params["min_lap"] = min_lap
        if max_lap is not None:
            params["max_lap"] = max_lap
        
        return self._make_request("GET", f"/api/v1/sessions/{session_id}/lap-times", params=params)
    
    def get_performance_analysis(self, session_id: str) -> APIResponse:
        """Get driver performance analysis"""
        return self._make_request("GET", f"/api/v1/sessions/{session_id}/performance")
    
    def get_performance_report(self, session_id: str) -> APIResponse:
        """Get comprehensive performance report"""
        return self._make_request("GET", f"/api/v1/sessions/{session_id}/performance/report")
    
    def get_quality_report(self, session_id: str) -> APIResponse:
        """Get data quality report"""
        return self._make_request("GET", f"/api/v1/sessions/{session_id}/quality")
    
    def train_lap_time_model(self, session_ids: List[str]) -> APIResponse:
        """Train lap time prediction model"""
        return self._make_request("POST", "/api/v1/models/train/lap-time", json=session_ids)
    
    def predict_lap_time(self, session_id: str, driver_code: str, lap_number: int,
                        tire_compound: str, tire_age: int) -> APIResponse:
        """Predict lap time"""
        data = {
            "session_id": session_id,
            "driver_code": driver_code,
            "lap_number": lap_number,
            "tire_compound": tire_compound,
            "tire_age": tire_age
        }
        return self._make_request("POST", "/api/v1/predictions/lap-time", json=data)
    
    def start_realtime_processing(self, session_id: str) -> APIResponse:
        """Start real-time processing"""
        return self._make_request("POST", f"/api/v1/realtime/{session_id}/start")
    
    def stop_realtime_processing(self, session_id: str) -> APIResponse:
        """Stop real-time processing"""
        return self._make_request("POST", f"/api/v1/realtime/{session_id}/stop")
    
    def get_live_leaderboard(self, session_id: str) -> APIResponse:
        """Get live leaderboard"""
        return self._make_request("GET", f"/api/v1/realtime/{session_id}/leaderboard")
    
    def get_driver_statistics(self) -> APIResponse:
        """Get driver statistics"""
        return self._make_request("GET", "/api/v1/stats/drivers")
    
    def get_circuit_statistics(self) -> APIResponse:
        """Get circuit statistics"""
        return self._make_request("GET", "/api/v1/stats/circuits")

# Usage example
if __name__ == "__main__":
    client = F1AnalyticsClient()
    
    # Test API connection
    health = client.health_check()
    if health.success:
        print("API is healthy!")
        print(health.data)
    else:
        print(f"API health check failed: {health.message}")
    
    # Get available sessions
    sessions = client.get_sessions()
    if sessions.success and sessions.data:
        print(f"\nAvailable sessions: {len(sessions.data)}")
        
        # Get summary for first session
        session_id = sessions.data[0]
        summary = client.get_session_summary(session_id)
        if summary.success:
            print(f"\nSession {session_id} summary:")
            print(f"Total laps: {summary.data['total_laps']}")
            print(f"Total drivers: {summary.data['total_drivers']}")
            print(f"Fastest lap: {summary.data['fastest_lap']:.3f}s")
