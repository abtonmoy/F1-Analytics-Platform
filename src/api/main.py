from fastapi import FastAPI, HTTPException, Depends, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import asyncio
from contextlib import asynccontextmanager

# Fixed imports with proper error handling
try:
    from config.settings import Config
except ImportError:
    print("Warning: Could not import Config from config.settings")
    Config = None

try:
    from src.utils.logger import setup_logger
except ImportError:
    print("Warning: Could not import setup_logger, using basic logging")
    import logging
    def setup_logger(name):
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)

try:
    from src.database.connection import db_pool
except ImportError:
    print("Warning: Could not import db_pool")
    db_pool = None

try:
    from src.analytics.performance import PerformanceAnalyzer
except ImportError:
    print("Warning: Could not import PerformanceAnalyzer")
    class PerformanceAnalyzer:
        def analyze_driver_performance(self, session_id):
            return []
        def generate_performance_report(self, session_id):
            return {}

try:
    from src.data_pipeline.quality import DataQualityChecker
except ImportError:
    print("Warning: Could not import DataQualityChecker")
    class DataQualityChecker:
        def generate_quality_report(self, session_id):
            return {}

try:
    from src.models.predictive import F1PredictiveModels
except ImportError:
    print("Warning: Could not import F1PredictiveModels")
    class F1PredictiveModels:
        def train_lap_time_model(self, session_ids):
            return {}
        def predict_lap_time(self, session_id, driver_code, lap_number, tire_compound, tire_age):
            return None

try:
    from src.streaming.realtime_processor import RealTimeProcessor
except ImportError:
    print("Warning: Could not import RealTimeProcessor")
    class RealTimeProcessor:
        def __init__(self, session_id):
            self.session_id = session_id
        def start_processing(self):
            pass
        def stop_processing(self):
            pass
        def get_live_leaderboard(self):
            return []

logger = setup_logger(__name__)

# Global instances
performance_analyzer = PerformanceAnalyzer()
quality_checker = DataQualityChecker()
ml_models = F1PredictiveModels()
realtime_processors = {}  # session_id -> processor

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting F1 Analytics API")
    yield
    logger.info("Shutting down F1 Analytics API")
    
    # Clean up real-time processors
    for processor in realtime_processors.values():
        processor.stop_processing()

app = FastAPI(
    title="F1 Analytics Platform API",
    description="Professional F1 data analytics and prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response models (Pydantic models)
from pydantic import BaseModel
from typing import Union

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class SessionSummary(BaseModel):
    session_id: str
    total_laps: int
    total_drivers: int
    fastest_lap: float
    avg_lap_time: float
    deleted_laps: int

class DriverPerformance(BaseModel):
    driver_code: str
    avg_lap_time: float
    pace_score: float
    consistency_score: float
    tire_management: float

class QualityReport(BaseModel):
    session_id: str
    overall_status: str
    quality_score: float
    total_checks: int
    passed_checks: int

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

# Session endpoints
@app.get("/api/v1/sessions", response_model=List[str])
async def get_sessions():
    """Get list of available sessions"""
    try:
        if not db_pool:
            return []
            
        with db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT session_id 
                FROM sessions 
                ORDER BY session_id DESC
            """)
            sessions = [row['session_id'] for row in cursor.fetchall()]
            return sessions
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sessions/{session_id}/summary", response_model=SessionSummary)
async def get_session_summary(
    session_id: str = Path(..., description="Session ID")
):
    """Get session summary statistics"""
    try:
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
            
        with db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_laps,
                    COUNT(DISTINCT driver_code) as total_drivers,
                    MIN(lap_time) as fastest_lap,
                    AVG(lap_time) as avg_lap_time,
                    COUNT(CASE WHEN deleted = 1 THEN 1 END) as deleted_laps
                FROM lap_times
                WHERE session_id = ? AND lap_time IS NOT NULL
            """, (session_id,))
            
            result = cursor.fetchone()
            
            if not result or result['total_laps'] == 0:
                raise HTTPException(status_code=404, detail="Session not found or no data")
            
            return SessionSummary(
                session_id=session_id,
                total_laps=result['total_laps'],
                total_drivers=result['total_drivers'],
                fastest_lap=result['fastest_lap'],
                avg_lap_time=result['avg_lap_time'],
                deleted_laps=result['deleted_laps']
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sessions/{session_id}/lap-times")
async def get_lap_times(
    session_id: str = Path(..., description="Session ID"),
    driver_code: Optional[str] = Query(None, description="Filter by driver code"),
    min_lap: Optional[int] = Query(None, description="Minimum lap number"),
    max_lap: Optional[int] = Query(None, description="Maximum lap number"),
    limit: int = Query(1000, description="Maximum number of records")
):
    """Get lap times for a session with optional filters"""
    try:
        if not db_pool:
            raise HTTPException(status_code=503, detail="Database not available")
            
        query = """
            SELECT * FROM lap_times
            WHERE session_id = ? AND COALESCE(deleted, 0) = 0
        """
        params = [session_id]
        
        if driver_code:
            query += " AND driver_code = ?"
            params.append(driver_code)
        
        if min_lap is not None:
            query += " AND lap_number >= ?"
            params.append(min_lap)
        
        if max_lap is not None:
            query += " AND lap_number <= ?"
            params.append(max_lap)
        
        query += " ORDER BY lap_number, driver_code LIMIT ?"
        params.append(limit)
        
        with db_pool.get_connection() as conn:
            cursor = conn.execute(query, params)
            lap_times = [dict(row) for row in cursor.fetchall()]
            
            return {
                "session_id": session_id,
                "total_records": len(lap_times),
                "lap_times": lap_times
            }
            
    except Exception as e:
        logger.error(f"Error getting lap times: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Performance analysis endpoints
@app.get("/api/v1/sessions/{session_id}/performance", response_model=List[DriverPerformance])
async def get_performance_analysis(
    session_id: str = Path(..., description="Session ID")
):
    """Get driver performance analysis for a session"""
    try:
        driver_metrics = performance_analyzer.analyze_driver_performance(session_id)
        
        if not driver_metrics:
            raise HTTPException(status_code=404, detail="No performance data found")
        
        return [
            DriverPerformance(
                driver_code=metrics.entity_id,
                avg_lap_time=metrics.avg_lap_time,
                pace_score=metrics.pace_score,
                consistency_score=metrics.consistency_score,
                tire_management=metrics.tire_management
            )
            for metrics in driver_metrics
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/sessions/{session_id}/performance/report")
async def get_performance_report(
    session_id: str = Path(..., description="Session ID")
):
    """Get comprehensive performance report"""
    try:
        report = performance_analyzer.generate_performance_report(session_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Unable to generate performance report")
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Data quality endpoints
@app.get("/api/v1/sessions/{session_id}/quality", response_model=QualityReport)
async def get_quality_report(
    session_id: str = Path(..., description="Session ID")
):
    """Get data quality report for a session"""
    try:
        report = quality_checker.generate_quality_report(session_id)
        
        if not report:
            raise HTTPException(status_code=404, detail="Unable to generate quality report")
        
        return QualityReport(
            session_id=report['session_id'],
            overall_status=report['overall_status'],
            quality_score=report['quality_score'],
            total_checks=report['total_checks'],
            passed_checks=report['passed_checks']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating quality report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Machine learning endpoints
@app.post("/api/v1/models/train/lap-time")
async def train_lap_time_model(
    session_ids: List[str]
):
    """Train lap time prediction model"""
    try:
        if not session_ids:
            raise HTTPException(status_code=400, detail="At least one session ID required")
        
        # Run training in background task (simplified for demo)
        performances = ml_models.train_lap_time_model(session_ids)
        
        return {
            "message": "Model training completed",
            "session_ids": session_ids,
            "model_performances": {
                name: {
                    "mae": perf.mae,
                    "rmse": perf.rmse,
                    "r2_score": perf.r2_score,
                    "cv_score": perf.cv_score
                }
                for name, perf in performances.items()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/predictions/lap-time")
async def predict_lap_time(
    session_id: str,
    driver_code: str,
    lap_number: int,
    tire_compound: str,
    tire_age: int
):
    """Predict lap time for given conditions"""
    try:
        prediction = ml_models.predict_lap_time(
            session_id, driver_code, lap_number, tire_compound, tire_age
        )
        
        if not prediction:
            raise HTTPException(status_code=400, detail="Unable to generate prediction")
        
        return {
            "session_id": session_id,
            "driver_code": driver_code,
            "predicted_lap_time": prediction.predicted_value,
            "confidence_interval": prediction.confidence_interval,
            "model_name": prediction.model_name,
            "feature_importance": prediction.feature_importance
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting lap time: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Real-time endpoints
@app.post("/api/v1/realtime/{session_id}/start")
async def start_realtime_processing(
    session_id: str = Path(..., description="Session ID")
):
    """Start real-time processing for a session"""
    try:
        if session_id in realtime_processors:
            return {"message": "Real-time processing already active", "session_id": session_id}
        
        processor = RealTimeProcessor(session_id)
        processor.start_processing()
        realtime_processors[session_id] = processor
        
        return {"message": "Real-time processing started", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"Error starting real-time processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/realtime/{session_id}/stop")
async def stop_realtime_processing(
    session_id: str = Path(..., description="Session ID")
):
    """Stop real-time processing for a session"""
    try:
        if session_id not in realtime_processors:
            raise HTTPException(status_code=404, detail="No active real-time processing found")
        
        processor = realtime_processors[session_id]
        processor.stop_processing()
        del realtime_processors[session_id]
        
        return {"message": "Real-time processing stopped", "session_id": session_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping real-time processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/realtime/{session_id}/leaderboard")
async def get_live_leaderboard(
    session_id: str = Path(..., description="Session ID")
):
    """Get live leaderboard for a session"""
    try:
        if session_id not in realtime_processors:
            raise HTTPException(status_code=404, detail="No active real-time processing found")
        
        processor = realtime_processors[session_id]
        leaderboard = processor.get_live_leaderboard()
        
        return {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "leaderboard": leaderboard
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting live leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Statistics endpoints
@app.get("/api/v1/stats/drivers")
async def get_driver_statistics():
    """Get driver statistics across all sessions"""
    try:
        if not db_pool:
            return {"driver_statistics": []}
            
        with db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    driver_code,
                    COUNT(*) as total_laps,
                    AVG(lap_time) as avg_lap_time,
                    MIN(lap_time) as best_lap_time,
                    COUNT(DISTINCT session_id) as sessions_participated
                FROM lap_times
                WHERE COALESCE(deleted, 0) = 0 AND lap_time IS NOT NULL
                GROUP BY driver_code
                ORDER BY avg_lap_time
            """)
            
            stats = [dict(row) for row in cursor.fetchall()]
            return {"driver_statistics": stats}
            
    except Exception as e:
        logger.error(f"Error getting driver statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v1/stats/circuits")
async def get_circuit_statistics():
    """Get circuit statistics"""
    try:
        if not db_pool:
            return {"circuit_statistics": []}
            
        with db_pool.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    r.circuit_name,
                    r.country,
                    COUNT(DISTINCT s.session_id) as total_sessions,
                    COUNT(l.lap_id) as total_laps,
                    AVG(l.lap_time) as avg_lap_time,
                    MIN(l.lap_time) as fastest_lap
                FROM races r
                JOIN sessions s ON r.race_id = s.race_id
                JOIN lap_times l ON s.session_id = l.session_id
                WHERE COALESCE(l.deleted, 0) = 0 AND l.lap_time IS NOT NULL
                GROUP BY r.circuit_name, r.country
                ORDER BY avg_lap_time
            """)
            
            stats = [dict(row) for row in cursor.fetchall()]
            return {"circuit_statistics": stats}
            
    except Exception as e:
        logger.error(f"Error getting circuit statistics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )