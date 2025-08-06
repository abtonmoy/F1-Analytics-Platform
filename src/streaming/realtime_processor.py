import time
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool
from src.analytics.performance import PerformanceAnalyzer
from src.models.predictive import F1PredictiveModels

logger = setup_logger(__name__)

@dataclass
class LiveTelemetryData:
    """Live telemetry data point"""
    driver_code: str
    timestamp: datetime
    lap_number: int
    lap_time: Optional[float] = None
    speed: Optional[float] = None
    tire_compound: Optional[str] = None
    tire_age: Optional[int] = None
    position: Optional[int] = None
    sector_1_time: Optional[float] = None
    sector_2_time: Optional[float] = None
    sector_3_time: Optional[float] = None

@dataclass
class RealTimeAlert:
    """Real-time alert/notification"""
    alert_type: str
    driver_code: str
    message: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)

class RealTimeProcessor:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = setup_logger(self.__class__.__name__)
        
        # Processing queues
        self.telemetry_queue = queue.Queue(maxsize=1000)
        self.alert_queue = queue.Queue(maxsize=100)
        
        # Analytics components
        self.performance_analyzer = PerformanceAnalyzer()
        self.ml_models = F1PredictiveModels()
        
        # Real-time data storage
        self.live_data = {}  # driver_code -> latest data
        self.lap_history = {}  # driver_code -> list of recent laps
        
        # Processing threads
        self.processing_threads = []
        self.is_running = False
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[RealTimeAlert], None]] = []
        
        # Performance tracking
        self.processed_count = 0
        self.start_time = None
    
    def start_processing(self):
        """Start real-time processing threads"""
        self.is_running = True
        self.start_time = datetime.now()
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._telemetry_processor, daemon=True),
            threading.Thread(target=self._analysis_processor, daemon=True),
            threading.Thread(target=self._alert_processor, daemon=True),
            threading.Thread(target=self._performance_monitor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
            self.processing_threads.append(thread)
        
        self.logger.info(f"Real-time processing started for session {self.session_id}")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        
        # Wait for threads to complete
        for thread in self.processing_threads:
            thread.join(timeout=5)
        
        self.logger.info("Real-time processing stopped")
    
    def ingest_telemetry(self, telemetry_data: LiveTelemetryData):
        """Ingest new telemetry data point"""
        try:
            if not self.telemetry_queue.full():
                self.telemetry_queue.put(telemetry_data, timeout=1)
            else:
                self.logger.warning("Telemetry queue full, dropping data point")
                
        except queue.Full:
            self.logger.warning("Failed to ingest telemetry data - queue full")
    
    def _telemetry_processor(self):
        """Process incoming telemetry data"""
        while self.is_running:
            try:
                # Get data from queue with timeout
                telemetry = self.telemetry_queue.get(timeout=1)
                
                # Update live data
                self.live_data[telemetry.driver_code] = telemetry
                
                # Update lap history
                if telemetry.driver_code not in self.lap_history:
                    self.lap_history[telemetry.driver_code] = []
                
                if telemetry.lap_time is not None:
                    self.lap_history[telemetry.driver_code].append({
                        'lap_number': telemetry.lap_number,
                        'lap_time': telemetry.lap_time,
                        'timestamp': telemetry.timestamp,
                        'tire_compound': telemetry.tire_compound,
                        'tire_age': telemetry.tire_age,
                        'position': telemetry.position
                    })
                    
                    # Keep only recent laps (last 10)
                    if len(self.lap_history[telemetry.driver_code]) > 10:
                        self.lap_history[telemetry.driver_code] = \
                            self.lap_history[telemetry.driver_code][-10:]
                
                # Check for alerts
                self._check_alerts(telemetry)
                
                self.processed_count += 1
                self.telemetry_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing telemetry: {e}")
    
    def _analysis_processor(self):
        """Perform real-time analysis"""
        while self.is_running:
            try:
                time.sleep(5)  # Run analysis every 5 seconds
                
                # Analyze current session state
                self._analyze_current_state()
                
            except Exception as e:
                self.logger.error(f"Error in analysis processor: {e}")
    
    def _check_alerts(self, telemetry: LiveTelemetryData):
        """Check for alert conditions"""
        try:
            alerts = []
            
            # Check for exceptional lap times
            if telemetry.lap_time is not None:
                # Personal best alert
                driver_history = self.lap_history.get(telemetry.driver_code, [])
                if driver_history:
                    personal_best = min(lap['lap_time'] for lap in driver_history)
                    if telemetry.lap_time < personal_best - 0.5:  # 0.5s improvement
                        alerts.append(RealTimeAlert(
                            alert_type="PERSONAL_BEST",
                            driver_code=telemetry.driver_code,
                            message=f"New personal best: {telemetry.lap_time:.3f}s",
                            severity="HIGH",
                            timestamp=telemetry.timestamp,
                            data={"lap_time": telemetry.lap_time, "improvement": personal_best - telemetry.lap_time}
                        ))
                
                # Slow lap alert
                if telemetry.lap_time > 120:  # Exceptionally slow lap
                    alerts.append(RealTimeAlert(
                        alert_type="SLOW_LAP",
                        driver_code=telemetry.driver_code,
                        message=f"Unusually slow lap: {telemetry.lap_time:.3f}s",
                        severity="MEDIUM",
                        timestamp=telemetry.timestamp,
                        data={"lap_time": telemetry.lap_time}
                    ))
            
            # Tire age alerts
            if telemetry.tire_age is not None and telemetry.tire_age > 25:
                alerts.append(RealTimeAlert(
                    alert_type="HIGH_TIRE_AGE",
                    driver_code=telemetry.driver_code,
                    message=f"High tire age: {telemetry.tire_age} laps",
                    severity="LOW",
                    timestamp=telemetry.timestamp,
                    data={"tire_age": telemetry.tire_age, "compound": telemetry.tire_compound}
                ))
            
            # Position change alerts
            if telemetry.position is not None:
                current_data = self.live_data.get(telemetry.driver_code)
                if current_data and current_data.position is not None:
                    position_change = current_data.position - telemetry.position
                    if abs(position_change) >= 3:  # Significant position change
                        direction = "gained" if position_change > 0 else "lost"
                        alerts.append(RealTimeAlert(
                            alert_type="POSITION_CHANGE",
                            driver_code=telemetry.driver_code,
                            message=f"{direction.title()} {abs(position_change)} positions (now P{telemetry.position})",
                            severity="MEDIUM",
                            timestamp=telemetry.timestamp,
                            data={"position_change": position_change, "new_position": telemetry.position}
                        ))
            
            # Queue alerts
            for alert in alerts:
                if not self.alert_queue.full():
                    self.alert_queue.put(alert)
                    
        except Exception as e:
            self.logger.error(f"Error checking alerts: {e}")
    
    def _alert_processor(self):
        """Process and dispatch alerts"""
        while self.is_running:
            try:
                alert = self.alert_queue.get(timeout=1)
                
                # Log alert
                self.logger.info(f"ALERT [{alert.severity}] {alert.driver_code}: {alert.message}")
                
                # Call registered callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        self.logger.error(f"Error in alert callback: {e}")
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error processing alerts: {e}")
    
    def _analyze_current_state(self):
        """Analyze current session state"""
        try:
            if not self.live_data:
                return
            
            # Current leader analysis
            drivers_with_position = {
                driver: data for driver, data in self.live_data.items()
                if data.position is not None
            }
            
            if drivers_with_position:
                leader = min(drivers_with_position.values(), key=lambda x: x.position)
                self.logger.info(f"Current leader: {leader.driver_code} (P{leader.position})")
            
            # Tire strategy analysis
            tire_strategies = {}
            for driver, data in self.live_data.items():
                if data.tire_compound and data.tire_age is not None:
                    tire_strategies[driver] = {
                        'compound': data.tire_compound,
                        'age': data.tire_age
                    }
            
            # Log tire strategy summary
            if tire_strategies:
                compound_counts = {}
                for strategy in tire_strategies.values():
                    compound = strategy['compound']
                    compound_counts[compound] = compound_counts.get(compound, 0) + 1
                
                self.logger.info(f"Tire strategy summary: {compound_counts}")
                
        except Exception as e:
            self.logger.error(f"Error analyzing current state: {e}")
    
    def _performance_monitor(self):
        """Monitor processing performance"""
        while self.is_running:
            try:
                time.sleep(30)  # Report every 30 seconds
                
                if self.start_time:
                    elapsed = (datetime.now() - self.start_time).total_seconds()
                    rate = self.processed_count / elapsed if elapsed > 0 else 0
                    
                    self.logger.info(f"Processing rate: {rate:.1f} messages/sec, "
                                   f"Queue size: {self.telemetry_queue.qsize()}, "
                                   f"Total processed: {self.processed_count}")
                    
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {e}")
    
    def add_alert_callback(self, callback: Callable[[RealTimeAlert], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_live_leaderboard(self) -> List[Dict[str, Any]]:
        """Get current live leaderboard"""
        try:
            drivers_with_position = [
                {
                    'driver_code': data.driver_code,
                    'position': data.position,
                    'last_lap_time': data.lap_time,
                    'tire_compound': data.tire_compound,
                    'tire_age': data.tire_age,
                    'last_update': data.timestamp.isoformat()
                }
                for data in self.live_data.values()
                if data.position is not None
            ]
            
            # Sort by position
            drivers_with_position.sort(key=lambda x: x['position'])
            return drivers_with_position
            
        except Exception as e:
            self.logger.error(f"Error getting live leaderboard: {e}")
            return []
    
    def get_driver_recent_performance(self, driver_code: str) -> Dict[str, Any]:
        """Get recent performance data for a driver"""
        try:
            history = self.lap_history.get(driver_code, [])
            current = self.live_data.get(driver_code)
            
            if not history:
                return {}
            
            recent_laps = history[-5:]  # Last 5 laps
            avg_lap_time = sum(lap['lap_time'] for lap in recent_laps) / len(recent_laps)
            best_lap = min(lap['lap_time'] for lap in recent_laps)
            
            return {
                'driver_code': driver_code,
                'current_position': current.position if current else None,
                'recent_avg_lap_time': avg_lap_time,
                'recent_best_lap': best_lap,
                'current_tire_compound': current.tire_compound if current else None,
                'current_tire_age': current.tire_age if current else None,
                'total_laps': len(history)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting driver performance: {e}")
            return {}

class LiveDataSimulator:
    """Simulate live F1 data for testing"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.logger = setup_logger(self.__class__.__name__)
        self.drivers = ['HAM', 'VER', 'LEC', 'RUS', 'SAI', 'NOR', 'PIA', 'ALO', 'STR', 'PER']
        self.current_lap = {driver: 1 for driver in self.drivers}
        self.tire_age = {driver: 1 for driver in self.drivers}
        self.tire_compound = {driver: 'MEDIUM' for driver in self.drivers}
        self.base_lap_time = {driver: 80 + np.random.normal(0, 2) for driver in self.drivers}
    
    def generate_telemetry_batch(self, count: int = 10) -> List[LiveTelemetryData]:
        """Generate batch of simulated telemetry data"""
        telemetry_batch = []
        
        for _ in range(count):
            driver = np.random.choice(self.drivers)
            
            # Simulate lap completion
            lap_time = None
            if np.random.random() < 0.1:  # 10% chance of lap completion
                # Generate lap time with some variation
                base_time = self.base_lap_time[driver]
                tire_degradation = self.tire_age[driver] * 0.02  # 0.02s per lap
                variation = np.random.normal(0, 0.5)
                lap_time = base_time + tire_degradation + variation
                
                self.current_lap[driver] += 1
                self.tire_age[driver] += 1
                
                # Pit stop simulation
                if self.tire_age[driver] > 20 and np.random.random() < 0.3:
                    compounds = ['SOFT', 'MEDIUM', 'HARD']
                    self.tire_compound[driver] = np.random.choice(compounds)
                    self.tire_age[driver] = 1
            
            telemetry = LiveTelemetryData(
                driver_code=driver,
                timestamp=datetime.now(),
                lap_number=self.current_lap[driver],
                lap_time=lap_time,
                speed=np.random.uniform(250, 320) if np.random.random() < 0.8 else None,
                tire_compound=self.tire_compound[driver],
                tire_age=self.tire_age[driver],
                position=self.drivers.index(driver) + 1 + np.random.randint(-2, 3),
                sector_1_time=np.random.uniform(20, 30) if lap_time else None,
                sector_2_time=np.random.uniform(25, 35) if lap_time else None,
                sector_3_time=np.random.uniform(20, 30) if lap_time else None
            )
            
            telemetry_batch.append(telemetry)
        
        return telemetry_batch

# Usage example
if __name__ == "__main__":
    # Create processor
    processor = RealTimeProcessor("2024_monaco_grand_prix_r")
    
    # Add alert callback
    def alert_handler(alert: RealTimeAlert):
        print(f"🚨 ALERT: {alert.message}")
    
    processor.add_alert_callback(alert_handler)
    
    # Start processing
    processor.start_processing()
    
    # Simulate live data
    simulator = LiveDataSimulator("2024_monaco_grand_prix_r")
    
    try:
        print("Starting live data simulation...")
        for i in range(100):  # Simulate 100 batches
            batch = simulator.generate_telemetry_batch(5)
            
            for telemetry in batch:
                processor.ingest_telemetry(telemetry)
            
            time.sleep(1)  # 1 second between batches
            
            # Print leaderboard every 10 batches
            if i % 10 == 0:
                leaderboard = processor.get_live_leaderboard()
                print(f"\n--- Live Leaderboard (Batch {i}) ---")
                for pos, driver_data in enumerate(leaderboard[:10], 1):
                    print(f"P{pos}: {driver_data['driver_code']} - "
                          f"{driver_data['tire_compound']} ({driver_data['tire_age']} laps)")
    
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    finally:
        processor.stop_processing()
