import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool

logger = setup_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for driver/team performance metrics"""
    entity_id: str
    avg_lap_time: float
    consistency_score: float
    pace_score: float
    tire_management: float
    qualifying_pace: Optional[float] = None
    race_pace: Optional[float] =None

class PerformanceAnalyzer:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def _load_clean_lap_data(self, session_id: str) -> pd.DataFrame:
        """Load clean lap data for analysis"""
        try:
            with db_pool.get_connection() as conn:
                # Fixed the SQL query formatting and method signature
                query = """
                    SELECT * FROM lap_times
                    WHERE session_id = ?
                    AND COALESCE(deleted, 0) = 0
                    AND lap_time IS NOT NULL
                    AND lap_time BETWEEN 40 AND 300
                    ORDER BY driver_code, lap_number
                """
                df = pd.read_sql_query(query, conn, params=(session_id,))
                
                self.logger.info(f"Loaded {len(df)} clean lap records for session {session_id}")
                return df
                
        except Exception as e:
            self.logger.error(f"Error loading clean lap data for {session_id}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error
        
    def _calculate_driver_metrics(self, driver_laps:pd.DataFrame, session_fastest:float)->PerformanceMetrics:
        driver_code = driver_laps['driver_code'].iloc[0]

        avg_lap_time = driver_laps['lap_time'].mean()
        consistency_score = driver_laps['lap_time'].std()
        pace_score = ((avg_lap_time - session_fastest) / session_fastest)*100
        tire_management = self._calculate_tire_managemnt(driver_laps)

        return PerformanceMetrics(
            entity_id=driver_code,
            avg_lap_time=avg_lap_time,
            consistency_score=consistency_score,
            pace_score=pace_score,
            tire_management=tire_management
        )
    
    def _calculate_tire_managemnt(self, driver_laps: pd.DataFrame) -> float:
        """Calculate tire management score (degradation rate)"""
        try:
            # Group by tire compound to analyze stints
            degradation_rates = []
            
            for compound in driver_laps['tire_compound'].dropna().unique():
                compound_laps = driver_laps[
                    (driver_laps['tire_compound'] == compound) & 
                    (driver_laps['tire_age'].notna())
                ].sort_values('tire_age')
                
                if len(compound_laps) > 3:  # Need enough laps
                    # Calculate degradation rate (seconds per lap of tire age)
                    correlation = stats.pearsonr(compound_laps['tire_age'], compound_laps['lap_time'])
                    if correlation[1] < 0.05:  # Significant correlation
                        degradation_rates.append(correlation[0])
            
            return np.mean(degradation_rates) if degradation_rates else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating tire management: {e}")
            return 0.0
        
    def analyze_stint_performance(self, session_id: str) -> Dict[str, Any]:
        """Analyze tire stint performance patterns"""
        self.logger.info(f"Analyzing stint performance for {session_id}")
        
        try:
            lap_data = self._load_clean_lap_data(session_id)
            stint_analysis = {}
            
            for compound in ['SOFT', 'MEDIUM', 'HARD']:
                compound_data = lap_data[lap_data['tire_compound'] == compound]
                
                if not compound_data.empty:
                    stint_analysis[compound] = self._analyze_compound_performance(compound_data)
            
            return stint_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing stint performance: {e}")
            return {}
        
    def _analyze_compound_performance(self, compound_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance characteristics of a tire compound"""
        # Group by driver to get individual stint data
        driver_stints = []
        
        for driver in compound_data['driver_code'].unique():
            driver_laps = compound_data[compound_data['driver_code'] == driver]
            
            # Find continuous stints (consecutive tire ages)
            stints = self._identify_stints(driver_laps)
            driver_stints.extend(stints)
        
        if not driver_stints:
            return {}
        
        # Calculate compound characteristics
        stint_lengths = [stint['length'] for stint in driver_stints]
        initial_pace = [stint['initial_lap_time'] for stint in driver_stints if stint['initial_lap_time']]
        degradation_rates = [stint['degradation_rate'] for stint in driver_stints if stint['degradation_rate']]
        
        return {
            'avg_stint_length': np.mean(stint_lengths),
            'max_stint_length': max(stint_lengths),
            'avg_initial_pace': np.mean(initial_pace) if initial_pace else None,
            'avg_degradation_rate': np.mean(degradation_rates) if degradation_rates else None,
            'total_stints': len(driver_stints)
        }

    def _identify_stints(self, driver_laps: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify continuous tire stints for a driver"""
        stints = []
        current_stint = []
        
        for _, lap in driver_laps.sort_values('lap_number').iterrows():
            if pd.notna(lap['tire_age']):
                if not current_stint or lap['tire_age'] == current_stint[-1]['tire_age'] + 1:
                    current_stint.append({
                        'lap_number': lap['lap_number'],
                        'tire_age': lap['tire_age'],
                        'lap_time': lap['lap_time']
                    })
                else:
                    # End current stint, start new one
                    if len(current_stint) > 2:  # Only consider stints with 3+ laps
                        stints.append(self._process_stint(current_stint))
                    current_stint = [{'lap_number': lap['lap_number'], 'tire_age': lap['tire_age'], 'lap_time': lap['lap_time']}]
        
        # Process final stint
        if len(current_stint) > 2:
            stints.append(self._process_stint(current_stint))
        
        return stints

    def analyze_driver_performance(self, session_id: str) -> List[PerformanceMetrics]:
        """Comprehensive driver performance analysis"""
        self.logger.info(f"Analyzing driver performance for {session_id}")
        
        try:
            # Load clean lap data
            lap_data = self._load_clean_lap_data(session_id)
            if lap_data.empty:
                self.logger.warning(f"No clean lap data found for {session_id}")
                return []
            
            driver_metrics = []
            session_fastest = lap_data['lap_time'].min()
            
            for driver in lap_data['driver_code'].unique():
                driver_laps = lap_data[lap_data['driver_code'] == driver]
                
                if len(driver_laps) < 3:  # Need minimum laps for analysis
                    continue
                
                metrics = self._calculate_driver_metrics(driver_laps, session_fastest)
                driver_metrics.append(metrics)
            
            # Sort by overall performance (combination of pace and consistency)
            driver_metrics.sort(key=lambda x: x.pace_score + (x.consistency_score * 0.1))
            
            self.logger.info(f"Analyzed performance for {len(driver_metrics)} drivers")
            return driver_metrics
            
        except Exception as e:
            self.logger.error(f"Error analyzing driver performance: {e}")
            return []

    def _process_stint(self, stint_laps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process stint data to extract performance metrics"""
        lap_times = [lap['lap_time'] for lap in stint_laps]
        tire_ages = [lap['tire_age'] for lap in stint_laps]
        
        # Calculate degradation rate
        degradation_rate = None
        if len(lap_times) > 3:
            correlation = stats.pearsonr(tire_ages, lap_times)
            if correlation[1] < 0.05:  # Significant correlation
                degradation_rate = correlation[0]
        
        return {
            'length': len(stint_laps),
            'initial_lap_time': lap_times[0] if lap_times else None,
            'final_lap_time': lap_times[-1] if lap_times else None,
            'avg_lap_time': np.mean(lap_times),
            'degradation_rate': degradation_rate,
            'stint_laps': stint_laps
        }
    
    def cluster_driver_performance(self, session_ids: List[str]) -> Dict[str, Any]:
        """Cluster drivers by performance characteristics across sessions"""
        self.logger.info(f"Clustering driver performance across {len(session_ids)} sessions")
        
        try:
            # Collect performance data across sessions
            all_performance_data = []
            
            for session_id in session_ids:
                session_metrics = self.analyze_driver_performance(session_id)
                for metrics in session_metrics:
                    all_performance_data.append({
                        'driver': metrics.entity_id,
                        'session': session_id,
                        'pace_score': metrics.pace_score,
                        'consistency_score': metrics.consistency_score,
                        'tire_management': metrics.tire_management
                    })
            
            if not all_performance_data:
                return {}
            
            # Create DataFrame and aggregate by driver
            df = pd.DataFrame(all_performance_data)
            driver_aggregates = df.groupby('driver').agg({
                'pace_score': 'mean',
                'consistency_score': 'mean', 
                'tire_management': 'mean'
            }).reset_index()
            
            # Perform clustering
            features = ['pace_score', 'consistency_score', 'tire_management']
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(driver_aggregates[features])
            
            # Use 4 clusters (e.g., Elite, Strong, Average, Struggling)
            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            driver_aggregates['cluster'] = clusters
            
            # Interpret clusters
            cluster_names = self._interpret_clusters(driver_aggregates, kmeans.cluster_centers_, scaler)
            
            return {
                'driver_clusters': driver_aggregates.to_dict('records'),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_names': cluster_names,
                'feature_names': features
            }
            
        except Exception as e:
            self.logger.error(f"Error clustering driver performance: {e}")
            return {}
    
    def _interpret_clusters(self, driver_data: pd.DataFrame, centers: np.ndarray, scaler: StandardScaler) -> Dict[int, str]:
        """Interpret cluster meanings based on centers"""
        # Convert scaled centers back to original scale
        original_centers = scaler.inverse_transform(centers)
        
        cluster_names = {}
        for i, center in enumerate(original_centers):
            pace_score, consistency_score, tire_management = center
            
            # Classify based on characteristics
            if pace_score < 1.0 and consistency_score < 1.0:  # Fast and consistent
                cluster_names[i] = "Elite Performers"
            elif pace_score < 2.0 and tire_management > 0.5:  # Good pace, good tire management
                cluster_names[i] = "Strong Racers"
            elif consistency_score > 2.0:  # Inconsistent
                cluster_names[i] = "Inconsistent Performers"
            else:
                cluster_names[i] = "Average Performers"
        
        return cluster_names
    
    def generate_performance_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        self.logger.info(f"Generating performance report for {session_id}")
        
        try:
            # Get basic performance metrics
            driver_metrics = self.analyze_driver_performance(session_id)
            
            # Get stint analysis
            stint_analysis = self.analyze_stint_performance(session_id)
            
            # Calculate session statistics
            lap_data = self._load_clean_lap_data(session_id)
            session_stats = self._calculate_session_statistics(lap_data)
            
            report = {
                'session_id': session_id,
                'generated_at': pd.Timestamp.now().isoformat(),
                'session_statistics': session_stats,
                'driver_performance': [
                    {
                        'driver': metrics.entity_id,
                        'avg_lap_time': round(metrics.avg_lap_time, 3),
                        'pace_score': round(metrics.pace_score, 2),
                        'consistency_score': round(metrics.consistency_score, 3),
                        'tire_management': round(metrics.tire_management, 3)
                    } for metrics in driver_metrics
                ],
                'tire_compound_analysis': stint_analysis,
                'insights': self._generate_insights(driver_metrics, stint_analysis, session_stats)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {}
    
    def _calculate_session_statistics(self, lap_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall session statistics"""
        return {
            'total_laps': len(lap_data),
            'total_drivers': lap_data['driver_code'].nunique(),
            'fastest_lap': float(lap_data['lap_time'].min()),
            'average_lap_time': float(lap_data['lap_time'].mean()),
            'lap_time_std': float(lap_data['lap_time'].std()),
            'tire_compounds_used': lap_data['tire_compound'].dropna().unique().tolist()
        }
    
    def _generate_insights(self, driver_metrics: List[PerformanceMetrics], 
                          stint_analysis: Dict[str, Any], 
                          session_stats: Dict[str, Any]) -> List[str]:
        """Generate performance insights"""
        insights = []
        
        if driver_metrics:
            # Fastest driver insight
            fastest_driver = min(driver_metrics, key=lambda x: x.pace_score)
            insights.append(f"{fastest_driver.entity_id} showed the strongest pace, {fastest_driver.pace_score:.2f}% off session fastest")
            
            # Most consistent driver
            most_consistent = min(driver_metrics, key=lambda x: x.consistency_score)
            insights.append(f"{most_consistent.entity_id} was most consistent with {most_consistent.consistency_score:.3f}s lap time variation")
            
            # Tire management insight
            if any(m.tire_management != 0 for m in driver_metrics):
                best_tire_mgmt = max(driver_metrics, key=lambda x: abs(x.tire_management) if x.tire_management != 0 else -1)
                if best_tire_mgmt.tire_management != 0:
                    insights.append(f"{best_tire_mgmt.entity_id} showed superior tire management")
        
        # Tire compound insights
        if 'SOFT' in stint_analysis and 'MEDIUM' in stint_analysis:
            soft_pace = stint_analysis['SOFT'].get('avg_initial_pace')
            medium_pace = stint_analysis['MEDIUM'].get('avg_initial_pace')
            
            if soft_pace and medium_pace:
                pace_diff = soft_pace - medium_pace
                insights.append(f"Soft tires were {abs(pace_diff):.3f}s {'faster' if pace_diff < 0 else 'slower'} than mediums on average")
        
        return insights

# Usage example
if __name__ == "__main__":
    analyzer = PerformanceAnalyzer()
    
    # Test with a session
    session_id = "2022_hungarian_grand_prix_r"
    
    # Generate performance report
    report = analyzer.generate_performance_report(session_id)
    
    if report:
        print(f"Performance Report for {session_id}")
        print(f"Generated at: {report['generated_at']}")
        print(f"Total laps analyzed: {report['session_statistics']['total_laps']}")
        print(f"Fastest lap: {report['session_statistics']['fastest_lap']:.3f}s")
        
        print("\nTop 10 Drivers by Pace:")
        for i, driver in enumerate(report['driver_performance'][:10]):
            print(f"{i+1}. {driver['driver']} - {driver['pace_score']:.2f}% off pace")
        
        print("\nKey Insights:")
        for insight in report['insights']:
            print(f"- {insight}")
