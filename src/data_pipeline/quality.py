import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool

logger = setup_logger(__name__)

class QualityCheckSeverity(Enum):
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

@dataclass
class QualityCheckResult:
    """Result of a data quality check"""
    check_name: str
    severity: QualityCheckSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_records: int = 0  # Fixed typo: was "affacted_records"
    total_records: int = 0

class DataQualityChecker:
    # Define constants for thresholds
    MAX_REASONABLE_LAP_TIME = 300  # seconds
    MIN_REASONABLE_LAP_TIME = 40   # seconds
    MIN_REASONABLE_SPEED = 20      # km/h
    MAX_REASONABLE_SPEED = 390     # km/h
    SECTOR_TIME_TOLERANCE = 1.0    # seconds
    IQR_MULTIPLIER = 1.5
    MIN_DRIVERS_EXPECTED = 10
    
    # Tire compound lifespans
    COMPOUND_LIFESPAN = {
        "SOFT": 15,
        "MEDIUM": 25,
        "HARD": 40,
        "INTERMEDIATE": 30,
        "WET": 20
    }
    
    VALID_COMPOUNDS = {'SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'}

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    def run_full_quality_check(self, session_id: str) -> List[QualityCheckResult]:
        """Run comprehensive quality checks for a session"""
        self.logger.info(f"Running quality checks for session: {session_id}")

        results = []

        try:
            session_data = self._load_session_data(session_id)

            if not session_data:
                return [QualityCheckResult(
                    check_name="data_availability",  # Fixed typo
                    severity=QualityCheckSeverity.CRITICAL,
                    passed=False,
                    message=f"No data found for session {session_id}"
                )]
            
            # Run individual checks
            results.extend(self._check_lap_time_consistency(session_data['lap_times']))
            results.extend(self._check_sector_time_logic(session_data['lap_times']))
            results.extend(self._check_tire_compound_logic(session_data['lap_times']))
            results.extend(self._check_position_consistency(session_data['lap_times']))
            results.extend(self._check_speed_anomalies(session_data['lap_times']))
            results.extend(self._check_data_completeness(session_data))
            results.extend(self._check_statistical_outliers(session_data['lap_times']))
            
            # Summary
            critical_issues = len([r for r in results if r.severity == QualityCheckSeverity.CRITICAL])
            error_issues = len([r for r in results if r.severity == QualityCheckSeverity.ERROR])
            warning_issues = len([r for r in results if r.severity == QualityCheckSeverity.WARNING])
            
            self.logger.info(f"Quality check complete: {critical_issues} critical, {error_issues} errors, {warning_issues} warnings")

        except Exception as e:
            self.logger.error(f"Error during quality checks: {e}")
            results.append(QualityCheckResult(
                check_name="quality_check_error",
                severity=QualityCheckSeverity.CRITICAL,
                passed=False,
                message=f"Quality check failed: {str(e)}"
            ))

        return results
    
    def _load_session_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data from database"""
        try:
            with db_pool.get_connection() as conn:
                lap_times_query = """
                    SELECT * FROM lap_times
                    WHERE session_id = ?
                    ORDER BY driver_code, lap_number
                """
                lap_times_df = pd.read_sql_query(lap_times_query, conn, params=(session_id,))

                drivers_query = """
                    SELECT DISTINCT d.* FROM drivers d
                    JOIN lap_times l ON d.driver_code = l.driver_code
                    WHERE l.session_id = ?
                """
                drivers_df = pd.read_sql_query(drivers_query, conn, params=(session_id,))

                session_query = "SELECT * FROM sessions WHERE session_id = ?"
                session_df = pd.read_sql_query(session_query, conn, params=(session_id,))  # Fixed: was using drivers_query

                return {
                    'lap_times': lap_times_df,
                    'drivers': drivers_df,
                    'session_info': session_df.iloc[0] if len(session_df) > 0 else None
                }
        except Exception as e:
            self.logger.error(f"Error loading session data: {e}")
            return None
        
    def _check_lap_time_consistency(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check lap time consistency and reasonableness"""
        results = []

        if lap_times_df.empty:
            return results
        
        try:
            null_lap_times = lap_times_df['lap_time'].isnull().sum()
            total_records = len(lap_times_df)

            if null_lap_times > 0:
                results.append(QualityCheckResult(
                    check_name="lap_time_null",
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"{null_lap_times} lap times are null",
                    affected_records=null_lap_times,  # Fixed field name
                    total_records=total_records
                ))

            valid_lap_times = lap_times_df.dropna(subset=['lap_time'])

            if not valid_lap_times.empty:
                unrealistic_slow = (valid_lap_times['lap_time'] > self.MAX_REASONABLE_LAP_TIME).sum()
                unrealistic_fast = (valid_lap_times['lap_time'] < self.MIN_REASONABLE_LAP_TIME).sum()

                if unrealistic_fast > 0:
                    results.append(QualityCheckResult(
                        check_name="unrealistic_fast_laps",
                        severity=QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"{unrealistic_fast} lap times are unrealistically fast (< {self.MIN_REASONABLE_LAP_TIME}s)",  # Fixed message
                        affected_records=unrealistic_fast,
                        total_records=len(valid_lap_times)
                    ))
                
                if unrealistic_slow > 0:
                    results.append(QualityCheckResult(
                        check_name="unrealistic_slow_laps",
                        severity=QualityCheckSeverity.WARNING,
                        passed=False,
                        message=f"{unrealistic_slow} lap times are unrealistically slow (> {self.MAX_REASONABLE_LAP_TIME}s)",
                        affected_records=unrealistic_slow,
                        total_records=len(valid_lap_times)
                    ))

            # Check lap time progression
            progression_issues = 0
            for driver in lap_times_df['driver_code'].unique():
                driver_laps = lap_times_df[lap_times_df['driver_code'] == driver].sort_values('lap_number')

                for compound in driver_laps['tire_compound'].dropna().unique():
                    stint_laps = driver_laps[driver_laps['tire_compound'] == compound]
                    stint_laps = stint_laps.dropna(subset=['lap_time'])

                    if len(stint_laps) > 3:  # Need at least 4 laps to check progression
                        # Check if lap times are consistently decreasing early in stint (should improve)
                        first_three = stint_laps.head(3)['lap_time'].tolist()
                        if len(first_three) == 3:
                            # Expect some improvement in first 3 laps
                            if first_three[2] > first_three[0] * 1.05:  # 5% slower than first lap
                                progression_issues += 1

            if progression_issues > 0:
                results.append(QualityCheckResult(
                    check_name="lap_time_progression",
                    severity=QualityCheckSeverity.INFO,
                    passed=False,
                    message=f"{progression_issues} stints show unusual lap time progression",
                    affected_records=progression_issues,
                    details={"check_type": "stint_progression"}
                ))
        
        except Exception as e:
            self.logger.error(f"Error in lap time consistency check: {e}")
            results.append(QualityCheckResult(
                check_name="lap_time_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Lap time consistency check failed: {str(e)}"
            ))
        
        return results
    
    def _check_sector_time_logic(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check sector time consistency"""
        results = []
        
        try:
            # Check if sector times add up to lap time (within tolerance)
            complete_laps = lap_times_df.dropna(subset=['lap_time', 'sector_1_time', 'sector_2_time', 'sector_3_time'])
            
            if not complete_laps.empty:
                sector_sum = complete_laps['sector_1_time'] + complete_laps['sector_2_time'] + complete_laps['sector_3_time']
                time_diff = abs(sector_sum - complete_laps['lap_time'])
                
                # Allow tolerance for timing precision
                inconsistent_sectors = (time_diff > self.SECTOR_TIME_TOLERANCE).sum()
                
                if inconsistent_sectors > 0:
                    results.append(QualityCheckResult(
                        check_name="sector_time_consistency",
                        severity=QualityCheckSeverity.WARNING,
                        passed=False,
                        message=f"{inconsistent_sectors} laps have inconsistent sector times",
                        affected_records=inconsistent_sectors,
                        total_records=len(complete_laps),
                        details={
                            "max_difference": float(time_diff.max()),
                            "avg_difference": float(time_diff.mean())
                        }
                    ))
        
        except Exception as e:
            self.logger.error(f"Error in sector time logic check: {e}")
            results.append(QualityCheckResult(
                check_name="sector_time_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Sector time consistency check failed: {str(e)}"
            ))
        
        return results
    
    def _check_tire_compound_logic(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check tire compound data consistency"""
        results = []
        
        try:
            # Check for valid tire compounds
            invalid_compounds = lap_times_df[
                ~lap_times_df['tire_compound'].isin(self.VALID_COMPOUNDS) & 
                lap_times_df['tire_compound'].notna()
            ]
            
            if not invalid_compounds.empty:
                unique_invalid = invalid_compounds['tire_compound'].unique()
                results.append(QualityCheckResult(
                    check_name="invalid_tire_compounds",
                    severity=QualityCheckSeverity.ERROR,
                    passed=False,
                    message=f"Invalid tire compounds found: {list(unique_invalid)}",
                    affected_records=len(invalid_compounds),
                    total_records=len(lap_times_df)
                ))
            
            # Check tire age progression
            tire_age_issues = 0
            for driver in lap_times_df['driver_code'].unique():
                driver_laps = lap_times_df[lap_times_df['driver_code'] == driver].sort_values('lap_number')
                
                current_compound = None
                expected_age = 0
                
                for _, lap in driver_laps.iterrows():
                    if pd.notna(lap['tire_compound']):
                        if lap['tire_compound'] != current_compound:
                            # New tire stint
                            current_compound = lap['tire_compound']
                            expected_age = 1
                        else:
                            expected_age += 1
                        
                        if pd.notna(lap['tire_age']) and abs(lap['tire_age'] - expected_age) > 1:
                            tire_age_issues += 1
            
            if tire_age_issues > 0:
                results.append(QualityCheckResult(
                    check_name="tire_age_progression",
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"{tire_age_issues} laps have inconsistent tire age progression",
                    affected_records=tire_age_issues
                ))
        
        except Exception as e:
            self.logger.error(f"Error in tire compound logic check: {e}")
            results.append(QualityCheckResult(
                check_name="tire_compound_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Tire compound logic check failed: {str(e)}"
            ))
        
        return results 

    def _check_position_consistency(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check position data consistency"""
        results = []
        
        try:
            # Check for valid position range
            positions_df = lap_times_df.dropna(subset=['position'])
            
            if not positions_df.empty:
                max_drivers = lap_times_df['driver_code'].nunique()
                invalid_positions = positions_df[
                    (positions_df['position'] < 1) | 
                    (positions_df['position'] > max_drivers)
                ]
                
                if not invalid_positions.empty:
                    results.append(QualityCheckResult(
                        check_name="invalid_positions",
                        severity=QualityCheckSeverity.ERROR,
                        passed=False,
                        message=f"{len(invalid_positions)} laps have invalid positions",
                        affected_records=len(invalid_positions),
                        total_records=len(positions_df)
                    ))
        
        except Exception as e:
            self.logger.error(f"Error in position consistency check: {e}")
            results.append(QualityCheckResult(
                check_name="position_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Position consistency check failed: {str(e)}"
            ))
        
        return results
    
    def _check_speed_anomalies(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check for speed anomalies"""
        results = []
        
        try:
            speed_columns = ['speed_i1', 'speed_i2', 'speed_fl', 'speed_st']
            
            for speed_col in speed_columns:
                if speed_col not in lap_times_df.columns:
                    continue
                    
                speed_data = lap_times_df.dropna(subset=[speed_col])
                
                if not speed_data.empty:
                    # Check for unrealistic speeds
                    too_slow = (speed_data[speed_col] < self.MIN_REASONABLE_SPEED).sum()
                    too_fast = (speed_data[speed_col] > self.MAX_REASONABLE_SPEED).sum()
                    
                    if too_slow > 0:
                        results.append(QualityCheckResult(
                            check_name=f"unrealistic_slow_speed_{speed_col}",
                            severity=QualityCheckSeverity.WARNING,
                            passed=False,
                            message=f"{too_slow} records have unrealistically slow {speed_col} (< {self.MIN_REASONABLE_SPEED} km/h)",
                            affected_records=too_slow,
                            total_records=len(speed_data)
                        ))
                    
                    if too_fast > 0:
                        results.append(QualityCheckResult(
                            check_name=f"unrealistic_fast_speed_{speed_col}",
                            severity=QualityCheckSeverity.ERROR,
                            passed=False,
                            message=f"{too_fast} records have unrealistically fast {speed_col} (> {self.MAX_REASONABLE_SPEED} km/h)",
                            affected_records=too_fast,
                            total_records=len(speed_data)
                        ))
        
        except Exception as e:
            self.logger.error(f"Error in speed anomalies check: {e}")
            results.append(QualityCheckResult(
                check_name="speed_anomalies_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Speed anomalies check failed: {str(e)}"
            ))
        
        return results

    def _check_data_completeness(self, session_data: Dict[str, Any]) -> List[QualityCheckResult]:
        """Check data completeness"""
        results = []
        
        try:
            lap_times_df = session_data['lap_times']
            
            if lap_times_df.empty:
                results.append(QualityCheckResult(
                    check_name="no_lap_data",
                    severity=QualityCheckSeverity.CRITICAL,
                    passed=False,
                    message="No lap time data available"
                ))
                return results
            
            # Check expected number of drivers (should be around 20-24)
            driver_count = lap_times_df['driver_code'].nunique()
            if driver_count < self.MIN_DRIVERS_EXPECTED:
                results.append(QualityCheckResult(
                    check_name="insufficient_drivers",
                    severity=QualityCheckSeverity.WARNING,
                    passed=False,
                    message=f"Only {driver_count} drivers found (expected ~20-24)",
                    details={"driver_count": driver_count}
                ))
            
            # Check data completeness for key fields
            key_fields = ['lap_time', 'tire_compound', 'sector_1_time', 'sector_2_time', 'sector_3_time']
            
            for field in key_fields:
                if field not in lap_times_df.columns:
                    results.append(QualityCheckResult(
                        check_name=f"missing_column_{field}",
                        severity=QualityCheckSeverity.CRITICAL,
                        passed=False,
                        message=f"Required column '{field}' is missing"
                    ))
                    continue
                    
                null_count = lap_times_df[field].isnull().sum()
                null_percentage = (null_count / len(lap_times_df)) * 100
                
                if null_percentage > 50:
                    severity = QualityCheckSeverity.ERROR
                elif null_percentage > 20:
                    severity = QualityCheckSeverity.WARNING
                else:
                    severity = QualityCheckSeverity.INFO
                
                if null_count > 0:
                    results.append(QualityCheckResult(
                        check_name=f"missing_{field}",
                        severity=severity,
                        passed=null_percentage < 10,
                        message=f"{null_count} records ({null_percentage:.1f}%) missing {field}",
                        affected_records=null_count,
                        total_records=len(lap_times_df)
                    ))
        
        except Exception as e:
            self.logger.error(f"Error in data completeness check: {e}")
            results.append(QualityCheckResult(
                check_name="completeness_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Data completeness check failed: {str(e)}"
            ))
        
        return results
    
    def _check_statistical_outliers(self, lap_times_df: pd.DataFrame) -> List[QualityCheckResult]:
        """Check for statistical outliers using IQR method"""
        results = []
        
        try:
            # Check lap time outliers per driver
            outlier_count = 0

            for driver in lap_times_df['driver_code'].unique():
                # Check if 'deleted' column exists, if not assume all records are valid
                if 'deleted' in lap_times_df.columns:
                    driver_laps = lap_times_df[
                        (lap_times_df['driver_code'] == driver) &
                        (lap_times_df['lap_time'].notna()) &
                        (lap_times_df['deleted'] == False)
                    ]
                else:
                    driver_laps = lap_times_df[
                        (lap_times_df['driver_code'] == driver) &
                        (lap_times_df['lap_time'].notna())
                    ]

                if len(driver_laps) > 5:
                    lap_times_series = driver_laps['lap_time']  # Fixed variable name

                    Q1 = lap_times_series.quantile(0.25)
                    Q3 = lap_times_series.quantile(0.75)
                    IQR = Q3 - Q1

                    lower_bound = Q1 - self.IQR_MULTIPLIER * IQR
                    upper_bound = Q3 + self.IQR_MULTIPLIER * IQR

                    # Fixed: use the series for comparison
                    outliers = driver_laps[
                        (lap_times_series < lower_bound) | 
                        (lap_times_series > upper_bound)
                    ]
                    outlier_count += len(outliers)

            if outlier_count > 0:
                results.append(QualityCheckResult(
                    check_name="statistical_outliers",
                    severity=QualityCheckSeverity.INFO,
                    passed=True,
                    message=f"{outlier_count} lap times identified as statistical outliers",
                    affected_records=outlier_count,
                    details={"method": "IQR", "threshold": f"{self.IQR_MULTIPLIER} * IQR"}
                ))
        
        except Exception as e:
            self.logger.error(f"Error in statistical outliers check: {e}")
            results.append(QualityCheckResult(
                check_name="outliers_check_error",
                severity=QualityCheckSeverity.ERROR,
                passed=False,
                message=f"Statistical outliers check failed: {str(e)}"
            ))
        
        return results

    def generate_quality_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        try:
            results = self.run_full_quality_check(session_id)
            
            # Categorize results
            critical = [r for r in results if r.severity == QualityCheckSeverity.CRITICAL]
            errors = [r for r in results if r.severity == QualityCheckSeverity.ERROR]
            warnings = [r for r in results if r.severity == QualityCheckSeverity.WARNING]
            info = [r for r in results if r.severity == QualityCheckSeverity.INFO]
            
            # Calculate overall score
            total_checks = len(results)
            passed_checks = len([r for r in results if r.passed])
            quality_score = (passed_checks / total_checks * 100) if total_checks > 0 else 0
            
            # Determine overall status
            if critical:
                overall_status = "CRITICAL"
            elif errors:
                overall_status = "POOR"
            elif warnings:
                overall_status = "FAIR"
            else:
                overall_status = "GOOD"
            
            return {
                'session_id': session_id,
                'overall_status': overall_status,
                'quality_score': quality_score,
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'summary': {
                    'critical': len(critical),
                    'errors': len(errors),
                    'warnings': len(warnings),
                    'info': len(info)
                },
                'issues': {
                    'critical': [{'check': r.check_name, 'message': r.message} for r in critical],
                    'errors': [{'check': r.check_name, 'message': r.message} for r in errors],
                    'warnings': [{'check': r.check_name, 'message': r.message} for r in warnings],
                    'info': [{'check': r.check_name, 'message': r.message} for r in info]
                },
                'generated_at': pd.Timestamp.now().isoformat()
            }
        
        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}")
            return {
                'session_id': session_id,
                'overall_status': "ERROR",
                'quality_score': 0,
                'error': str(e),
                'generated_at': pd.Timestamp.now().isoformat()
            }


class DataPreprocessor:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
    
    def clean_session_data(self, session_id: str) -> bool:
        """Clean data for a session based on quality checks"""
        try:
            quality_checker = DataQualityChecker()
            results = quality_checker.run_full_quality_check(session_id)

            with db_pool.get_connection() as conn:
                self._mark_outliers_as_deleted(conn, session_id)
                self._fix_tire_age_issues(conn, session_id)  # Fixed method name
                self._interpolate_missing_sector_times(conn, session_id)

                conn.commit()
            self.logger.info(f"Data cleaning completed for session {session_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cleaning session data: {e}")
            return False
        
    def _fix_tire_age_issues(self, conn, session_id: str):  # Fixed method name
        """Recalculate and fix missing or inconsistent tire age values, respecting compound lifespans"""
        try:
            # Use constants from DataQualityChecker
            compound_lifespan = DataQualityChecker.COMPOUND_LIFESPAN

            cursor = conn.execute("""
                SELECT * FROM lap_times 
                WHERE session_id = ? 
                ORDER BY driver_code, lap_number
            """, (session_id,))
            laps = cursor.fetchall()

            if not laps:
                self.logger.warning(f"No lap data found to fix tire age for session {session_id}")
                return

            from collections import defaultdict
            laps_by_driver = defaultdict(list)
            for lap in laps:
                laps_by_driver[lap['driver_code']].append(dict(lap))

            update_count = 0
            skipped_due_to_lifespan = 0

            for driver_code, driver_laps in laps_by_driver.items():
                current_compound = None
                current_age = 0

                for lap in driver_laps:
                    lap_id = lap['lap_id']
                    compound = lap['tire_compound']
                    if compound is None:
                        continue

                    compound = compound.upper()
                    lifespan = compound_lifespan.get(compound, 30)  # default to 30 laps if unknown

                    if compound != current_compound:
                        current_compound = compound
                        current_age = 1
                    else:
                        current_age += 1

                    if current_age > lifespan:
                        skipped_due_to_lifespan += 1
                        continue

                    if lap.get('tire_age') != current_age:
                        conn.execute("""
                            UPDATE lap_times 
                            SET tire_age = ?
                            WHERE lap_id = ?
                        """, (current_age, lap_id))
                        update_count += 1

            self.logger.info(f"Updated tire_age for {update_count} laps in session {session_id}")
            if skipped_due_to_lifespan > 0:
                self.logger.warning(f"Skipped {skipped_due_to_lifespan} laps that exceeded expected compound lifespan")
        
        except Exception as e:
            self.logger.error(f"Failed to fix tire age issues: {e}")

    def _mark_outliers_as_deleted(self, conn, session_id: str):
        """Marks lap time outliers as deleted based on domain-specific thresholds"""
        try:
            # Use constants from DataQualityChecker
            deletion_rules = [
                {
                    "condition": f"lap_time > {DataQualityChecker.MAX_REASONABLE_LAP_TIME}",
                    "reason": f"Unrealistic lap time (>{DataQualityChecker.MAX_REASONABLE_LAP_TIME}s)"
                },
                {
                    "condition": f"lap_time < {DataQualityChecker.MIN_REASONABLE_LAP_TIME}",
                    "reason": f"Unrealistic lap time (<{DataQualityChecker.MIN_REASONABLE_LAP_TIME}s)"
                },
                # Future rules can be added here
            ]

            total_deleted = 0

            for rule in deletion_rules:
                # Use parameterized query to avoid SQL injection risks
                query = f"""
                    UPDATE lap_times
                    SET deleted = 1,
                        deleted_reason = ?
                    WHERE session_id = ? AND {rule['condition']} AND COALESCE(deleted, 0) = 0
                """
                res = conn.execute(query, (rule["reason"], session_id))
                total_deleted += res.rowcount
                if res.rowcount > 0:
                    self.logger.info(f"Deleted {res.rowcount} laps for reason: {rule['reason']}")

            self.logger.info(f"Total {total_deleted} outliers marked as deleted for session {session_id}")
        
        except Exception as e:
            self.logger.error(f"Outlier deletion failed for session {session_id}: {e}")

    def _interpolate_missing_sector_times(self, conn, session_id: str):
        """Interpolate exactly one missing sector time by subtracting known sectors from lap_time"""
        try:
            interpolated_count = 0

            cursor = conn.execute("""
                SELECT lap_id, lap_time, sector_1_time, sector_2_time, sector_3_time
                FROM lap_times
                WHERE session_id = ? AND COALESCE(deleted, 0) = 0 AND lap_time IS NOT NULL
                AND (
                    (sector_1_time IS NULL AND sector_2_time IS NOT NULL AND sector_3_time IS NOT NULL) OR
                    (sector_2_time IS NULL AND sector_1_time IS NOT NULL AND sector_3_time IS NOT NULL) OR
                    (sector_3_time IS NULL AND sector_1_time IS NOT NULL AND sector_2_time IS NOT NULL)
                )
            """, (session_id,))
            
            for row in cursor.fetchall():
                lap_id = row['lap_id']
                lap_time = row['lap_time']
                s1, s2, s3 = row['sector_1_time'], row['sector_2_time'], row['sector_3_time']
                
                if s1 is None:
                    missing_sector = 'sector_1_time'
                    interpolated_value = lap_time - s2 - s3
                elif s2 is None:
                    missing_sector = 'sector_2_time'
                    interpolated_value = lap_time - s1 - s3
                elif s3 is None:
                    missing_sector = 'sector_3_time'
                    interpolated_value = lap_time - s1 - s2
                else:
                    continue  # This shouldn't happen due to WHERE clause

                # Sanity check
                if interpolated_value <= 0:
                    self.logger.warning(f"Skipped interpolation for lap_id={lap_id} due to invalid value: {interpolated_value}")
                    continue

                conn.execute(f"""
                    UPDATE lap_times
                    SET {missing_sector} = ?
                    WHERE lap_id = ?
                """, (interpolated_value, lap_id))
                interpolated_count += 1

            self.logger.info(f"Interpolated {interpolated_count} missing sector values for session {session_id}")

        except Exception as e:
            self.logger.error(f"Interpolation error in session {session_id}: {e}")


if __name__ == "__main__":
    # Test quality checking
    quality_checker = DataQualityChecker()
    
    # Test with a session (you'll need to have data first)
    session_id = "2024_monaco_grand_prix_r"  # Example session ID
    
    try:
        # Run quality checks
        report = quality_checker.generate_quality_report(session_id)
        
        print("Quality Report:")
        print(f"Status: {report['overall_status']}")
        print(f"Score: {report['quality_score']:.1f}%")
        print(f"Issues: {report['summary']}")
        
        # Test data preprocessing
        preprocessor = DataPreprocessor()
        success = preprocessor.clean_session_data(session_id)
        print(f"Data cleaning successful: {success}")
        
    except Exception as e:
        print(f"Error during testing: {e}")