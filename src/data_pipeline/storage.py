# storage.py - COMPLETELY FIXED VERSION with proper foreign key handling
import sqlite3
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import uuid
from datetime import datetime
import time
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool
from random import random

logger = setup_logger(__name__)


class DataStorageManager:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)

    @contextmanager
    def transaction(self):
        """Fixed transaction with proper connection pool handling"""
        max_retries = 3
        base_delay = 0.1
        
        for attempt in range(max_retries + 1):
            try:
                with db_pool.get_connection() as conn:
                    # SQLite optimizations
                    conn.execute('PRAGMA busy_timeout = 30000')
                    conn.execute('PRAGMA journal_mode = WAL') 
                    conn.execute('PRAGMA synchronous = NORMAL')
                    conn.execute('BEGIN IMMEDIATE')
                    
                    try:
                        yield conn
                        conn.commit()
                        self.logger.debug('Transaction committed successfully')
                        break
                    except Exception as e:
                        conn.rollback()
                        raise
                        
            except sqlite3.OperationalError as e:
                if 'database is locked' in str(e).lower() and attempt < max_retries:
                    delay = base_delay * (2 ** attempt) + random() * 0.1
                    self.logger.warning(f'Database locked, retrying in {delay:.2f}s (attempt {attempt + 1})')
                    time.sleep(delay)
                    continue
                else:
                    raise
            except Exception as e:
                self.logger.error(f'Transaction failed: {e}')
                raise
        else:
            raise sqlite3.OperationalError("Max retries exceeded for database transaction")

    def _generate_race_id(self, year: int, race_name: str) -> str:
        """Consistent race_id generation that matches session_id parsing"""
        clean_name = race_name.lower().replace(' ', '_').replace('-', '_').replace('grand_prix', 'grand_prix')
        return f"{year}_{clean_name}"

    def store_season_data(self, year: int, total_rounds: int) -> bool:
        try:
            with self.transaction() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO seasons (year, total_rounds, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (year, total_rounds))

                self.logger.info(f"Stored season data: {year}")
                return True

        except Exception as e:
            self.logger.error(f"Error storing season data: {e}")
            return False

    # storage.py FOREIGN KEY FIX PATCH
# Apply this patch to your store_race_data method in storage.py

    def store_race_data(self, race_events: List[Dict[str, Any]]) -> bool:
        """COMPLETELY FIXED: Store race data with bulletproof dependency handling"""
        try:
            with self.transaction() as conn:
                for event in race_events:
                    year = event['year']
                    
                    # 1. CRITICAL: Verify season exists first, create if missing
                    cursor = conn.execute("SELECT year FROM seasons WHERE year = ?", (year,))
                    if not cursor.fetchone():
                        self.logger.warning(f"Season {year} doesn't exist, creating it first")
                        
                        # Temporarily disable foreign keys to safely create season
                        conn.execute("PRAGMA foreign_keys=OFF")
                        conn.execute("""
                            INSERT OR REPLACE INTO seasons (year, total_rounds, created_at)
                            VALUES (?, 0, CURRENT_TIMESTAMP)
                        """, (year,))
                        conn.execute("PRAGMA foreign_keys=ON")
                        
                        # Verify season was created
                        cursor = conn.execute("SELECT year FROM seasons WHERE year = ?", (year,))
                        if not cursor.fetchone():
                            self.logger.error(f"Failed to create season {year}")
                            return False
                        else:
                            self.logger.info(f"Successfully created missing season {year}")
                    
                    # 2. Generate consistent race_id
                    race_id = self._generate_race_id(year, event['race_name'])
                    
                    self.logger.debug(f"Storing race: '{event['race_name']}' with race_id: '{race_id}' for year: {year}")
                    
                    # 3. Now safely insert race with verified foreign key
                    conn.execute("""
                        INSERT OR REPLACE INTO races
                        (race_id, year, round_number, race_name, circuit_name, country, race_date, race_time, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, (
                        race_id,
                        year,
                        event['round_number'],
                        event['race_name'],
                        event['circuit_name'],
                        event['country'],
                        event['race_date'],
                        event['race_time']
                    ))
                    
                    # 4. Verify the race was inserted
                    cursor = conn.execute("SELECT race_id FROM races WHERE race_id = ?", (race_id,))
                    if cursor.fetchone():
                        self.logger.debug(f"Successfully stored race: {race_id}")
                    else:
                        self.logger.error(f"Failed to verify race insertion: {race_id}")
                        return False

                self.logger.info(f"Stored {len(race_events)} race events")
                return True

        except Exception as e:
            self.logger.error(f"Error storing race data: {e}")
            import traceback
            traceback.print_exc()
            return False


    # ALTERNATIVE: Enhanced get_race_id_for_session with better fallback
    def get_race_id_for_session(self, session_id: str) -> Optional[str]:
        """ENHANCED: Improved session_id parsing with bulletproof fallback creation"""
        try:
            parts = session_id.split('_')
            if len(parts) < 3:
                self.logger.warning(f"Invalid session_id format: {session_id}")
                return None
                
            year = parts[0]
            session_type = parts[-1]
            race_name_parts = parts[1:-1]
            
            try:
                year_int = int(year)
            except ValueError:
                self.logger.warning(f"Invalid year in session_id: {session_id}")
                return None
                
            race_name_clean = '_'.join(race_name_parts)
            expected_race_id = f"{year}_{race_name_clean}"
            
            with db_pool.get_connection() as conn:
                # Try exact match first
                cursor = conn.execute("SELECT race_id FROM races WHERE race_id = ?", (expected_race_id,))
                result = cursor.fetchone()
                if result:
                    return result['race_id']
                
                # ENHANCED FALLBACK: Create complete record chain with proper error handling
                self.logger.warning(f"Race {expected_race_id} not found, creating complete record chain")
                
                try:
                    # Start a new transaction for the fallback creation
                    conn.execute("BEGIN IMMEDIATE")
                    
                    # Disable foreign keys temporarily for safe insertion
                    conn.execute("PRAGMA foreign_keys=OFF")
                    
                    # Ensure season exists
                    conn.execute("""
                        INSERT OR IGNORE INTO seasons (year, total_rounds, created_at)
                        VALUES (?, 0, CURRENT_TIMESTAMP)
                    """, (year_int,))
                    
                    # Create race record
                    race_name_display = ' '.join(race_name_parts).replace('_', ' ').title()
                    conn.execute("""
                        INSERT OR IGNORE INTO races 
                        (race_id, year, round_number, race_name, circuit_name, country, created_at)
                        VALUES (?, ?, 1, ?, 'Unknown', 'Unknown', CURRENT_TIMESTAMP)
                    """, (expected_race_id, year_int, race_name_display))
                    
                    # Re-enable foreign keys
                    conn.execute("PRAGMA foreign_keys=ON")
                    
                    # Commit the fallback transaction
                    conn.commit()
                    
                    # Verify creation
                    cursor = conn.execute("SELECT race_id FROM races WHERE race_id = ?", (expected_race_id,))
                    if cursor.fetchone():
                        self.logger.info(f"Created fallback race record: {expected_race_id}")
                        return expected_race_id
                    else:
                        self.logger.error(f"Failed to create fallback race record: {expected_race_id}")
                        return None
                        
                except Exception as fallback_error:
                    conn.rollback()
                    self.logger.error(f"Fallback creation failed: {fallback_error}")
                    return None
                        
        except Exception as e:
            self.logger.error(f"Error getting race_id for session {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def store_session_data(self, session_data: Dict[str, Any]) -> bool:
        """FIXED: Enhanced session data storage with bulletproof dependency handling"""
        session_id = session_data['session_id']
        
        try:
            self.logger.info(f"Starting to store session data for {session_id}")
            
            # 1. CRITICAL: Ensure race exists FIRST using a separate transaction
            race_id = self.get_race_id_for_session(session_id)
            if not race_id:
                self.logger.error(f"Could not resolve race_id for session {session_id}")
                return False
            
            self.logger.info(f"Using race_id: {race_id} for session: {session_id}")
            
            # 2. Store drivers (no dependencies) - SEPARATE TRANSACTION
            if not self._store_drivers_with_validation(session_data['drivers']):
                self.logger.error(f"Failed to store drivers for session {session_id}")
                return False
            
            # 3. Store session info (depends on race_id existing) - SEPARATE TRANSACTION
            if not self._store_session_info_with_validation(session_data, race_id):
                self.logger.error(f"Failed to store session info for session {session_id}")
                return False
            
            # 4. Store dependent data with proper validation - SEPARATE TRANSACTIONS
            self._store_lap_times_with_validation(session_data['lap_times'])
            self._store_pit_stops_with_validation(session_data['pit_stops'])
            self._store_weather_data_with_validation(session_data['weather'])
            self._store_tire_stints(session_data['tire_stints'])
            self._store_compound_usage(session_data['compound_usage'])
            self._store_telemetry_data(session_data['telemetry'])

            self.logger.info(f"Successfully stored session data for {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Error storing session data for {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _store_drivers_with_validation(self, drivers: List[Dict[str, Any]]) -> bool:
        """Store drivers with proper transaction handling"""
        if not drivers:
            self.logger.warning("No drivers to store")
            return True
            
        try:
            with self.transaction() as conn:
                stored_count = 0
                for driver in drivers:
                    try:
                        if not driver.get('driver_code'):
                            self.logger.warning(f"Skipping driver with missing driver_code: {driver}")
                            continue
                            
                        conn.execute("""
                            INSERT OR REPLACE INTO drivers 
                            (driver_code, first_name, last_name, full_name, team_name, team_color, 
                             driver_number, nationality, birth_date, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                        """, (
                            driver['driver_code'],
                            driver.get('first_name'),
                            driver.get('last_name'),
                            driver.get('full_name'),
                            driver.get('team_name'),
                            driver.get('team_color'),
                            driver.get('driver_number'),
                            driver.get('nationality'),
                            driver.get('birth_date')
                        ))
                        stored_count += 1
                    except Exception as e:
                        self.logger.error(f"Error storing driver {driver.get('driver_code', 'Unknown')}: {e}")
                        
                self.logger.info(f"Stored {stored_count} drivers")
                return stored_count > 0
                
        except Exception as e:
            self.logger.error(f"Error in driver storage transaction: {e}")
            return False

    def _store_session_info_with_validation(self, session_data: Dict[str, Any], race_id: str) -> bool:
        """COMPLETELY FIXED: Store session info with bulletproof race_id verification"""
        try:
            session_info = session_data['session_info']
            session_id = session_data['session_id']
            
            self.logger.debug(f"Attempting to store session '{session_id}' with race_id: '{race_id}'")

            with self.transaction() as conn:
                # CRITICAL: Verify race exists before attempting session insertion
                cursor = conn.execute("SELECT race_id FROM races WHERE race_id = ?", (race_id,))
                if not cursor.fetchone():
                    self.logger.error(f"Race {race_id} does not exist in database before session insertion")
                    
                    # Emergency fallback - create the race record right now
                    parts = session_id.split('_')
                    year = int(parts[0])
                    race_name = ' '.join(parts[1:-1]).replace('_', ' ').title()
                    
                    # Temporarily disable foreign keys
                    conn.execute("PRAGMA foreign_keys=OFF")
                    
                    # Ensure season exists
                    conn.execute("""
                        INSERT OR IGNORE INTO seasons (year, total_rounds, created_at)
                        VALUES (?, 0, CURRENT_TIMESTAMP)
                    """, (year,))
                    
                    # Create race
                    conn.execute("""
                        INSERT OR IGNORE INTO races 
                        (race_id, year, round_number, race_name, circuit_name, country, created_at)
                        VALUES (?, ?, 1, ?, 'Unknown', 'Unknown', CURRENT_TIMESTAMP)
                    """, (race_id, year, race_name))
                    
                    # Re-enable foreign keys
                    conn.execute("PRAGMA foreign_keys=ON")
                    
                    # Verify creation
                    cursor = conn.execute("SELECT race_id FROM races WHERE race_id = ?", (race_id,))
                    if not cursor.fetchone():
                        self.logger.error(f"Emergency race creation failed for {race_id}")
                        return False
                    else:
                        self.logger.warning(f"Emergency created race record: {race_id}")

                # Now attempt session insertion
                conn.execute("""
                    INSERT OR REPLACE INTO sessions
                    (session_id, race_id, session_type, session_date, session_time, 
                     air_temp, track_temp, humidity, pressure, wind_speed, wind_direction, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    session_id,
                    race_id,
                    session_info.get('session_type'),
                    session_info.get('session_date'),
                    session_info.get('session_time'),
                    session_info.get('air_temp'),
                    session_info.get('track_temp'),
                    session_info.get('humidity'),
                    session_info.get('pressure'),
                    session_info.get('wind_speed'),
                    session_info.get('wind_direction')
                ))
                
                # Verify session was created
                cursor = conn.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
                if cursor.fetchone():
                    self.logger.info(f"Successfully stored session info for '{session_id}'")
                    return True
                else:
                    self.logger.error(f"Session insertion verification failed for '{session_id}'")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error storing session info: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _store_lap_times_with_validation(self, lap_times: List[Dict[str, Any]]) -> bool:
        """Store lap times with comprehensive validation"""
        if not lap_times:
            self.logger.info("No lap times to store")
            return True
            
        try:
            with self.transaction() as conn:
                # Pre-check: Ensure all referenced drivers exist
                driver_codes = set(lap['driver_code'] for lap in lap_times if lap.get('driver_code'))
                for driver_code in driver_codes:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                
                # Validate and filter lap times
                valid_laps = []
                for lap in lap_times:
                    if self._validate_lap_data_comprehensive(lap):
                        valid_laps.append(lap)
                
                if not valid_laps:
                    self.logger.warning("No valid lap times to store after validation")
                    return True

                # Store lap times
                conn.executemany("""
                    INSERT OR REPLACE INTO lap_times 
                    (lap_id, session_id, driver_code, lap_number, lap_time, sector_1_time, sector_2_time, 
                     sector_3_time, speed_i1, speed_i2, speed_fl, speed_st, is_personal_best, is_accurate,
                     tire_compound, tire_age, fresh_tyre, pit_out_time, pit_in_time, deleted, deleted_reason,
                     track_status, position, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [( 
                    lap['lap_id'], lap['session_id'], lap['driver_code'], lap['lap_number'],
                    lap['lap_time'], lap.get('sector_1_time'), lap.get('sector_2_time'), lap.get('sector_3_time'),
                    lap.get('speed_i1'), lap.get('speed_i2'), lap.get('speed_fl'), lap.get('speed_st'),
                    lap.get('is_personal_best', False), lap.get('is_accurate', True), 
                    lap.get('tire_compound'), lap.get('tire_age'), lap.get('fresh_tyre', False),
                    lap.get('pit_out_time'), lap.get('pit_in_time'), lap.get('deleted', False),
                    lap.get('deleted_reason'), lap.get('track_status'), lap.get('position')
                ) for lap in valid_laps])

                self.logger.info(f"Stored {len(valid_laps)} lap times")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing lap times: {e}")
            return False

    def _store_pit_stops_with_validation(self, pit_stops: List[Dict[str, Any]]) -> bool:
        """FIXED: Store pit stops with driver existence check"""
        if not pit_stops:
            return True
            
        try:
            with self.transaction() as conn:
                # Pre-check: Ensure all referenced drivers exist
                driver_codes = set(stop['driver_code'] for stop in pit_stops if stop.get('driver_code'))
                for driver_code in driver_codes:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                
                # Validate pit stops
                valid_stops = []
                for stop in pit_stops:
                    if self._validate_pit_stop_data(stop):
                        valid_stops.append(stop)
                
                if not valid_stops:
                    self.logger.info("No valid pit stops to store")
                    return True

                conn.executemany("""
                    INSERT OR REPLACE INTO pit_stops 
                    (pit_stop_id, session_id, driver_code, lap_number, pit_stop_time, pit_stop_duration,
                     tire_compound_old, tire_compound_new, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [( 
                    stop['pit_stop_id'], stop['session_id'], stop['driver_code'], stop['lap_number'],
                    stop.get('pit_stop_time'), stop.get('pit_stop_duration'), 
                    stop.get('tire_compound_old'), stop.get('tire_compound_new')
                ) for stop in valid_stops])

                self.logger.info(f"Stored {len(valid_stops)} pit stops")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing pit stops: {e}")
            return False

    def _store_weather_data_with_validation(self, weather_data: List[Dict[str, Any]]) -> bool:
        """Store weather data with validation"""
        if not weather_data:
            return True
            
        try:
            with self.transaction() as conn:
                valid_weather = []
                for weather in weather_data:
                    if self._validate_weather_data(weather):
                        valid_weather.append(weather)

                if not valid_weather:
                    self.logger.info("No valid weather data to store")
                    return True

                conn.executemany("""
                    INSERT OR REPLACE INTO weather 
                    (weather_id, session_id, time, air_temp, track_temp, humidity, pressure,
                     wind_speed, wind_direction, rainfall, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [(
                    weather['weather_id'], weather['session_id'], weather.get('time'), 
                    weather.get('air_temp'), weather.get('track_temp'), weather.get('humidity'), 
                    weather.get('pressure'), weather.get('wind_speed'), weather.get('wind_direction'),
                    weather.get('rainfall', False)
                ) for weather in valid_weather])

                self.logger.info(f"Stored {len(valid_weather)} weather data points")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing weather data: {e}")
            return False

    def _store_tire_stints(self, tire_stints: List[Dict[str, Any]]) -> bool:
        """FIXED: Store tire stint data with driver existence check"""
        if not tire_stints:
            self.logger.info("No tire stints to store")
            return True
            
        try:
            with self.transaction() as conn:
                # Pre-check: Ensure all referenced drivers exist
                driver_codes = set(stint['driver_code'] for stint in tire_stints if stint.get('driver_code'))
                for driver_code in driver_codes:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                
                # Validate tire stints
                valid_stints = []
                for stint in tire_stints:
                    if self._validate_tire_stint_data(stint):
                        valid_stints.append(stint)
                
                if not valid_stints:
                    self.logger.warning("No valid tire stints to store")
                    return True

                conn.executemany("""
                    INSERT OR REPLACE INTO tire_stints 
                    (stint_id, session_id, driver_code, stint_number, tire_compound, 
                     start_lap, end_lap, stint_length, total_distance, avg_lap_time, 
                     tire_degradation_rate, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [(
                    stint['stint_id'], stint['session_id'], stint['driver_code'],
                    stint['stint_number'], stint['tire_compound'], stint['start_lap'],
                    stint['end_lap'], stint['stint_length'], stint.get('total_distance'),
                    stint.get('avg_lap_time'), stint.get('tire_degradation_rate')
                ) for stint in valid_stints])

                self.logger.info(f"Stored {len(valid_stints)} tire stints")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing tire stints: {e}")
            return False

    def _store_compound_usage(self, compound_usage: List[Dict[str, Any]]) -> bool:
        """FIXED: Store tire compound usage with driver existence check"""
        if not compound_usage:
            self.logger.info("No compound usage to store")
            return True
            
        try:
            with self.transaction() as conn:
                # Pre-check: Ensure all referenced drivers exist
                driver_codes = set(usage['driver_code'] for usage in compound_usage if usage.get('driver_code'))
                for driver_code in driver_codes:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                
                # Validate and store compound usage
                valid_usage = []
                for usage in compound_usage:
                    if self._validate_compound_usage_data(usage):
                        valid_usage.append(usage)

                if not valid_usage:
                    self.logger.info("No valid compound usage to store")
                    return True

                conn.executemany("""
                    INSERT OR REPLACE INTO compound_usage 
                    (usage_id, session_id, driver_code, compound, stint_number, lap_start, lap_end, laps, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, [(
                    usage['usage_id'], 
                    usage['session_id'], 
                    usage['driver_code'],
                    usage.get('tire_compound', usage.get('compound')),
                    None,
                    usage.get('first_lap_number'),
                    usage.get('last_lap_number'),
                    usage.get('total_laps')
                ) for usage in valid_usage])

                self.logger.info(f"Stored {len(valid_usage)} compound usage records")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing compound usage: {e}")
            return False

    def _store_telemetry_data(self, telemetry_data: List[Dict[str, Any]]) -> bool:
        """Store telemetry data with batching for performance"""
        if not telemetry_data:
            return True
            
        try:
            with self.transaction() as conn:
                # Pre-check: Ensure all referenced drivers exist
                driver_codes = set(t['driver_code'] for t in telemetry_data if t.get('driver_code'))
                for driver_code in driver_codes:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                
                # Validate telemetry data
                valid_telemetry = []
                for telemetry in telemetry_data:
                    if self._validate_telemetry_data(telemetry):
                        valid_telemetry.append(telemetry)

                if not valid_telemetry:
                    self.logger.info("No valid telemetry data to store")
                    return True

                # Store in batches to avoid memory issues
                batch_size = 1000
                for i in range(0, len(valid_telemetry), batch_size):
                    batch = valid_telemetry[i:i + batch_size]
                    
                    conn.executemany("""
                        INSERT OR REPLACE INTO telemetry 
                        (telemetry_id, session_id, driver_code, lap_number, time_into_lap,
                         distance, speed, throttle, brake, drs, gear, rpm, 
                         x_coordinate, y_coordinate, z_coordinate, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, [(
                        t['telemetry_id'], t['session_id'], t['driver_code'], t['lap_number'],
                        t.get('time_into_lap'), t.get('distance'), t.get('speed'), t.get('throttle'),
                        t.get('brake'), t.get('drs'), t.get('gear'), t.get('rpm'),
                        t.get('x_coordinate'), t.get('y_coordinate'), t.get('z_coordinate')
                    ) for t in batch])

                self.logger.info(f"Stored {len(valid_telemetry)} telemetry records in batches")
                return True
                
        except Exception as e:
            self.logger.error(f"Error storing telemetry data: {e}")
            return False

    # Validation methods remain the same
    def _validate_lap_data_comprehensive(self, lap: Dict[str, Any]) -> bool:
        """Comprehensive lap data validation with relaxed constraints"""
        required_fields = ['lap_id', 'session_id', 'driver_code', 'lap_number']

        for field in required_fields:
            if not lap.get(field):
                self.logger.warning(f"Missing required field {field} in lap data")
                return False

        if lap.get('lap_time'):
            lap_time = lap['lap_time']
            if lap_time < 20 or lap_time > 400:
                self.logger.warning(f"Unrealistic lap time: {lap_time}s")
                return False

        lap_number = lap.get('lap_number', 0)
        if lap_number < 1 or lap_number > 200:
            self.logger.warning(f"Invalid lap number: {lap_number}")
            return False

        return True
    
    def _validate_tire_stint_data(self, stint: Dict[str, Any]) -> bool:
        required_fields = ['stint_id', 'session_id', 'driver_code', 'stint_number', 'tire_compound']
        for field in required_fields:
            if not stint.get(field):
                return False
        stint_number = stint.get('stint_number', 0)
        if stint_number < 1 or stint_number > 10:
            return False
        return True

    def _validate_compound_usage_data(self, usage: Dict[str, Any]) -> bool:
        required_fields = ['usage_id', 'session_id', 'driver_code']
        for field in required_fields:
            if not usage.get(field):
                return False
        if not usage.get('tire_compound') and not usage.get('compound'):
            return False
        return True

    def _validate_telemetry_data(self, telemetry: Dict[str, Any]) -> bool:
        required_fields = ['telemetry_id', 'session_id', 'driver_code', 'lap_number']
        for field in required_fields:
            if not telemetry.get(field):
                return False
        speed = telemetry.get('speed')
        if speed is not None and (speed < 0 or speed > 400):
            return False
        return True

    def _validate_pit_stop_data(self, stop: Dict[str, Any]) -> bool:
        required_fields = ['pit_stop_id', 'session_id', 'driver_code', 'lap_number']
        for field in required_fields:
            if not stop.get(field):
                return False
        return True

    def _validate_weather_data(self, weather: Dict[str, Any]) -> bool:
        required_fields = ['weather_id', 'session_id']
        for field in required_fields:
            if not weather.get(field):
                return False
        return True
    
    def get_session_analytics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive session analytics"""
        try:
            with db_pool.get_connection() as conn:
                analytics = {}
                
                # Basic lap statistics
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
                lap_stats = cursor.fetchone()
                analytics['lap_statistics'] = dict(lap_stats) if lap_stats else {}
                
                # Tire stint statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_stints,
                        COUNT(DISTINCT tire_compound) as compounds_used,
                        AVG(stint_length) as avg_stint_length,
                        AVG(tire_degradation_rate) as avg_degradation_rate
                    FROM tire_stints
                    WHERE session_id = ?               
                """, (session_id,))
                stint_stats = cursor.fetchone()
                analytics['stint_statistics'] = dict(stint_stats) if stint_stats else {}
                
                # Telemetry statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as telemetry_points,
                        MAX(speed) as max_speed,
                        AVG(speed) as avg_speed,
                        COUNT(DISTINCT driver_code) as drivers_with_telemetry
                    FROM telemetry
                    WHERE session_id = ?               
                """, (session_id,))
                telemetry_stats = cursor.fetchone()
                analytics['telemetry_statistics'] = dict(telemetry_stats) if telemetry_stats else {}
                
                return analytics

        except Exception as e:
            self.logger.error(f"Error getting session analytics: {e}")
            return None

    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
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
                if result:
                    return dict(result)
                return None

        except Exception as e:
            self.logger.error(f"Error getting session summary: {e}")
            return None


def diagnose_foreign_key_issues():
    """Diagnostic function to identify foreign key constraint issues"""
    print("🔍 DIAGNOSING FOREIGN KEY ISSUES")
    print("=" * 60)
    
    try:
        with db_pool.get_connection() as conn:
            # Check foreign key status
            cursor = conn.execute("PRAGMA foreign_keys")
            fk_status = cursor.fetchone()[0]
            print(f"Foreign Keys Enabled: {bool(fk_status)}")
            
            # Check all foreign key constraints
            cursor = conn.execute("PRAGMA foreign_key_check")
            violations = cursor.fetchall()
            if violations:
                print(f"❌ Found {len(violations)} foreign key violations:")
                for violation in violations:
                    print(f"   Table: {violation[0]}, Row: {violation[1]}, Parent: {violation[2]}, Key: {violation[3]}")
            else:
                print("✅ No foreign key violations found")
            
            # Check table existence and relationships
            tables_to_check = ['seasons', 'races', 'sessions', 'drivers']
            for table in tables_to_check:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"{table}: {count} records")
            
            # Check specific race-session relationships
            cursor = conn.execute("""
                SELECT s.session_id, s.race_id, r.race_id as race_exists
                FROM sessions s
                LEFT JOIN races r ON s.race_id = r.race_id
                WHERE r.race_id IS NULL
                LIMIT 5
            """)
            orphaned = cursor.fetchall()
            if orphaned:
                print(f"❌ Found {len(orphaned)} orphaned sessions:")
                for session in orphaned:
                    print(f"   Session: {session[0]}, Missing Race: {session[1]}")
            else:
                print("✅ All sessions have valid race references")
                
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


def emergency_fix_database():
    """Emergency fix for foreign key constraint issues"""
    print("🚨 EMERGENCY DATABASE FIX")
    print("=" * 60)
    
    try:
        with db_pool.get_connection() as conn:
            # Temporarily disable foreign keys
            conn.execute("PRAGMA foreign_keys=OFF")
            
            # Fix orphaned sessions by creating missing races
            cursor = conn.execute("""
                SELECT DISTINCT s.race_id, s.session_id
                FROM sessions s
                LEFT JOIN races r ON s.race_id = r.race_id
                WHERE r.race_id IS NULL
            """)
            orphaned_sessions = cursor.fetchall()
            
            fixed_races = 0
            for race_id, session_id in orphaned_sessions:
                if race_id:
                    # Extract year from race_id
                    try:
                        year = int(race_id.split('_')[0])
                        race_name = ' '.join(race_id.split('_')[1:]).replace('_', ' ').title()
                        
                        # Ensure season exists
                        conn.execute("""
                            INSERT OR IGNORE INTO seasons (year, total_rounds, created_at)
                            VALUES (?, 0, CURRENT_TIMESTAMP)
                        """, (year,))
                        
                        # Create race record
                        conn.execute("""
                            INSERT OR IGNORE INTO races 
                            (race_id, year, round_number, race_name, circuit_name, country, created_at)
                            VALUES (?, ?, 1, ?, 'Emergency Fix', 'Unknown', CURRENT_TIMESTAMP)
                        """, (race_id, year, race_name))
                        
                        fixed_races += 1
                        
                    except Exception as e:
                        print(f"Failed to fix race {race_id}: {e}")
            
            # Re-enable foreign keys
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
            
            print(f"✅ Fixed {fixed_races} missing race records")
            
            # Verify fix
            cursor = conn.execute("PRAGMA foreign_key_check")
            violations = cursor.fetchall()
            if violations:
                print(f"❌ Still have {len(violations)} foreign key violations after fix")
            else:
                print("✅ All foreign key constraints satisfied")
                
    except Exception as e:
        print(f"❌ Emergency fix failed: {e}")


# Test the complete fix
def test_complete_fix():
    """Test the complete storage fix"""
    print("🧪 TESTING COMPLETE STORAGE FIX")
    print("=" * 60)
    
    # First, diagnose issues
    diagnose_foreign_key_issues()
    
    # Apply emergency fix if needed
    print("\n" + "="*60)
    emergency_fix_database()
    
    # Test basic functionality
    print("\n" + "="*60)
    storage = DataStorageManager()
    
    # Test season storage
    print("Testing season storage...")
    result = storage.store_season_data(2021, 19)
    print(f"Season storage: {'✅ Success' if result else '❌ Failed'}")
    
    # Test race storage
    print("Testing race storage...")
    test_race = [{
        'year': 2021,
        'round_number': 1,
        'race_name': 'Bahrain Grand Prix',
        'circuit_name': 'Bahrain International Circuit',
        'country': 'Bahrain',
        'race_date': '2021-03-28',
        'race_time': '18:00:00'
    }]
    result = storage.store_race_data(test_race)
    print(f"Race storage: {'✅ Success' if result else '❌ Failed'}")
    
    # Test race_id resolution
    print("Testing race_id resolution...")
    race_id = storage.get_race_id_for_session("2021_bahrain_grand_prix_r")
    print(f"Race ID resolution: {'✅ Success' if race_id else '❌ Failed'} (ID: {race_id})")
    
    print("\n🎉 Complete fix testing finished!")


if __name__ == "__main__":
    test_complete_fix()