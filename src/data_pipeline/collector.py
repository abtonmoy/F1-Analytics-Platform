# collector.py - F1DataCollector with telemetry support
import os
import fastf1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
from dataclasses import dataclass
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool

logger = setup_logger(__name__)

class F1DataCollector:
    """F1 Data Collector with telemetry, tire compounds, and stint analysis"""

    def __init__(self):
        os.makedirs(Config.FASTF1_CACHE_DIR, exist_ok=True)
        fastf1.Cache.enable_cache(Config.FASTF1_CACHE_DIR)
        logger.info(f"FastF1 cache enabled at {Config.FASTF1_CACHE_DIR}")
        
        # Correct FastF1 session types
        self.valid_session_types = {
            'FP1': 'Practice 1',
            'FP2': 'Practice 2', 
            'FP3': 'Practice 3',
            'Q': 'Qualifying',
            'R': 'Race',
            'Sprint': 'Sprint',
            'SQ': 'Sprint Qualifying'
        }
        
        # Session type mapping from FastF1 names to database values
        self.session_type_mapping = {
            'Practice 1': 'FP1',
            'Practice 2': 'FP2',
            'Practice 3': 'FP3',
            'Qualifying': 'Q',
            'Race': 'R',
            'Sprint': 'Sprint',
            'Sprint Qualifying': 'SQ',
            'Sprint Shootout': 'SQ'
        }

        # Telemetry channels to collect
        self.telemetry_channels = [
            'Speed', 'Throttle', 'Brake', 'DRS', 'nGear', 'RPM',
            'Distance', 'RelativeDistance', 'X', 'Y', 'Z'
        ]

    def collect_session_data(self, year: int, race_name: str, session_type: str) -> Optional[Dict[str, Any]]:
        """Session data collection with telemetry, stints, and compound analysis"""
        logger.info(f"Collecting {session_type} data for {race_name} {year}")

        if session_type not in self.valid_session_types:
            logger.error(f"Invalid session type '{session_type}'. Valid types: {list(self.valid_session_types.keys())}")
            return None

        try:
            time.sleep(0.5)  # Rate limiting
            
            fastf1_session_type = self.valid_session_types[session_type]
            session = fastf1.get_session(year, race_name, fastf1_session_type)
            session.load()

            if session.results.empty:
                logger.warning(f"No data available for {race_name} {year} {session_type}")
                return None

            # Create session_id
            race_name_formatted = race_name.lower().replace(" ", "_")
            session_id = f"{year}_{race_name_formatted}_{session_type}".lower()

            # Collect all data
            session_data = {
                'session_id': session_id,
                'session_info': self._extract_session_info(session, session_type),
                'drivers': self._extract_drivers_info(session),
                'lap_times': self._extract_lap_times(session, session_id),
                'tire_stints': self._extract_tire_stints(session, session_id),
                'compound_usage': self._extract_compound_usage(session, session_id),
                'telemetry': self._extract_telemetry_data(session, session_id),
                'pit_stops': self._extract_pit_stops(session, session_id),
                'weather': self._extract_weather_data(session, session_id)
            }

            logger.info(f'Successfully collected {session_type} data for {race_name} {year}')
            logger.info(f'  - {len(session_data["lap_times"])} lap times')
            logger.info(f'  - {len(session_data["tire_stints"])} tire stints') 
            logger.info(f'  - {len(session_data["telemetry"])} telemetry records')
            
            return session_data
        
        except Exception as e:
            logger.error(f"Error collecting {session_type} data for {race_name} {year}: {e}")
            return None

    def _extract_lap_times(self, session, session_id: str) -> List[Dict[str, Any]]:
        """Lap times extraction with compound and stint information"""
        lap_times = []

        try:
            laps = session.laps
            for idx, lap in laps.iterrows():
                lap_id = f'{session_id}_{lap["Driver"]}_{lap["LapNumber"]}'

                lap_time_seconds = self._time_to_seconds(lap['LapTime'])
                
                # Relaxed lap time validation
                if lap_time_seconds and (lap_time_seconds > 400 or lap_time_seconds < 20):
                    logger.debug(f"Skipping unrealistic lap time: {lap_time_seconds}s")
                    continue
                    
                driver_number = lap['DriverNumber']
                driver_info = session.get_driver(driver_number)
                driver_code = driver_info.get('Abbreviation', driver_number)

                # Lap data with stint information
                lap_data = {
                    'lap_id': lap_id,
                    'session_id': session_id,
                    'driver_code': driver_code,
                    'lap_number': int(lap['LapNumber']),
                    'lap_time': lap_time_seconds,
                    'sector_1_time': self._time_to_seconds(lap.get('Sector1Time')),
                    'sector_2_time': self._time_to_seconds(lap.get('Sector2Time')),
                    'sector_3_time': self._time_to_seconds(lap.get('Sector3Time')),
                    'speed_i1': self._safe_numeric_extract(lap, 'SpeedI1'),
                    'speed_i2': self._safe_numeric_extract(lap, 'SpeedI2'),
                    'speed_fl': self._safe_numeric_extract(lap, 'SpeedFL'),
                    'speed_st': self._safe_numeric_extract(lap, 'SpeedST'),
                    'is_personal_best': bool(lap.get('IsPersonalBest', False)),
                    'is_accurate': bool(lap.get('IsAccurate', True)),
                    'tire_compound': self._safe_string_extract(lap, 'Compound'),
                    'tire_age': self._safe_numeric_extract(lap, 'TyreLife'),
                    'fresh_tyre': bool(lap.get('FreshTyre', False)),
                    'pit_out_time': self._time_to_seconds(lap.get('PitOutTime')),
                    'pit_in_time': self._time_to_seconds(lap.get('PitInTime')),
                    'deleted': bool(lap.get('Deleted', False)),
                    'deleted_reason': self._safe_string_extract(lap, 'DeletedReason'),
                    'track_status': self._safe_string_extract(lap, 'TrackStatus'),
                    'position': self._safe_numeric_extract(lap, 'Position'),
                    # Additional fields
                    'stint_number': None,  # Will be calculated in stint analysis
                    'compound_age': self._safe_numeric_extract(lap, 'TyreLife'),
                    'fuel_load_estimated': None,  # Could be estimated from lap time progression
                    'weather_condition': self._safe_string_extract(lap, 'TrackStatus')
                }

                lap_times.append(lap_data)

        except Exception as e:
            logger.error(f"Error extracting lap times: {e}")

        logger.info(f'Extracted {len(lap_times)} lap times')
        return lap_times

    def _extract_tire_stints(self, session, session_id: str) -> List[Dict[str, Any]]:
        """Extract tire stints with detailed analysis"""
        tire_stints = []

        try:
            laps = session.laps
            
            # Group laps by driver
            for driver_code in session.drivers:
                driver_laps = laps.pick_drivers(driver_code).sort_values('LapNumber')
                
                if driver_laps.empty:
                    continue

                stint_number = 1
                current_compound = None
                stint_start_lap = None
                stint_laps = []
                
                for idx, lap in driver_laps.iterrows():
                    lap_compound = self._safe_string_extract(lap, 'Compound')
                    lap_time = self._time_to_seconds(lap['LapTime'])
                    lap_number = int(lap['LapNumber'])
                    
                    # Detect stint change (pit stop required)
                    is_pit_lap = pd.notna(lap.get('PitOutTime')) or pd.notna(lap.get('PitInTime'))
                    compound_changed = (lap_compound != current_compound and lap_compound is not None)
                    
                    if (is_pit_lap and current_compound is not None) or (compound_changed and is_pit_lap):
                        # End current stint
                        if current_compound is not None and stint_laps:
                            stint_id = f"{session_id}_{driver_code}_stint_{stint_number}"
                            stint_data = self._calculate_stint_statistics(
                                stint_id, session_id, str(driver_code), stint_number,
                                current_compound, stint_start_lap, stint_laps
                            )
                            if stint_data:
                                tire_stints.append(stint_data)
                            stint_number += 1
                        
                        # Start new stint
                        current_compound = lap_compound
                        stint_start_lap = lap_number
                        stint_laps = []
                    
                    # Add lap to current stint
                    if current_compound is not None and lap_time is not None:
                        stint_laps.append({
                            'lap_number': lap_number,
                            'lap_time': lap_time,
                            'tire_age': self._safe_numeric_extract(lap, 'TyreLife'),
                            'sector_times': [
                                self._time_to_seconds(lap.get('Sector1Time')),
                                self._time_to_seconds(lap.get('Sector2Time')),
                                self._time_to_seconds(lap.get('Sector3Time'))
                            ]
                        })
                
                # End final stint
                if current_compound is not None and stint_laps:
                    stint_id = f"{session_id}_{driver_code}_stint_{stint_number}"
                    stint_data = self._calculate_stint_statistics(
                        stint_id, session_id, str(driver_code), stint_number,
                        current_compound, stint_start_lap, stint_laps
                    )
                    if stint_data:
                        tire_stints.append(stint_data)

        except Exception as e:
            logger.error(f"Error extracting tire stints: {e}")

        logger.info(f'Extracted {len(tire_stints)} tire stints')
        return tire_stints

    def _calculate_stint_statistics(self, stint_id: str, session_id: str, driver_code: str, 
                                  stint_number: int, compound: str, start_lap: int, 
                                  stint_laps: List[Dict]) -> Optional[Dict[str, Any]]:
        """Calculate detailed stint statistics"""
        if not stint_laps:
            return None
        
        try:
            # Basic stint info
            end_lap = stint_laps[-1]['lap_number']
            stint_length = len(stint_laps)
            
            # Lap time analysis
            lap_times = [lap['lap_time'] for lap in stint_laps if lap['lap_time'] is not None]
            if not lap_times:
                return None
            
            avg_lap_time = np.mean(lap_times)
            fastest_lap_time = min(lap_times)
            slowest_lap_time = max(lap_times)
            
            # Tire degradation analysis (lap time progression)
            tire_degradation_rate = None
            if len(lap_times) >= 3:
                # Simple linear regression for degradation
                x = np.arange(len(lap_times))
                y = np.array(lap_times)
                try:
                    coeffs = np.polyfit(x, y, 1)
                    tire_degradation_rate = float(coeffs[0])  # seconds per lap degradation
                except:
                    tire_degradation_rate = None
            
            # Tire age analysis
            tire_ages = [lap.get('tire_age') for lap in stint_laps if lap.get('tire_age') is not None]
            starting_tire_age = min(tire_ages) if tire_ages else None
            ending_tire_age = max(tire_ages) if tire_ages else None
            
            # Estimated distance (would need circuit length)
            total_distance = None  # stint_length * circuit_length_km
            
            return {
                'stint_id': stint_id,
                'session_id': session_id,
                'driver_code': driver_code,
                'stint_number': stint_number,
                'tire_compound': compound,
                'start_lap': start_lap,
                'end_lap': end_lap,
                'stint_length': stint_length,
                'total_distance': total_distance,
                'avg_lap_time': avg_lap_time,
                'fastest_lap_time': fastest_lap_time,
                'slowest_lap_time': slowest_lap_time,
                'tire_degradation_rate': tire_degradation_rate,
                'starting_tire_age': starting_tire_age,
                'ending_tire_age': ending_tire_age,
                'lap_time_std': np.std(lap_times) if len(lap_times) > 1 else None,
                'performance_consistency': 1.0 / (np.std(lap_times) + 0.001) if len(lap_times) > 1 else None
            }
            
        except Exception as e:
            logger.error(f"Error calculating stint statistics: {e}")
            return None

    def _extract_compound_usage(self, session, session_id: str) -> List[Dict[str, Any]]:
        """Extract detailed tire compound usage analysis"""
        compound_usage = []
        
        try:
            laps = session.laps
            
            # Analyze compound usage per driver
            for driver_code in session.drivers:
                driver_laps = laps.pick_drivers(driver_code)
                
                if driver_laps.empty:
                    continue
                
                # Get compound usage statistics
                compound_stats = {}
                for idx, lap in driver_laps.iterrows():
                    compound = self._safe_string_extract(lap, 'Compound')
                    lap_time = self._time_to_seconds(lap['LapTime'])
                    tire_age = self._safe_numeric_extract(lap, 'TyreLife')
                    
                    if compound and lap_time:
                        if compound not in compound_stats:
                            compound_stats[compound] = {
                                'laps_count': 0,
                                'lap_times': [],
                                'tire_ages': [],
                                'first_lap': float('inf'),
                                'last_lap': 0
                            }
                        
                        compound_stats[compound]['laps_count'] += 1
                        compound_stats[compound]['lap_times'].append(lap_time)
                        if tire_age is not None:
                            compound_stats[compound]['tire_ages'].append(tire_age)
                        compound_stats[compound]['first_lap'] = min(compound_stats[compound]['first_lap'], int(lap['LapNumber']))
                        compound_stats[compound]['last_lap'] = max(compound_stats[compound]['last_lap'], int(lap['LapNumber']))
                
                # Create compound usage records
                for compound, stats in compound_stats.items():
                    if stats['lap_times']:
                        usage_id = f"{session_id}_{driver_code}_{compound}"
                        
                        # Fixed tire performance calculation
                        avg_lap_time = np.mean(stats['lap_times'])
                        fastest_lap_time = min(stats['lap_times'])
                        tire_performance = fastest_lap_time / avg_lap_time
                        
                        compound_usage.append({
                            'usage_id': usage_id,
                            'session_id': session_id,
                            'driver_code': str(driver_code),
                            'tire_compound': compound,
                            'total_laps': stats['laps_count'],
                            'first_lap_number': stats['first_lap'],
                            'last_lap_number': stats['last_lap'],
                            'avg_lap_time': avg_lap_time,
                            'fastest_lap_time': fastest_lap_time,
                            'tire_performance': tire_performance,
                            'avg_tire_age': np.mean(stats['tire_ages']) if stats['tire_ages'] else None,
                            'max_tire_age': max(stats['tire_ages']) if stats['tire_ages'] else None
                        })
        
        except Exception as e:
            logger.error(f"Error extracting compound usage: {e}")
        
        logger.info(f'Extracted {len(compound_usage)} compound usage records')
        return compound_usage

    def _extract_telemetry_data(self, session, session_id: str, 
                               max_drivers: int = 5, max_laps_per_driver: int = 3) -> List[Dict[str, Any]]:
        """Extract telemetry data with distance-based sampling"""
        telemetry_data = []
        
        try:
            # Limit telemetry collection to avoid massive datasets
            drivers = list(session.drivers)[:max_drivers]  # First N drivers
            
            for driver_code in drivers:
                driver_laps = session.laps.pick_drivers(driver_code)
                
                if driver_laps.empty:
                    continue
                
                # Get fastest laps for telemetry
                valid_laps = driver_laps[driver_laps['LapTime'].notna()]
                if valid_laps.empty:
                    continue
                
                fastest_laps = valid_laps.nsmallest(max_laps_per_driver, 'LapTime')
                
                for idx, lap in fastest_laps.iterrows():
                    try:
                        # Get telemetry for this lap
                        lap_telemetry = lap.get_telemetry()
                        
                        if lap_telemetry.empty:
                            continue
                        
                        # Distance-based sampling (every 10 meters)
                        if 'Distance' in lap_telemetry.columns:
                            lap_telemetry['DistanceBucket'] = (lap_telemetry['Distance'] // 10).astype(int)
                            sampled_telemetry = lap_telemetry.groupby('DistanceBucket').first().reset_index()
                        else:
                            # Fallback to time-based sampling if distance not available
                            sampled_telemetry = lap_telemetry.iloc[::10]
                        
                        for tel_idx, tel_point in sampled_telemetry.iterrows():
                            telemetry_id = f'{session_id}_{driver_code}_{lap["LapNumber"]}_{tel_idx}'
                            
                            telemetry_record = {
                                'telemetry_id': telemetry_id,
                                'session_id': session_id,
                                'driver_code': str(driver_code),
                                'lap_number': int(lap['LapNumber']),
                                'time_into_lap': self._safe_numeric_extract(tel_point, 'Time'),
                                'distance': self._safe_numeric_extract(tel_point, 'Distance'),
                                'speed': self._safe_numeric_extract(tel_point, 'Speed'),
                                'throttle': self._safe_numeric_extract(tel_point, 'Throttle'),
                                'brake': self._safe_numeric_extract(tel_point, 'Brake'),
                                'drs': self._safe_numeric_extract(tel_point, 'DRS'),
                                'gear': self._safe_numeric_extract(tel_point, 'nGear'),
                                'rpm': self._safe_numeric_extract(tel_point, 'RPM'),
                                'x_coordinate': self._safe_numeric_extract(tel_point, 'X'),
                                'y_coordinate': self._safe_numeric_extract(tel_point, 'Y'),
                                'z_coordinate': self._safe_numeric_extract(tel_point, 'Z')
                            }
                            
                            telemetry_data.append(telemetry_record)
                    
                    except Exception as lap_error:
                        logger.debug(f"Could not get telemetry for {driver_code} lap {lap['LapNumber']}: {lap_error}")
                        continue
        
        except Exception as e:
            logger.error(f"Error extracting telemetry data: {e}")
        
        logger.info(f'Extracted {len(telemetry_data)} telemetry records')
        return telemetry_data

    def _extract_pit_stops(self, session, session_id: str) -> List[Dict[str, Any]]:
        """Pit stop extraction with compound change tracking"""
        pit_stops = []

        try:
            laps = session.laps
            
            for driver_code in session.drivers:
                driver_laps = laps.pick_drivers(driver_code).sort_values('LapNumber')
                
                previous_compound = None
                
                for idx, lap in driver_laps.iterrows():
                    current_compound = self._safe_string_extract(lap, 'Compound')
                    is_pit_in = pd.notna(lap.get('PitInTime'))
                    is_pit_out = pd.notna(lap.get('PitOutTime'))
                    
                    # Only register pit stops when pit timing exists
                    if is_pit_in or is_pit_out:
                        pit_stop_id = f"{session_id}_{driver_code}_{lap['LapNumber']}_pit"
                        
                        # Try to calculate pit stop duration
                        pit_duration = None
                        if is_pit_in and is_pit_out:
                            pit_in_time = self._time_to_seconds(lap.get('PitInTime'))
                            pit_out_time = self._time_to_seconds(lap.get('PitOutTime'))
                            if pit_in_time and pit_out_time:
                                pit_duration = pit_out_time - pit_in_time
                        
                        pit_stop = {
                            'pit_stop_id': pit_stop_id,
                            'session_id': session_id,
                            'driver_code': str(driver_code),
                            'lap_number': int(lap['LapNumber']),
                            'pit_stop_time': self._time_to_seconds(lap.get('PitInTime')),
                            'pit_stop_duration': pit_duration,
                            'tire_compound_old': previous_compound,
                            'tire_compound_new': current_compound,
                            'tire_age_old': None,  # Would need to track from previous stint
                            'tire_age_new': self._safe_numeric_extract(lap, 'TyreLife'),
                            'pit_stop_type': 'compound_change' if current_compound != previous_compound else 'standard'
                        }
                        
                        pit_stops.append(pit_stop)
                    
                    if current_compound:
                        previous_compound = current_compound
                        
        except Exception as e:
            logger.error(f"Error extracting pit stops: {e}")
        
        logger.info(f"Extracted {len(pit_stops)} pit stops")
        return pit_stops

    # Utility methods
    def _extract_session_info(self, session, requested_session_type: str) -> Dict[str, Any]:
        session_name = getattr(session, 'name', requested_session_type)
        mapped_session_type = self.session_type_mapping.get(session_name, requested_session_type)
        
        return {
            'session_type': mapped_session_type,
            'session_date': session.date.strftime('%Y-%m-%d') if session.date else None,
            'session_time': session.date.strftime('%H:%M:%S') if session.date else None,
            'air_temp': getattr(session, 'air_temp', None),
            'track_temp': getattr(session, 'track_temp', None),
            'humidity': getattr(session, 'humidity', None),
            'pressure': getattr(session, 'pressure', None),
            'wind_speed': getattr(session, 'wind_speed', None),
            'wind_direction': getattr(session, 'wind_direction', None)
        }
    
    def _extract_drivers_info(self, session) -> List[Dict[str, Any]]:
        drivers = []
        
        for driver_code in session.drivers:
            try:
                driver_info = session.get_driver(driver_code)
                driver_abbr = driver_info.get('Abbreviation', str(driver_code))
                
                team_color = getattr(driver_info, 'TeamColor', None)
                if team_color is not None:
                    team_color = str(team_color)
                
                drivers.append({
                    'driver_code': driver_abbr,
                    'first_name': self._safe_string_extract(driver_info, 'FirstName'),
                    'last_name': self._safe_string_extract(driver_info, 'LastName'),
                    'full_name': self._safe_string_extract(driver_info, 'FullName'),
                    'team_name': self._safe_string_extract(driver_info, 'TeamName'),
                    'team_color': team_color,
                    'driver_number': self._safe_numeric_extract(driver_info, 'DriverNumber'),
                    'nationality': self._safe_string_extract(driver_info, 'CountryCode')
                })
            except Exception as e:
                logger.warning(f"Error extracting info for driver {driver_code}: {e}")
                continue
        
        return drivers

    def _extract_weather_data(self, session, session_id: str) -> List[Dict[str, Any]]:
        """Extract weather data"""
        weather_data = []
        
        try:
            if hasattr(session, 'weather_data') and session.weather_data is not None:
                for idx, weather in session.weather_data.iterrows():
                    weather_id = f"{session_id}_weather_{idx}"
                    
                    weather_point = {
                        'weather_id': weather_id,
                        'session_id': session_id,
                        'time': self._safe_numeric_extract(weather, 'Time'),
                        'air_temp': self._safe_numeric_extract(weather, 'AirTemp'),
                        'track_temp': self._safe_numeric_extract(weather, 'TrackTemp'),
                        'humidity': self._safe_numeric_extract(weather, 'Humidity'),
                        'pressure': self._safe_numeric_extract(weather, 'Pressure'),
                        'wind_speed': self._safe_numeric_extract(weather, 'WindSpeed'),
                        'wind_direction': self._safe_numeric_extract(weather, 'WindDirection'),
                        'rainfall': bool(weather.get('Rainfall', False))
                    }
                    
                    weather_data.append(weather_point)
            
        except Exception as e:
            logger.warning(f"Weather data not available: {e}")
        
        return weather_data

    def collect_season_schedule(self, year: int) -> List:
        """Collect season schedule"""
        logger.info(f"Collecting schedule for {year} season")

        try:
            schedule = fastf1.get_event_schedule(year)
            events = []

            for idx, event in schedule.iterrows():
                if event.get('EventFormat') == 'conventional':
                    from dataclasses import dataclass
                    
                    @dataclass
                    class RaceEvent:
                        year: int
                        round_number: int
                        race_name: str
                        circuit_name: str
                        country: str
                        race_date: str
                        race_time: str = None
                    
                    race_event = RaceEvent(
                        year=year,
                        round_number=event['RoundNumber'],
                        race_name=event['EventName'],
                        circuit_name=event.get('Location', 'Unknown'),
                        country=event.get('Country', 'Unknown'),
                        race_date=event['Session5Date'].strftime('%Y-%m-%d') if pd.notna(event['Session5Date']) else None,
                        race_time=event['Session5Date'].strftime('%H:%M:%S') if pd.notna(event['Session5Date']) else None
                    )
                    events.append(race_event)

            logger.info(f'Found {len(events)} race events for {year}')
            return events
        except Exception as e:
            logger.error(f'Error collecting schedule for {year}: {e}')
            return []

    def _time_to_seconds(self, time_value) -> Optional[float]:
        """Convert various time formats to seconds"""
        if pd.isna(time_value) or time_value is None:
            return None
        
        try:
            if isinstance(time_value, pd.Timedelta):
                return float(time_value.total_seconds())
            if isinstance(time_value, timedelta):
                return float(time_value.total_seconds())
            if isinstance(time_value, (int, float)):
                return float(time_value)
            if isinstance(time_value, str):
                return float(time_value)
            if hasattr(time_value, 'total_seconds'):
                return float(time_value.total_seconds())
            return float(time_value)
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Could not convert time value {time_value} to seconds: {e}")
            return None

    def _safe_numeric_extract(self, obj, attr_name) -> Optional[float]:
        """Safely extract numeric values"""
        try:
            if hasattr(obj, attr_name):
                value = getattr(obj, attr_name, None)
            elif hasattr(obj, 'get'):
                value = obj.get(attr_name)
            elif isinstance(obj, dict):
                value = obj.get(attr_name)
            else:
                try:
                    value = obj[attr_name] if attr_name in obj else None
                except (KeyError, TypeError):
                    return None
            
            if pd.isna(value) or value is None:
                return None
            return float(value)
        except (ValueError, TypeError, AttributeError):
            return None

    def _safe_string_extract(self, obj, attr_name) -> Optional[str]:
        """Safely extract string values"""
        try:
            if hasattr(obj, attr_name):
                value = getattr(obj, attr_name, None)
            elif hasattr(obj, 'get'):
                value = obj.get(attr_name)
            elif isinstance(obj, dict):
                value = obj.get(attr_name)
            else:
                try:
                    value = obj[attr_name] if attr_name in obj else None
                except (KeyError, TypeError):
                    return None
            
            if pd.isna(value) or value is None:
                return None
            return str(value)
        except (ValueError, TypeError, AttributeError):
            return None


if __name__ == "__main__":
    collector = F1DataCollector()
    
    # Test with one race
    session_data = collector.collect_session_data(2024, "Monaco", "R")
    if session_data:
        print(f"Collected data with {len(session_data['lap_times'])} lap times")
        print("Driver codes:", [d['driver_code'] for d in session_data['drivers']])