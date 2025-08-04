# schema.py - CORRECTED VERSION with proper foreign key handling
import sqlite3
from pathlib import Path
from config.settings import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(Config.DATA_DIR / "f1_analytics.db")
        self.init_database()

    def get_connection(self):
        """Get db connection with optimizations"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row

        # IMPORTANT: Enable foreign keys FIRST
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")

        return conn

    def init_database(self):
        """Initialize the db with all the tables in correct dependency order"""
        logger.info(f"Initializing db at {self.db_path}")

        with self.get_connection() as conn:
            # Create tables in dependency order (parents first)
            self.create_seasons_table(conn)      # No dependencies
            self.create_races_table(conn)        # Depends on seasons
            self.create_drivers_table(conn)      # No dependencies
            self.create_sessions_table(conn)     # Depends on races
            self.create_lap_times_table(conn)    # Depends on sessions, drivers
            self.create_pit_stops_table(conn)    # Depends on sessions, drivers
            self.create_weather_table(conn)      # Depends on sessions
            self.create_tire_stints_table(conn)  # Depends on sessions, drivers
            self.create_compound_usage_table(conn) # Depends on sessions, drivers
            self.create_telemetry_table(conn)    # Depends on sessions, drivers
            
            # Check and fix existing tables if needed
            self.migrate_existing_tables(conn)
            
            # Create indexes AFTER migration
            self.create_indexes(conn)
            
            conn.commit()
            logger.info("Database initialization complete")

    def migrate_existing_tables(self, conn):
        """FIXED: Check and migrate existing tables to match expected schema"""
        try:
            # 1. Ensure all necessary columns exist
            self._add_missing_columns(conn)
            
            # 2. Fix orphaned sessions
            self._fix_orphaned_sessions(conn)
            
            # 3. Create minimal driver records for existing lap times
            self._ensure_driver_records_exist(conn)
            
        except Exception as e:
            logger.error(f"Error during table migration: {e}")

    def _add_missing_columns(self, conn):
        """Add any missing columns to existing tables"""
        try:
            # Check sessions table for race_id column
            cursor = conn.execute("PRAGMA table_info(sessions)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            if 'race_id' not in columns:
                logger.info("Adding missing race_id column to sessions table")
                conn.execute("ALTER TABLE sessions ADD COLUMN race_id TEXT")
            
            # Check telemetry table for required columns
            cursor = conn.execute("PRAGMA table_info(telemetry)")
            telemetry_columns = {row[1]: row[2] for row in cursor.fetchall()}
            
            missing_telemetry_cols = [
                ('session_id', 'TEXT'),
                ('driver_code', 'TEXT'), 
                ('lap_number', 'INTEGER'),
                ('time_into_lap', 'REAL')
            ]
            
            for col_name, col_type in missing_telemetry_cols:
                if col_name not in telemetry_columns:
                    logger.info(f"Adding {col_name} column to telemetry table")
                    conn.execute(f"ALTER TABLE telemetry ADD COLUMN {col_name} {col_type}")
            
            # Ensure compound_usage table exists
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='compound_usage'")
            if not cursor.fetchone():
                logger.info("Creating missing compound_usage table")
                self.create_compound_usage_table(conn)
                
        except Exception as e:
            logger.error(f"Error adding missing columns: {e}")

    def _fix_orphaned_sessions(self, conn):
        """Fix sessions that reference non-existent races"""
        try:
            cursor = conn.execute("""
                SELECT s.session_id, s.race_id 
                FROM sessions s
                LEFT JOIN races r ON s.race_id = r.race_id
                WHERE r.race_id IS NULL AND s.race_id IS NOT NULL
            """)
            orphaned_sessions = cursor.fetchall()
            
            logger.info(f"Found {len(orphaned_sessions)} orphaned sessions")
            
            for session in orphaned_sessions:
                session_id = session['session_id']
                race_id = session['race_id']
                
                # Create minimal race record
                race_name = self._generate_race_name_from_id(race_id)
                year = self._extract_year_from_race_id(race_id)
                
                if year:
                    # Ensure season exists
                    conn.execute("""
                        INSERT OR IGNORE INTO seasons (year, total_rounds, created_at)
                        VALUES (?, 0, CURRENT_TIMESTAMP)
                    """, (year,))
                    
                    # Create race record
                    conn.execute("""
                        INSERT OR IGNORE INTO races 
                        (race_id, year, round_number, race_name, circuit_name, country, created_at)
                        VALUES (?, ?, 1, ?, 'Unknown', 'Unknown', CURRENT_TIMESTAMP)
                    """, (race_id, year, race_name))
                    
                    logger.info(f"Created race record: {race_id}")
                    
        except Exception as e:
            logger.error(f"Error fixing orphaned sessions: {e}")

    def _ensure_driver_records_exist(self, conn):
        """Ensure driver records exist for all referenced drivers"""
        try:
            # Get all driver codes referenced in lap_times
            cursor = conn.execute("""
                SELECT DISTINCT driver_code 
                FROM lap_times 
                WHERE driver_code NOT IN (SELECT driver_code FROM drivers)
            """)
            missing_drivers = [row[0] for row in cursor.fetchall()]
            
            for driver_code in missing_drivers:
                conn.execute("""
                    INSERT OR IGNORE INTO drivers (driver_code, created_at)
                    VALUES (?, CURRENT_TIMESTAMP)
                """, (driver_code,))
            
            if missing_drivers:
                logger.info(f"Created {len(missing_drivers)} missing driver records")
                
        except Exception as e:
            logger.error(f"Error ensuring driver records: {e}")

    def _generate_race_name_from_id(self, race_id: str) -> str:
        """Generate a readable race name from race_id"""
        try:
            parts = race_id.split('_')[1:]  # Skip year
            return ' '.join(word.capitalize() for word in parts)
        except:
            return "Unknown Race"

    def _extract_year_from_race_id(self, race_id: str) -> int:
        """Extract year from race_id"""
        try:
            return int(race_id.split('_')[0])
        except:
            return None

    def create_seasons_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS seasons (
                year INTEGER PRIMARY KEY,
                total_rounds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def create_races_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS races (
                race_id TEXT PRIMARY KEY,
                year INTEGER,
                round_number INTEGER,
                race_name TEXT NOT NULL,
                circuit_name TEXT NOT NULL,
                country TEXT,
                race_date DATE,
                race_time TIME,
                total_laps INTEGER,
                race_distance REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (year) REFERENCES seasons (year) ON DELETE SET NULL,
                UNIQUE(year, round_number)
            )
        """)

    def create_sessions_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                race_id TEXT,
                session_type TEXT CHECK(session_type IN ('FP1', 'FP2', 'FP3', 'Sprint', 'SQ', 'Q', 'R')),
                session_date DATE,
                session_time TIME,
                air_temp REAL,
                track_temp REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (race_id) REFERENCES races (race_id) ON DELETE CASCADE
            )
        """)

    def create_drivers_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_code TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                full_name TEXT,
                team_name TEXT,
                team_color TEXT,
                driver_number INTEGER,
                nationality TEXT,
                birth_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def create_lap_times_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lap_times (
                lap_id TEXT PRIMARY KEY,
                session_id TEXT,
                driver_code TEXT,
                lap_number INTEGER,
                lap_time REAL,
                sector_1_time REAL,
                sector_2_time REAL,
                sector_3_time REAL,
                speed_i1 REAL,
                speed_i2 REAL,
                speed_fl REAL,
                speed_st REAL,
                is_personal_best BOOLEAN,
                is_accurate BOOLEAN,
                tire_compound TEXT,
                tire_age INTEGER,
                fresh_tyre BOOLEAN,
                pit_out_time REAL,
                pit_in_time REAL,
                deleted BOOLEAN DEFAULT FALSE,
                deleted_reason TEXT,
                track_status TEXT,
                position INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                FOREIGN KEY (driver_code) REFERENCES drivers (driver_code) ON DELETE CASCADE,
                UNIQUE(session_id, driver_code, lap_number)
            )
        """)

    def create_telemetry_table(self, conn):
        """FIXED: Telemetry table with proper foreign keys"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                telemetry_id TEXT PRIMARY KEY,
                session_id TEXT,
                driver_code TEXT,
                lap_number INTEGER,
                time_into_lap REAL,
                distance REAL,
                speed REAL,
                throttle REAL,
                brake REAL,
                drs INTEGER,
                gear INTEGER,
                rpm INTEGER,
                x_coordinate REAL,
                y_coordinate REAL,
                z_coordinate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                FOREIGN KEY (driver_code) REFERENCES drivers (driver_code) ON DELETE CASCADE
            )
        """)

    def create_pit_stops_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pit_stops (
                pit_stop_id TEXT PRIMARY KEY,
                session_id TEXT,
                driver_code TEXT,
                lap_number INTEGER,
                pit_stop_time REAL,
                pit_stop_duration REAL,
                tire_compound_old TEXT,
                tire_compound_new TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                FOREIGN KEY (driver_code) REFERENCES drivers (driver_code) ON DELETE CASCADE,
                UNIQUE(session_id, driver_code, lap_number)
            )
        """)

    def create_weather_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather (
                weather_id TEXT PRIMARY KEY,
                session_id TEXT,
                time REAL,
                air_temp REAL,
                track_temp REAL,
                humidity REAL,
                pressure REAL,
                wind_speed REAL,
                wind_direction REAL,
                rainfall BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
            )
        """)

    def create_tire_stints_table(self, conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tire_stints (
                stint_id TEXT PRIMARY KEY,
                session_id TEXT,
                driver_code TEXT,
                stint_number INTEGER,
                tire_compound TEXT,
                start_lap INTEGER,
                end_lap INTEGER,
                stint_length INTEGER,
                total_distance REAL,
                avg_lap_time REAL,
                tire_degradation_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                FOREIGN KEY (driver_code) REFERENCES drivers (driver_code) ON DELETE CASCADE,
                UNIQUE(session_id, driver_code, stint_number)
            )
        """)

    def create_compound_usage_table(self, conn):
        """Create table for tire compound usage data"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compound_usage (
                usage_id TEXT PRIMARY KEY,
                session_id TEXT,
                driver_code TEXT,
                compound TEXT NOT NULL,
                stint_number INTEGER,
                lap_start INTEGER,
                lap_end INTEGER,
                laps INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
                FOREIGN KEY (driver_code) REFERENCES drivers (driver_code) ON DELETE CASCADE,
                UNIQUE(session_id, driver_code, compound, stint_number)
            )
        """)

    def create_indexes(self, conn):
        """Create performance indexes"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_lap_times_session_driver ON lap_times(session_id, driver_code)",
            "CREATE INDEX IF NOT EXISTS idx_lap_times_lap_number ON lap_times(lap_number)",
            "CREATE INDEX IF NOT EXISTS idx_lap_times_compound ON lap_times(tire_compound)",
            "CREATE INDEX IF NOT EXISTS idx_telemetry_session_driver ON telemetry(session_id, driver_code)",
            "CREATE INDEX IF NOT EXISTS idx_pit_stops_session ON pit_stops(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_tire_stints_session_driver ON tire_stints(session_id, driver_code)",
            "CREATE INDEX IF NOT EXISTS idx_races_year ON races(year)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_race_type ON sessions(race_id, session_type)",
            "CREATE INDEX IF NOT EXISTS idx_compound_usage_session ON compound_usage(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_weather_session ON weather(session_id)"
        ]

        for index in indexes:
            try:
                conn.execute(index)
                logger.debug(f"Created index: {index.split()[-1]}")
            except sqlite3.OperationalError as e:
                if "already exists" not in str(e):
                    logger.warning(f"Index creation warning: {e}")

    def rebuild_database(self):
        """Completely rebuild the database with correct structure"""
        logger.info("Rebuilding database with correct structure...")
        
        # Backup existing data if it exists
        backup_path = f"{self.db_path}.backup"
        if Path(self.db_path).exists():
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Backed up existing database to {backup_path}")
        
        # Remove existing database
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
        
        # Recreate database
        self.init_database()
        logger.info("Database rebuilt successfully")


# Initialize database when module is imported
if __name__ == "__main__":
    db_manager = DatabaseManager()
    
    # Option to rebuild database if there are issues
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--rebuild":
        db_manager.rebuild_database()
        print("✅ Database rebuilt!")
    else:
        print("✅ Database schema verified!")
    
    logger.info("Database initialization complete!")