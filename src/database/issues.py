#!/usr/bin/env python3
"""
Quick Fix Script for F1 Database Foreign Key Issues
Run this before your main data population script
"""

import sqlite3
import sys
from pathlib import Path
from config.settings import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def fix_database_constraints():
    """Fix database constraints and add missing records"""
    db_path = str(Config.DATA_DIR / "f1_analytics.db")
    
    print("🔧 Fixing F1 Database Constraints...")
    print("=" * 50)
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.row_factory = sqlite3.Row
            
            # 1. Fix seasons - ensure basic seasons exist
            seasons_to_add = [2021, 2022, 2023, 2024]
            for year in seasons_to_add:
                conn.execute("""
                    INSERT OR IGNORE INTO seasons (year, total_rounds, created_at)
                    VALUES (?, 0, CURRENT_TIMESTAMP)
                """, (year,))
            print(f"✅ Ensured seasons exist: {seasons_to_add}")
            
            # 2. Check for orphaned sessions and fix their race_id references
            cursor = conn.execute("""
                SELECT s.session_id, s.race_id 
                FROM sessions s
                LEFT JOIN races r ON s.race_id = r.race_id
                WHERE r.race_id IS NULL AND s.race_id IS NOT NULL
            """)
            orphaned_sessions = cursor.fetchall()
            
            print(f"Found {len(orphaned_sessions)} orphaned sessions")
            
            # Create minimal race records for orphaned sessions
            for session in orphaned_sessions:
                session_id = session['session_id']
                race_id = session['race_id']
                
                # Parse race info from race_id
                parts = race_id.split('_')
                if len(parts) >= 2:
                    try:
                        year = int(parts[0])
                        race_name = ' '.join(parts[1:]).replace('_', ' ').title()
                        
                        conn.execute("""
                            INSERT OR IGNORE INTO races 
                            (race_id, year, round_number, race_name, circuit_name, country, created_at)
                            VALUES (?, ?, 1, ?, 'Unknown', 'Unknown', CURRENT_TIMESTAMP)
                        """, (race_id, year, race_name))
                        
                        print(f"   Created race: {race_id} -> {race_name}")
                    except ValueError:
                        print(f"   Skipped invalid race_id: {race_id}")
            
            # 3. Check for missing driver records
            cursor = conn.execute("""
                SELECT DISTINCT driver_code 
                FROM lap_times 
                WHERE driver_code NOT IN (SELECT driver_code FROM drivers)
            """)
            missing_drivers = [row[0] for row in cursor.fetchall()]
            
            if missing_drivers:
                print(f"Found {len(missing_drivers)} missing drivers: {missing_drivers}")
                for driver_code in missing_drivers:
                    conn.execute("""
                        INSERT OR IGNORE INTO drivers (driver_code, created_at)
                        VALUES (?, CURRENT_TIMESTAMP)
                    """, (driver_code,))
                print("✅ Created minimal driver records")
            
            # 4. Clean up any remaining orphaned records
            # Remove pit stops that reference non-existent drivers
            cursor = conn.execute("""
                DELETE FROM pit_stops 
                WHERE driver_code NOT IN (SELECT driver_code FROM drivers)
            """)
            deleted_pit_stops = cursor.rowcount
            
            # Remove tire stints that reference non-existent drivers  
            cursor = conn.execute("""
                DELETE FROM tire_stints
                WHERE driver_code NOT IN (SELECT driver_code FROM drivers)
            """)
            deleted_stints = cursor.rowcount
            
            # Remove compound usage that references non-existent drivers
            cursor = conn.execute("""
                DELETE FROM compound_usage
                WHERE driver_code NOT IN (SELECT driver_code FROM drivers)
            """)
            deleted_compounds = cursor.rowcount
            
            if deleted_pit_stops or deleted_stints or deleted_compounds:
                print(f"🧹 Cleaned up orphaned records:")
                print(f"   - Pit stops: {deleted_pit_stops}")
                print(f"   - Tire stints: {deleted_stints}")
                print(f"   - Compound usage: {deleted_compounds}")
            
            # 5. Update database statistics
            cursor = conn.execute("SELECT COUNT(*) FROM seasons")
            season_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM races")
            race_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM drivers")
            driver_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            session_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM lap_times")
            lap_count = cursor.fetchone()[0]
            
            conn.commit()
            
            print("\n📊 Database Statistics:")
            print(f"   - Seasons: {season_count}")
            print(f"   - Races: {race_count}")
            print(f"   - Drivers: {driver_count}")
            print(f"   - Sessions: {session_count}")
            print(f"   - Lap times: {lap_count}")
            
            print("\n✅ Database constraints fixed successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing database constraints: {e}")
        return False

def test_constraints():
    """Test that foreign key constraints are working properly"""
    db_path = str(Config.DATA_DIR / "f1_analytics.db")
    
    print("\n🧪 Testing Foreign Key Constraints...")
    
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys=ON")
            
            # Test 1: Try to insert session with non-existent race_id
            try:
                conn.execute("""
                    INSERT INTO sessions (session_id, race_id, session_type)
                    VALUES ('test_session', 'non_existent_race', 'R')
                """)
                conn.rollback()
                print("❌ Foreign key constraint not working - should have failed")
                return False
            except sqlite3.IntegrityError:
                print("✅ Foreign key constraint working correctly")
                conn.rollback()
            
            # Test 2: Check if we can insert valid session
            cursor = conn.execute("SELECT race_id FROM races LIMIT 1")
            result = cursor.fetchone()
            if result:
                race_id = result[0]
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO sessions (session_id, race_id, session_type)
                        VALUES ('test_valid_session', ?, 'R')
                    """, (race_id,))
                    conn.execute("DELETE FROM sessions WHERE session_id = 'test_valid_session'")
                    print("✅ Valid session insertion works")
                except Exception as e:
                    print(f"❌ Valid session insertion failed: {e}")
                    return False
            
            conn.rollback()
            return True
            
    except Exception as e:
        print(f"❌ Error testing constraints: {e}")
        return False

def backup_database():
    """Create a backup of the current database"""
    db_path = Path(Config.DATA_DIR / "f1_analytics.db")
    if db_path.exists():
        backup_path = db_path.with_suffix('.db.backup')
        
        try:
            import shutil
            shutil.copy2(db_path, backup_path)
            print(f"✅ Database backed up to: {backup_path}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    else:
        print("ℹ️  No existing database to backup")
        return True

def main():
    """Main function to fix all database issues"""
    print("🚀 F1 Database Constraint Fixer")
    print("=" * 50)
    
    # Step 1: Backup existing database
    if not backup_database():
        print("⚠️  Backup failed, continuing anyway...")
    
    # Step 2: Fix constraints
    if not fix_database_constraints():
        print("❌ Failed to fix database constraints")
        return False
    
    # Step 3: Test constraints
    if not test_constraints():
        print("❌ Foreign key constraints still not working properly")
        return False
    
    print("\n🎉 All database issues fixed!")
    print("You can now run your data population script.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)