#!/usr/bin/env python3
"""
Enhanced F1 Database Population Script

This script provides improved strategies for populating the F1 database
with better error handling, rate limiting, and session type management.
"""

import time
import sys
from typing import List, Dict, Tuple
from datetime import datetime

# Import your ETL pipeline (adjust import path as needed)
from src.data_pipeline.etl import F1ETLPipeline, ETLJobResult


class F1DatabasePopulator:
    """Enhanced helper class for populating F1 database with multiple races"""
    
    def __init__(self, max_workers: int = 2):
        # Reduced workers to avoid API rate limits
        self.pipeline = F1ETLPipeline(max_workers=max_workers)
    
    def populate_full_seasons(self, seasons: List[int], session_types: List[str] = None) -> None:
        """
        Populate database with complete seasons using correct session types
        
        Args:
            seasons: List of years to process (e.g., [2023, 2024])
            session_types: List of session types to include (corrected defaults)
        """
        # FIXED: Use correct session types
        if session_types is None:
            session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']  # Corrected session types
        
        print(f"Starting full season population for: {seasons}")
        print(f"Session types: {session_types}")
        print("=" * 60)
        
        start_time = time.time()
        results = self.pipeline.run_full_pipeline(seasons, session_types)
        total_time = time.time() - start_time
        
        self._print_summary(results, total_time, "Full Season Population")
    
    def populate_specific_races(self, race_configs: List[Dict]) -> None:
        """
        Populate database with specific races using corrected session types
        
        Args:
            race_configs: List of dicts with keys: year, race_name, session_types
            Example: [
                {'year': 2024, 'race_name': 'Monaco Grand Prix', 'session_types': ['Q', 'R']},
                {'year': 2024, 'race_name': 'Spanish Grand Prix', 'session_types': ['R']}
            ]
        """
        print(f"Starting specific race population for {len(race_configs)} races")
        print("=" * 60)
        
        all_results = []
        start_time = time.time()
        
        for i, config in enumerate(race_configs, 1):
            year = config['year']
            race_name = config['race_name']
            session_types = config.get('session_types', ['R'])
            
            print(f"\n[{i}/{len(race_configs)}] Processing: {race_name} {year}")
            print(f"Sessions: {session_types}")
            
            race_start = time.time()
            results = self.pipeline.run_incremental_update(year, race_name, session_types)
            race_time = time.time() - race_start
            
            all_results.extend(results)
            
            # Print race summary
            successful = len([r for r in results if r.success])
            total_records = sum(r.records_processed for r in results)
            print(f"Completed in {race_time:.2f}s | Success: {successful}/{len(results)} | Records: {total_records}")
            
            # Add delay between races to respect API limits
            time.sleep(1)
        
        total_time = time.time() - start_time
        self._print_summary(all_results, total_time, "Specific Race Population")
    
    def populate_recent_season_highlights(self, year: int) -> None:
        """
        Populate database with highlights from a recent season
        (Qualifying and Race sessions only for faster processing)
        """
        print(f"Populating {year} season highlights (Qualifying + Race sessions)")
        print("=" * 60)
        
        start_time = time.time()
        results = self.pipeline.run_full_pipeline([year], session_types=['Q', 'R'])
        total_time = time.time() - start_time
        
        self._print_summary(results, total_time, f"{year} Season Highlights")
    
    def populate_practice_sessions_only(self, year: int) -> None:
        """
        Populate only practice sessions for a year (useful for setup data)
        """
        print(f"Populating {year} practice sessions only")
        print("=" * 60)
        
        start_time = time.time()
        results = self.pipeline.run_full_pipeline([year], session_types=['FP1', 'FP2', 'FP3'])
        total_time = time.time() - start_time
        
        self._print_summary(results, total_time, f"{year} Practice Sessions")
    
    def populate_conservative_batch(self, seasons: List[int]) -> None:
        """
        Conservative population strategy with minimal load on APIs
        Only Race sessions to get core data quickly
        """
        print(f"Conservative population for seasons: {seasons}")
        print("Only Race sessions to minimize API calls")
        print("=" * 60)
        
        start_time = time.time()
        results = self.pipeline.run_full_pipeline(seasons, session_types=['R'])
        total_time = time.time() - start_time
        
        self._print_summary(results, total_time, "Conservative Batch Population")
    
    def _print_summary(self, results: List[ETLJobResult], total_time: float, operation_name: str) -> None:
        """Print detailed summary of ETL operation"""
        print("\n" + "=" * 60)
        print(f"{operation_name.upper()} SUMMARY")
        print("=" * 60)
        
        successful_jobs = [r for r in results if r.success]
        failed_jobs = [r for r in results if not r.success]
        total_records = sum(r.records_processed for r in results)
        
        print(f"Total Duration: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"Successful Jobs: {len(successful_jobs)}")
        print(f"Failed Jobs: {len(failed_jobs)}")
        print(f"Total Records Processed: {total_records:,}")
        if len(results) > 0:
            print(f"Success Rate: {len(successful_jobs)/len(results)*100:.1f}%")
        
        if failed_jobs:
            print(f"\nFAILED JOBS:")
            for job in failed_jobs[:10]:  # Show first 10 failures
                print(f"   • {job.job_id}: {job.message}")
            if len(failed_jobs) > 10:
                print(f"   ... and {len(failed_jobs) - 10} more failures")
        
        if successful_jobs:
            print(f"\nTOP PERFORMING JOBS:")
            top_jobs = sorted(successful_jobs, key=lambda x: x.records_processed, reverse=True)[:5]
            for job in top_jobs:
                print(f"   • {job.job_id}: {job.records_processed:,} records in {job.duration:.2f}s")


def main():
    """Main function with improved population strategies"""
    
    print("Enhanced F1 Database Population Tool")
    print("=" * 60)
    
    # Initialize populator with reduced workers to avoid rate limits
    populator = F1DatabasePopulator(max_workers=2)
    
    # Strategy 1: Conservative approach - Start with recent seasons, races only
    print("STRATEGY 1: Conservative Population (Race sessions only)")
    # populator.populate_conservative_batch([2023, 2024])
    
    # Strategy 2: Add qualifying data for recent seasons
    # Uncomment after conservative batch completes successfully
    # print("\nSTRATEGY 2: Adding Qualifying Data")
    # populator.populate_recent_season_highlights(2024)
    # populator.populate_recent_season_highlights(2023)
    
    # Strategy 3: Full data for specific important races
    # Uncomment for comprehensive data on key races
    # important_races = [
    #     {'year': 2024, 'race_name': 'Monaco Grand Prix', 'session_types': ['FP3', 'Q', 'R']},
    #     {'year': 2024, 'race_name': 'British Grand Prix', 'session_types': ['FP3', 'Q', 'R']},
    #     {'year': 2024, 'race_name': 'Italian Grand Prix', 'session_types': ['FP3', 'Q', 'R']},
    #     {'year': 2023, 'race_name': 'Monaco Grand Prix', 'session_types': ['FP3', 'Q', 'R']},
    #     {'year': 2023, 'race_name': 'British Grand Prix', 'session_types': ['FP3', 'Q', 'R']},
    # ]
    # populator.populate_specific_races(important_races)


# Preset configurations for common use cases
def quick_2024_races():
    """Quick population of major 2024 races with corrected session types"""
    populator = F1DatabasePopulator(max_workers=2)
    
    major_races = [
        {'year': 2024, 'race_name': 'Bahrain Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Monaco Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'British Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Italian Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Singapore Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Japanese Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Las Vegas Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2024, 'race_name': 'Abu Dhabi Grand Prix', 'session_types': ['Q', 'R']},
    ]
    
    # populator.populate_specific_races(major_races)

def races_only_multiple_seasons():
    """Populate race sessions only for multiple seasons - fastest approach"""
    populator = F1DatabasePopulator(max_workers=1)
    populator.populate_conservative_batch([2021, 2022, 2023, 2024])

def championship_deciding_races():
    """Populate historically significant championship-deciding races"""
    populator = F1DatabasePopulator(max_workers=2)
    
    historic_races = [
        {'year': 2021, 'race_name': 'Abu Dhabi Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2008, 'race_name': 'Brazilian Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2010, 'race_name': 'Abu Dhabi Grand Prix', 'session_types': ['Q', 'R']},
        {'year': 2016, 'race_name': 'Abu Dhabi Grand Prix', 'session_types': ['Q', 'R']},
    ]
    
    # populator.populate_specific_races(historic_races)

def test_single_race():
    """Test with a single race to verify everything works"""
    populator = F1DatabasePopulator(max_workers=1)
    
    test_races = [
        {'year': 2024, 'race_name': 'Monaco Grand Prix', 'session_types': ['R']},
    ]
    
    # populator.populate_specific_races(test_races)

def gradual_buildup_strategy():
    """Gradual buildup strategy - start small and expand"""
    populator = F1DatabasePopulator(max_workers=2)
    
    print("Phase 1: Test with single race")
    # test_single_race()
    
    # print("\nPhase 2: 2024 races only")
    # populator.populate_conservative_batch([2024])
    
    # print("\nPhase 3: Add 2023 races")
    # populator.populate_conservative_batch([2023])
    
    # print("\nPhase 4: Add qualifying for recent seasons")
    # populator.populate_recent_season_highlights(2024)
    # populator.populate_recent_season_highlights(2023)


if __name__ == '__main__':
    # Choose your strategy:
    
    # For testing - start here
    # print("=== TESTING STRATEGY ===")
    # test_single_race()
    
    # For gradual buildup
    # print("=== GRADUAL BUILDUP STRATEGY ===")
    # gradual_buildup_strategy()
    
    # For quick major races
    # print("=== QUICK 2024 RACES ===")
    # quick_2024_races()
    
    # For conservative multi-season
    print("=== RACES ONLY MULTI-SEASON ===")
    races_only_multiple_seasons()
    
    # For main comprehensive approach
    # main()