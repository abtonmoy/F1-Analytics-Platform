import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from config.settings import Config
from src.utils.logger import setup_logger
from src.data_pipeline.storage import DataStorageManager
from src.data_pipeline.collector import F1DataCollector

logger = setup_logger(__name__)

@dataclass
class ETLJobResult:
    success: bool
    job_id: str
    message: str
    duration: float
    records_processed: int = 0

class F1ETLPipeline:

    def __init__(self, max_workers: int = 1):
        self.collector = F1DataCollector()
        self.storage = DataStorageManager()
        self.max_workers = max_workers
        self.logger = setup_logger(self.__class__.__name__)

    def run_full_pipeline(self, seasons: List[int], session_types: List[str] = None) -> List[ETLJobResult]:
        """Run complete ETL pipeline for multiple seasons"""
        # Fixed: Use correct session types
        if session_types is None:
            session_types = ['FP1', 'FP2', 'FP3', 'Q', 'R']  # Removed invalid Q1, Q2, Q3

        results = []
        total_start_time = time.time()

        self.logger.info(f"Starting ETL pipeline for seasons: {seasons}")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_season = {
                executor.submit(self.process_season, year, session_types): year
                for year in seasons
            }

            for future in as_completed(future_to_season):
                year = future_to_season[future]
                try:
                    season_results = future.result()
                    results.extend(season_results)
                except Exception as e:
                    self.logger.error(f"Error processing season {year}: {e}")
                    results.append(ETLJobResult(
                        success=False,
                        job_id=f"season_{year}",
                        message=f"Error processing season: {str(e)}",
                        duration=0
                    ))

        total_duration = time.time() - total_start_time
        successful_jobs = len([r for r in results if r.success])
        total_records = sum(r.records_processed for r in results)

        self.logger.info(f"ETL Pipeline completed in {total_duration:.2f}s")
        self.logger.info(f"Success rate: {successful_jobs}/{len(results)} jobs")
        self.logger.info(f"Total records processed: {total_records}")

        return results

    def process_season(self, year: int, session_types: List[str]) -> List[ETLJobResult]:
        start_time = time.time()
        results = []

        self.logger.info(f"Processing {year} season")

        # Collect race events for the season
        race_events = self.collector.collect_season_schedule(year)
        if not race_events:
            return [ETLJobResult(
                success=False,
                job_id=f"season_{year}",
                message=f"No race events found for {year}",
                duration=time.time() - start_time
            )]

        # Store season data
        try:
            self.storage.store_season_data(year, len(race_events))
            self.logger.info(f"Stored season data for {year}: {len(race_events)} races")
        except Exception as e:
            self.logger.error(f"Failed to store season data for {year}: {e}")
            return [ETLJobResult(
                success=False,
                job_id=f"season_{year}",
                message=f"Failed to store season data: {str(e)}",
                duration=time.time() - start_time
            )]

        # Store race data
        try:
            race_data = [{
                'year': event.year,
                'round_number': event.round_number,
                'race_name': event.race_name,
                'circuit_name': event.circuit_name,
                'country': event.country,
                'race_date': event.race_date,
                'race_time': event.race_time
            } for event in race_events]
            
            self.storage.store_race_data(race_data)
            self.logger.info(f"Stored race data for {year}: {len(race_data)} races")
        except Exception as e:
            self.logger.error(f"Failed to store race data for {year}: {e}")
            return [ETLJobResult(
                success=False,
                job_id=f"season_{year}",
                message=f"Failed to store race data: {str(e)}",
                duration=time.time() - start_time
            )]

        # Create session jobs
        jobs = []
        for event in race_events:
            for session_type in session_types:
                job_id = f"{year}_{event.race_name}_{session_type}".replace(' ', '_')
                jobs.append({
                    'job_id': job_id,
                    'year': year,
                    'race_name': event.race_name,
                    'session_type': session_type
                })

        self.logger.info(f"Created {len(jobs)} session jobs for {year}")

        # Process session jobs in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_job = {
                executor.submit(self.process_session_jobs, job): job
                for job in jobs
            }

            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Log progress
                    completed = len(results)
                    if completed % 5 == 0 or completed == len(jobs):
                        self.logger.info(f"Progress: {completed}/{len(jobs)} jobs completed for {year}")
                        
                except Exception as e:
                    self.logger.exception(f"Job {job['job_id']} failed with exception")
                    results.append(ETLJobResult(
                        success=False,
                        job_id=job['job_id'],
                        message=f"Exception: {str(e)}",
                        duration=0
                    ))

        # Final summary
        season_duration = time.time() - start_time
        successful_sessions = len([r for r in results if r.success])
        total_records = sum(r.records_processed for r in results)

        self.logger.info(f"Season {year} completed in {season_duration:.2f}s")
        self.logger.info(f"Success rate: {successful_sessions}/{len(results)} sessions")
        self.logger.info(f"Total records processed: {total_records}")

        return results

    def process_session_jobs(self, job: Dict[str, Any]) -> ETLJobResult:
        """Process single session job with retry mechanism"""
        max_retries = 2
        attempts = 0
        job_id = job['job_id']
        result = None
        
        while attempts <= max_retries:
            attempts += 1
            start_time = time.time()
            
            try:
                self.logger.info(f"Starting job (attempt {attempts}): {job_id}")

                session_data = self.collector.collect_session_data(
                    job['year'],
                    job['race_name'],
                    job['session_type']
                )

                if not session_data:
                    result = ETLJobResult(
                        success=False,
                        job_id=job_id,
                        message="No session data collected",
                        duration=time.time() - start_time
                    )
                    continue

                success = self.storage.store_session_data(session_data)

                if success:
                    records_count = len(session_data.get('lap_times', []))
                    self.logger.info(f"Job {job_id} completed successfully: {records_count} lap times")

                    return ETLJobResult(
                        success=True,
                        job_id=job_id,
                        message=f"Successfully processed {records_count} lap times",
                        duration=time.time() - start_time,
                        records_processed=records_count
                    )
                else:
                    result = ETLJobResult(
                        success=False,
                        job_id=job_id,
                        message="Failed to store session data",
                        duration=time.time() - start_time
                    )
            except Exception as e:
                self.logger.exception(f"Job {job_id} failed with error (attempt {attempts})")
                result = ETLJobResult(
                    success=False,
                    job_id=job_id,
                    message=f"Error: {str(e)}",
                    duration=time.time() - start_time
                )
                
            # Add delay before retry
            if attempts <= max_retries:
                time.sleep(2 ** attempts)  # Exponential backoff
        
        return result

    def run_incremental_update(self, year: int, race_name: str, session_types: List[str] = None) -> List[ETLJobResult]:
        """Run incremental update for specific race"""
        if session_types is None:
            session_types = ['R']

        self.logger.info(f"Running incremental update for {race_name} {year}")
        
        # First, ensure the season and race data exists
        self.logger.info(f"Ensuring season and race data exists for {year}")
        race_events = self.collector.collect_season_schedule(year)
        
        if race_events:
            # Store season data
            self.storage.store_season_data(year, len(race_events))
            
            # Find the specific race and store its data
            target_race = None
            for event in race_events:
                if event.race_name == race_name:
                    target_race = event
                    break
            
            if target_race:
                self.storage.store_race_data([{
                    'year': target_race.year,
                    'round_number': target_race.round_number,
                    'race_name': target_race.race_name,
                    'circuit_name': target_race.circuit_name,
                    'country': target_race.country,
                    'race_date': target_race.race_date,
                    'race_time': target_race.race_time
                }])
                self.logger.info(f"Stored race data for {race_name}")
            else:
                self.logger.warning(f"Race {race_name} not found in {year} schedule")

        # Now process the session jobs
        jobs = []
        for session_type in session_types:
            job_id = f"{year}_{race_name}_{session_type}".replace(' ', '_')
            jobs.append({
                'job_id': job_id,
                'year': year,
                'race_name': race_name,
                'session_type': session_type
            })

        results = []
        for job in jobs:
            result = self.process_session_jobs(job)
            results.append(result)

        return results


if __name__ == '__main__':
    pipeline = F1ETLPipeline(max_workers=2)
    results = pipeline.run_incremental_update(2024, 'Monaco Grand Prix', ["R"])

    for result in results:
        print(f"Job: {result.job_id}")
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Records: {result.records_processed}")
        print("-" * 50)