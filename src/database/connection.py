import sqlite3
import threading
from contextlib import contextmanager
from queue import Queue, Empty
from config.settings import Config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ConnectionPool:

    def __init__(self, database_path:str, pool_size:int=5):
        self.database_path = database_path
        self.pool_size = pool_size
        self.connections = Queue(maxsize=pool_size)
        self.lock = threading.Lock()

        for _ in range(pool_size):
            conn = self._create_connection()
            self.connections.put(conn)

    def _create_connection(self):
        conn = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
            timeout=30.0
        )
        conn.row_factory = sqlite3.Row

        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=10000")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=ON")

        return conn
    
    @contextmanager
    def get_connection(self, timeout:   float=5.0):
        conn = None
        try:
            conn = self.connections.get(timeout)
            yield conn

        except Empty:
            logger.error("CONNECTION POOL EXHAUSTED")
            raise Exception("Unable to get database connection")
        
        finally:
            if conn:
                self.connections.put(conn)

    def close_all(self):
        while not self.connections.empty():
            try:
                conn = self.connections.get_nowait()
                conn.close()

            except Empty:
                break

db_pool = ConnectionPool(str(Config.DATA_DIR / "f1_analytics.db"))