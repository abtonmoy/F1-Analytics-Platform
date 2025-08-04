import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sqlite3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from config.settings import Config
from src.utils.logger import setup_logger
from src.database.connection import db_pool

logger = setup_logger(__name__)


@dataclass
class ModelPrediction:
    '''container for model prediction'''
    predicted_value: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_name: str

@dataclass
class ModelPerformace:
    '''container for model performance matrix'''
    model_name: str
    mae: float
    rmse: float
    r2_score: float
    cv_score: float

class F1PredictiveModel:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_dir = Config.BASE_DIR/ "models" / "saved"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # initialize models
        self.models = {
            'lap_time_rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'lap_time_gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lap_time_lr': LinearRegression(),
            'tire_degradation': RandomForestRegressor(n_estimators=50, random_state=42),
            'pit_stop_strategy': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }

    def prepare_lap_time_features(self, session_ids: List[str]) -> pd.DataFrame:
        """Prepare features for lap time prediction"""
        self.logger.info(f"Preparing lap time features for {len(session_ids)} sessions")


        try: 
            all_features = []

            with db_pool.get_connection() as conn:
                for session_id in session_ids:
                    query = """
                        SELECT 
                            l.lap_time,
                            l.driver_code,
                            l.lap_number,
                            l.tire_compound,
                            l.tire_age,
                            l.sector_1_time,
                            l.sector_2_time,
                            l.sector_3_time,
                            l.speed_i1,
                            l.speed_i2,
                            l.spped_fl,
                            l.position,
                            s.air_temp,
                            s.track_temp,
                            s.humidity,
                            s.session_type,
                            r.circuit_name
                        FROM lap_times l
                        JOIN sessions s ON l.session_id = s.session_id
                        JOIN races r ON s.race_id = r.race_id
                        WHERE l.session_id = ?
                        AND COALESCE(l.deleted, 0) = 0
                        AND l.lap_time IS NOT NULL
                        AND l.lap_time BETWEEN 20 AND 400
                    """

                    session_data = pd.read_sql_query(query, conn, params=(session_id,))

                    if not session_data.empty:
                        session_data = self._add_derived_features(session_data)
                        all_features.append(session_data)

            if all_features:
                combined_df = pd.concat(all_features,ignore_index=True)
                self.logger.info(f'Prepared {len(combined_df)} future records')
                return combined_df
            
            else:
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f'Error preparing lap time features: {e}')
            return pd.DataFrame()
        
    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        '''derivated features for better predictions'''
        try:
            # previous lap performance
            df['prev_lap_time'] = df.groupby('driver_code')['lap_time'].shift(1)
            df['lap_time_trend'] = df['lap_time'] - df['prev_lap_time']

            # tire stint
            df['stint_lap'] = df.groupby(['driver_code', 'tire_compound']).cumcount()+1

            # position changes
            df['prev_position'] = df.groupby('driver_code')['postion'].shift(1)
            df['position_change'] = df['prev_position'] - df['position']

            # Weather impact
            df['temp_difference'] = df['track_temp'] - df['air_temp']

            # compound performance relative to session
            compound_avg = df.groupby('tire_compound')['lap_time'].transform('mean')
            df['compound_relative_pace'] = df['lap_time'] - compound_avg

            # driver form - rolling avg of last 5 laps
            df['driver_form'] = df.groupby('driver_code')['lap_time'].rolling(
                window=5, min_periods=1
            ).mean().reset_index(drop=True)

            return df
        
        except Exception as e:
            self.logger.error(f'Error adding drived features: {e}')
            return df
        
    def train_lap_time_model(self, session_ids: List[str]) -> Dict[str, ModelPerformace]:
        '''train lap time prediction model'''
        self.logger.info('Training lap time prediction models')

        try:
            # prepare features
            data = self.prepare_lap_time_features(session_ids)

            if data.empty:
                self.logger.error('No data available for trainning')
                return {}
            
            # Prepare feature and target
            X, y = self._prepare_model_data(data, target = 'lap_time')

            if X.empty:
                return {}
            
            # split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # train models
            model_performances = {}
            model_names = ['lap_time_rf', 'lap_time_gb', 'lap_time_lr']
            for model_name in model_names:
                performance = self._train_single_model(
                    model_name, X_train, X_test, y_train, y_test
                )
                model_performances[model_name] = performance

            # save model
            best_model_name = min(model_performances.keys(), key=lambda x: model_performances[x].mae)
            self._save_model(best_model_name, X.columns.tolist())

            self.logger.info(f"Training completed. Best model: {best_model_name}")
            return model_performances
            
        except Exception as e:
            self.logger.error(f"Error training lap time models: {e}")
            return {}
        

    def _prepare_model_data(self, data: pd.DataFrame, target:str) -> Tuple[pd.DataFrame, pd.Series]:
        try:
            feature_columns = [
                'lap_number', 'tire_age', 'sector_1_time', 'sector_2_time',
                'sector_3_time', 'speed_i1', 'speed_i2', 'speed_fl',
                'position', 'air_temp', 'track_temp', 'humidity',
                'prev_lap_time', 'lap_time_trend', 'stint_lap',
                'position_change', 'temp_difference', 'compound_relative_pace',
                'driver_form'
            ]

            categorical_features = ['tire_compound', 'session_type', 'circuit_name', 'driver_code']

            available_features = [col for col in feature_columns if col in data.columns]
            available_categorical = [col for col in categorical_features if col in data.columns]

            X = data[available_features + available_categorical].copy()
            X[available_features] = X[available_features].fillna(X[available_features].mean())

            for col in available_categorical:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()

                X[col] = X[col].fillna('Unknown')
                unique_values = X[col].unique()

                if not hasattr(self.label_encoders[col], 'classes_'):
                    self.label_encoders[col].fit(unique_values)

                else:
                    known_classes = set(self.label_encoders[col].classes_)
                    new_classes = set(unique_values) - known_classes

                    if new_classes:
                        all_classes = list(self.label_encoders[col].classes_) + list(new_classes)
                        self.label_encoders[col].classes_ = np.array(all_classes)
                        
                X[col] = self.label_encoders[col].transform(X[col])

            if 'lap_time_scaler' not in self.scalers:
                self.scalers['lap_time_scaler'] = StandardScaler()
                X = pd.DataFrame(
                    self.scalers['lap_time_scaler'].fit_transform(X), columns=X.columns
                )
            else:
                X = pd.DataFrame(self.scalers['lap_time_scaler'].transform(X), columns=X.columns)
            
            y = data[target].dropna()

            X = X.loc[y.index]

            return X, y
        
        except Exception as e:
            self.logger.error(f"Error preparing model data: {e}")
            return pd.DataFrame(), pd.Series()