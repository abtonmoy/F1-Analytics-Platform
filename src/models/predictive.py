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
from sklearn.impute import SimpleImputer
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
class ModelPerformance:
    '''container for model performance matrix'''
    model_name: str
    mae: float
    rmse: float
    r2_score: float
    cv_score: float


class F1PredictiveModels:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.imputers = {}
        self.model_dir = Config.BASE_DIR / "models" / "saved"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Initialize models with better parameters
        self.models = {
            'lap_time_rf': RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'lap_time_gb': GradientBoostingRegressor(
                n_estimators=100, 
                random_state=42,
                max_depth=8,
                learning_rate=0.1
            ),
            'lap_time_lr': LinearRegression(),
            'tire_degradation': RandomForestRegressor(
                n_estimators=50, 
                random_state=42,
                max_depth=10
            ),
            'pit_stop_strategy': GradientBoostingRegressor(
                n_estimators=50, 
                random_state=42
            )
        }

    def prepare_lap_time_features(self, session_ids: List[str]) -> pd.DataFrame:
        """Prepare features for lap time prediction with better error handling"""
        self.logger.info(f"Preparing lap time features for {len(session_ids)} sessions")

        try: 
            with db_pool.get_connection() as conn:
                session_placeholders = ','.join(['?' for _ in session_ids])
                query = f"""
                    SELECT 
                        l.lap_time,
                        l.driver_code,
                        l.lap_number,
                        l.tire_compound,
                        COALESCE(l.tire_age, 1) as tire_age,
                        l.sector_1_time,
                        l.sector_2_time,
                        l.sector_3_time,
                        l.speed_i1,
                        l.speed_i2,
                        l.speed_fl,
                        COALESCE(l.position, 10) as position,
                        COALESCE(s.air_temp, 25.0) as air_temp,
                        COALESCE(s.track_temp, 35.0) as track_temp,
                        COALESCE(s.humidity, 60.0) as humidity,
                        COALESCE(s.session_type, 'R') as session_type,
                        COALESCE(r.circuit_name, 'Unknown') as circuit_name,
                        l.session_id
                    FROM lap_times l
                    JOIN sessions s ON l.session_id = s.session_id
                    JOIN races r ON s.race_id = r.race_id
                    WHERE l.session_id IN ({session_placeholders})
                    AND COALESCE(l.deleted, 0) = 0
                    AND l.lap_time IS NOT NULL
                    AND l.lap_time BETWEEN 60 AND 120
                    ORDER BY l.session_id, l.driver_code, l.lap_number
                """

                combined_data = pd.read_sql_query(query, conn, params=tuple(session_ids))

                if not combined_data.empty:
                    # Add derived features with better error handling
                    combined_data = self._add_derived_features_safe(combined_data)
                    self.logger.info(f'Prepared {len(combined_data)} feature records')
                    return combined_data
                else:
                    self.logger.warning("No data found for provided session IDs")
                    return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f'Error preparing lap time features: {e}')
            return pd.DataFrame()
        
    def _add_derived_features_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Add derived features with robust error handling'''
        try:
            # Sort by session, driver, and lap number for proper shift operations
            df = df.sort_values(['session_id', 'driver_code', 'lap_number'])
            
            # Previous lap performance with safe defaults
            df['prev_lap_time'] = df.groupby(['session_id', 'driver_code'])['lap_time'].shift(1)
            df['prev_lap_time'] = df['prev_lap_time'].fillna(df['lap_time'])
            
            df['lap_time_trend'] = df['lap_time'] - df['prev_lap_time']
            df['lap_time_trend'] = df['lap_time_trend'].fillna(0.0)

            # Tire stint calculation
            df['stint_lap'] = df.groupby(['session_id', 'driver_code', 'tire_compound']).cumcount() + 1

            # Position changes with safe handling
            df['prev_position'] = df.groupby(['session_id', 'driver_code'])['position'].shift(1)
            df['prev_position'] = df['prev_position'].fillna(df['position'])
            df['position_change'] = df['prev_position'] - df['position']
            df['position_change'] = df['position_change'].fillna(0.0)

            # Weather impact
            df['temp_difference'] = df['track_temp'] - df['air_temp']

            # Compound performance relative to session (safe calculation)
            try:
                compound_avg = df.groupby(['session_id', 'tire_compound'])['lap_time'].transform('mean')
                df['compound_relative_pace'] = df['lap_time'] - compound_avg
            except:
                df['compound_relative_pace'] = 0.0

            # Driver form - rolling avg of last 5 laps with safe window
            try:
                df['driver_form'] = df.groupby(['session_id', 'driver_code'])['lap_time'].transform(
                    lambda x: x.rolling(window=min(5, len(x)), min_periods=1).mean()
                )
            except:
                df['driver_form'] = df['lap_time']

            # Fill any remaining NaN values with sensible defaults
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

            return df
        
        except Exception as e:
            self.logger.error(f'Error adding derived features: {e}')
            return df
        
    def train_lap_time_model(self, session_ids: List[str]) -> Dict[str, ModelPerformance]:
        '''Train lap time prediction model with improved data handling'''
        self.logger.info('Training lap time prediction models')

        try:
            # Prepare features
            data = self.prepare_lap_time_features(session_ids)

            if data.empty:
                self.logger.error('No data available for training')
                return {}
            
            # Prepare features and target with better preprocessing
            X, y = self._prepare_model_data_safe(data, target='lap_time')

            if X.empty or len(y) == 0:
                self.logger.error('No valid features prepared for training')
                return {}
            
            # Remove any remaining invalid values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 10:
                self.logger.error('Insufficient clean data for training')
                return {}

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train models
            model_performances = {}
            model_names = ['lap_time_rf', 'lap_time_gb', 'lap_time_lr']
            for model_name in model_names:
                performance = self._train_single_model_safe(
                    model_name, X_train, X_test, y_train, y_test
                )
                model_performances[model_name] = performance

            # Save best model
            if model_performances:
                # Filter out failed models
                valid_models = {k: v for k, v in model_performances.items() if v.mae != float('inf')}
                
                if valid_models:
                    best_model_name = min(valid_models.keys(), key=lambda x: valid_models[x].mae)
                    self._save_model(best_model_name, X.columns.tolist())
                    
                    # Save best model name for future reference
                    best_model_file = self.model_dir / "best_model_name.txt"
                    with open(best_model_file, 'w') as f:
                        f.write(best_model_name)

                    self.logger.info(f"Training completed. Best model: {best_model_name}")
                else:
                    self.logger.error("All models failed to train properly")
            
            return model_performances
            
        except Exception as e:
            self.logger.error(f"Error training lap time models: {e}")
            return {}

    def _prepare_model_data_safe(self, data: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare model data with comprehensive data cleaning"""
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

            # Filter available features
            available_features = [col for col in feature_columns if col in data.columns]
            available_categorical = [col for col in categorical_features if col in data.columns]

            if not available_features and not available_categorical:
                self.logger.error("No valid features found in data")
                return pd.DataFrame(), pd.Series()

            # Create feature matrix
            X = data[available_features + available_categorical].copy()
            
            # Handle numerical features with imputation
            if available_features:
                # Create imputer for numerical features
                imputer_name = f'{target}_numerical_imputer'
                if imputer_name not in self.imputers:
                    self.imputers[imputer_name] = SimpleImputer(strategy='median')
                
                X[available_features] = self.imputers[imputer_name].fit_transform(X[available_features])

            # Encode categorical features safely
            for col in available_categorical:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()

                # Handle missing values in categorical features
                X[col] = X[col].fillna('Unknown').astype(str)
                
                # Fit encoder on all unique values
                unique_values = X[col].unique()
                self.label_encoders[col].fit(unique_values)
                X[col] = self.label_encoders[col].transform(X[col])

            # Scale features
            scaler_name = f'{target}_scaler'
            if scaler_name not in self.scalers:
                self.scalers[scaler_name] = StandardScaler()
                X_scaled = self.scalers[scaler_name].fit_transform(X)
            else:
                X_scaled = self.scalers[scaler_name].transform(X)
                
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            
            # Prepare target with cleaning
            y = data[target].dropna()
            
            # Remove outliers from target (beyond 3 standard deviations)
            y_mean, y_std = y.mean(), y.std()
            y_mask = np.abs(y - y_mean) <= 3 * y_std
            y = y[y_mask]
            
            # Align X and y indices
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]

            return X, y
        
        except Exception as e:
            self.logger.error(f"Error preparing model data: {e}")
            return pd.DataFrame(), pd.Series()

    def _train_single_model_safe(self, model_name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                                y_train: pd.Series, y_test: pd.Series) -> ModelPerformance:
        """Train a single model with comprehensive error handling"""
        try:
            model = self.models[model_name]

            # Check for NaN or infinite values before training
            if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
                self.logger.error(f"X_train contains NaN or inf values for {model_name}")
                return ModelPerformance(model_name, float('inf'), float('inf'), -1, float('inf'))
            
            if np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
                self.logger.error(f"y_train contains NaN or inf values for {model_name}")
                return ModelPerformance(model_name, float('inf'), float('inf'), -1, float('inf'))

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Check predictions for validity
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                self.logger.error(f"Model {model_name} produced invalid predictions")
                return ModelPerformance(model_name, float('inf'), float('inf'), -1, float('inf'))

            # Cross-validation with error handling
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//10), 
                                          scoring='neg_mean_absolute_error')
                cv_score = -cv_scores.mean() if len(cv_scores) > 0 else float('inf')
            except:
                cv_score = float('inf')

            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            performance = ModelPerformance(
                model_name=model_name, 
                mae=mae, 
                rmse=rmse, 
                r2_score=r2, 
                cv_score=cv_score
            )

            self.logger.info(f'{model_name} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}')
            return performance
        
        except Exception as e:
            self.logger.error(f"Error training model {model_name}: {e}")
            return ModelPerformance(model_name, float('inf'), float('inf'), -1, float('inf'))

    def predict_lap_time(self, session_id: str, driver_code: str, 
                        lap_number: int, tire_compound: str, 
                        tire_age: int, **kwargs) -> Optional[ModelPrediction]:
        """
        Predict lap time with improved error handling and fallbacks
        """
        try:
            # Load best model if not already loaded
            best_model_name = self._get_best_model_name()
            if best_model_name not in self.models:
                self._load_model(best_model_name)
                
            if best_model_name not in self.models:
                self.logger.error("No trained model available")
                return None
            
            # Gather session context with fallbacks
            session_context = self._get_session_context_safe(session_id)
            driver_history = self._get_driver_recent_performance_safe(session_id, driver_code, lap_number)
            
            # Build comprehensive feature vector with safe defaults
            feature_vector = self._build_prediction_features_safe(
                session_id=session_id,
                driver_code=driver_code,
                lap_number=lap_number,
                tire_compound=tire_compound,
                tire_age=tire_age,
                session_context=session_context,
                driver_history=driver_history,
                **kwargs
            )
            
            if feature_vector is None:
                return None
            
            # Make prediction with the best model
            model = self.models[best_model_name]
            predicted_time = model.predict([feature_vector])[0]
            
            # Validate prediction
            if np.isnan(predicted_time) or np.isinf(predicted_time) or predicted_time <= 0:
                self.logger.error("Invalid prediction generated")
                return None
            
            # Calculate prediction confidence
            confidence_interval = self._calculate_confidence_interval_safe(
                feature_vector, predicted_time, best_model_name
            )
            
            # Get feature importance
            feature_importance = self._get_prediction_feature_importance_safe(
                model, feature_vector, best_model_name
            )
            
            # Apply racing-specific adjustments
            predicted_time = self._apply_racing_adjustments_safe(
                predicted_time, session_context, tire_compound, tire_age
            )
            
            prediction = ModelPrediction(
                predicted_value=round(max(predicted_time, 60.0), 3),  # Ensure minimum realistic lap time
                confidence_interval=confidence_interval,
                feature_importance=feature_importance,
                model_name=best_model_name
            )
            
            self.logger.info(
                f"Predicted lap time for {driver_code}: {prediction.predicted_value:.3f}s "
                f"(±{(confidence_interval[1] - confidence_interval[0])/2:.3f}s)"
            )
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting lap time: {e}")
            return None

    def _get_session_context_safe(self, session_id: str) -> Dict[str, Any]:
        """Retrieve session context with safe defaults"""
        try:
            with db_pool.get_connection() as conn:
                query = """
                    SELECT 
                        COALESCE(s.air_temp, 25.0) as air_temp,
                        COALESCE(s.track_temp, 35.0) as track_temp,
                        COALESCE(s.humidity, 60.0) as humidity,
                        COALESCE(s.session_type, 'R') as session_type,
                        COALESCE(r.circuit_name, 'Unknown') as circuit_name,
                        COALESCE(AVG(l.lap_time), 90.0) as session_avg_pace
                    FROM sessions s
                    JOIN races r ON s.race_id = r.race_id
                    LEFT JOIN lap_times l ON s.session_id = l.session_id 
                        AND COALESCE(l.deleted, 0) = 0
                        AND l.lap_time BETWEEN 60 AND 120
                    WHERE s.session_id = ?
                    GROUP BY s.session_id
                """
                
                result = pd.read_sql_query(query, conn, params=(session_id,))
                
                if not result.empty:
                    return result.iloc[0].to_dict()
                else:
                    # Return safe defaults
                    return {
                        'air_temp': 25.0,
                        'track_temp': 35.0,
                        'humidity': 60.0,
                        'session_type': 'R',
                        'circuit_name': 'Unknown',
                        'session_avg_pace': 90.0
                    }
                    
        except Exception as e:
            self.logger.error(f"Error getting session context: {e}")
            return {
                'air_temp': 25.0,
                'track_temp': 35.0,
                'humidity': 60.0,
                'session_type': 'R',
                'circuit_name': 'Unknown',
                'session_avg_pace': 90.0
            }

    def _get_driver_recent_performance_safe(self, session_id: str, driver_code: str, 
                                          current_lap: int) -> Dict[str, Any]:
        """Get driver's recent performance with safe defaults"""
        try:
            with db_pool.get_connection() as conn:
                query = """
                    SELECT 
                        lap_time,
                        lap_number,
                        COALESCE(tire_compound, 'MEDIUM') as tire_compound,
                        COALESCE(tire_age, 1) as tire_age,
                        COALESCE(position, 10) as position,
                        sector_1_time,
                        sector_2_time,
                        sector_3_time,
                        speed_i1,
                        speed_i2,
                        speed_fl
                    FROM lap_times
                    WHERE session_id = ? 
                        AND driver_code = ?
                        AND lap_number < ?
                        AND COALESCE(deleted, 0) = 0
                        AND lap_time IS NOT NULL
                    ORDER BY lap_number DESC
                    LIMIT 10
                """
                
                history = pd.read_sql_query(query, conn, params=(session_id, driver_code, current_lap))
                
                if history.empty:
                    return {'has_history': False}
                
                # Calculate performance metrics with safe defaults
                recent_laps = history.head(5)
                
                performance_data = {
                    'has_history': True,
                    'prev_lap_time': float(history.iloc[0]['lap_time']) if len(history) > 0 else None,
                    'avg_recent_pace': float(recent_laps['lap_time'].mean()),
                    'pace_trend': self._calculate_pace_trend_safe(history),
                    'current_position': int(history.iloc[0]['position']) if len(history) > 0 else 10,
                    'position_trend': self._calculate_position_trend_safe(history),
                    'current_tire_compound': str(history.iloc[0]['tire_compound']) if len(history) > 0 else 'MEDIUM',
                    'stint_length': self._calculate_current_stint_length_safe(history),
                    'sector_performance': self._analyze_sector_performance_safe(recent_laps)
                }
                
                return performance_data
                
        except Exception as e:
            self.logger.error(f"Error getting driver history: {e}")
            return {'has_history': False}

    def _build_prediction_features_safe(self, session_id: str, driver_code: str, 
                                      lap_number: int, tire_compound: str, tire_age: int,
                                      session_context: Dict[str, Any], 
                                      driver_history: Dict[str, Any], **kwargs) -> Optional[np.ndarray]:
        """Build feature vector with comprehensive error handling"""
        try:
            # Base features with safe defaults
            features = {
                'lap_number': float(lap_number),
                'tire_age': float(tire_age),
                'air_temp': float(session_context.get('air_temp', 25.0)),
                'track_temp': float(session_context.get('track_temp', 35.0)),
                'humidity': float(session_context.get('humidity', 60.0)),
            }
            
            # Add weather-derived features
            features['temp_difference'] = features['track_temp'] - features['air_temp']
            
            # Driver performance features with safe defaults
            avg_pace = session_context.get('session_avg_pace', 90.0)
            
            if driver_history.get('has_history', False):
                features.update({
                    'prev_lap_time': driver_history.get('prev_lap_time', avg_pace),
                    'driver_form': driver_history.get('avg_recent_pace', avg_pace),
                    'position': float(driver_history.get('current_position', 10)),
                    'stint_lap': float(driver_history.get('stint_length', tire_age)),
                })
                
                # Pace trend
                prev_lap = features['prev_lap_time']
                if prev_lap and not np.isnan(prev_lap):
                    features['lap_time_trend'] = prev_lap - features['driver_form']
                else:
                    features['lap_time_trend'] = 0.0
                    
                # Position change
                features['position_change'] = float(driver_history.get('position_trend', 0.0))
            else:
                # Use session averages for new drivers
                features.update({
                    'prev_lap_time': avg_pace,
                    'driver_form': avg_pace,
                    'lap_time_trend': 0.0,
                    'position': float(kwargs.get('position', 10)),
                    'stint_lap': float(tire_age),
                    'position_change': 0.0
                })
            
            # Sector times (estimated if not provided)
            sector_perf = driver_history.get('sector_performance', {})
            features.update({
                'sector_1_time': sector_perf.get('sector_1', features['prev_lap_time'] * 0.35),
                'sector_2_time': sector_perf.get('sector_2', features['prev_lap_time'] * 0.35), 
                'sector_3_time': sector_perf.get('sector_3', features['prev_lap_time'] * 0.30),
            })
            
            # Speed trap estimates
            features.update({
                'speed_i1': float(kwargs.get('speed_i1', 280.0)),
                'speed_i2': float(kwargs.get('speed_i2', 290.0)),
                'speed_fl': float(kwargs.get('speed_fl', 310.0)),
            })
            
            # Compound relative pace with safe calculation
            compound_avg = self._get_compound_average_pace_safe(session_id, tire_compound)
            features['compound_relative_pace'] = features['prev_lap_time'] - compound_avg
            
            # Categorical features
            categorical_features = {
                'tire_compound': str(tire_compound),
                'session_type': str(session_context.get('session_type', 'R')),
                'circuit_name': str(session_context.get('circuit_name', 'Unknown')),
                'driver_code': str(driver_code)
            }
            
            # Convert to feature vector using safe preprocessing
            feature_vector = self._encode_and_scale_features_safe(features, categorical_features)
            
            return feature_vector
            
        except Exception as e:
            self.logger.error(f"Error building prediction features: {e}")
            return None

    def _encode_and_scale_features_safe(self, features: Dict[str, Any], 
                                      categorical_features: Dict[str, str]) -> Optional[np.ndarray]:
        """Encode categorical features and scale all features safely"""
        try:
            # Combine all features
            all_features = {**features, **categorical_features}
            
            # Create DataFrame with single row
            feature_df = pd.DataFrame([all_features])
            
            # Encode categorical features safely
            for col, value in categorical_features.items():
                if col in self.label_encoders:
                    encoder = self.label_encoders[col]
                    try:
                        # Handle unseen categories gracefully
                        if str(value) in encoder.classes_:
                            encoded_value = encoder.transform([str(value)])[0]
                        else:
                            # Use most frequent class or 0 as fallback
                            encoded_value = 0
                        feature_df[col] = encoded_value
                    except:
                        feature_df[col] = 0
                else:
                    # If encoder doesn't exist, use simple mapping
                    feature_df[col] = 0
            
            # Apply imputation if available
            numeric_cols = [col for col in features.keys()]
            if 'lap_time_numerical_imputer' in self.imputers:
                try:
                    feature_df[numeric_cols] = self.imputers['lap_time_numerical_imputer'].transform(feature_df[numeric_cols])
                except:
                    # Fill with medians if imputer fails
                    feature_df[numeric_cols] = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())
            
            # Scale features
            if 'lap_time_scaler' in self.scalers:
                scaler = self.scalers['lap_time_scaler']
                try:
                    scaled_features = scaler.transform(feature_df)
                    return scaled_features[0]
                except Exception as scale_error:
                    self.logger.warning(f"Scaling failed, using normalized features: {scale_error}")
                    # Fallback: simple normalization
                    numeric_data = feature_df.select_dtypes(include=[np.number]).values[0]
                    mean_val = np.nanmean(numeric_data)
                    std_val = np.nanstd(numeric_data)
                    if std_val > 0:
                        return (numeric_data - mean_val) / std_val
                    else:
                        return numeric_data
            else:
                return feature_df.values[0]
                
        except Exception as e:
            self.logger.error(f"Error encoding features: {e}")
            # Return a safe default feature vector
            return np.zeros(15)

    # Helper methods with safe implementations
    def _calculate_pace_trend_safe(self, history: pd.DataFrame) -> float:
        """Calculate pace trend safely"""
        try:
            if len(history) < 3:
                return 0.0
            
            recent_laps = history.head(3)['lap_time'].values
            older_laps = history.tail(3)['lap_time'].values
            
            if len(recent_laps) > 0 and len(older_laps) > 0:
                return float(np.nanmean(older_laps) - np.nanmean(recent_laps))
            return 0.0
        except:
            return 0.0

    def _calculate_position_trend_safe(self, history: pd.DataFrame) -> float:
        """Calculate position change trend safely"""
        try:
            if len(history) < 2:
                return 0.0
            return float(history.iloc[-1]['position'] - history.iloc[0]['position'])
        except:
            return 0.0

    def _calculate_current_stint_length_safe(self, history: pd.DataFrame) -> int:
        """Calculate current stint length safely"""
        try:
            if history.empty:
                return 1
            
            current_compound = history.iloc[0]['tire_compound']
            stint_length = 0
            
            for _, lap in history.iterrows():
                if lap['tire_compound'] == current_compound:
                    stint_length += 1
                else:
                    break
            
            return max(stint_length, 1)
        except:
            return 1

    def _analyze_sector_performance_safe(self, recent_laps: pd.DataFrame) -> Dict[str, float]:
        """Analyze sector performance safely"""
        try:
            if recent_laps.empty:
                return {}
            
            result = {}
            for sector in ['sector_1_time', 'sector_2_time', 'sector_3_time']:
                if sector in recent_laps.columns:
                    sector_data = recent_laps[sector].dropna()
                    if not sector_data.empty:
                        result[sector.replace('_time', '')] = float(sector_data.mean())
                    else:
                        result[sector.replace('_time', '')] = 0.0
                else:
                    result[sector.replace('_time', '')] = 0.0
            
            return result
        except:
            return {}

    def _get_compound_average_pace_safe(self, session_id: str, tire_compound: str) -> float:
        """Get compound average pace safely"""
        try:
            with db_pool.get_connection() as conn:
                query = """
                    SELECT AVG(lap_time) as avg_pace
                    FROM lap_times
                    WHERE session_id = ? 
                        AND tire_compound = ?
                        AND COALESCE(deleted, 0) = 0
                        AND lap_time BETWEEN 60 AND 120
                """
                
                result = pd.read_sql_query(query, conn, params=(session_id, tire_compound))
                
                if not result.empty and result.iloc[0]['avg_pace'] is not None:
                    return float(result.iloc[0]['avg_pace'])
                else:
                    return 90.0  # Fallback average
                    
        except Exception as e:
            self.logger.error(f"Error getting compound average: {e}")
            return 90.0

    def _calculate_confidence_interval_safe(self, feature_vector: np.ndarray, 
                                          prediction: float, model_name: str) -> Tuple[float, float]:
        """Calculate confidence interval safely"""
        try:
            predictions = []
            
            # Get predictions from available models
            for name, model in self.models.items():
                if name.startswith('lap_time_') and hasattr(model, 'predict'):
                    try:
                        pred = model.predict([feature_vector])[0]
                        if not (np.isnan(pred) or np.isinf(pred)):
                            predictions.append(pred)
                    except:
                        continue
            
            if len(predictions) > 1:
                pred_std = np.std(predictions)
                confidence_width = max(1.96 * pred_std, prediction * 0.01)
            else:
                confidence_width = prediction * 0.02  # ±2% of prediction
            
            lower_bound = max(prediction - confidence_width, 60.0)  # Minimum realistic lap time
            upper_bound = prediction + confidence_width
            
            return (round(lower_bound, 3), round(upper_bound, 3))
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            return (round(prediction * 0.98, 3), round(prediction * 1.02, 3))

    def _get_prediction_feature_importance_safe(self, model, feature_vector: np.ndarray, 
                                              model_name: str) -> Dict[str, float]:
        """Get feature importance safely"""
        try:
            # Default feature importance mapping
            default_importance = {
                'tire_age': 0.25, 'driver_form': 0.20, 'track_temp': 0.15, 
                'lap_number': 0.15, 'position': 0.10, 'tire_compound': 0.08,
                'stint_lap': 0.07
            }
            
            # Try to get actual feature importance
            if hasattr(model, 'feature_importances_'):
                importance_values = model.feature_importances_
                
                # Load feature names from metadata
                try:
                    metadata_path = self.model_dir / f"{model_name}_metadata.joblib"
                    if metadata_path.exists():
                        metadata = joblib.load(metadata_path)
                        feature_names = metadata.get('feature_names', [])
                        
                        if len(feature_names) == len(importance_values):
                            importance_dict = {}
                            for name, importance in zip(feature_names, importance_values):
                                importance_dict[name] = float(importance)
                            
                            # Sort and return top 5
                            sorted_importance = sorted(importance_dict.items(), 
                                                     key=lambda x: x[1], reverse=True)
                            return dict(sorted_importance[:5])
                except:
                    pass
            
            return default_importance
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {e}")
            return {'tire_age': 0.25, 'driver_form': 0.20, 'track_temp': 0.15, 
                    'lap_number': 0.15, 'position': 0.10}

    def _apply_racing_adjustments_safe(self, predicted_time: float, session_context: Dict[str, Any],
                                     tire_compound: str, tire_age: int) -> float:
        """Apply racing adjustments safely"""
        try:
            adjusted_time = predicted_time
            
            # Session type adjustments
            session_type = session_context.get('session_type', 'R')
            if session_type == 'Q':
                adjusted_time *= 0.985  # Qualifying is typically faster
            elif session_type in ['FP1', 'FP2', 'FP3']:
                adjusted_time *= 1.01   # Practice is typically slower
            
            # Extreme tire age penalty
            if tire_age > 40:
                age_penalty = (tire_age - 40) * 0.02
                adjusted_time += age_penalty
            
            # Track temperature extremes
            track_temp = session_context.get('track_temp', 35.0)
            if track_temp > 50:
                adjusted_time += (track_temp - 50) * 0.01
            elif track_temp < 20:
                adjusted_time += (20 - track_temp) * 0.005
            
            return max(adjusted_time, 60.0)  # Ensure minimum realistic time
            
        except Exception as e:
            self.logger.error(f"Error applying racing adjustments: {e}")
            return max(predicted_time, 60.0)

    def _get_best_model_name(self) -> str:
        """Get the name of the best performing model"""
        best_model_file = self.model_dir / "best_model_name.txt"
        
        if best_model_file.exists():
            try:
                with open(best_model_file, 'r') as f:
                    return f.read().strip()
            except:
                pass
        
        return 'lap_time_rf'  # Default fallback

    def _save_model(self, model_name: str, feature_names: List[str]):
        """Save trained model with metadata"""
        try:
            model_path = self.model_dir / f"{model_name}.joblib"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            encoder_path = self.model_dir / f"{model_name}_encoders.joblib"
            imputer_path = self.model_dir / f"{model_name}_imputers.joblib"
            metadata_path = self.model_dir / f"{model_name}_metadata.joblib"

            # Save model
            joblib.dump(self.models[model_name], model_path)

            # Save scalers
            relevant_scalers = {k: v for k, v in self.scalers.items() if model_name.split('_')[0] in k}
            if relevant_scalers:
                joblib.dump(relevant_scalers, scaler_path)

            # Save encoders
            if self.label_encoders:
                joblib.dump(self.label_encoders, encoder_path)

            # Save imputers
            relevant_imputers = {k: v for k, v in self.imputers.items() if model_name.split('_')[0] in k}
            if relevant_imputers:
                joblib.dump(relevant_imputers, imputer_path)

            # Save metadata
            metadata = {
                'feature_names': feature_names,
                'model_type': type(self.models[model_name]).__name__,
                'trained_at': pd.Timestamp.now().isoformat(),
                'n_features': len(feature_names)
            }
            joblib.dump(metadata, metadata_path)

            self.logger.info(f"Model {model_name} saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model {model_name}: {e}")

    def _load_model(self, model_name: str):
        """Load saved model and metadata"""
        try:
            model_path = self.model_dir / f"{model_name}.joblib"
            scaler_path = self.model_dir / f"{model_name}_scaler.joblib"
            encoder_path = self.model_dir / f"{model_name}_encoders.joblib"
            imputer_path = self.model_dir / f"{model_name}_imputers.joblib"
            
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                self.logger.info(f"Model {model_name} loaded successfully")
                
                # Load scalers
                if scaler_path.exists():
                    loaded_scalers = joblib.load(scaler_path)
                    self.scalers.update(loaded_scalers)
                    
                # Load encoders
                if encoder_path.exists():
                    loaded_encoders = joblib.load(encoder_path)
                    self.label_encoders.update(loaded_encoders)
                
                # Load imputers
                if imputer_path.exists():
                    loaded_imputers = joblib.load(imputer_path)
                    self.imputers.update(loaded_imputers)
                    
            else:
                self.logger.warning(f"Model {model_name} not found at {model_path}")
                
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")

    # Tire degradation methods (simplified for space)
    def prepare_tire_degradation_features(self, session_ids: List[str]) -> pd.DataFrame:
        """Prepare tire degradation features safely"""
        try:
            with db_pool.get_connection() as conn:
                session_placeholders = ','.join(['?' for _ in session_ids])
                query = f"""
                    SELECT 
                        l.driver_code,
                        COALESCE(l.tire_compound, 'MEDIUM') as tire_compound,
                        COALESCE(l.tire_age, 1) as tire_age,
                        l.lap_time,
                        l.lap_number,
                        COALESCE(s.air_temp, 25.0) as air_temp,
                        COALESCE(s.track_temp, 35.0) as track_temp,
                        l.session_id
                    FROM lap_times l
                    JOIN sessions s ON l.session_id = s.session_id
                    WHERE l.session_id IN ({session_placeholders})
                    AND COALESCE(l.deleted, 0) = 0
                    AND l.lap_time IS NOT NULL
                    AND l.lap_time BETWEEN 60 AND 120
                    AND l.tire_age > 0
                    ORDER BY l.session_id, l.driver_code, l.lap_number
                """

                data = pd.read_sql_query(query, conn, params=tuple(session_ids))

                if data.empty:
                    return pd.DataFrame()

                # Calculate degradation features safely
                degradation_data = []
                
                for (session_id, driver, compound), group in data.groupby(['session_id', 'driver_code', 'tire_compound']):
                    if len(group) < 3:
                        continue
                    
                    group = group.sort_values('lap_number')
                    
                    # Safe degradation calculation
                    try:
                        initial_pace = group['lap_time'].iloc[:3].mean()
                        final_pace = group['lap_time'].iloc[-3:].mean()
                        stint_length = len(group)
                        
                        if stint_length > 1 and not np.isnan(initial_pace) and not np.isnan(final_pace):
                            degradation_rate = (final_pace - initial_pace) / stint_length
                        else:
                            degradation_rate = 0.0
                        
                        degradation_data.append({
                            'driver_code': driver,
                            'tire_compound': compound,
                            'stint_length': stint_length,
                            'initial_pace': initial_pace,
                            'degradation_rate': degradation_rate,
                            'avg_air_temp': group['air_temp'].mean(),
                            'avg_track_temp': group['track_temp'].mean(),
                            'session_id': session_id
                        })
                    except:
                        continue

                if degradation_data:
                    result_df = pd.DataFrame(degradation_data)
                    # Remove any rows with NaN values
                    result_df = result_df.dropna()
                    self.logger.info(f'Prepared {len(result_df)} degradation records')
                    return result_df
                else:
                    return pd.DataFrame()

        except Exception as e:
            self.logger.error(f'Error preparing tire degradation features: {e}')
            return pd.DataFrame()

    def train_tire_degradation_model(self, session_ids: List[str]) -> ModelPerformance:
        """Train tire degradation model safely"""
        try:
            data = self.prepare_tire_degradation_features(session_ids)

            if data.empty:
                self.logger.error('No data available for tire degradation training')
                return ModelPerformance('tire_degradation', float('inf'), float('inf'), -1, float('inf'))
            
            # Prepare features
            X, y = self._prepare_degradation_model_data_safe(data)

            if X.empty or len(y) == 0:
                self.logger.error('No valid features for degradation training')
                return ModelPerformance('tire_degradation', float('inf'), float('inf'), -1, float('inf'))
            
            # Clean data
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
            X = X[mask]
            y = y[mask]
            
            if len(X) < 5:
                self.logger.error('Insufficient clean data for degradation training')
                return ModelPerformance('tire_degradation', float('inf'), float('inf'), -1, float('inf'))
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            performance = self._train_single_model_safe('tire_degradation', X_train, X_test, y_train, y_test)
            
            if performance.mae != float('inf'):
                self._save_model('tire_degradation', X.columns.tolist())

            return performance
            
        except Exception as e:
            self.logger.error(f"Error training tire degradation model: {e}")
            return ModelPerformance('tire_degradation', float('inf'), float('inf'), -1, float('inf'))

    def _prepare_degradation_model_data_safe(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare degradation model data safely"""
        try:
            feature_columns = ['stint_length', 'avg_track_temp', 'avg_air_temp', 'initial_pace']
            categorical_columns = ['tire_compound', 'driver_code']

            # Check columns exist
            missing_cols = [col for col in feature_columns + categorical_columns if col not in data.columns]
            if missing_cols:
                self.logger.error(f"Missing columns: {missing_cols}")
                return pd.DataFrame(), pd.Series()

            X = data[feature_columns + categorical_columns].copy()
            
            # Handle numerical features
            if 'degradation_numerical_imputer' not in self.imputers:
                self.imputers['degradation_numerical_imputer'] = SimpleImputer(strategy='median')
            
            X[feature_columns] = self.imputers['degradation_numerical_imputer'].fit_transform(X[feature_columns])
            
            # Encode categorical variables safely
            for col in categorical_columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = X[col].fillna('Unknown').astype(str)
                X[col] = self.label_encoders[col].fit_transform(X[col])
            
            # Scale features
            if 'degradation_scaler' not in self.scalers:
                self.scalers['degradation_scaler'] = StandardScaler()
                X_scaled = self.scalers['degradation_scaler'].fit_transform(X)
            else:
                X_scaled = self.scalers['degradation_scaler'].transform(X)
            
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
            y = data['degradation_rate'].dropna()
            
            # Align indices
            common_indices = X.index.intersection(y.index)
            X = X.loc[common_indices]
            y = y.loc[common_indices]
            
            return X, y
        
        except Exception as e:
            self.logger.error(f"Error preparing degradation model data: {e}")
            return pd.DataFrame(), pd.Series()

    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get model performance summary"""
        summary = {
            'available_models': list(self.models.keys()),
            'saved_models': [],
            'best_lap_time_model': self._get_best_model_name()
        }
        
        # Check for saved models
        try:
            for model_file in self.model_dir.glob("*.joblib"):
                if not model_file.name.endswith(('_scaler.joblib', '_encoders.joblib', '_metadata.joblib', '_imputers.joblib')):
                    summary['saved_models'].append(model_file.stem)
        except:
            pass
        
        return summary


# Example usage with error handling
if __name__ == "__main__":
    try:
        ml_models = F1PredictiveModels()
        
        # Use actual session IDs from your database
        session_ids = ["2024_abu_dhabi_grand_prix_r", "2022_abu_dhabi_grand_prix_r"]
        
        print("Training lap time prediction models...")
        performances = ml_models.train_lap_time_model(session_ids)
        
        for model_name, performance in performances.items():
            if performance.mae != float('inf'):
                print(f"{model_name}:")
                print(f"  MAE: {performance.mae:.3f}s")
                print(f"  RMSE: {performance.rmse:.3f}s")
                print(f"  R²: {performance.r2_score:.3f}")
            else:
                print(f"{model_name}: Training failed")
        
        # Train tire degradation model
        print("\nTraining tire degradation model...")
        degradation_performance = ml_models.train_tire_degradation_model(session_ids)
        if degradation_performance.mae != float('inf'):
            print(f"Tire Degradation - MAE: {degradation_performance.mae:.4f}")
        else:
            print("Tire degradation training failed")
        
        # Example prediction
        if any(p.mae != float('inf') for p in performances.values()):
            print("\nMaking example prediction...")
            prediction = ml_models.predict_lap_time(
                session_id="2024_abu_dhabi_grand_prix_r",
                driver_code="HUL",
                lap_number=25,
                tire_compound="HARD",
                tire_age=23,
                position=7
            )
            
            if prediction:
                print(f"Predicted lap time: {prediction.predicted_value:.3f}s")
                print(f"Confidence: {prediction.confidence_interval}")
            else:
                print("Prediction failed")
        
        # Model summary
        summary = ml_models.get_model_performance_summary()
        print(f"\nModel Summary: {summary}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")