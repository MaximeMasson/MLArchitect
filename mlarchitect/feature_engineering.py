import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import os

class FeatureEngineering:
    def __init__(self, data_dir='../data/raw', fillna_method='median'):
        """
        Initialize the FeatureEngineering class
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data files (x_train.csv and y_train.csv)
        fillna_method : str
            Method to fill missing values ('median' or 'mean')
        """
        self.data_dir = data_dir
        self.fillna_method = fillna_method
        self.windows = [5, 10, 20]  # Standard time windows
        self.data = None
        self.x_train = None
        self.y_train = None
        self.category_cols = ['INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']
        
    def load_data(self):
        """
        Load and merge x_train and y_train data
        
        Returns:
        --------
        self : FeatureEngineering
            The instance itself for method chaining
        """
        # Load training data
        x_train_path = os.path.join(self.data_dir, 'x_train.csv')
        y_train_path = os.path.join(self.data_dir, 'y_train.csv')
        
        # Check if files exist
        if not os.path.exists(x_train_path) or not os.path.exists(y_train_path):
            raise FileNotFoundError(f"Data files not found in directory: {self.data_dir}")
        
        # Load data
        try:
            self.x_train = pd.read_csv(x_train_path, index_col='ID')
            self.y_train = pd.read_csv(y_train_path, index_col='ID')
        except Exception as e:
            raise RuntimeError(f"Error reading CSV files: {str(e)}")
        
        # Merge the data
        try:
            self.data = pd.concat([self.x_train, self.y_train], axis=1)
        except Exception as e:
            raise RuntimeError(f"Error merging data: {str(e)}")
            
        # Multiply returns by 100 for better scaling
        ret_cols = [col for col in self.data.columns if col.startswith('RET')]
        self.data[ret_cols] *= 100
        
        # Identify important column groups
        self.ret_cols = ret_cols
        self.vol_cols = [col for col in self.data.columns if col.startswith('VOLUME')]
        self.cat_cols = ['STOCK', 'INDUSTRY', 'INDUSTRY_GROUP', 'SECTOR', 'SUB_INDUSTRY']
        
        return self
    
    def get_data_info(self):
        """
        Get information about the loaded data
        
        Returns:
        --------
        dict : Dictionary containing data information
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': {
                'returns': len(self.ret_cols),
                'volumes': len(self.vol_cols),
                'categorical': len(self.cat_cols)
            },
            'date_range': {
                'start': self.data['DATE'].min(),
                'end': self.data['DATE'].max()
            },
            'stocks': {
                'count': self.data['STOCK'].nunique(),
                'sectors': self.data['SECTOR'].nunique() if 'SECTOR' in self.data.columns else 0
            },
            'missing_values': self.data.isnull().sum().to_dict()
        }
        
        return info

    def handle_nan(self, fill_methods=None, date_threshold=10):
        """
        Handle missing values with customized strategies per column
        
        Parameters:
        -----------
        fill_methods : dict
            Dictionary mapping column names to their fill methods.
            Valid methods are: 'mean', 'median', '0'
            Example: {'RET': 'median', 'VOLUME': '0', 'PRICE': 'mean'}
            Columns not specified will be left untouched
        date_threshold : int
            Dates with more than this many NaN values will be removed
        
        Returns:
        --------
        self : FeatureEngineering
            The instance itself for method chaining
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Remove dates with too many NaN values
        # Calculate the total number of possible values for each date
        total_values_per_date = self.data.groupby('DATE').size() * len(self.data.columns)
        # Count NaN values by date
        nan_counts_by_date = self.data.groupby('DATE', group_keys=False).apply(lambda x: x.isna().sum().sum())
        nan_percentage_by_date = (nan_counts_by_date / total_values_per_date) 
        dates_to_drop = nan_percentage_by_date[nan_percentage_by_date > date_threshold].index
        self.data = self.data[~self.data['DATE'].isin(dates_to_drop)]
        print(f"Dates dropped: {dates_to_drop}")
        
        # Apply specified fill methods for each column
        if fill_methods:
            for column, method in fill_methods.items():
                if column not in self.data.columns:
                    print(f"Warning: Column '{column}' not found in data")
                    continue
                
                if method == 'mean':
                    fill_value = self.data[column].mean()
                elif method == 'median':
                    fill_value = self.data[column].median()
                elif method == '0':
                    fill_value = 0
                else:
                    print(f"Warning: Invalid fill method '{method}' for column '{column}'")
                    continue
                    
                self.data.loc[:, column] = self.data[column].fillna(fill_value)
        
        return self
    
    def add_basic_statistical_features(self):
        """
        Category 1: Basic Statistical Features
        - Moving averages
        - Moving medians
        - Moving standard deviations
        - Moving quantiles
        - Skewness and Kurtosis
        - Historical extrema
        - Cumulative momentum
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Dictionary to store all new features
        new_features = {}
        
        # Group by stock to calculate features for each stock separately
        for stock in self.data['STOCK'].unique():
            stock_data = self.data[self.data['STOCK'] == stock].copy()
            
            # Sort by date to ensure proper calculation of features
            stock_data = stock_data.sort_values('DATE')
            
            for col_prefix in ['RET', 'VOLUME']:
                
                # For each date, calculate features across the available columns
                for window in self.windows:
                        
                    # Get the relevant columns for this window size
                    window_cols = [f"{col_prefix}_{col}" for col in range(1, window+1)]
                    
                    # Calculate statistics across these columns for each date
                    stock_data.loc[:, f'{col_prefix}_ma_{window}'] = stock_data[window_cols].mean(axis=1)
                    stock_data.loc[:, f'{col_prefix}_median_{window}'] = stock_data[window_cols].median(axis=1)
                    stock_data.loc[:, f'{col_prefix}_std_{window}'] = stock_data[window_cols].std(axis=1)
                    
                    # Quantiles
                    if window >= 10:
                        stock_data.loc[:, f'{col_prefix}_q25_{window}'] = stock_data[window_cols].quantile(0.25, axis=1)
                        stock_data.loc[:, f'{col_prefix}_q75_{window}'] = stock_data[window_cols].quantile(0.75, axis=1)
                    
                    # Historical extrema
                    stock_data.loc[:, f'{col_prefix}_max_{window}'] = stock_data[window_cols].max(axis=1)
                    stock_data.loc[:, f'{col_prefix}_min_{window}'] = stock_data[window_cols].min(axis=1)
                    
                    # Skewness and Kurtosis (only for window >= 10)
                    if window >= 10:
                        stock_data.loc[:, f'{col_prefix}_skew_{window}'] = stock_data[window_cols].skew(axis=1)
                        stock_data.loc[:, f'{col_prefix}_kurt_{window}'] = stock_data[window_cols].kurtosis(axis=1)
                    
                    # Cumulative momentum for returns
                    if col_prefix == 'RET' and window <= 5:
                        ret_cols_for_cum = [c for c in window_cols if int(c.split('_')[1]) <= 5]
                        if len(ret_cols_for_cum) > 0:
                            # Convert to decimal returns, calculate product, convert back to percentage
                            stock_data.loc[:, f'{col_prefix}_cum_ret_5'] = (
                                (1 + stock_data[ret_cols_for_cum]/100).prod(axis=1) - 1
                            ) * 100
            
            # Update the new features dictionary with this stock's data
            for col in stock_data.columns:
                if col not in self.data.columns:
                    if col not in new_features:
                        new_features[col] = pd.Series(index=self.data.index)
                    new_features[col].update(stock_data[col])
        
        # Combine all new features with existing data
        self.data = pd.concat([self.data, pd.DataFrame(new_features, index=self.data.index)], axis=1)
        
        return self

    def add_temporal_features(self):
        """
        Category 2: Temporal Features
        - Technical indicators based on return and volume data
        - Momentum indicators
        - Volatility measures
        - Relative strength indicators
        - Pattern detection features
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Dictionary to store all new features
        new_features = {}
        
        # Process each stock individually
        for stock_id in self.data['STOCK'].unique():
            stock_data = self.data[self.data['STOCK'] == stock_id].copy()
            # Sort by date to ensure proper feature calculation
            stock_data = stock_data.sort_values('DATE')
            
            # Get available return columns for this stock
            ret_cols = self.ret_cols
            vol_cols = self.vol_cols
            
            # 1. Momentum indicators
            # Rate of change (ROC) - comparing most recent return to older returns
            stock_data['MOMENTUM_1_5'] = stock_data['RET_1'] / (stock_data['RET_5'] + 0.01)
            stock_data['MOMENTUM_1_10'] = stock_data['RET_1'] / (stock_data['RET_10'] + 0.01)
            stock_data['MOMENTUM_1_20'] = stock_data['RET_1'] / (stock_data['RET_20'] + 0.01)
            
            # 2. Volatility measures
            # High-Low range for returns - Optimisé sans apply
            stock_data['RET_RANGE_5'] = stock_data[ret_cols[:5]].max(axis=1) - stock_data[ret_cols[:5]].min(axis=1)
            stock_data['RET_RANGE_10'] = stock_data[ret_cols[:10]].max(axis=1) - stock_data[ret_cols[:10]].min(axis=1)
            stock_data['RET_RANGE_20'] = stock_data[ret_cols[:20]].max(axis=1) - stock_data[ret_cols[:20]].min(axis=1)
            
            # 3. Relative Strength Index (RSI) adaptation
            # Optimisé sans fonction apply et calculate_rsi
            ret_data = stock_data[ret_cols[:14]]
            # Créer des masques pour les gains et les pertes
            gains_mask = ret_data > 0
            losses_mask = ret_data < 0
            
            # Calculer les gains et les pertes moyens
            gains = ret_data.copy()
            losses = ret_data.copy()
            gains[~gains_mask] = 0
            losses[~losses_mask] = 0
            losses = losses.abs()
            
            avg_gains = gains.mean(axis=1)
            avg_losses = losses.mean(axis=1)
            
            # Éviter la division par zéro
            avg_losses_safe = avg_losses.copy()
            avg_losses_safe[avg_losses_safe == 0] = 0.01
            
            # Calculer le RSI
            rs = avg_gains / avg_losses_safe
            stock_data['RSI_14'] = 100 - (100 / (1 + rs))
            
            # 4. Moving Average Convergence Divergence (MACD) adaptation
            # Optimisé sans apply
            # Calculer les EMA pour chaque colonne de rendement, puis prendre la moyenne
            ema_10_cols = []
            ema_20_cols = []
            
            for col in ret_cols[:10]:
                ema_10_cols.append(stock_data[col].ewm(span=10, adjust=False).mean())
            
            for col in ret_cols[:20]:
                ema_20_cols.append(stock_data[col].ewm(span=20, adjust=False).mean())
            
            ema_10 = pd.concat(ema_10_cols, axis=1).mean(axis=1)
            ema_20 = pd.concat(ema_20_cols, axis=1).mean(axis=1)
            stock_data['MACD'] = ema_10 - ema_20
            
            # 5. Bollinger Bands
            # Optimisé sans apply
            sma_20 = stock_data[ret_cols[:20]].mean(axis=1)
            std_20 = stock_data[ret_cols[:20]].std(axis=1)
            stock_data['BB_UPPER'] = sma_20 + (std_20 * 2)
            stock_data['BB_LOWER'] = sma_20 - (std_20 * 2)
            stock_data['BB_WIDTH'] = (stock_data['BB_UPPER'] - stock_data['BB_LOWER']) / sma_20
            stock_data['BB_POSITION'] = (stock_data['RET_1'] - sma_20) / (std_20 * 2)
            
            # 6. Volume-based indicators
            # Volume-weighted return
            stock_data['VOL_WEIGHTED_RET'] = (stock_data[ret_cols[0]] * stock_data[vol_cols[0]]) / (stock_data[vol_cols[0]] + 0.0001)
            
            # On-Balance Volume (OBV) adaptation
            obv = pd.Series(0, index=stock_data.index)
            for i in range(5):
                obv += np.where(stock_data[ret_cols[i]] >= 0, 
                                stock_data[vol_cols[i]], 
                                -stock_data[vol_cols[i]])
            stock_data['OBV_5'] = obv
            
            # Volume Oscillator - Optimisé sans apply
            vol_ma_5 = stock_data[vol_cols[:5]].mean(axis=1)
            vol_ma_10 = stock_data[vol_cols[:10]].mean(axis=1)
            stock_data['VOL_OSC'] = ((vol_ma_5 - vol_ma_10) / vol_ma_10) * 100
            
            # 7. Pattern detection features
            # Detect trend reversals
            stock_data['REVERSAL_PATTERN'] = ((stock_data['RET_1'] > 0) & 
                                                (stock_data['RET_2'] < 0) & 
                                                (stock_data['RET_3'] < 0)).astype(int) - \
                                            ((stock_data['RET_1'] < 0) & 
                                                (stock_data['RET_2'] > 0) & 
                                                (stock_data['RET_3'] > 0)).astype(int)
        
            # 8. Acceleration/Deceleration
            stock_data['RET_ACCEL'] = stock_data['RET_1'] - 2*stock_data['RET_2'] + stock_data['RET_3']
            
            # Update the new features dictionary with this stock's data
            for col in stock_data.columns:
                if col not in self.data.columns:
                    if col not in new_features:
                        new_features[col] = pd.Series(index=self.data.index)
                    new_features[col].update(stock_data[col])
        
        # Combine all new features with existing data
        self.data = pd.concat([self.data, pd.DataFrame(new_features, index=self.data.index)], axis=1)
        
        return self

    def add_stock_comparison_features(self):
        """
        - Sector mean deviations
        - Relative z-scores
        - Cross-sectional rankings
        - Mean distances
        - Relative volatilities
        - Performance relative par catégorie
        - Score de tendance composite
        - Ratio volatilité/rendement
        - Anomalie de volume
        - Persistance du momentum
        - Distance aux bandes de Bollinger
        - Score RSI ajusté
        - Contribution au volume sectoriel
        - Accélération corrigée du rendement
        - Probabilité de retournement
        - Efficacité du volume
        - Stabilité inter-périodes
        - Divergence MACD/RSI
        - Poids dans la volatilité sectorielle
        - Score de liquidité ajustée
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if 'SECTOR' not in self.data.columns:
            return self
            
        # Dictionary to store all new features
        new_features = {}
        
        specific_ret_cols = ['RET_1', 'RET_5', 'RET_10', 'RET_20']
        specific_vol_cols = ['VOLUME_1', 'VOLUME_5', 'VOLUME_10', 'VOLUME_20']
        
        # Precompute sector-level statistics for efficiency
        sector_groups = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)
        sector_stats = {}
        for col in specific_ret_cols + specific_vol_cols:
            sector_stats[col] = {
                'mean': sector_groups[col].transform('mean'),
                'std': sector_groups[col].transform('std').replace(0, np.finfo(float).eps)
            }
        
        # Dictionary to collect features before concatenation
        feature_dict = {}
        
        # Basic features for each return and volume column
        for col in specific_ret_cols + specific_vol_cols:
            # Sector mean deviations
            feature_dict[f'{col}_sector_diff'] = self.data[col].fillna(0) - sector_stats[col]['mean']
            
            # Sector z-scores
            feature_dict[f'{col}_sector_zscore'] = (self.data[col].fillna(0) - sector_stats[col]['mean']) / sector_stats[col]['std']
            
            # Cross-sectional rankings
            feature_dict[f'{col}_sector_rank'] = sector_groups[col].rank(pct=True).fillna(0.5)
            
            # Relative volatilities (using fixed 20-day window instead of rolling)
            window_cols = [f'{col.split("_")[0]}_{i}' for i in range(1, 21)]  # e.g., RET_1 to RET_20
            
            rolling_std = self.data[window_cols].std(axis=1)
            sector_vol = self.data.groupby('SECTOR')[window_cols].std().mean(axis=1).reindex(self.data.index, fill_value=np.finfo(float).eps)
            feature_dict[f'{col}_rel_vol'] = rolling_std / sector_vol
    
        # Combine features into new_features DataFrame (assuming this is part of a larger new_features)
        new_features = pd.DataFrame(feature_dict, index=self.data.index)
        
        # 1. Performance relative par catégorie
        if 'RET_ma_20' in self.data.columns:
            # Par SECTOR
            sector_ret_ma_20_mean = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['RET_ma_20'].transform('mean')
            sector_ret_ma_20_mean = sector_ret_ma_20_mean.replace(0, np.finfo(float).eps)
            new_features['RET_ma_20_sector_rel_perf'] = (self.data['RET_ma_20'] - sector_ret_ma_20_mean) / sector_ret_ma_20_mean
            
            # Par INDUSTRY
            if 'INDUSTRY' in self.data.columns:
                industry_ret_ma_20_mean = self.data.groupby(['DATE', 'INDUSTRY'], group_keys=False)['RET_ma_20'].transform('mean')
                industry_ret_ma_20_mean = industry_ret_ma_20_mean.replace(0, np.finfo(float).eps)
                new_features['RET_ma_20_industry_rel_perf'] = (self.data['RET_ma_20'] - industry_ret_ma_20_mean) / industry_ret_ma_20_mean
            
            # Par SUB_INDUSTRY
            if 'SUB_INDUSTRY' in self.data.columns:
                sub_industry_ret_ma_20_mean = self.data.groupby(['DATE', 'SUB_INDUSTRY'], group_keys=False)['RET_ma_20'].transform('mean')
                sub_industry_ret_ma_20_mean = sub_industry_ret_ma_20_mean.replace(0, np.finfo(float).eps)
                new_features['RET_ma_20_sub_industry_rel_perf'] = (self.data['RET_ma_20'] - sub_industry_ret_ma_20_mean) / sub_industry_ret_ma_20_mean
        
        # 2. Score de tendance composite
        if 'MACD' in self.data.columns and 'MOMENTUM_1_20' in self.data.columns and 'RET_ma_20' in self.data.columns and 'RET_ma_10' in self.data.columns:
            # Calculer la pente de RET_ma_20
            slope_ret_ma = self.data['RET_ma_10'] - self.data['RET_ma_20']
            
            # Normaliser les composants entre 0 et 1
            macd_norm = (self.data['MACD'] - self.data['MACD'].min()) / (self.data['MACD'].max() - self.data['MACD'].min() + np.finfo(float).eps)
            momentum_norm = (self.data['MOMENTUM_1_20'] - self.data['MOMENTUM_1_20'].min()) / (self.data['MOMENTUM_1_20'].max() - self.data['MOMENTUM_1_20'].min() + np.finfo(float).eps)
            slope_norm = (slope_ret_ma - slope_ret_ma.min()) / (slope_ret_ma.max() - slope_ret_ma.min() + np.finfo(float).eps)
            
            # Calculer le score composite (pondération égale)
            new_features['TREND_COMPOSITE_SCORE'] = (macd_norm + momentum_norm + slope_norm) / 3
        
        # 3. Ratio volatilité/rendement
        if 'RET_ma_20' in self.data.columns and 'RET_std_20' in self.data.columns:
            # Éviter division par zéro
            ret_std_20_safe = self.data['RET_std_20'].replace(0, np.finfo(float).eps)
            new_features['RET_SHARPE_RATIO'] = self.data['RET_ma_20'] / ret_std_20_safe
        
        # 4. Anomalie de volume
        if 'VOLUME_ma_20' in self.data.columns:
            # Calculer la moyenne et l'écart-type du volume par secteur
            sector_vol_mean = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('mean')
            sector_vol_std = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('std')
            sector_vol_std = sector_vol_std.replace(0, np.finfo(float).eps)
            
            # Calculer l'anomalie de volume (z-score)
            new_features['VOLUME_ANOMALY'] = (self.data['VOLUME_ma_20'] - sector_vol_mean) / sector_vol_std
        
        # 5. Persistance du momentum
        if 'MOMENTUM_1_5' in self.data.columns and 'MOMENTUM_1_10' in self.data.columns and 'MOMENTUM_1_20' in self.data.columns:
            # Normaliser chaque momentum
            mom_5_norm = (self.data['MOMENTUM_1_5'] - self.data['MOMENTUM_1_5'].mean()) / (self.data['MOMENTUM_1_5'].std() + np.finfo(float).eps)
            mom_10_norm = (self.data['MOMENTUM_1_10'] - self.data['MOMENTUM_1_10'].mean()) / (self.data['MOMENTUM_1_10'].std() + np.finfo(float).eps)
            mom_20_norm = (self.data['MOMENTUM_1_20'] - self.data['MOMENTUM_1_20'].mean()) / (self.data['MOMENTUM_1_20'].std() + np.finfo(float).eps)
            
            # Calculer les écarts entre les périodes
            diff_5_10 = np.abs(mom_5_norm - mom_10_norm)
            diff_10_20 = np.abs(mom_10_norm - mom_20_norm)
            diff_5_20 = np.abs(mom_5_norm - mom_20_norm)
            
            # Calculer la persistance (inverse de la moyenne des écarts)
            avg_diff = (diff_5_10 + diff_10_20 + diff_5_20) / 3
            new_features['MOMENTUM_PERSISTENCE'] = 1 / (1 + avg_diff)
        
        # 6. Distance aux bandes de Bollinger
        if 'RET_1' in self.data.columns and 'BB_UPPER' in self.data.columns and 'BB_LOWER' in self.data.columns and 'BB_WIDTH' in self.data.columns:
            # Calculer la distance normalisée
            bb_width_safe = self.data['BB_WIDTH'].replace(0, np.finfo(float).eps)
            
            # Distance au bord supérieur (positif si au-dessus, négatif si en-dessous)
            new_features['BB_UPPER_DISTANCE'] = (self.data['RET_1'] - self.data['BB_UPPER']) / bb_width_safe
            
            # Distance au bord inférieur (positif si au-dessus, négatif si en-dessous)
            new_features['BB_LOWER_DISTANCE'] = (self.data['RET_1'] - self.data['BB_LOWER']) / bb_width_safe
        
        # 7. Score RSI ajusté
        if 'RSI_14' in self.data.columns and 'RET_std_20' in self.data.columns:
            # Ajuster le RSI en fonction de la volatilité
            new_features['RSI_VOLATILITY_ADJUSTED'] = self.data['RSI_14'] / (1 + self.data['RET_std_20'])
        
        # 8. Contribution au volume sectoriel
        if 'VOLUME_ma_20' in self.data.columns:
            # Calculer la somme du volume par secteur et date
            sector_vol_sum = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('sum')
            sector_vol_sum = sector_vol_sum.replace(0, np.finfo(float).eps)
            
            # Calculer la contribution en pourcentage
            new_features['VOLUME_SECTOR_CONTRIBUTION'] = (self.data['VOLUME_ma_20'] / sector_vol_sum) * 100
        
        # 9. Accélération corrigée du rendement
        if 'RET_ACCEL' in self.data.columns and 'RET_std_20' in self.data.columns:
            # Normaliser l'accélération par la volatilité
            ret_std_20_safe = self.data['RET_std_20'].replace(0, np.finfo(float).eps)
            new_features['RET_ACCEL_NORMALIZED'] = self.data['RET_ACCEL'] / ret_std_20_safe
        
        # 10. Probabilité de retournement
        if 'REVERSAL_PATTERN' in self.data.columns and 'RSI_14' in self.data.columns and 'BB_POSITION' in self.data.columns:
            # Créer des indicateurs de conditions extrêmes
            rsi_extreme = ((self.data['RSI_14'] > 70) | (self.data['RSI_14'] < 30)).astype(float)
            bb_extreme = ((self.data['BB_POSITION'] > 0.8) | (self.data['BB_POSITION'] < -0.8)).astype(float)
            
            # Combiner les indicateurs
            reversal_pattern = self.data['REVERSAL_PATTERN'].astype(float).abs()  # Assurer que c'est positif
            new_features['REVERSAL_PROBABILITY'] = (0.4 * rsi_extreme + 0.4 * bb_extreme + 0.2 * reversal_pattern)
        
        # 11. Efficacité du volume
        if 'VOL_WEIGHTED_RET' in self.data.columns and 'VOLUME_ma_20' in self.data.columns:
            # Éviter division par zéro
            volume_ma_20_safe = self.data['VOLUME_ma_20'].replace(0, np.finfo(float).eps)
            new_features['VOLUME_EFFICIENCY'] = self.data['VOL_WEIGHTED_RET'] / volume_ma_20_safe
        
        # 12. Stabilité inter-périodes
        if 'RET_ma_5' in self.data.columns and 'RET_ma_10' in self.data.columns and 'RET_ma_20' in self.data.columns:
            # Calculer l'écart-type des moyennes mobiles
            ma_cols = ['RET_ma_5', 'RET_ma_10', 'RET_ma_20']
            ma_std = self.data[ma_cols].std(axis=1)
            new_features['MA_STABILITY'] = 1 / (1 + ma_std)  # Inverse pour que les valeurs élevées indiquent plus de stabilité
        
        # 13. Divergence MACD/RSI
        if 'MACD' in self.data.columns and 'RSI_14' in self.data.columns:
            # Normaliser MACD et RSI
            macd_norm = (self.data['MACD'] - self.data['MACD'].mean()) / (self.data['MACD'].std() + np.finfo(float).eps)
            rsi_norm = (self.data['RSI_14'] - 50) / 50  # Centrer autour de 0
            
            # Calculer la divergence (produit des signes)
            new_features['MACD_RSI_DIVERGENCE'] = np.sign(macd_norm) * np.sign(rsi_norm) * -1
            # -1 signifie divergence (signes opposés), 1 signifie convergence (mêmes signes)
        
        # 14. Poids dans la volatilité sectorielle
        if 'RET_std_20' in self.data.columns:
            # Calculer la volatilité moyenne par secteur
            sector_std_mean = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['RET_std_20'].transform('mean')
            sector_std_mean = sector_std_mean.replace(0, np.finfo(float).eps)
            
            # Calculer le poids relatif
            new_features['VOLATILITY_SECTOR_WEIGHT'] = self.data['RET_std_20'] / sector_std_mean
        
        # 15. Score de liquidité ajustée
        if 'VOLUME_ma_20' in self.data.columns and 'VOL_OSC' in self.data.columns:
            # Normaliser le volume et l'oscillateur
            vol_ma_norm = (self.data['VOLUME_ma_20'] - self.data['VOLUME_ma_20'].min()) / (self.data['VOLUME_ma_20'].max() - self.data['VOLUME_ma_20'].min() + np.finfo(float).eps)
            vol_osc_norm = (self.data['VOL_OSC'] - self.data['VOL_OSC'].min()) / (self.data['VOL_OSC'].max() - self.data['VOL_OSC'].min() + np.finfo(float).eps)
            
            # Calculer le score de liquidité (70% volume, 30% oscillateur)
            liquidity_score = 0.7 * vol_ma_norm + 0.3 * vol_osc_norm
            self.data['liquidity_score'] = liquidity_score  # Add liquidity_score to the DataFrame
            
            # Normaliser par la médiane sectorielle
            sector_liquidity_median = self.data.groupby(['DATE', 'SECTOR'], group_keys=False)['liquidity_score'].transform('median')
            sector_liquidity_median = sector_liquidity_median.replace(0, np.finfo(float).eps)
            
            new_features['LIQUIDITY_ADJUSTED_SCORE'] = liquidity_score / sector_liquidity_median
        
        # Basic features for SECTOR, SUB_INDUSTRY, INDUSTRY_GROUP, and INDUSTRY
        for group in ['SECTOR', 'SUB_INDUSTRY', 'INDUSTRY_GROUP', 'INDUSTRY']:
            if group in self.data.columns:
                for col in ['RET_1']:
                    group_mean = self.data.groupby(['DATE', group], group_keys=False)[col].transform('mean')
                    group_std = self.data.groupby(['DATE', group], group_keys=False)[col].transform('std')
                    new_features[f'{col}_{group.lower()}_mean'] = group_mean
                    new_features[f'{col}_{group.lower()}_std'] = group_std
            
        # Combine all new features with existing data
        self.data = pd.concat([self.data, pd.DataFrame(new_features, index=self.data.index)], axis=1)
        
        return self

    def add_date_comparison_features(self):
        """
        - Fractal or Multiscale Features
        - Seasonal and Cyclical Features
        - Time-Based Anomaly Detection
        - Dynamic Market Regimes
        - Network-Based Metrics
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Dictionary to store all new features, initialized with NaN
        new_features = pd.DataFrame(index=self.data.index)
        
        # Define return columns
        ret_cols = [f'RET_{i}' for i in range(1, 21)]
        
        # 1. Fractal or Multiscale Features (Hurst Exponent)
        cum_ret = self.data[ret_cols].cumsum(axis=1)
        R = cum_ret.max(axis=1) - cum_ret.min(axis=1)
        S = self.data[ret_cols].std(axis=1).replace(0, np.finfo(float).eps)  # Avoid division by zero
        hurst = np.log(R / S) / np.log(20)
        new_features['hurst_exponent'] = hurst.fillna(0)  # NaN if all returns are NaN
        
        # 3. Seasonal and Cyclical Features
        # Assume a 5-day cycle; adjust if DATE encoding differs
        self.data['cycle_label'] = self.data['DATE'] % 5
        cycle_means = self.data.groupby(['STOCK', 'cycle_label'])['RET_1'].transform('mean')
        new_features['cycle_avg_ret_1'] = cycle_means.fillna(0)  # NaN if no data for cycle
        
        # 4. Time-Based Anomaly Detection
        past_ret_cols = [f'RET_{i}' for i in range(2, 21)]
        mean_ret = self.data[past_ret_cols].mean(axis=1)
        std_ret = self.data[past_ret_cols].std(axis=1).replace(0, np.finfo(float).eps)  # Avoid division by zero
        z_score = (self.data['RET_1'].fillna(0) - mean_ret) / std_ret
        new_features['anomaly_z_score'] = z_score.fillna(0)  # NaN if insufficient data
        
        # 5. Dynamic Market Regimes
        # Use RET_1 to RET_5 for short-term, RET_1 to RET_20 for long-term, averaged across stocks per DATE
        market_ret_short = self.data.groupby('DATE')[[f'RET_{i}' for i in range(1, 6)]].mean().mean(axis=1)
        market_ret_long = self.data.groupby('DATE')[ret_cols].mean().mean(axis=1)
        market_regime = (market_ret_short > market_ret_long).astype(int)
        new_features['market_regime'] = self.data['DATE'].map(market_regime).fillna(0)  # NaN if DATE missing
        
        # 6. Network-Based Metrics for SECTOR - Optimized implementation
        new_features['network_centrality_sector'] = 0  # Initialize with zeros
        
        for sector in self.data['SECTOR'].unique():
            # Get data for this sector
            sector_mask = self.data['SECTOR'] == sector
            sector_data = self.data[sector_mask]
            
            # Calculate mean returns across all stocks in the sector
            sector_ret_mean = sector_data[ret_cols].mean(axis=0)
            
            # For each stock-date row in this sector, calculate correlation with sector average
            for stock_id in sector_data['STOCK'].unique():
                # Get data for this stock in this sector
                stock_mask = sector_data['STOCK'] == stock_id
                stock_indices = sector_data[stock_mask].index
                
                # Get stock return data
                stock_ret = sector_data.loc[stock_indices, ret_cols]
                
                # Calculate correlation efficiently for all rows at once
                valid_mask = ~(stock_ret.isna() | sector_ret_mean.isna())
                if valid_mask.sum().sum() > 0:  # If at least some valid data points
                    # Calculate correlation for each row with the sector average
                    corrs = stock_ret.apply(
                        lambda row: np.corrcoef(
                            row[valid_mask.loc[row.name]], 
                            sector_ret_mean[valid_mask.loc[row.name]]
                        )[0, 1] if valid_mask.loc[row.name].sum() > 1 else 0, 
                        axis=1
                    )
                    # Update the feature dataframe with calculated correlations
                    new_features.loc[stock_indices, 'network_centrality_sector'] = corrs.fillna(0)

        # 7. Network-Based Metrics for INDUSTRY - Optimized implementation
        new_features['network_centrality_industry'] = 0  # Initialize with zeros
        
        for industry in self.data['INDUSTRY'].unique():
            # Get data for this industry
            industry_mask = self.data['INDUSTRY'] == industry
            industry_data = self.data[industry_mask]
            
            # Calculate mean returns across all stocks in the industry
            industry_ret_mean = industry_data[ret_cols].mean(axis=0)
            
            # For each stock-date row in this industry, calculate correlation with industry average
            for stock_id in industry_data['STOCK'].unique():
                # Get data for this stock in this industry
                stock_mask = industry_data['STOCK'] == stock_id
                stock_indices = industry_data[stock_mask].index
                
                # Get stock return data
                stock_ret = industry_data.loc[stock_indices, ret_cols]
                
                # Calculate correlation efficiently for all rows at once
                valid_mask = ~(stock_ret.isna() | industry_ret_mean.isna())
                if valid_mask.sum().sum() > 0:  # If at least some valid data points
                    # Calculate correlation for each row with the industry average
                    corrs = stock_ret.apply(
                        lambda row: np.corrcoef(
                            row[valid_mask.loc[row.name]], 
                            industry_ret_mean[valid_mask.loc[row.name]]
                        )[0, 1] if valid_mask.loc[row.name].sum() > 1 else 0, 
                        axis=1
                    )
                    # Update the feature dataframe with calculated correlations
                    new_features.loc[stock_indices, 'network_centrality_industry'] = corrs.fillna(0)
        
        # Combine all new features with existing data
        self.data = pd.concat([self.data, new_features], axis=1)
        
        return self

    def add_clustering_features(self):
        """
        Category 6: Clustering Features
        - Stock clusters
        - Temporal clusters
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if len(self.ret_cols) > 0:
            # Prepare data for clustering
            features_for_clustering = self.data[self.ret_cols].copy()
            # Gérer les NaN avant le clustering
            features_for_clustering = features_for_clustering.fillna(features_for_clustering.mean())
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features_for_clustering)
            
            # Stock clustering
            kmeans_stocks = KMeans(n_clusters=5, random_state=42)
            self.data['stock_cluster'] = kmeans_stocks.fit_predict(scaled_features)
            
            # Temporal clustering - utiliser group_keys=False
            daily_features = self.data.groupby('DATE', group_keys=False)[self.ret_cols].mean()
            # Gérer les NaN avant le clustering
            daily_features = daily_features.fillna(daily_features.mean())
            kmeans_temporal = KMeans(n_clusters=3, random_state=42)
            temporal_clusters = kmeans_temporal.fit_predict(daily_features)
            cluster_map = pd.Series(temporal_clusters, index=daily_features.index)
            self.data['temporal_cluster'] = self.data['DATE'].map(cluster_map)
        
        return self

    def add_categorical_features(self):
        """
        Category 7: Categorical Information Based Features
        - Sector means
        - Sector impact
        - Frequency encoding
        - Target encoding
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if 'SECTOR' in self.data.columns:
            # Dictionary to store all new features
            new_features = {}
            
            # Historical sector means - utiliser group_keys=False
            for col in self.ret_cols + self.vol_cols:
                sector_means = self.data.groupby('SECTOR', group_keys=False)[col].transform('mean')
                new_features[f'{col}_sector_mean'] = sector_means
            
            # Recent sector impact - utiliser group_keys=False
            for window in [5, 10]:
                recent_impact = self.data.groupby('SECTOR', group_keys=False)[self.ret_cols[0]].transform(
                    lambda x: x.rolling(window=window).mean()
                )
                new_features[f'sector_impact_{window}'] = recent_impact
            
            # Frequency encoding
            sector_freq = self.data['SECTOR'].value_counts(normalize=True)
            new_features['sector_freq'] = self.data['SECTOR'].map(sector_freq)
            
            # Target encoding (if applicable)
            if 'RET' in self.data.columns:  # Utiliser RET comme cible
                target_means = self.data.groupby('SECTOR', group_keys=False)['RET'].transform('mean')
                new_features['sector_target_encoding'] = target_means
            
            # Combine all new features with existing data
            self.data = pd.concat([self.data, pd.DataFrame(new_features, index=self.data.index)], axis=1)
        
        return self

    def calculate_slopes(self, series, window):
        """Calculate slopes over a moving window"""
        slopes = []
        for i in range(len(series)):
            if i < window:
                slopes.append(np.nan)
                continue
                
            # Extraire les données pour la fenêtre actuelle
            y_data = series.iloc[i-window+1:i+1].values
            
            # Vérifier s'il y a des NaN dans les données
            if np.isnan(y_data).any():
                slopes.append(np.nan)
                continue
                
            x = np.arange(window)
            try:
                slope = LinearRegression().fit(x.reshape(-1, 1), y_data).coef_[0]
                slopes.append(slope)
            except:
                slopes.append(np.nan)
                
        return slopes

    def calculate_rolling_correlation(self, series1, series2, window):
        """Calculate rolling correlation between two series"""
        # Gérer les NaN dans le calcul de la corrélation
        return series1.rolling(window).corr(series2)
    
    def safe_autocorr(self, series, lag):
        """Calculate autocorrelation safely handling NaN values"""
        # Créer une série temporaire sans NaN
        temp_series = series.copy()
        if temp_series.isna().any():
            # Remplacer les NaN par la moyenne de la série
            temp_series = temp_series.fillna(temp_series.mean())
        
        try:
            return temp_series.autocorr(lag)
        except:
            return np.nan

    def create_feature_set(self, features=['all']):
        """
        Create complete feature set according to specified categories
        
        Parameters:
        -----------
        features : list
            List of feature categories to include:
            - 'all': All categories
            - 'basic': Basic statistical features
            - 'temporal': Temporal features
            - 'stock_comparison': Stock comparison features
            - 'date_comparison': Date comparison features
            - 'nonlinear': Nonlinear interaction features
            - 'clustering': Clustering features
            - 'categorical': Categorical features
        
        Returns:
        --------
        pandas.DataFrame: DataFrame with all generated features
        """
        # Handle missing values first
        self.handle_nan()
        
        if 'all' in features or 'basic' in features:
            self.add_basic_statistical_features()
        
        if 'all' in features or 'temporal' in features:
            self.add_temporal_features()
        
        if 'all' in features or 'stock_comparison' in features:
            self.add_stock_comparison_features()
        
        if 'all' in features or 'date_comparison' in features:
            self.add_date_comparison_features()
        
        if 'all' in features or 'nonlinear' in features:
            self.add_nonlinear_interaction_features()
        
        if 'all' in features or 'clustering' in features:
            self.add_clustering_features()
        
        if 'all' in features or 'categorical' in features:
            self.add_categorical_features()
        
        return self.data