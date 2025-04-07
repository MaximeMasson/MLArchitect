from mlarchitect.feature_engineering import FeatureEngineering 
import pandas as pd
import numpy as np
import warnings

# Global constant for numerical stability
EPSILON = np.finfo(float).eps

def nan_handling(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle NaN values in the dataframe using the FeatureEngineering class.
    
    Parameters:
        original_df (pd.DataFrame): The original dataframe with possible NaN values
        
    Returns:
        pd.DataFrame: DataFrame with NaN values handled
    """
    # Suppress potential warnings from the FeatureEngineering class
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        fe = FeatureEngineering()
        fe.data = original_df.copy(deep=True)
        
        # Create a dictionary mapping columns with "VOLUME" or "RET" to use median filling
        fill_cols = {
            col: "median" for col in original_df.columns 
            if "VOLUME" in col or "RET" in col
        }
        
        # Handle NaN values with the specified methods
        fe.handle_nan(fill_methods=fill_cols, date_threshold=1)
        
        return fe.data

def sum(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a new feature that sums RET_1 and RET_2
    Return ONLY the newly created column
    
    Parameters:
        original_df (pd.DataFrame): The original dataframe with RET_1 and RET_2 columns
        
    Returns:
        pd.DataFrame: DataFrame containing only the newly created sum column
    """
    # Check if required columns exist
    if "RET_1" not in original_df.columns or "RET_2" not in original_df.columns:
        # Return empty DataFrame with same index if columns don't exist
        return pd.DataFrame({"RET_1_2_sum": np.nan}, index=original_df.index)
    
    # Create a new DataFrame directly instead of copying the original
    result = pd.DataFrame(index=original_df.index)
    
    # Calculate sum of RET_1 and RET_2, handling potential NaN values
    result["RET_1_2_sum"] = original_df["RET_1"].fillna(0) + original_df["RET_2"].fillna(0)
    
    return result

def basic_statistical_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create basic statistical features including:
    - Moving averages
    - Moving medians
    - Moving standard deviations
    - Moving quantiles (25th and 75th percentiles)
    - Skewness and Kurtosis (for window sizes >= 10)
    - Historical extrema (max and min)
    - Cumulative momentum (for RET, computed for windows <= 5)
    
    Parameters:
        original_df (pd.DataFrame): The original dataframe with columns like 'STOCK', 'DATE',
                                    and for each stock, columns for returns (RET_1, RET_2, ...)
                                    and volume (VOLUME_1, VOLUME_2, ...).
                                    
    Returns:
        pd.DataFrame: A DataFrame containing only the newly generated features.
    """
    # Validate input
    required_cols = ['STOCK', 'DATE']
    for col in required_cols:
        if col not in original_df.columns:
            return pd.DataFrame(index=original_df.index)
    
    # Initialize result DataFrame using the original index
    new_features = pd.DataFrame(index=original_df.index)
    
    # Define window sizes - keep small for performance
    windows = [5, 10, 20]
    
    # Pre-check which columns are available
    all_cols = {}
    for col_prefix in ['RET', 'VOLUME']:
        all_cols[col_prefix] = {}
        for i in range(1, max(windows) + 1):
            col_name = f"{col_prefix}_{i}"
            if col_name in original_df.columns:
                all_cols[col_prefix][i] = col_name
    
    # Using groupby to avoid the loop over stocks - faster processing
    for stock_idx, stock_data in original_df.groupby('STOCK'):
        # Sort by DATE to ensure temporal ordering
        stock_data = stock_data.sort_values('DATE')
        
        # Get indices for updating the result DataFrame
        stock_indices = stock_data.index
        
        # Process each column prefix (returns and volume)
        for col_prefix, col_dict in all_cols.items():
            available_cols = list(col_dict.values())
            
            # Skip if no columns are available
            if not available_cols:
                continue
                
            # For each window size, calculate the features
            for window in windows:
                window_cols = [col_dict[i] for i in range(1, window + 1) if i in col_dict]
                
                # Skip if we don't have enough columns for this window
                if len(window_cols) < 2:  # Need at least 2 for most statistics
                    continue
                
                # Process window data in one go for all stocks of this type
                window_data = stock_data[window_cols]
                
                # For small windows with few calculations, process all at once
                feature_dict = {
                    f'{col_prefix}_ma_{window}': window_data.mean(axis=1),
                    f'{col_prefix}_median_{window}': window_data.median(axis=1),
                    f'{col_prefix}_std_{window}': window_data.std(axis=1) + EPSILON,  # Add epsilon to avoid division by zero
                    f'{col_prefix}_max_{window}': window_data.max(axis=1),
                    f'{col_prefix}_min_{window}': window_data.min(axis=1)
                }
                
                # For windows >= 10, compute quantiles, skewness, and kurtosis
                if window >= 10 and len(window_cols) >= 3:  # Need at least 3 for skew/kurt
                    feature_dict.update({
                        f'{col_prefix}_q25_{window}': window_data.quantile(0.25, axis=1),
                        f'{col_prefix}_q75_{window}': window_data.quantile(0.75, axis=1),
                        f'{col_prefix}_skew_{window}': window_data.skew(axis=1),
                        f'{col_prefix}_kurt_{window}': window_data.kurtosis(axis=1)
                    })
                
                # Cumulative momentum for returns (only for RET and window sizes <= 5)
                if col_prefix == 'RET' and window <= 5:
                    ret_cols_for_cum = [c for c in window_cols if int(c.split('_')[1]) <= 5]
                    if ret_cols_for_cum:
                        # Vectorized calculation of cumulative returns - faster than row-wise operations
                        # Adding 1 to avoid negative percentages causing issues
                        cum_returns = (1 + stock_data[ret_cols_for_cum] / 100).prod(axis=1) - 1
                        feature_dict[f'{col_prefix}_cum_ret_{window}'] = cum_returns * 100
                
                # Update new_features in one batch operation per stock - more efficient
                for feat_name, values in feature_dict.items():
                    if feat_name not in new_features:
                        new_features[feat_name] = np.nan
                    new_features.loc[stock_indices, feat_name] = values
    
    # Replace NaN values with 0 and inf/extreme values with reasonable bounds
    new_features = new_features.fillna(0)
    
    # Clip extreme values to prevent outliers
    for col in new_features.columns:
        if 'std' in col or 'kurt' in col:  # Columns that tend to have extreme values
            q_low = new_features[col].quantile(0.01)
            q_high = new_features[col].quantile(0.99)
            new_features[col] = new_features[col].clip(q_low, q_high)
    
    return new_features

def temporal_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features including momentum indicators, volatility measures,
    RSI, MACD, Bollinger Bands, volume-based indicators, pattern detection,
    and acceleration/deceleration metrics.

    Parameters:
        original_df (pd.DataFrame): Input DataFrame with 'STOCK', 'DATE', return and volume columns

    Returns:
        pd.DataFrame: DataFrame containing only the newly generated temporal features
    """
    # Validate input
    required_cols = ['STOCK', 'DATE']
    for col in required_cols:
        if col not in original_df.columns:
            return pd.DataFrame(index=original_df.index)
            
    # Initialize empty DataFrame for new features
    new_features = pd.DataFrame(index=original_df.index)
    
    # Identify columns efficiently and cache the results
    ret_cols = [col for col in original_df.columns if col.startswith("RET_") and col.split('_')[1].isdigit()]
    vol_cols = [col for col in original_df.columns if col.startswith("VOLUME") and col.split('_')[1].isdigit()]
    
    # If essential columns are missing, return empty DataFrame
    if not ret_cols or not vol_cols:
        return new_features
    
    # Sort columns by their number for consistent processing
    ret_cols.sort(key=lambda x: int(x.split('_')[1]))
    vol_cols.sort(key=lambda x: int(x.split('_')[1]))
    
    # Pre-select subsets of columns for different window sizes
    ret_cols_5 = [col for col in ret_cols if int(col.split('_')[1]) <= 5]
    ret_cols_10 = [col for col in ret_cols if int(col.split('_')[1]) <= 10]
    ret_cols_14 = [col for col in ret_cols if int(col.split('_')[1]) <= 14]
    ret_cols_20 = [col for col in ret_cols if int(col.split('_')[1]) <= 20]
    
    vol_cols_5 = [col for col in vol_cols if int(col.split('_')[1]) <= 5]
    vol_cols_10 = [col for col in vol_cols if int(col.split('_')[1]) <= 10]
    
    # Process each stock efficiently using groupby
    for stock_id, stock_data in original_df.groupby('STOCK'):
        # Sort by date for temporal calculations
        stock_data = stock_data.sort_values('DATE')
        
        # Get indices for updating the result DataFrame
        stock_indices = stock_data.index
        
        # Dictionary to store features for this stock
        stock_features = {}
        
        # 1. Momentum Indicators - more robust calculation with epsilon
        if all(col in ret_cols for col in ['RET_1', 'RET_5', 'RET_10', 'RET_20']):
            # Add small constant to avoid division by zero
            stock_features['MOMENTUM_1_5'] = stock_data['RET_1'] / (stock_data['RET_5'] + EPSILON)
            stock_features['MOMENTUM_1_10'] = stock_data['RET_1'] / (stock_data['RET_10'] + EPSILON)
            stock_features['MOMENTUM_1_20'] = stock_data['RET_1'] / (stock_data['RET_20'] + EPSILON)
            
            # Clip extreme values
            for col in ['MOMENTUM_1_5', 'MOMENTUM_1_10', 'MOMENTUM_1_20']:
                stock_features[col] = stock_features[col].clip(-10, 10)
        
        # 2. Volatility Measures - vectorized range calculation
        if len(ret_cols_5) >= 5:
            stock_features['RET_RANGE_5'] = stock_data[ret_cols_5].max(axis=1) - stock_data[ret_cols_5].min(axis=1)
        if len(ret_cols_10) >= 10:
            stock_features['RET_RANGE_10'] = stock_data[ret_cols_10].max(axis=1) - stock_data[ret_cols_10].min(axis=1)
        if len(ret_cols_20) >= 20:
            stock_features['RET_RANGE_20'] = stock_data[ret_cols_20].max(axis=1) - stock_data[ret_cols_20].min(axis=1)
        
        # 3. RSI - using vectorized operations with improved handling of edge cases
        if len(ret_cols_14) >= 14:
            ret_data = stock_data[ret_cols_14]
            gains = ret_data.clip(lower=0)
            losses = -ret_data.clip(upper=0)
            avg_gains = gains.mean(axis=1)
            avg_losses = losses.mean(axis=1)
            # Handle division by zero by adding epsilon
            rs = avg_gains / (avg_losses + EPSILON)
            stock_features['RSI_14'] = 100 - (100 / (1 + rs))
            # Ensure RSI is within valid range [0, 100]
            stock_features['RSI_14'] = stock_features['RSI_14'].clip(0, 100)
        
        # 4. MACD - optimized calculation
        if len(ret_cols_10) >= 10 and len(ret_cols_20) >= 20:
            # Use ewm for accurate EMA calculation, handling NaNs properly
            ema_10 = stock_data[ret_cols_10].mean(axis=1).ewm(span=10, adjust=False, min_periods=1).mean()
            ema_20 = stock_data[ret_cols_20].mean(axis=1).ewm(span=20, adjust=False, min_periods=1).mean()
            stock_features['MACD'] = ema_10 - ema_20
        
        # 5. Bollinger Bands - vectorized calculation with improved numerical stability
        if len(ret_cols_20) >= 20:
            sma_20 = stock_data[ret_cols_20].mean(axis=1)
            std_20 = stock_data[ret_cols_20].std(axis=1) + EPSILON  # Add epsilon to avoid zero std
            
            stock_features['BB_UPPER'] = sma_20 + (std_20 * 2)
            stock_features['BB_LOWER'] = sma_20 - (std_20 * 2)
            # Use epsilon in denominator to avoid division by zero
            stock_features['BB_WIDTH'] = (stock_features['BB_UPPER'] - stock_features['BB_LOWER']) / (sma_20 + EPSILON)
            
            if 'RET_1' in ret_cols:
                # Normalized position within the bands
                stock_features['BB_POSITION'] = (stock_data['RET_1'] - sma_20) / (std_20 * 2)
        
        # 6. Volume-Based Indicators with improved numerical stability
        if 'RET_1' in ret_cols and 'VOLUME_1' in vol_cols:
            # Volume-weighted return with small constant to avoid division by zero
            denominator = stock_data['VOLUME_1'] + EPSILON
            stock_features['VOL_WEIGHTED_RET'] = (stock_data['RET_1'] * stock_data['VOLUME_1']) / denominator
        
        # On-Balance Volume - vectorized calculation
        if len(vol_cols_5) >= 5 and len(ret_cols_5) >= 5:
            # Initialize as Series for cleaner vectorized operations
            obv = pd.Series(0, index=stock_data.index)
            
            # Vectorized calculation for OBV
            for i in range(min(5, len(ret_cols_5), len(vol_cols_5))):
                obv += np.where(stock_data[ret_cols_5[i]] >= 0, 
                               stock_data[vol_cols_5[i]], 
                               -stock_data[vol_cols_5[i]])
            stock_features['OBV_5'] = obv
        
        # Volume Oscillator with proper handling of division
        if len(vol_cols_5) >= 5 and len(vol_cols_10) >= 10:
            vol_ma_5 = stock_data[vol_cols_5].mean(axis=1)
            vol_ma_10 = stock_data[vol_cols_10].mean(axis=1)
            # Add epsilon to denominator to avoid division by zero
            stock_features['VOL_OSC'] = ((vol_ma_5 - vol_ma_10) / (vol_ma_10 + EPSILON)) * 100
            # Clip to reasonable range
            stock_features['VOL_OSC'] = stock_features['VOL_OSC'].clip(-100, 100)
        
        # 7. Pattern Detection - vectorized boolean operations
        if all(col in ret_cols for col in ['RET_1', 'RET_2', 'RET_3']):
            # Calculate reversal pattern signal
            stock_features['REVERSAL_PATTERN'] = (
                ((stock_data['RET_1'] > 0) & (stock_data['RET_2'] < 0) & (stock_data['RET_3'] < 0)).astype(int) -
                ((stock_data['RET_1'] < 0) & (stock_data['RET_2'] > 0) & (stock_data['RET_3'] > 0)).astype(int)
            )
        
        # 8. Acceleration/Deceleration
        if all(col in ret_cols for col in ['RET_1', 'RET_2', 'RET_3']):
            stock_features['RET_ACCEL'] = stock_data['RET_1'] - 2 * stock_data['RET_2'] + stock_data['RET_3']
        
        # Update new features with this stock's data efficiently
        for col, values in stock_features.items():
            if col not in new_features:
                new_features[col] = pd.Series(np.nan, index=original_df.index)
            new_features.loc[stock_indices, col] = values
    
    # Fill NaN values with 0 and handle extreme values
    new_features = new_features.fillna(0)
    
    # Clip extreme values to prevent outliers - using different thresholds based on feature types
    for col in new_features.columns:
        if any(x in col for x in ['MOMENTUM', 'VOL_OSC', 'RET_ACCEL']):
            # These metrics can have high variance
            q_low = new_features[col].quantile(0.005)
            q_high = new_features[col].quantile(0.995)
            new_features[col] = new_features[col].clip(q_low, q_high)
    
    return new_features

def stock_comparison_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create stock comparison features including:
      - Sector mean deviations
      - Relative z-scores
      - Cross-sectional rankings
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

    Parameters:
        original_df (pd.DataFrame): Input DataFrame containing at least:
            - 'DATE', 'SECTOR'
            - Return columns (e.g. 'RET_1', 'RET_5', 'RET_10', 'RET_20', 'RET_ma_20', 'RET_ma_10', 'RET_std_20', 'RET_ACCEL', etc.)
            - Volume columns (e.g. 'VOLUME_1', 'VOLUME_5', 'VOLUME_10', 'VOLUME_20', 'VOLUME_ma_20', 'VOL_OSC', etc.)
            - Other columns such as 'MACD', 'MOMENTUM_1_20', 'RSI_14', 'BB_UPPER', 'BB_LOWER', 'BB_WIDTH', 'BB_POSITION',
              'REVERSAL_PATTERN'
            - Optional grouping columns: 'INDUSTRY', 'SUB_INDUSTRY', 'INDUSTRY_GROUP'
    Returns:
        pd.DataFrame: A DataFrame with only the newly generated features.
    """
    
    # If the input does not have SECTOR information, return an empty DataFrame.
    if 'SECTOR' not in original_df.columns:
        return pd.DataFrame(index=original_df.index)

    # Define the specific return and volume columns for comparison.
    specific_ret_cols = ['RET_1', 'RET_5', 'RET_10', 'RET_20']
    specific_vol_cols = ['VOLUME_1', 'VOLUME_5', 'VOLUME_10', 'VOLUME_20']

    # First filter the columns that actually exist
    specific_ret_cols = [col for col in specific_ret_cols if col in original_df.columns]
    specific_vol_cols = [col for col in specific_vol_cols if col in original_df.columns]
    
    # If no columns exist, return empty DataFrame
    if not specific_ret_cols and not specific_vol_cols:
        return pd.DataFrame(index=original_df.index)

    # Precompute sector-level statistics for efficiency.
    sector_groups = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)
    sector_stats = {}
    for col in specific_ret_cols + specific_vol_cols:
        sector_stats[col] = {
            'mean': sector_groups[col].transform('mean'),
            'std': sector_groups[col].transform('std').replace(0, EPSILON)
        }

    # Dictionary to collect basic features.
    feature_dict = {}

    for col in specific_ret_cols + specific_vol_cols:
        # Sector mean deviations.
        feature_dict[f'{col}_sector_diff'] = original_df[col].fillna(0) - sector_stats[col]['mean']
        # Sector z-scores.
        feature_dict[f'{col}_sector_zscore'] = (original_df[col].fillna(0) - sector_stats[col]['mean']) / sector_stats[col]['std']
        # Cross-sectional rankings (percentile ranks within each DATE/SECTOR group).
        feature_dict[f'{col}_sector_rank'] = sector_groups[col].rank(pct=True).fillna(0.5)
        
        # Relative volatilities using a fixed 20-day window.
        base = col.split('_')[0]
        window_cols = [f"{base}_{i}" for i in range(1, 21) if f"{base}_{i}" in original_df.columns]
        
        if window_cols:  # Only calculate if the window columns exist
            rolling_std = original_df[window_cols].std(axis=1)
            # Group by SECTOR to get sector volatility - using fillna to handle missing values
            sector_vol = original_df.groupby('SECTOR')[window_cols].std().mean(axis=1).reindex(original_df.index, 
                                                                                           fill_value=EPSILON)
            feature_dict[f'{col}_rel_vol'] = rolling_std / sector_vol

    # Start with the basic features.
    new_features = pd.DataFrame(feature_dict, index=original_df.index)

    # 1. Performance relative par catégorie.
    if 'RET_ma_20' in original_df.columns:
        # By SECTOR.
        sector_ret_ma_20_mean = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)['RET_ma_20'].transform('mean')
        sector_ret_ma_20_mean = sector_ret_ma_20_mean.replace(0, EPSILON)
        new_features['RET_ma_20_sector_rel_perf'] = (original_df['RET_ma_20'] - sector_ret_ma_20_mean) / sector_ret_ma_20_mean

        # By INDUSTRY.
        if 'INDUSTRY' in original_df.columns:
            industry_ret_ma_20_mean = original_df.groupby(['DATE', 'INDUSTRY'], group_keys=False)['RET_ma_20'].transform('mean')
            industry_ret_ma_20_mean = industry_ret_ma_20_mean.replace(0, EPSILON)
            new_features['RET_ma_20_industry_rel_perf'] = (original_df['RET_ma_20'] - industry_ret_ma_20_mean) / industry_ret_ma_20_mean

        # By SUB_INDUSTRY.
        if 'SUB_INDUSTRY' in original_df.columns:
            sub_industry_ret_ma_20_mean = original_df.groupby(['DATE', 'SUB_INDUSTRY'], group_keys=False)['RET_ma_20'].transform('mean')
            sub_industry_ret_ma_20_mean = sub_industry_ret_ma_20_mean.replace(0, EPSILON)
            new_features['RET_ma_20_sub_industry_rel_perf'] = (original_df['RET_ma_20'] - sub_industry_ret_ma_20_mean) / sub_industry_ret_ma_20_mean

    # 2. Score de tendance composite.
    if all(col in original_df.columns for col in ['MACD', 'MOMENTUM_1_20', 'RET_ma_20', 'RET_ma_10']):
        # Calculate composite trend score safely
        slope_ret_ma = original_df['RET_ma_10'] - original_df['RET_ma_20']
        
        # Calculate normalization safely with epsilon to prevent division by zero
        macd_range = original_df['MACD'].max() - original_df['MACD'].min() + EPSILON
        momentum_range = original_df['MOMENTUM_1_20'].max() - original_df['MOMENTUM_1_20'].min() + EPSILON
        slope_range = slope_ret_ma.max() - slope_ret_ma.min() + EPSILON
        
        macd_norm = (original_df['MACD'] - original_df['MACD'].min()) / macd_range
        momentum_norm = (original_df['MOMENTUM_1_20'] - original_df['MOMENTUM_1_20'].min()) / momentum_range
        slope_norm = (slope_ret_ma - slope_ret_ma.min()) / slope_range
        
        new_features['TREND_COMPOSITE_SCORE'] = (macd_norm + momentum_norm + slope_norm) / 3

    # 3. Ratio volatilité/rendement (RET Sharpe Ratio surrogate).
    if all(col in original_df.columns for col in ['RET_ma_20', 'RET_std_20']):
        ret_std_20_safe = original_df['RET_std_20'].replace(0, EPSILON)
        new_features['RET_SHARPE_RATIO'] = original_df['RET_ma_20'] / ret_std_20_safe

    # 4. Anomalie de volume.
    if 'VOLUME_ma_20' in original_df.columns:
        sector_vol_mean = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('mean')
        sector_vol_std = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('std').replace(0, EPSILON)
        new_features['VOLUME_ANOMALY'] = (original_df['VOLUME_ma_20'] - sector_vol_mean) / sector_vol_std

    # 5. Persistance du momentum.
    if all(col in original_df.columns for col in ['MOMENTUM_1_5', 'MOMENTUM_1_10', 'MOMENTUM_1_20']):
        # Calculate persistence with proper handling of division by zero
        mom_5_std = original_df['MOMENTUM_1_5'].std() + EPSILON
        mom_10_std = original_df['MOMENTUM_1_10'].std() + EPSILON
        mom_20_std = original_df['MOMENTUM_1_20'].std() + EPSILON
        
        mom_5_norm = (original_df['MOMENTUM_1_5'] - original_df['MOMENTUM_1_5'].mean()) / mom_5_std
        mom_10_norm = (original_df['MOMENTUM_1_10'] - original_df['MOMENTUM_1_10'].mean()) / mom_10_std
        mom_20_norm = (original_df['MOMENTUM_1_20'] - original_df['MOMENTUM_1_20'].mean()) / mom_20_std
        
        diff_5_10 = np.abs(mom_5_norm - mom_10_norm)
        diff_10_20 = np.abs(mom_10_norm - mom_20_norm)
        diff_5_20 = np.abs(mom_5_norm - mom_20_norm)
        avg_diff = (diff_5_10 + diff_10_20 + diff_5_20) / 3
        
        new_features['MOMENTUM_PERSISTENCE'] = 1 / (1 + avg_diff)

    # 6. Distance aux bandes de Bollinger.
    if all(col in original_df.columns for col in ['RET_1', 'BB_UPPER', 'BB_LOWER', 'BB_WIDTH']):
        bb_width_safe = original_df['BB_WIDTH'].replace(0, EPSILON)
        new_features['BB_UPPER_DISTANCE'] = (original_df['RET_1'] - original_df['BB_UPPER']) / bb_width_safe
        new_features['BB_LOWER_DISTANCE'] = (original_df['RET_1'] - original_df['BB_LOWER']) / bb_width_safe

    # 7. Score RSI ajusté.
    if all(col in original_df.columns for col in ['RSI_14', 'RET_std_20']):
        new_features['RSI_VOLATILITY_ADJUSTED'] = original_df['RSI_14'] / (1 + original_df['RET_std_20'])

    # 8. Contribution au volume sectoriel.
    if 'VOLUME_ma_20' in original_df.columns:
        sector_vol_sum = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)['VOLUME_ma_20'].transform('sum').replace(0, EPSILON)
        new_features['VOLUME_SECTOR_CONTRIBUTION'] = (original_df['VOLUME_ma_20'] / sector_vol_sum) * 100

    # 9. Accélération corrigée du rendement.
    if all(col in original_df.columns for col in ['RET_ACCEL', 'RET_std_20']):
        ret_std_20_safe = original_df['RET_std_20'].replace(0, EPSILON)
        new_features['RET_ACCEL_NORMALIZED'] = original_df['RET_ACCEL'] / ret_std_20_safe

    # 10. Probabilité de retournement.
    if all(col in original_df.columns for col in ['REVERSAL_PATTERN', 'RSI_14', 'BB_POSITION']):
        # Calculate reversal probability components safely
        rsi_extreme = ((original_df['RSI_14'] > 70) | (original_df['RSI_14'] < 30)).astype(float)
        bb_extreme = ((original_df['BB_POSITION'] > 0.8) | (original_df['BB_POSITION'] < -0.8)).astype(float)
        reversal_pattern = original_df['REVERSAL_PATTERN'].astype(float).abs()
        
        new_features['REVERSAL_PROBABILITY'] = (0.4 * rsi_extreme + 0.4 * bb_extreme + 0.2 * reversal_pattern)

    # 11. Efficacité du volume.
    if all(col in original_df.columns for col in ['VOL_WEIGHTED_RET', 'VOLUME_ma_20']):
        volume_ma_20_safe = original_df['VOLUME_ma_20'].replace(0, EPSILON)
        new_features['VOLUME_EFFICIENCY'] = original_df['VOL_WEIGHTED_RET'] / volume_ma_20_safe

    # 12. Stabilité inter-périodes.
    if all(col in original_df.columns for col in ['RET_ma_5', 'RET_ma_10', 'RET_ma_20']):
        ma_cols = ['RET_ma_5', 'RET_ma_10', 'RET_ma_20']
        ma_std = original_df[ma_cols].std(axis=1)
        new_features['MA_STABILITY'] = 1 / (1 + ma_std)

    # 13. Divergence MACD/RSI.
    if all(col in original_df.columns for col in ['MACD', 'RSI_14']):
        # Safe normalization of MACD with proper handling of zeros
        macd_std = original_df['MACD'].std() + EPSILON
        macd_norm = (original_df['MACD'] - original_df['MACD'].mean()) / macd_std
        
        # RSI normalized around 0 (where 50 is the center)
        rsi_norm = (original_df['RSI_14'] - 50) / 50
        
        # Calculate divergence (-1 for divergence, 1 for convergence)
        new_features['MACD_RSI_DIVERGENCE'] = np.sign(macd_norm) * np.sign(rsi_norm) * -1

    # 14. Poids dans la volatilité sectorielle.
    if 'RET_std_20' in original_df.columns:
        sector_std_mean = original_df.groupby(['DATE', 'SECTOR'], group_keys=False)['RET_std_20'].transform('mean').replace(0, EPSILON)
        new_features['VOLATILITY_SECTOR_WEIGHT'] = original_df['RET_std_20'] / sector_std_mean

    # 15. Score de liquidité ajustée.
    if all(col in original_df.columns for col in ['VOLUME_ma_20', 'VOL_OSC']):
        # Normalize volume metrics safely
        vol_ma_range = original_df['VOLUME_ma_20'].max() - original_df['VOLUME_ma_20'].min() + EPSILON
        vol_osc_range = original_df['VOL_OSC'].max() - original_df['VOL_OSC'].min() + EPSILON
        
        vol_ma_norm = (original_df['VOLUME_ma_20'] - original_df['VOLUME_ma_20'].min()) / vol_ma_range
        vol_osc_norm = (original_df['VOL_OSC'] - original_df['VOL_OSC'].min()) / vol_osc_range
        
        # Calculate liquidity score
        liquidity_score = 0.7 * vol_ma_norm + 0.3 * vol_osc_norm
        
        # Compute sector median safely
        tmp = pd.DataFrame({'liquidity_score': liquidity_score}, index=original_df.index)
        sector_liquidity_median = original_df.groupby(['DATE', 'SECTOR'])['SECTOR'].transform(
            lambda x: tmp.loc[x.index, 'liquidity_score'].median()
        ).replace(0, EPSILON)
        
        new_features['LIQUIDITY_ADJUSTED_SCORE'] = liquidity_score / sector_liquidity_median

    # 16. Basic features for groupings.
    for group in ['SECTOR', 'SUB_INDUSTRY', 'INDUSTRY_GROUP', 'INDUSTRY']:
        if group in original_df.columns:
            for col in ['RET_1']:
                if col in original_df.columns:
                    group_mean = original_df.groupby(['DATE', group], group_keys=False)[col].transform('mean')
                    group_std = original_df.groupby(['DATE', group], group_keys=False)[col].transform('std')
                    new_features[f'{col}_{group.lower()}_mean'] = group_mean
                    new_features[f'{col}_{group.lower()}_std'] = group_std

    # Clip extreme values to prevent outliers
    for col in new_features.columns:
        if col.endswith('_zscore') or col.endswith('_normalized') or 'ANOMALY' in col:
            q_low = new_features[col].quantile(0.001)
            q_high = new_features[col].quantile(0.999)
            new_features[col] = new_features[col].clip(q_low, q_high)

    return pd.DataFrame(new_features, index=original_df.index)

def date_comparison_features(original_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create date comparison features including:
        - Fractal or Multiscale Features (Hurst Exponent)
        - Seasonal and Cyclical Features
        - Time-Based Anomaly Detection
        - Dynamic Market Regimes
        - Network-Based Metrics for SECTOR and INDUSTRY

    Parameters:
        original_df (pd.DataFrame): Input DataFrame that must contain:
            - 'DATE' (numeric or datetime encoded as a number for modulo operation)
            - 'STOCK'
            - Return columns: 'RET_1' to 'RET_20'
            - Optionally, 'SECTOR' and 'INDUSTRY'

    Returns:
        pd.DataFrame: A DataFrame with only the newly generated features.
    """
    if original_df is None:
        raise ValueError("Data not loaded. Provide a valid DataFrame.")

    # Validate required columns
    required_cols = ['DATE', 'STOCK']
    for col in required_cols:
        if col not in original_df.columns:
            return pd.DataFrame(index=original_df.index)

    new_features = pd.DataFrame(index=original_df.index)

    # Define return columns that exist in the data
    ret_cols = [f'RET_{i}' for i in range(1, 21) if f'RET_{i}' in original_df.columns]
    if not ret_cols:
        return new_features

    # 1. Fractal or Multiscale Features (Hurst Exponent) - with numerical stability
    try:
        cum_ret = original_df[ret_cols].cumsum(axis=1)
        R = (cum_ret.max(axis=1) - cum_ret.min(axis=1)).clip(lower=EPSILON)
        S = original_df[ret_cols].std(axis=1).clip(lower=EPSILON)
        hurst = np.log(R / S) / np.log(max(len(ret_cols), 2))
        # Clip hurst to valid range [0, 1] and handle potential NaN values
        new_features['hurst_exponent'] = hurst.clip(0, 1).fillna(0.5)
    except Exception as e:
        # Fallback in case of error
        print(f"Warning: Error calculating Hurst exponent: {e}")
        new_features['hurst_exponent'] = 0.5
        
    # 2. Seasonal and Cyclical Features - simplified to prevent memory issues
    try:
        if 'DATE' in original_df.columns and 'STOCK' in original_df.columns and 'RET_1' in original_df.columns:
            # Use modulo operator for cycle detection - fast and memory efficient
            df = original_df.copy()[['DATE', 'STOCK', 'RET_1']]
            df['cycle_label'] = df['DATE'] % 5
            
            # Group and aggregate instead of transform for memory efficiency
            cycle_means = df.groupby(['STOCK', 'cycle_label'])['RET_1'].mean().reset_index()
            
            # Create a mapping dictionary for efficient updating
            cycle_map = cycle_means.set_index(['STOCK', 'cycle_label']).to_dict()['RET_1']
            
            # Apply mapping efficiently
            stock_cycle_labels = list(zip(df['STOCK'], df['cycle_label']))
            cycle_values = [cycle_map.get(key, 0) for key in stock_cycle_labels]
            
            new_features['cycle_avg_ret_1'] = cycle_values
    except Exception as e:
        print(f"Warning: Error calculating seasonal features: {e}")
        new_features['cycle_avg_ret_1'] = 0
        
    # 3. Time-Based Anomaly Detection - more robust handling
    try:
        past_ret_cols = [f'RET_{i}' for i in range(2, 21) if f'RET_{i}' in original_df.columns]
        if past_ret_cols and 'RET_1' in original_df.columns:
            mean_ret = original_df[past_ret_cols].mean(axis=1)
            std_ret = original_df[past_ret_cols].std(axis=1).clip(lower=EPSILON)
            z_score = (original_df['RET_1'].fillna(0) - mean_ret) / std_ret
            # Clip z-score to reasonable range to prevent extreme values
            new_features['anomaly_z_score'] = z_score.clip(-5, 5).fillna(0)
    except Exception as e:
        print(f"Warning: Error calculating anomaly scores: {e}")
        new_features['anomaly_z_score'] = 0
        
    # 4. Dynamic Market Regimes - with error handling
    try:
        if 'DATE' in original_df.columns:
            short_term_cols = [f'RET_{i}' for i in range(1, 6) if f'RET_{i}' in original_df.columns]
            if short_term_cols:
                # Use more efficient aggregation
                market_ret_short = original_df.groupby('DATE')[short_term_cols].mean().mean(axis=1)
                market_ret_long = original_df.groupby('DATE')[ret_cols].mean().mean(axis=1)
                
                # Only use valid dates
                valid_dates = market_ret_short.index.intersection(market_ret_long.index)
                market_regime = pd.Series(0, index=market_ret_short.index)
                
                if not valid_dates.empty:
                    # Determine market regime (1 for bullish, 0 for bearish)
                    market_regime.loc[valid_dates] = (market_ret_short.loc[valid_dates] > 
                                                     market_ret_long.loc[valid_dates]).astype(int)
                
                # Map regime values to original dataframe
                new_features['market_regime'] = original_df['DATE'].map(market_regime).fillna(0)
    except Exception as e:
        print(f"Warning: Error calculating market regimes: {e}")
        new_features['market_regime'] = 0
    
    # 5. Network-Based Metrics with improved numerical stability and error handling
    # Initialize network centrality columns
    new_features['network_centrality_sector'] = 0.0
    new_features['network_centrality_industry'] = 0.0
    
    # Only calculate if we have less than 10,000 rows to ensure performance
    if len(original_df) < 10000:
        try:
            # Sector centrality
            if 'SECTOR' in original_df.columns and 'STOCK' in original_df.columns:
                for sector, group in original_df.groupby('SECTOR'):
                    if group.empty or pd.isna(sector):
                        continue
                    # Skip computation for very small groups
                    if len(group) < 3:
                        continue
                    
                    # Calculate sector mean returns
                    sector_mean = group[ret_cols].mean().values
                    center_M = sector_mean - sector_mean.mean()
                    norm_M = np.linalg.norm(center_M)
                    
                    # Skip if the norm is too small
                    if norm_M < EPSILON:
                        continue
                        
                    # Calculate stock-specific metrics efficiently
                    ret_matrix = group[ret_cols].values
                    row_mean = ret_matrix.mean(axis=1, keepdims=True)
                    centered_ret = ret_matrix - row_mean
                    
                    # Ensure norm is never zero
                    norm_S = np.linalg.norm(centered_ret, axis=1) + EPSILON
                    
                    # Calculate correlation
                    dotprod = np.sum(centered_ret * center_M, axis=1)
                    corr_values = dotprod / (norm_S * norm_M)
                    
                    # Clip to valid correlation range [-1, 1]
                    corr_values = np.clip(corr_values, -1, 1)
                    
                    # Update the feature with correlation values
                    new_features.loc[group.index, 'network_centrality_sector'] = corr_values
                    
            # Industry centrality - similar to sector calculation
            if 'INDUSTRY' in original_df.columns and 'STOCK' in original_df.columns:
                for industry, group in original_df.groupby('INDUSTRY'):
                    if group.empty or pd.isna(industry):
                        continue
                    if len(group) < 3:
                        continue
                        
                    industry_mean = group[ret_cols].mean().values
                    center_M = industry_mean - industry_mean.mean()
                    norm_M = np.linalg.norm(center_M)
                    
                    if norm_M < EPSILON:
                        continue
                        
                    ret_matrix = group[ret_cols].values
                    row_mean = ret_matrix.mean(axis=1, keepdims=True)
                    centered_ret = ret_matrix - row_mean
                    norm_S = np.linalg.norm(centered_ret, axis=1) + EPSILON
                    
                    dotprod = np.sum(centered_ret * center_M, axis=1)
                    corr_values = dotprod / (norm_S * norm_M)
                    corr_values = np.clip(corr_values, -1, 1)
                    
                    new_features.loc[group.index, 'network_centrality_industry'] = corr_values
                    
        except Exception as e:
            print(f"Warning: Error calculating network metrics: {e}")
            # Keep default values in case of error
    
    # Handle any NaN values in the results
    new_features = new_features.fillna(0)
    
    # Final check for extreme values
    for col in new_features.columns:
        if col in ['network_centrality_sector', 'network_centrality_industry']:
            # These should be correlations between -1 and 1
            new_features[col] = new_features[col].clip(-1, 1)
        elif col == 'hurst_exponent':
            # Hurst exponent is between 0 and 1
            new_features[col] = new_features[col].clip(0, 1)
        elif col == 'anomaly_z_score':
            # Z-scores typically don't exceed ±5
            new_features[col] = new_features[col].clip(-5, 5)
    
    return pd.DataFrame(new_features, index=original_df.index)

def mean_std_group(original_df: pd.DataFrame) -> pd.DataFrame:
    # Initialize result DataFrame with original index
    new_features = pd.DataFrame(index=original_df.index)
    
    # Check which grouping columns are available
    group_cols = ['SECTOR', 'SUB_INDUSTRY', 'INDUSTRY_GROUP', 'INDUSTRY']
    available_groups = [group for group in group_cols if group in original_df.columns]
    
    # Apply transformations for each available group and specified column
    col = 'RET_1'
    for group in available_groups:
        group_mean = original_df.groupby(['DATE', group], group_keys=False)[col].transform('mean')
        group_std = original_df.groupby(['DATE', group], group_keys=False)[col].transform('std')
        
        # Add to result DataFrame
        new_features[f'{col}_{group.lower()}_mean'] = group_mean
        new_features[f'{col}_{group.lower()}_std'] = group_std
    
    return new_features