import pandas as pd
import numpy as np

class Alpha:
    pass

    @staticmethod
    def alpha_quanta_full_base_001_rank(df, window=10):
        seq = pd.Series(np.arange(len(df)), index=df.index)
        x = seq.rolling(window).apply(lambda s: s, raw=True)
        y = df['close'].rolling(window).apply(lambda s: s, raw=True)
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = (x - x_mean) * (y - y_mean)
        var_x = (x - x_mean) ** 2
        regbeta = cov.rolling(window).sum() / (var_x.rolling(window).sum() + 1e-8)
        raw = (regbeta * seq.rolling(window).var()) ** 2 / (df['close'].rolling(window).var() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_001_tanh(df, window=65):
        seq = pd.Series(np.arange(len(df)), index=df.index)
        x_mean = seq.rolling(window).mean()
        y_mean = df['close'].rolling(window).mean()
        cov = (seq - x_mean) * (df['close'] - y_mean)
        var_x = (seq - x_mean) ** 2
        regbeta = cov.rolling(window).sum() / (var_x.rolling(window).sum() + 1e-8)
        raw = (regbeta * seq.rolling(window).var()) ** 2 / (df['close'].rolling(window).var() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_001_zscore(df, window=55):
        seq = pd.Series(np.arange(len(df)), index=df.index)
        x_mean = seq.rolling(window).mean()
        y_mean = df['close'].rolling(window).mean()
        cov = (seq - x_mean) * (df['close'] - y_mean)
        var_x = (seq - x_mean) ** 2
        regbeta = cov.rolling(window).sum() / (var_x.rolling(window).sum() + 1e-8)
        raw = (regbeta * seq.rolling(window).var()) ** 2 / (df['close'].rolling(window).var() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_001_sign(df, window=10):
        seq = pd.Series(np.arange(len(df)), index=df.index)
        x_mean = seq.rolling(window).mean()
        y_mean = df['close'].rolling(window).mean()
        cov = (seq - x_mean) * (df['close'] - y_mean)
        var_x = (seq - x_mean) ** 2
        regbeta = cov.rolling(window).sum() / (var_x.rolling(window).sum() + 1e-8)
        raw = (regbeta * seq.rolling(window).var()) ** 2 / (df['close'].rolling(window).var() + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_001_wf(df, p1=0.1, p2=50):
        window = max(int(p1 * p2), 2)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        x_mean = seq.rolling(window).mean()
        y_mean = df['close'].rolling(window).mean()
        cov = (seq - x_mean) * (df['close'] - y_mean)
        var_x = (seq - x_mean) ** 2
        regbeta = cov.rolling(window).sum() / (var_x.rolling(window).sum() + 1e-8)
        raw = (regbeta * seq.rolling(window).var()) ** 2 / (df['close'].rolling(window).var() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_002_rank(df, window=35):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_002_tanh(df, window=40):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_002_zscore(df, window=60):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_002_sign(df, window=95):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_002_wf(df, window=10, p1=0.7, p2=50):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_003_rank(df, window=70):
        ret = df.close / df.close.shift(1) - 1
        volume_log = np.log1p(df['matchingVolume'].clip(lower=0))
        volume_delta = volume_log - volume_log.shift(1)
        raw = volume_delta.rolling(window).corr(ret.abs())
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_003_tanh(df, window=10):
        ret = df.close / df.close.shift(1) - 1
        volume_log = np.log1p(df['matchingVolume'].clip(lower=0))
        volume_delta = volume_log - volume_log.shift(1)
        raw = volume_delta.rolling(window).corr(ret.abs())
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_003_zscore(df, window=90):
        ret = df.close / df.close.shift(1) - 1
        volume_log = np.log1p(df['matchingVolume'].clip(lower=0))
        volume_delta = volume_log - volume_log.shift(1)
        raw = volume_delta.rolling(window).corr(ret.abs())
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_003_sign(df, window=15):
        ret = df.close / df.close.shift(1) - 1
        volume_log = np.log1p(df['matchingVolume'].clip(lower=0))
        volume_delta = volume_log - volume_log.shift(1)
        raw = volume_delta.rolling(window).corr(ret.abs())
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_003_wf(df, window=90, p1=0.3):
        ret = df.close / df.close.shift(1) - 1
        volume_log = np.log1p(df['matchingVolume'].clip(lower=0))
        volume_delta = volume_log - volume_log.shift(1)
        raw = volume_delta.rolling(window).corr(ret.abs())
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_004_rank(df, window=75):
        close = df['close'].values
        high = df['high'].values
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).values
        log_vol = np.log1p(np.maximum(vol, 0))
        n = len(df)
        # Vectorized rolling correlation close vs log_vol
        close_mean = pd.Series(close).rolling(window, min_periods=window).mean().values
        vol_mean = pd.Series(log_vol).rolling(window, min_periods=window).mean().values
        cov = pd.Series(close).rolling(window, min_periods=window).cov(pd.Series(log_vol)).values
        var_close = pd.Series(close).rolling(window, min_periods=window).var().values
        var_vol = pd.Series(log_vol).rolling(window, min_periods=window).var().values
        denom = np.sqrt(var_close * var_vol)
        corr = np.where(denom != 0, cov / denom, 0)
        # Second term: close / max(high, window)
        max_high = pd.Series(high).rolling(window, min_periods=window).max().values
        term2 = close / (max_high + 1e-8)
        raw = corr * term2
        # Rolling rank normalization
        signal = (pd.Series(raw).rolling(window, min_periods=window).rank(pct=True).values * 2) - 1
        signal = pd.Series(signal, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_004_tanh(df, window=5):
        close = df['close'].values
        high = df['high'].values
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).values
        log_vol = np.log1p(np.maximum(vol, 0))
        n = len(df)
        # Rolling correlation
        close_mean = pd.Series(close).rolling(window, min_periods=window).mean().values
        vol_mean = pd.Series(log_vol).rolling(window, min_periods=window).mean().values
        cov = pd.Series(close).rolling(window, min_periods=window).cov(pd.Series(log_vol)).values
        var_close = pd.Series(close).rolling(window, min_periods=window).var().values
        var_vol = pd.Series(log_vol).rolling(window, min_periods=window).var().values
        denom = np.sqrt(var_close * var_vol)
        corr = np.where(denom != 0, cov / denom, 0)
        max_high = pd.Series(high).rolling(window, min_periods=window).max().values
        term2 = close / (max_high + 1e-8)
        raw = corr * term2
        # Dynamic tanh normalization
        std = pd.Series(raw).rolling(window, min_periods=window).std().values
        signal = np.tanh(raw / (std + 1e-8))
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_004_zscore(df, window=65):
        close = df['close'].values
        high = df['high'].values
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).values
        log_vol = np.log1p(np.maximum(vol, 0))
        n = len(df)
        # Rolling correlation
        close_mean = pd.Series(close).rolling(window, min_periods=window).mean().values
        vol_mean = pd.Series(log_vol).rolling(window, min_periods=window).mean().values
        cov = pd.Series(close).rolling(window, min_periods=window).cov(pd.Series(log_vol)).values
        var_close = pd.Series(close).rolling(window, min_periods=window).var().values
        var_vol = pd.Series(log_vol).rolling(window, min_periods=window).var().values
        denom = np.sqrt(var_close * var_vol)
        corr = np.where(denom != 0, cov / denom, 0)
        max_high = pd.Series(high).rolling(window, min_periods=window).max().values
        term2 = close / (max_high + 1e-8)
        raw = corr * term2
        # Z-score clip
        mean = pd.Series(raw).rolling(window, min_periods=window).mean().values
        std = pd.Series(raw).rolling(window, min_periods=window).std().values
        signal = (raw - mean) / (std + 1e-8)
        signal = np.clip(signal, -1, 1)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_004_sign(df, window=5):
        close = df['close'].values
        high = df['high'].values
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).values
        log_vol = np.log1p(np.maximum(vol, 0))
        n = len(df)
        # Rolling correlation
        close_mean = pd.Series(close).rolling(window, min_periods=window).mean().values
        vol_mean = pd.Series(log_vol).rolling(window, min_periods=window).mean().values
        cov = pd.Series(close).rolling(window, min_periods=window).cov(pd.Series(log_vol)).values
        var_close = pd.Series(close).rolling(window, min_periods=window).var().values
        var_vol = pd.Series(log_vol).rolling(window, min_periods=window).var().values
        denom = np.sqrt(var_close * var_vol)
        corr = np.where(denom != 0, cov / denom, 0)
        max_high = pd.Series(high).rolling(window, min_periods=window).max().values
        term2 = close / (max_high + 1e-8)
        raw = corr * term2
        # Sign normalization
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_004_wf(df, window=70, p1=0.3):
        close = df['close'].values
        high = df['high'].values
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).values
        log_vol = np.log1p(np.maximum(vol, 0))
        n = len(df)
        # Rolling correlation
        close_mean = pd.Series(close).rolling(window, min_periods=window).mean().values
        vol_mean = pd.Series(log_vol).rolling(window, min_periods=window).mean().values
        cov = pd.Series(close).rolling(window, min_periods=window).cov(pd.Series(log_vol)).values
        var_close = pd.Series(close).rolling(window, min_periods=window).var().values
        var_vol = pd.Series(log_vol).rolling(window, min_periods=window).var().values
        denom = np.sqrt(var_close * var_vol)
        corr = np.where(denom != 0, cov / denom, 0)
        max_high = pd.Series(high).rolling(window, min_periods=window).max().values
        term2 = close / (max_high + 1e-8)
        raw = corr * term2
        # Winsorized Fisher
        low = pd.Series(raw).rolling(window, min_periods=window).quantile(p1).values
        high_q = pd.Series(raw).rolling(window, min_periods=window).quantile(1 - p1).values
        winsorized = np.clip(raw, low, high_q)
        x = (winsorized - low) / (high_q - low + 1e-9)
        x = x * 1.98 - 0.99
        x = np.clip(x, -0.99, 0.99)
        signal = np.arctanh(x)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_005_rank(df, window=15):
        # Tính khối lượng giao dịch trong 5 phiên tăng giá (close>close.shift(1))
        up_volume = df['close'].gt(df['close'].shift(1)) * df.get('matchingVolume', df.get('volume',1))
        sum_up_volume = up_volume.rolling(window).sum()
        # Tổng khối lượng 5 phiên
        total_volume = df.get('matchingVolume', df.get('volume',1)).rolling(window).sum()
        # Tỷ lệ khối lượng tăng giá / tổng khối lượng (tránh chia 0)
        volume_ratio = sum_up_volume / (total_volume + 1e-8)
        # Giá đóng cửa / giá cao nhất 5 phiên (tránh chia 0)
        price_ratio = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        # Kết hợp thành raw alpha
        raw = volume_ratio * price_ratio
        # Chuẩn hóa Rolling Rank (A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_005_tanh(df, window=65):
        up_volume = df['close'].gt(df['close'].shift(1)) * df.get('matchingVolume', df.get('volume',1))
        sum_up_volume = up_volume.rolling(window).sum()
        total_volume = df.get('matchingVolume', df.get('volume',1)).rolling(window).sum()
        volume_ratio = sum_up_volume / (total_volume + 1e-8)
        price_ratio = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        raw = volume_ratio * price_ratio
        # Chuẩn hóa Dynamic Tanh (B)
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_005_zscore(df, window=15):
        up_volume = df['close'].gt(df['close'].shift(1)) * df.get('matchingVolume', df.get('volume',1))
        sum_up_volume = up_volume.rolling(window).sum()
        total_volume = df.get('matchingVolume', df.get('volume',1)).rolling(window).sum()
        volume_ratio = sum_up_volume / (total_volume + 1e-8)
        price_ratio = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        raw = volume_ratio * price_ratio
        # Chuẩn hóa Rolling Z-Score Clip (C)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_005_sign(df, window=30):
        up_volume = df['close'].gt(df['close'].shift(1)) * df.get('matchingVolume', df.get('volume',1))
        sum_up_volume = up_volume.rolling(window).sum()
        total_volume = df.get('matchingVolume', df.get('volume',1)).rolling(window).sum()
        volume_ratio = sum_up_volume / (total_volume + 1e-8)
        price_ratio = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        raw = volume_ratio * price_ratio
        # Chuẩn hóa Sign/Binary Soft (D)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_005_wf(df, p1=0.1, p2=10):
        window = p2
        up_volume = df['close'].gt(df['close'].shift(1)) * df.get('matchingVolume', df.get('volume',1))
        sum_up_volume = up_volume.rolling(window).sum()
        total_volume = df.get('matchingVolume', df.get('volume',1)).rolling(window).sum()
        volume_ratio = sum_up_volume / (total_volume + 1e-8)
        price_ratio = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        raw = volume_ratio * price_ratio
        # Chuẩn hóa Winsorized Fisher (E)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_006_6(df, window=15):
        ret = df['close'].pct_change(window)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        close_max_high = df['close'] / (df['high'].rolling(window).max() + 1e-8)
        raw = ret * vol_ratio * close_max_high
        normalized = pd.Series(np.tanh(raw / raw.rolling(window).std()), index=df.index)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_006_rank(df, window=25):
        pct_chg = df['close'].pct_change(periods=window)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8)
        close_ratio = df['close'] / (df['high'].rolling(window=window).max() + 1e-8)
        raw = pct_chg * vol_ratio * close_ratio
        normalized = (raw.rolling(window=window*2).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_006_tanh(df, window=10):
        pct_chg = df['close'].pct_change(periods=window)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8)
        close_ratio = df['close'] / (df['high'].rolling(window=window).max() + 1e-8)
        raw = pct_chg * vol_ratio * close_ratio
        normalized = np.tanh(raw / (raw.rolling(window=window*2).std() + 1e-8))
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_006_zscore(df, window=15):
        pct_chg = df['close'].pct_change(periods=window)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8)
        close_ratio = df['close'] / (df['high'].rolling(window=window).max() + 1e-8)
        raw = pct_chg * vol_ratio * close_ratio
        roll_mean = raw.rolling(window=window*2).mean()
        roll_std = raw.rolling(window=window*2).std()
        normalized = ((raw - roll_mean) / (roll_std + 1e-8)).clip(-1, 1)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_006_sign(df, window=20):
        pct_chg = df['close'].pct_change(periods=window)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8)
        close_ratio = df['close'] / (df['high'].rolling(window=window).max() + 1e-8)
        raw = pct_chg * vol_ratio * close_ratio
        normalized = np.sign(raw)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_006_wf(df, p1=0.1, p2=80):
        pct_chg = df['close'].pct_change(periods=5)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window=5).mean() + 1e-8)
        close_ratio = df['close'] / (df['high'].rolling(window=5).max() + 1e-8)
        raw = pct_chg * vol_ratio * close_ratio
        low = raw.rolling(window=p2).quantile(p1)
        high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_007_rank(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_close = close.rolling(window).mean()
        mean_volume = volume.rolling(window).mean()
        raw_close = close / (mean_close + 1e-8) - 1
        raw_volume = volume / (mean_volume + 1e-8) - 1
        zscore_close = (raw_close - raw_close.rolling(window).mean()) / raw_close.rolling(window).std().replace(0, np.nan)
        zscore_volume = (raw_volume - raw_volume.rolling(window).mean()) / raw_volume.rolling(window).std().replace(0, np.nan)
        raw = np.sign(zscore_close) * zscore_volume
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_007_tanh(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_close = close.rolling(window).mean()
        mean_volume = volume.rolling(window).mean()
        raw_close = close / (mean_close + 1e-8) - 1
        raw_volume = volume / (mean_volume + 1e-8) - 1
        zscore_close = (raw_close - raw_close.rolling(window).mean()) / raw_close.rolling(window).std().replace(0, np.nan)
        zscore_volume = (raw_volume - raw_volume.rolling(window).mean()) / raw_volume.rolling(window).std().replace(0, np.nan)
        raw = np.sign(zscore_close) * zscore_volume
        std = raw.rolling(window).std().replace(0, np.nan)
        result = np.tanh(raw / std)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_007_zscore(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_close = close.rolling(window).mean()
        mean_volume = volume.rolling(window).mean()
        raw_close = close / (mean_close + 1e-8) - 1
        raw_volume = volume / (mean_volume + 1e-8) - 1
        zscore_close = (raw_close - raw_close.rolling(window).mean()) / raw_close.rolling(window).std().replace(0, np.nan)
        zscore_volume = (raw_volume - raw_volume.rolling(window).mean()) / raw_volume.rolling(window).std().replace(0, np.nan)
        raw = np.sign(zscore_close) * zscore_volume
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_007_sign(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_close = close.rolling(window).mean()
        mean_volume = volume.rolling(window).mean()
        raw_close = close / (mean_close + 1e-8) - 1
        raw_volume = volume / (mean_volume + 1e-8) - 1
        zscore_close = (raw_close - raw_close.rolling(window).mean()) / raw_close.rolling(window).std().replace(0, np.nan)
        zscore_volume = (raw_volume - raw_volume.rolling(window).mean()) / raw_volume.rolling(window).std().replace(0, np.nan)
        raw = np.sign(zscore_close) * zscore_volume
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_007_wf(df, p1=0.1, p2=80):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_close = close.rolling(5).mean()
        mean_volume = volume.rolling(5).mean()
        raw_close = close / (mean_close + 1e-8) - 1
        raw_volume = volume / (mean_volume + 1e-8) - 1
        zscore_close = (raw_close - raw_close.rolling(5).mean()) / raw_close.rolling(5).std().replace(0, np.nan)
        zscore_volume = (raw_volume - raw_volume.rolling(5).mean()) / raw_volume.rolling(5).std().replace(0, np.nan)
        raw = np.sign(zscore_close) * zscore_volume
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        n = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return n.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_008_rank(df, window=85):
        candle = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        mean_close = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - mean_close)
        raw = candle * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_008_tanh(df, window=85):
        candle = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        mean_close = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - mean_close)
        raw = candle * sign
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_008_zscore(df, window=100):
        candle = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        mean_close = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - mean_close)
        raw = candle * sign
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_008_sign(df, window=25):
        candle = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        mean_close = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - mean_close)
        raw = candle * sign
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_008_wf(df, p1=0.9, p2=20):
        candle = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        mean_close = df['close'].rolling(p2).mean()
        sign = np.sign(df['close'] - mean_close)
        raw = candle * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_009_k(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close_ma = close.rolling(window, min_periods=1).mean()
        volume_ma = volume.rolling(window, min_periods=1).mean()
        sign1 = np.sign(close - close_ma)
        sign2 = np.sign(volume - volume_ma)
        sign3 = np.sign(close - open_)
        raw = sign1 * sign2 * sign3
        ts_zscore_num = (close - open_) / (close + 1e-8)
        ts_mean = ts_zscore_num.rolling(window, min_periods=1).mean()
        ts_std = ts_zscore_num.rolling(window, min_periods=1).std().replace(0, np.nan)
        ts_z = (ts_zscore_num - ts_mean) / ts_std
        final_raw = raw * ts_z
        signal = final_raw.rolling(window, min_periods=1).rank(pct=True) * 2 - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_009_h(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close_ma = close.rolling(window, min_periods=1).mean()
        volume_ma = volume.rolling(window, min_periods=1).mean()
        sign1 = np.sign(close - close_ma)
        sign2 = np.sign(volume - volume_ma)
        sign3 = np.sign(close - open_)
        raw = sign1 * sign2 * sign3
        ts_zscore_num = (close - open_) / (close + 1e-8)
        ts_mean = ts_zscore_num.rolling(window, min_periods=1).mean()
        ts_std = ts_zscore_num.rolling(window, min_periods=1).std().replace(0, np.nan)
        ts_z = (ts_zscore_num - ts_mean) / ts_std
        final_raw = raw * ts_z
        signal = np.tanh(final_raw / (final_raw.rolling(window, min_periods=1).std().replace(0, np.nan)))
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_009_e(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close_ma = close.rolling(window, min_periods=1).mean()
        volume_ma = volume.rolling(window, min_periods=1).mean()
        sign1 = np.sign(close - close_ma)
        sign2 = np.sign(volume - volume_ma)
        sign3 = np.sign(close - open_)
        raw = sign1 * sign2 * sign3
        ts_zscore_num = (close - open_) / (close + 1e-8)
        ts_mean = ts_zscore_num.rolling(window, min_periods=1).mean()
        ts_std = ts_zscore_num.rolling(window, min_periods=1).std().replace(0, np.nan)
        ts_z = (ts_zscore_num - ts_mean) / ts_std
        final_raw = raw * ts_z
        rolling_mean = final_raw.rolling(window, min_periods=1).mean()
        rolling_std = final_raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        signal = ((final_raw - rolling_mean) / rolling_std).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_009_n(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close_ma = close.rolling(window, min_periods=1).mean()
        volume_ma = volume.rolling(window, min_periods=1).mean()
        sign1 = np.sign(close - close_ma)
        sign2 = np.sign(volume - volume_ma)
        sign3 = np.sign(close - open_)
        raw = sign1 * sign2 * sign3
        ts_zscore_num = (close - open_) / (close + 1e-8)
        ts_mean = ts_zscore_num.rolling(window, min_periods=1).mean()
        ts_std = ts_zscore_num.rolling(window, min_periods=1).std().replace(0, np.nan)
        ts_z = (ts_zscore_num - ts_mean) / ts_std
        final_raw = raw * ts_z
        signal = np.sign(final_raw)
        signal = pd.Series(signal, index=df.index)
        return signal

    @staticmethod
    def alpha_quanta_full_base_009_r(df, window=10, p=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close_ma = close.rolling(window, min_periods=1).mean()
        volume_ma = volume.rolling(window, min_periods=1).mean()
        sign1 = np.sign(close - close_ma)
        sign2 = np.sign(volume - volume_ma)
        sign3 = np.sign(close - open_)
        raw = sign1 * sign2 * sign3
        ts_zscore_num = (close - open_) / (close + 1e-8)
        ts_mean = ts_zscore_num.rolling(window, min_periods=1).mean()
        ts_std = ts_zscore_num.rolling(window, min_periods=1).std().replace(0, np.nan)
        ts_z = (ts_zscore_num - ts_mean) / ts_std
        final_raw = raw * ts_z
        low = final_raw.rolling(window * 2, min_periods=1).quantile(p)
        high = final_raw.rolling(window * 2, min_periods=1).quantile(1 - p)
        winsorized = final_raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_010_rank(df, window=5):
        close = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope) * slope.abs() / (close.rolling(window).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_010_tanh(df, window=5):
        close = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope) * slope.abs() / (close.rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_010_zscore(df, window=5):
        close = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope) * slope.abs() / (close.rolling(window).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_010_sign(df, window=5):
        close = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope) * slope.abs() / (close.rolling(window).std() + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_010_wf(df, window=30, quantile_factor=0.3):
        close = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope) * slope.abs() / (close.rolling(window).std() + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(quantile_factor)
        high = raw.rolling(p2).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_011_rank(df, window=80):
        raw = df['close'] - df['open']
        spread = df['high'] - df['low']
        numerator = raw.abs().rolling(window).mean()
        denominator = spread.rolling(window).mean() + 1e-8
        alpha_raw = numerator / denominator
        norm = (alpha_raw.rolling(window).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_011_tanh(df, window=20):
        raw = df['close'] - df['open']
        spread = df['high'] - df['low']
        numerator = raw.abs().rolling(window).mean()
        denominator = spread.rolling(window).mean() + 1e-8
        alpha_raw = numerator / denominator
        norm = np.tanh(alpha_raw / alpha_raw.rolling(window).std())
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_011_zscore(df, window=75):
        raw = df['close'] - df['open']
        spread = df['high'] - df['low']
        numerator = raw.abs().rolling(window).mean()
        denominator = spread.rolling(window).mean() + 1e-8
        alpha_raw = numerator / denominator
        norm = ((alpha_raw - alpha_raw.rolling(window).mean()) / alpha_raw.rolling(window).std()).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_011_sign(df, window=70):
        raw = df['close'] - df['open']
        spread = df['high'] - df['low']
        numerator = raw.abs().rolling(window).mean()
        denominator = spread.rolling(window).mean() + 1e-8
        alpha_raw = numerator / denominator
        norm = np.sign(alpha_raw)
        return pd.Series(norm, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_011_wf(df, window=70, p1=0.3):
        raw = df['close'] - df['open']
        spread = df['high'] - df['low']
        numerator = raw.abs().rolling(window).mean()
        denominator = spread.rolling(window).mean() + 1e-8
        alpha_raw = numerator / denominator
        p2 = window
        low = alpha_raw.rolling(p2).quantile(p1)
        high = alpha_raw.rolling(p2).quantile(1 - p1)
        winsorized = alpha_raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_012_rank(df, window=10):
        low = df['close'].rolling(window=window).min()
        high = df['close'].rolling(window=window).max()
        raw = (df['close'] - low) / (high - low + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_012_tanh(df, window=5):
        low = df['close'].rolling(window=window).min()
        high = df['close'].rolling(window=window).max()
        raw = (df['close'] - low) / (high - low + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_012_zscore(df, window=10):
        low = df['close'].rolling(window=window).min()
        high = df['close'].rolling(window=window).max()
        raw = (df['close'] - low) / (high - low + 1e-8)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_) / std_).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_012_sign(df, window=5):
        low = df['close'].rolling(window=window).min()
        high = df['close'].rolling(window=window).max()
        raw = (df['close'] - low) / (high - low + 1e-8)
        signal = np.sign(raw - 0.5)
        signal = pd.Series(signal, index=df.index)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_012_wf(df, window_rank=10, p1=0.3):
        low = df['close'].rolling(window=window_rank).min()
        high = df['close'].rolling(window=window_rank).max()
        raw = (df['close'] - low) / (high - low + 1e-8)
        p2 = window_rank
        q_low = raw.rolling(window=p2).quantile(p1)
        q_high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=q_low, upper=q_high, axis=0)
        normalized = np.arctanh(((winsorized - q_low) / (q_high - q_low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_013_rank(df, window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / close.rolling(window).var().replace(0, np.nan)
        raw_regbeta = slope
        raw_regbeta_z = (raw_regbeta - raw_regbeta.rolling(window).mean()) / raw_regbeta.rolling(window).std()

        returns = close.pct_change()
        pos_count = (returns > 0).rolling(window).sum()
        neg_count = (returns < 0).rolling(window).sum()
        diff_count = pos_count - neg_count
        diff_z = (diff_count - diff_count.rolling(window).mean()) / diff_count.rolling(window).std()

        argmax = high.rolling(window).apply(np.argmax, raw=True) if window > 0 else pd.Series(0, index=df.index)
        argmin = low.rolling(window).apply(np.argmin, raw=True) if window > 0 else pd.Series(0, index=df.index)
        ts_diff = argmax - argmin
        ts_diff_z = (ts_diff - ts_diff.rolling(window).mean()) / ts_diff.rolling(window).std()

        raw_signal = (raw_regbeta_z.fillna(0) + diff_z.fillna(0) + ts_diff_z.fillna(0)) / 3.0
        normalized = (raw_signal.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_013_tanh(df, window=15):
        close = df['close']
        high = df['high']
        low = df['low']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / close.rolling(window).var().replace(0, np.nan)
        raw_regbeta = slope

        returns = close.pct_change()
        pos_count = (returns > 0).rolling(window).sum()
        neg_count = (returns < 0).rolling(window).sum()
        diff_count = pos_count - neg_count

        argmax = high.rolling(window).apply(np.argmax, raw=True) if window > 0 else pd.Series(0, index=df.index)
        argmin = low.rolling(window).apply(np.argmin, raw=True) if window > 0 else pd.Series(0, index=df.index)
        ts_diff = argmax - argmin

        raw_signal = (raw_regbeta.fillna(0) + diff_count.fillna(0) + ts_diff.fillna(0)) / 3.0
        normalized = np.tanh(raw_signal / raw_signal.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_013_zscore(df, window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / close.rolling(window).var().replace(0, np.nan)
        raw_regbeta = slope
        raw_regbeta_z = (raw_regbeta - raw_regbeta.rolling(window).mean()) / raw_regbeta.rolling(window).std()

        returns = close.pct_change()
        pos_count = (returns > 0).rolling(window).sum()
        neg_count = (returns < 0).rolling(window).sum()
        diff_count = pos_count - neg_count
        diff_z = (diff_count - diff_count.rolling(window).mean()) / diff_count.rolling(window).std()

        argmax = high.rolling(window).apply(np.argmax, raw=True) if window > 0 else pd.Series(0, index=df.index)
        argmin = low.rolling(window).apply(np.argmin, raw=True) if window > 0 else pd.Series(0, index=df.index)
        ts_diff = argmax - argmin
        ts_diff_z = (ts_diff - ts_diff.rolling(window).mean()) / ts_diff.rolling(window).std()

        raw_signal = (raw_regbeta_z.fillna(0) + diff_z.fillna(0) + ts_diff_z.fillna(0)) / 3.0
        normalized = ((raw_signal - raw_signal.rolling(window).mean()) / raw_signal.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_013_sign(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / close.rolling(window).var().replace(0, np.nan)
        raw_regbeta = slope

        returns = close.pct_change()
        pos_count = (returns > 0).rolling(window).sum()
        neg_count = (returns < 0).rolling(window).sum()
        diff_count = pos_count - neg_count

        argmax = high.rolling(window).apply(np.argmax, raw=True) if window > 0 else pd.Series(0, index=df.index)
        argmin = low.rolling(window).apply(np.argmin, raw=True) if window > 0 else pd.Series(0, index=df.index)
        ts_diff = argmax - argmin

        raw_signal = (raw_regbeta.fillna(0) + diff_count.fillna(0) + ts_diff.fillna(0)) / 3.0
        normalized = np.sign(raw_signal)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_013_wf(df, window=40, winsor_p=0.1):
        close = df['close']
        high = df['high']
        low = df['low']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / close.rolling(window).var().replace(0, np.nan)
        raw_regbeta = slope
        raw_regbeta_z = (raw_regbeta - raw_regbeta.rolling(window).mean()) / raw_regbeta.rolling(window).std()

        returns = close.pct_change()
        pos_count = (returns > 0).rolling(window).sum()
        neg_count = (returns < 0).rolling(window).sum()
        diff_count = pos_count - neg_count
        diff_z = (diff_count - diff_count.rolling(window).mean()) / diff_count.rolling(window).std()

        argmax = high.rolling(window).apply(np.argmax, raw=True) if window > 0 else pd.Series(0, index=df.index)
        argmin = low.rolling(window).apply(np.argmin, raw=True) if window > 0 else pd.Series(0, index=df.index)
        ts_diff = argmax - argmin
        ts_diff_z = (ts_diff - ts_diff.rolling(window).mean()) / ts_diff.rolling(window).std()

        raw_signal = (raw_regbeta_z.fillna(0) + diff_z.fillna(0) + ts_diff_z.fillna(0)) / 3.0
        low_bd = raw_signal.rolling(window).quantile(winsor_p)
        high_bd = raw_signal.rolling(window).quantile(1 - winsor_p)
        winsorized = raw_signal.clip(lower=low_bd, upper=high_bd, axis=0)
        normalized = np.arctanh(((winsorized - low_bd) / (high_bd - low_bd + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_014_k(df, window=55):
        # Tính returns
        returns = df['close'].pct_change()
        # Tạo biến sequence (days) 
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Tính beta (REGBETA) với window=5
        cov = days.rolling(5).cov(df['close'])
        var = df['close'].rolling(5).var().replace(0, np.nan)
        beta = cov / var
        # TS_SUM của return *5 sign
        sign_sum_ret = np.sign(returns.rolling(5).sum())
        # Nhân beta với sign
        raw_signal = beta * sign_sum_ret
        # TS_ZSCORE với window=20 và chuẩn hóa rolling rank
        raw = raw_signal.fillna(0)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_014_h(df, window=45):
        # Tính returns
        returns = df['close'].pct_change()
        # Tạo biến sequence (days) 
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Tính beta (REGBETA) với window=5
        cov = days.rolling(5).cov(df['close'])
        var = df['close'].rolling(5).var().replace(0, np.nan)
        beta = cov / var
        # TS_SUM của return *5 sign
        sign_sum_ret = np.sign(returns.rolling(5).sum())
        # Nhân beta với sign
        raw_signal = beta * sign_sum_ret
        # Dynamic Tanh chuẩn hóa
        std = raw_signal.rolling(window).std().replace(0, np.nan)
        result = np.tanh(raw_signal / std)
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_014_e(df, window=100):
        # Tính returns
        returns = df['close'].pct_change()
        # Tạo biến sequence (days) 
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Tính beta (REGBETA) với window=5
        cov = days.rolling(5).cov(df['close'])
        var = df['close'].rolling(5).var().replace(0, np.nan)
        beta = cov / var
        # TS_SUM của return *5 sign
        sign_sum_ret = np.sign(returns.rolling(5).sum())
        # Nhân beta với sign
        raw_signal = beta * sign_sum_ret
        # Rolling Z-Score/Clip chuẩn hóa
        mean = raw_signal.rolling(window).mean()
        std = raw_signal.rolling(window).std().replace(0, np.nan)
        result = ((raw_signal - mean) / std).clip(-1, 1)
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_014_y(df, window=75):
        # Tính returns
        returns = df['close'].pct_change()
        # Tạo biến sequence (days) 
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Tính beta (REGBETA) với window=5
        cov = days.rolling(window).cov(df['close'])
        var = df['close'].rolling(window).var().replace(0, np.nan)
        beta = cov / var
        # TS_SUM của return -*5 sign
        sign_sum_ret = np.sign(returns.rolling(window).sum())
        # Nhân beta với sign
        raw_signal = beta * sign_sum_ret
        # Sign/Binary Soft chuẩn hóa
        result = np.sign(raw_signal)
        return result.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_014_r(df, p2=40, p1=0.1):
        # Tính returns
        returns = df['close'].pct_change()
        # Tạo biến sequence (days) 
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Tính beta (REGBETA) với window=5
        cov = days.rolling(5).cov(df['close'])
        var = df['close'].rolling(5).var().replace(0, np.nan)
        beta = cov / var
        # TS_SUM của return *5 sign
        sign_sum_ret = np.sign(returns.rolling(5).sum())
        # Nhân beta với sign
        raw_signal = beta * sign_sum_ret
        raw = raw_signal.fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.replace([np.inf, -np.inf], 0).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_015_rank(df, window=5):
        ret = df['close'].pct_change()
        sum_ret = ret.rolling(window).sum()
        argmax_high = df['high'].rolling(window).apply(lambda x: x.values.argmax(), raw=True)
        argmin_low = df['low'].rolling(window).apply(lambda x: x.values.argmin(), raw=True)
        raw = sum_ret * (argmax_high - argmin_low)
        ranked = raw.rolling(window * 3).rank(pct=True) * 2 - 1
        signal = ranked.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_015_tanh(df, window=5):
        ret = df['close'].pct_change()
        sum_ret = ret.rolling(window).sum()
        argmax_high = df['high'].rolling(window).apply(lambda x: x.values.argmax(), raw=True)
        argmin_low = df['low'].rolling(window).apply(lambda x: x.values.argmin(), raw=True)
        raw = sum_ret * (argmax_high - argmin_low)
        denominator = raw.rolling(window * 3).std().replace(0, np.nan)
        signal = np.tanh(raw / denominator)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_015_zscore(df, window=5):
        ret = df['close'].pct_change()
        sum_ret = ret.rolling(window).sum()
        argmax_high = df['high'].rolling(window).apply(lambda x: x.values.argmax(), raw=True)
        argmin_low = df['low'].rolling(window).apply(lambda x: x.values.argmin(), raw=True)
        raw = sum_ret * (argmax_high - argmin_low)
        mean_ = raw.rolling(window * 3).mean()
        std_ = raw.rolling(window * 3).std().replace(0, np.nan)
        signal = ((raw - mean_) / std_).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_015_sign(df, window=5):
        ret = df['close'].pct_change()
        sum_ret = ret.rolling(window).sum()
        argmax_high = df['high'].rolling(window).apply(lambda x: x.values.argmax(), raw=True)
        argmin_low = df['low'].rolling(window).apply(lambda x: x.values.argmin(), raw=True)
        raw = sum_ret * (argmax_high - argmin_low)
        signal = np.sign(raw).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_015_wf(df, window=5, p1=0.05):
        ret = df['close'].pct_change()
        sum_ret = ret.rolling(window).sum()
        argmax_high = df['high'].rolling(window).apply(lambda x: x.values.argmax(), raw=True)
        argmin_low = df['low'].rolling(window).apply(lambda x: x.values.argmin(), raw=True)
        raw = sum_ret * (argmax_high - argmin_low)
        p2 = window * 3
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        num = (winsorized - low) / (high - low + 1e-9)
        num_clip = num.clip(0.01, 0.99)
        signal = np.arctanh(num_clip * 1.98 - 0.99)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_016_rank(df, window=5):
        close = df['close']
        low = df['low']
        open_ = df['open']
        slope = (close.rolling(window).cov(pd.Series(np.arange(window, dtype=float), index=close.index)) / close.rolling(window).var().replace(0, np.nan)).ffill().fillna(0)
        sign = np.sign((low - open_) / (open_ + 1e-8))
        raw = slope * sign
        std = close.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        raw = raw / (std + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_016_tanh(df, window=5):
        close = df['close']
        low = df['low']
        open_ = df['open']
        slope = (close.rolling(window).cov(pd.Series(np.arange(window, dtype=float), index=close.index)) / close.rolling(window).var().replace(0, np.nan)).ffill().fillna(0)
        sign = np.sign((low - open_) / (open_ + 1e-8))
        raw = slope * sign
        std = close.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        raw = raw / (std + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_016_zscore(df, window=5):
        close = df['close']
        low = df['low']
        open_ = df['open']
        slope = (close.rolling(window).cov(pd.Series(np.arange(window, dtype=float), index=close.index)) / close.rolling(window).var().replace(0, np.nan)).ffill().fillna(0)
        sign = np.sign((low - open_) / (open_ + 1e-8))
        raw = slope * sign
        std = close.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        raw = raw / (std + 1e-8)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_016_sign(df, window=5):
        close = df['close']
        low = df['low']
        open_ = df['open']
        slope = (close.rolling(window).cov(pd.Series(np.arange(window, dtype=float), index=close.index)) / close.rolling(window).var().replace(0, np.nan)).ffill().fillna(0)
        sign = np.sign((low - open_) / (open_ + 1e-8))
        raw = slope * sign
        std = close.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        raw = raw / (std + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_016_wf(df, window=5):
        close = df['close']
        low = df['low']
        open_ = df['open']
        slope = (close.rolling(window).cov(pd.Series(np.arange(window, dtype=float), index=close.index)) / close.rolling(window).var().replace(0, np.nan)).ffill().fillna(0)
        sign = np.sign((low - open_) / (open_ + 1e-8))
        raw = slope * sign
        std = close.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        raw = raw / (std + 1e-8)
        p1 = 0.05
        p2 = window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_017_rank(df, window=5):
        open_ = df['open']
        low = df['low']
        close = df['close']
        raw = ((low - open_) / (open_ + 1e-8))
        seq = pd.Series(np.arange(len(df)), index=df.index)
        cov = seq.rolling(5).cov(close)
        var = close.rolling(5).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        std = close.rolling(5).std().replace(0, np.nan)
        raw_final = result / (std + 1e-8)
        normalized = (raw_final.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_017_tanh(df, window=60):
        open_ = df['open']
        low = df['low']
        close = df['close']
        raw = ((low - open_) / (open_ + 1e-8))
        seq = pd.Series(np.arange(len(df)), index=df.index)
        cov = seq.rolling(5).cov(close)
        var = close.rolling(5).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        std = close.rolling(5).std().replace(0, np.nan)
        raw_final = result / (std + 1e-8)
        normalized = np.tanh(raw_final / raw_final.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_017_zscore(df, window=10):
        open_ = df['open']
        low = df['low']
        close = df['close']
        raw = ((low - open_) / (open_ + 1e-8))
        seq = pd.Series(np.arange(len(df)), index=df.index)
        cov = seq.rolling(5).cov(close)
        var = close.rolling(5).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        std = close.rolling(5).std().replace(0, np.nan)
        raw_final = result / (std + 1e-8)
        mean_ = raw_final.rolling(window).mean()
        std_ = raw_final.rolling(window).std().replace(0, np.nan)
        normalized = ((raw_final - mean_) / std_).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_017_sign(df):
        open_ = df['open']
        low = df['low']
        close = df['close']
        raw = ((low - open_) / (open_ + 1e-8))
        seq = pd.Series(np.arange(len(df)), index=df.index)
        cov = seq.rolling(5).cov(close)
        var = close.rolling(5).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        std = close.rolling(5).std().replace(0, np.nan)
        raw_final = result / (std + 1e-8)
        normalized = np.sign(raw_final)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_017_wf(df, window=30, quantile_factor=0.3):
        open_ = df['open']
        low = df['low']
        close = df['close']
        raw = ((low - open_) / (open_ + 1e-8))
        seq = pd.Series(np.arange(len(df)), index=df.index)
        cov = seq.rolling(5).cov(close)
        var = close.rolling(5).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        std = close.rolling(5).std().replace(0, np.nan)
        raw_final = result / (std + 1e-8)
        p2 = max(window, 5)
        low_q = raw_final.rolling(p2).quantile(quantile_factor)
        high_q = raw_final.rolling(p2).quantile(1 - quantile_factor)
        winsorized = raw_final.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_018_rank(df, window=100):
        # REGRESSION slope of close over time using covariance method
        y = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = y.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        # absolute slope
        abs_slope = slope.abs()
        # term2: (low - open) / (open + 1e-8)
        term2 = (df['low'] - df['open']) / (df['open'] + 1e-8)
        # raw = abs_slope * term2 / (TS_STD(close,5) + 1e-8)
        std_close = df['close'].rolling(window).std()
        raw = abs_slope * term2 / (std_close + 1e-8)
        # Rolling rank normalization: Rolling Rank (case A)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_018_tanh(df, window=100):
        y = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = y.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        abs_slope = slope.abs()
        term2 = (df['low'] - df['open']) / (df['open'] + 1e-8)
        std_close = df['close'].rolling(window).std()
        raw = abs_slope * term2 / (std_close + 1e-8)
        # Dynamic Tanh normalization (case B)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_018_zscore(df, window=75):
        y = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = y.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        abs_slope = slope.abs()
        term2 = (df['low'] - df['open']) / (df['open'] + 1e-8)
        std_close = df['close'].rolling(window).std()
        raw = abs_slope * term2 / (std_close + 1e-8)
        # Rolling Z-Score normalization (case C)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_018_sign(df, window=35):
        y = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = y.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        abs_slope = slope.abs()
        term2 = (df['low'] - df['open']) / (df['open'] + 1e-8)
        std_close = df['close'].rolling(window).std()
        raw = abs_slope * term2 / (std_close + 1e-8)
        # Sign/Binary Soft normalization (case D)
        result = np.sign(raw)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_018_wf(df, window=100):
        y = df['close']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = y.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        abs_slope = slope.abs()
        term2 = (df['low'] - df['open']) / (df['open'] + 1e-8)
        std_close = df['close'].rolling(window).std()
        raw = abs_slope * term2 / (std_close + 1e-8)
        # Winsorized Fisher normalization (case E)
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0).clip(-1, 1)
        return result

    @staticmethod
    def alpha_quanta_full_base_019_rank(df, window=70):
        sub_window = 10
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = close
        cov = y.rolling(window).cov(x)
        var_ = x.rolling(window).var().replace(0, np.nan)
        beta = cov / var_
        sq_beta = np.square(beta)
        corr = ret.rolling(sub_window).corr(volume.rolling(sub_window).mean() if sub_window else volume)
        corr = np.where(corr.isna(), 0, corr)
        corr = pd.Series(corr, index=df.index)
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        avg_vol = volume.rolling(window).mean()
        raw = sq_beta * corr * std_ret / (avg_vol + 1e-8)
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_019_tanh(df, window=15):
        sub_window = 10
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = close
        cov = y.rolling(window).cov(x)
        var_ = x.rolling(window).var().replace(0, np.nan)
        beta = cov / var_
        sq_beta = np.square(beta)
        corr = ret.rolling(sub_window).corr(volume.rolling(sub_window).mean() if sub_window else volume)
        corr = np.where(corr.isna(), 0, corr)
        corr = pd.Series(corr, index=df.index)
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        avg_vol = volume.rolling(window).mean()
        raw = sq_beta * corr * std_ret / (avg_vol + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_019_zscore(df, window=80):
        sub_window = 10
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = close
        cov = y.rolling(window).cov(x)
        var_ = x.rolling(window).var().replace(0, np.nan)
        beta = cov / var_
        sq_beta = np.square(beta)
        corr = ret.rolling(sub_window).corr(volume.rolling(sub_window).mean() if sub_window else volume)
        corr = np.where(corr.isna(), 0, corr)
        corr = pd.Series(corr, index=df.index)
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        avg_vol = volume.rolling(window).mean()
        raw = sq_beta * corr * std_ret / (avg_vol + 1e-8)
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_019_sign(df, window=5):
        sub_window = 10
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = close
        cov = y.rolling(window).cov(x)
        var_ = x.rolling(window).var().replace(0, np.nan)
        beta = cov / var_
        sq_beta = np.square(beta)
        corr = ret.rolling(sub_window).corr(volume.rolling(sub_window).mean() if sub_window else volume)
        corr = np.where(corr.isna(), 0, corr)
        corr = pd.Series(corr, index=df.index)
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        avg_vol = volume.rolling(window).mean()
        raw = sq_beta * corr * std_ret / (avg_vol + 1e-8)
        sign = np.sign(raw)
        signal = pd.Series(sign, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_019_wf(df, window=30, p1=0.9):
        sub_window = 10
        p2 = window
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = close
        cov = y.rolling(window).cov(x)
        var_ = x.rolling(window).var().replace(0, np.nan)
        beta = cov / var_
        sq_beta = np.square(beta)
        corr = ret.rolling(sub_window).corr(volume.rolling(sub_window).mean() if sub_window else volume)
        corr = np.where(corr.isna(), 0, corr)
        corr = pd.Series(corr, index=df.index)
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        avg_vol = volume.rolling(window).mean()
        raw = sq_beta * corr * std_ret / (avg_vol + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_020_rank(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct = close.pct_change(window)
        sign = np.sign(pct)
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        beta_sq = slope ** 2
        raw = sign * vol_ratio * beta_sq
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_020_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct = close.pct_change(window)
        sign = np.sign(pct)
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        beta_sq = slope ** 2
        raw = sign * vol_ratio * beta_sq
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_020_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct = close.pct_change(window)
        sign = np.sign(pct)
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        beta_sq = slope ** 2
        raw = sign * vol_ratio * beta_sq
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_020_sign(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        pct = close.pct_change(window)
        sign = np.sign(pct)
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        beta_sq = slope ** 2
        raw = sign * vol_ratio * beta_sq
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_020_wf(df, window=10, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        pct = close.pct_change(window)
        sign = np.sign(pct)
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        beta_sq = slope ** 2
        raw = sign * vol_ratio * beta_sq
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_021_rank(df, window=15):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol)
        pct = df['close'].pct_change(window)
        std = ret.rolling(window * 6).std() + 1e-8
        raw = corr * pct / std
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_021_tanh(df, window=100):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol)
        pct = df['close'].pct_change(window)
        std = ret.rolling(window * 6).std() + 1e-8
        raw = corr * pct / std
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_021_zscore(df, window=40):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol)
        pct = df['close'].pct_change(window)
        std = ret.rolling(window * 6).std() + 1e-8
        raw = corr * pct / std
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_021_sign(df, window=80):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol)
        pct = df['close'].pct_change(window)
        std = ret.rolling(window * 6).std() + 1e-8
        raw = corr * pct / std
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_021_wf(df, window=20, p1=0.9):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol)
        pct = df['close'].pct_change(window)
        std = ret.rolling(window * 6).std() + 1e-8
        raw = corr * pct / std
        p2 = window * 5
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_022_rank(df, window=85):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        n = len(df)
        days = pd.Series(np.arange(n), index=df.index)
        # REGRESSI: slope = cov(days, close) / var(days)
        cov = days.rolling(window).cov(close)
        var_days = days.rolling(window).var().replace(0, np.nan)
        slope = cov / var_days
        # TS_VAR of slope
        var_slope = slope.rolling(window).var()
        # TS_VAR of close
        var_close = close.rolling(window).var()
        # RANK 1
        raw1 = 1 - var_slope / (var_close + 1e-8)
        rank1 = raw1.rolling(window).rank(pct=True)
        # Part 2
        vol_delay = volume.shift(1)
        sum_vol5 = volume.rolling(5).sum()
        sum_vol_delay5 = vol_delay.rolling(5).sum()
        raw2 = sum_vol5 / (sum_vol_delay5 + 1e-8)
        rank2 = raw2.rolling(window).rank(pct=True)
        # Part 3
        min_low5 = low.rolling(5).min()
        max_high5 = high.rolling(5).max()
        raw3 = (close - min_low5) / (max_high5 - min_low5 + 1e-8)
        rank3 = raw3.rolling(window).rank(pct=True)
        # Combine
        raw = rank1 / 3 + rank2 / 3 + rank3 / 3
        raw = raw.ffill().fillna(0)
        # A: Rolling Rank
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result

    @staticmethod
    def alpha_quanta_full_base_022_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        n = len(df)
        days = pd.Series(np.arange(n), index=df.index)
        cov = days.rolling(window).cov(close)
        var_days = days.rolling(window).var().replace(0, np.nan)
        slope = cov / var_days
        var_slope = slope.rolling(window).var()
        var_close = close.rolling(window).var()
        raw1 = 1 - var_slope / (var_close + 1e-8)
        rank1 = raw1.rolling(window).rank(pct=True)
        vol_delay = volume.shift(1)
        sum_vol5 = volume.rolling(5).sum()
        sum_vol_delay5 = vol_delay.rolling(5).sum()
        raw2 = sum_vol5 / (sum_vol_delay5 + 1e-8)
        rank2 = raw2.rolling(window).rank(pct=True)
        min_low5 = low.rolling(5).min()
        max_high5 = high.rolling(5).max()
        raw3 = (close - min_low5) / (max_high5 - min_low5 + 1e-8)
        rank3 = raw3.rolling(window).rank(pct=True)
        raw = rank1 / 3 + rank2 / 3 + rank3 / 3
        raw = raw.ffill().fillna(0)
        # B: Dynamic Tanh
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result

    @staticmethod
    def alpha_quanta_full_base_022_zscore(df, window=60):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        n = len(df)
        days = pd.Series(np.arange(n), index=df.index)
        cov = days.rolling(window).cov(close)
        var_days = days.rolling(window).var().replace(0, np.nan)
        slope = cov / var_days
        var_slope = slope.rolling(window).var()
        var_close = close.rolling(window).var()
        raw1 = 1 - var_slope / (var_close + 1e-8)
        rank1 = raw1.rolling(window).rank(pct=True)
        vol_delay = volume.shift(1)
        sum_vol5 = volume.rolling(5).sum()
        sum_vol_delay5 = vol_delay.rolling(5).sum()
        raw2 = sum_vol5 / (sum_vol_delay5 + 1e-8)
        rank2 = raw2.rolling(window).rank(pct=True)
        min_low5 = low.rolling(5).min()
        max_high5 = high.rolling(5).max()
        raw3 = (close - min_low5) / (max_high5 - min_low5 + 1e-8)
        rank3 = raw3.rolling(window).rank(pct=True)
        raw = rank1 / 3 + rank2 / 3 + rank3 / 3
        raw = raw.ffill().fillna(0)
        # C: Rolling Z-Score Clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return result

    @staticmethod
    def alpha_quanta_full_base_022_sign(df, window=60):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        n = len(df)
        days = pd.Series(np.arange(n), index=df.index)
        cov = days.rolling(window).cov(close)
        var_days = days.rolling(window).var().replace(0, np.nan)
        slope = cov / var_days
        var_slope = slope.rolling(window).var()
        var_close = close.rolling(window).var()
        raw1 = 1 - var_slope / (var_close + 1e-8)
        rank1 = raw1.rolling(window).rank(pct=True)
        vol_delay = volume.shift(1)
        sum_vol5 = volume.rolling(5).sum()
        sum_vol_delay5 = vol_delay.rolling(5).sum()
        raw2 = sum_vol5 / (sum_vol_delay5 + 1e-8)
        rank2 = raw2.rolling(window).rank(pct=True)
        min_low5 = low.rolling(5).min()
        max_high5 = high.rolling(5).max()
        raw3 = (close - min_low5) / (max_high5 - min_low5 + 1e-8)
        rank3 = raw3.rolling(window).rank(pct=True)
        raw = rank1 / 3 + rank2 / 3 + rank3 / 3
        raw = raw.ffill().fillna(0)
        # D: Sign Binary Soft
        result = np.sign(raw)
        return result

    @staticmethod
    def alpha_quanta_full_base_022_wf(df, window=65):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        n = len(df)
        days = pd.Series(np.arange(n), index=df.index)
        cov = days.rolling(window).cov(close)
        var_days = days.rolling(window).var().replace(0, np.nan)
        slope = cov / var_days
        var_slope = slope.rolling(window).var()
        var_close = close.rolling(window).var()
        raw1 = 1 - var_slope / (var_close + 1e-8)
        rank1 = raw1.rolling(window).rank(pct=True)
        vol_delay = volume.shift(1)
        sum_vol5 = volume.rolling(5).sum()
        sum_vol_delay5 = vol_delay.rolling(5).sum()
        raw2 = sum_vol5 / (sum_vol_delay5 + 1e-8)
        rank2 = raw2.rolling(window).rank(pct=True)
        min_low5 = low.rolling(5).min()
        max_high5 = high.rolling(5).max()
        raw3 = (close - min_low5) / (max_high5 - min_low5 + 1e-8)
        rank3 = raw3.rolling(window).rank(pct=True)
        raw = rank1 / 3 + rank2 / 3 + rank3 / 3
        raw = raw.ffill().fillna(0)
        # E: Winsorized Fisher
        p1 = 0.05
        p2 = window
        low_w = raw.rolling(p2).quantile(p1)
        high_w = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_w, upper=high_w, axis=0)
        result = np.arctanh(((winsorized - low_w) / (high_w - low_w + 1e-9)) * 1.98 - 0.99)
        return result

    @staticmethod
    def alpha_quanta_full_base_023_rank(df, window=20, sub_window=20):
        close = df['close'].values
        volume = df['matchingVolume'].values
        n = len(df)
        days = np.arange(n)
        weight = lambda w: np.concatenate([np.linspace(1, w, w) / np.sum(np.linspace(1, w, w)), np.zeros(n - w)])
        def alpha_quanta_full_base_023_rank(close, window):
            x = np.arange(window)
            x_mean = (window - 1) / 2
            x_var = np.sum((x - x_mean) ** 2)
            slope = np.full(n, np.nan)
            for i in range(window - 1, n):
                y = close[i - window + 1:i + 1]
                y_mean = np.mean(y)
                cov = np.sum((x - x_mean) * (y - y_mean))
                slope[i] = cov / x_var
            return slope
        slope = rolling_regression_slope(close, sub_window)
        slope_series = pd.Series(slope, index=df.index)
        var_slope = slope_series.rolling(window).var()
        var_close = pd.Series(close, index=df.index).rolling(window).var() + 1e-8
        ratio = 1 - var_slope / var_close
        rank_ratio = ratio.rolling(window).rank(pct=True) * 2 - 1
        vol = pd.Series(volume, index=df.index)
        vol_ratio = vol.rolling(window=5).sum() / (vol.shift(1).rolling(window=5).sum() + 1e-8)
        rank_vol = vol_ratio.rolling(window).rank(pct=True) * 2 - 1
        signal = rank_ratio * rank_vol
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_023_tanh(df, window=20, sub_window=20):
        close = df['close'].values
        volume = df['matchingVolume'].values
        n = len(df)
        days = np.arange(n)
        weight = lambda w: np.concatenate([np.linspace(1, w, w) / np.sum(np.linspace(1, w, w)), np.zeros(n - w)])
        def alpha_quanta_full_base_023_tanh(close, window):
            x = np.arange(window)
            x_mean = (window - 1) / 2
            x_var = np.sum((x - x_mean) ** 2)
            slope = np.full(n, np.nan)
            for i in range(window - 1, n):
                y = close[i - window + 1:i + 1]
                y_mean = np.mean(y)
                cov = np.sum((x - x_mean) * (y - y_mean))
                slope[i] = cov / x_var
            return slope
        slope = rolling_regression_slope(close, sub_window)
        slope_series = pd.Series(slope, index=df.index)
        var_slope = slope_series.rolling(window).var()
        var_close = pd.Series(close, index=df.index).rolling(window).var() + 1e-8
        ratio = 1 - var_slope / var_close
        raw_signal = ratio
        tanh_signal = np.tanh(raw_signal / raw_signal.rolling(window).std())
        vol = pd.Series(volume, index=df.index)
        vol_ratio = vol.rolling(window=5).sum() / (vol.shift(1).rolling(window=5).sum() + 1e-8)
        raw_vol = vol_ratio
        tanh_vol = np.tanh(raw_vol / raw_vol.rolling(window).std())
        signal = tanh_signal * tanh_vol
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_023_zscore(df, window=20, sub_window=20):
        close = df['close'].values
        volume = df['matchingVolume'].values
        n = len(df)
        days = np.arange(n)
        weight = lambda w: np.concatenate([np.linspace(1, w, w) / np.sum(np.linspace(1, w, w)), np.zeros(n - w)])
        def alpha_quanta_full_base_023_zscore(close, window):
            x = np.arange(window)
            x_mean = (window - 1) / 2
            x_var = np.sum((x - x_mean) ** 2)
            slope = np.full(n, np.nan)
            for i in range(window - 1, n):
                y = close[i - window + 1:i + 1]
                y_mean = np.mean(y)
                cov = np.sum((x - x_mean) * (y - y_mean))
                slope[i] = cov / x_var
            return slope
        slope = rolling_regression_slope(close, sub_window)
        slope_series = pd.Series(slope, index=df.index)
        var_slope = slope_series.rolling(window).var()
        var_close = pd.Series(close, index=df.index).rolling(window).var() + 1e-8
        ratio = 1 - var_slope / var_close
        zscore_ratio = ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()).clip(-1, 1)
        vol = pd.Series(volume, index=df.index)
        vol_ratio = vol.rolling(window=5).sum() / (vol.shift(1).rolling(window=5).sum() + 1e-8)
        zscore_vol = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std()).clip(-1, 1)
        signal = zscore_ratio * zscore_vol
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_023_sign(df, window=20, sub_window=20):
        close = df['close'].values
        volume = df['matchingVolume'].values
        n = len(df)
        days = np.arange(n)
        weight = lambda w: np.concatenate([np.linspace(1, w, w) / np.sum(np.linspace(1, w, w)), np.zeros(n - w)])
        def alpha_quanta_full_base_023_sign(close, window):
            x = np.arange(window)
            x_mean = (window - 1) / 2
            x_var = np.sum((x - x_mean) ** 2)
            slope = np.full(n, np.nan)
            for i in range(window - 1, n):
                y = close[i - window + 1:i + 1]
                y_mean = np.mean(y)
                cov = np.sum((x - x_mean) * (y - y_mean))
                slope[i] = cov / x_var
            return slope
        slope = rolling_regression_slope(close, sub_window)
        slope_series = pd.Series(slope, index=df.index)
        var_slope = slope_series.rolling(window).var()
        var_close = pd.Series(close, index=df.index).rolling(window).var() + 1e-8
        ratio = 1 - var_slope / var_close
        vol = pd.Series(volume, index=df.index)
        vol_ratio = vol.rolling(window=5).sum() / (vol.shift(1).rolling(window=5).sum() + 1e-8)
        sign_ratio = np.sign(ratio)
        sign_vol = np.sign(vol_ratio)
        signal = pd.Series(sign_ratio * sign_vol, index=df.index)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_023_wf(df, window=20, sub_window=20, p1=0.1, p2=20):
        close = df['close'].values
        volume = df['matchingVolume'].values
        n = len(df)
        days = np.arange(n)
        weight = lambda w: np.concatenate([np.linspace(1, w, w) / np.sum(np.linspace(1, w, w)), np.zeros(n - w)])
        def alpha_quanta_full_base_023_wf(close, window):
            x = np.arange(window)
            x_mean = (window - 1) / 2
            x_var = np.sum((x - x_mean) ** 2)
            slope = np.full(n, np.nan)
            for i in range(window - 1, n):
                y = close[i - window + 1:i + 1]
                y_mean = np.mean(y)
                cov = np.sum((x - x_mean) * (y - y_mean))
                slope[i] = cov / x_var
            return slope
        slope = rolling_regression_slope(close, sub_window)
        slope_series = pd.Series(slope, index=df.index)
        var_slope = slope_series.rolling(window).var()
        var_close = pd.Series(close, index=df.index).rolling(window).var() + 1e-8
        ratio = 1 - var_slope / var_close
        vol = pd.Series(volume, index=df.index)
        vol_ratio = vol.rolling(window=5).sum() / (vol.shift(1).rolling(window=5).sum() + 1e-8)
        raw = ratio * vol_ratio
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_024_rank(df, window=10, sub_window=20):
        close = df['close']
        low = df['low']
        high = df['high']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / days.rolling(window).var().replace(0, np.nan)
        regresi = slope * days + (close - slope * days).rolling(window).mean()
        var_resid = ((close - regresi) ** 2).rolling(window).var().replace(0, np.nan)
        var_close = close.rolling(window).var().replace(0, np.nan)
        ratio = var_resid / (var_close + 1e-8)
        raw_1 = 1 - ratio
        rank_1 = raw_1.rolling(window).rank(pct=True)
        norm_1 = rank_1 * 2 - 1
        min_low = low.rolling(sub_window).min()
        max_high = high.rolling(sub_window).max()
        raw_2 = (close - min_low) / (max_high - min_low + 1e-8)
        rank_2 = raw_2.rolling(window).rank(pct=True)
        norm_2 = rank_2 * 2 - 1
        raw_signal = norm_1 * norm_2
        signal = raw_signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_024_tanh(df, window=20, sub_window=5):
        close = df['close']
        low = df['low']
        high = df['high']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / days.rolling(window).var().replace(0, np.nan)
        regresi = slope * days + (close - slope * days).rolling(window).mean()
        var_resid = ((close - regresi) ** 2).rolling(window).var().replace(0, np.nan)
        var_close = close.rolling(window).var().replace(0, np.nan)
        ratio = var_resid / (var_close + 1e-8)
        raw_1 = 1 - ratio
        norm_1 = np.tanh(raw_1 / raw_1.rolling(window).std().replace(0, np.nan))
        min_low = low.rolling(sub_window).min()
        max_high = high.rolling(sub_window).max()
        raw_2 = (close - min_low) / (max_high - min_low + 1e-8)
        norm_2 = np.tanh(raw_2 / raw_2.rolling(window).std().replace(0, np.nan))
        raw_signal = norm_1 * norm_2
        signal = raw_signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_024_zscore(df, window=10, sub_window=40):
        close = df['close']
        low = df['low']
        high = df['high']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / days.rolling(window).var().replace(0, np.nan)
        regresi = slope * days + (close - slope * days).rolling(window).mean()
        var_resid = ((close - regresi) ** 2).rolling(window).var().replace(0, np.nan)
        var_close = close.rolling(window).var().replace(0, np.nan)
        ratio = var_resid / (var_close + 1e-8)
        raw_1 = 1 - ratio
        norm_1 = ((raw_1 - raw_1.rolling(window).mean()) / raw_1.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        min_low = low.rolling(sub_window).min()
        max_high = high.rolling(sub_window).max()
        raw_2 = (close - min_low) / (max_high - min_low + 1e-8)
        norm_2 = ((raw_2 - raw_2.rolling(window).mean()) / raw_2.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        raw_signal = norm_1 * norm_2
        signal = raw_signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_024_sign(df, window=30, sub_window=3):
        close = df['close']
        low = df['low']
        high = df['high']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / days.rolling(window).var().replace(0, np.nan)
        regresi = slope * days + (close - slope * days).rolling(window).mean()
        var_resid = ((close - regresi) ** 2).rolling(window).var().replace(0, np.nan)
        var_close = close.rolling(window).var().replace(0, np.nan)
        ratio = var_resid / (var_close + 1e-8)
        raw_1 = 1 - ratio
        norm_1 = np.sign(raw_1)
        min_low = low.rolling(sub_window).min()
        max_high = high.rolling(sub_window).max()
        raw_2 = (close - min_low) / (max_high - min_low + 1e-8)
        norm_2 = np.sign(raw_2 - 0.5)
        raw_signal = norm_1 * norm_2
        signal = raw_signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_024_wf(df, window=10, sub_window=3, p1=0.05, p2=20):
        close = df['close']
        low = df['low']
        high = df['high']
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = days.rolling(window).cov(close) / days.rolling(window).var().replace(0, np.nan)
        regresi = slope * days + (close - slope * days).rolling(window).mean()
        var_resid = ((close - regresi) ** 2).rolling(window).var().replace(0, np.nan)
        var_close = close.rolling(window).var().replace(0, np.nan)
        ratio = var_resid / (var_close + 1e-8)
        raw_1 = 1 - ratio
        low_1 = raw_1.rolling(p2).quantile(p1)
        high_1 = raw_1.rolling(p2).quantile(1 - p1)
        winsorized_1 = raw_1.clip(lower=low_1, upper=high_1, axis=0)
        norm_1 = np.arctanh(((winsorized_1 - low_1) / (high_1 - low_1 + 1e-9)) * 1.98 - 0.99)
        min_low = low.rolling(sub_window).min()
        max_high = high.rolling(sub_window).max()
        raw_2 = (close - min_low) / (max_high - min_low + 1e-8)
        low_2 = raw_2.rolling(p2).quantile(p1)
        high_2 = raw_2.rolling(p2).quantile(1 - p1)
        winsorized_2 = raw_2.clip(lower=low_2, upper=high_2, axis=0)
        norm_2 = np.arctanh(((winsorized_2 - low_2) / (high_2 - low_2 + 1e-9)) * 1.98 - 0.99)
        raw_signal = norm_1 * norm_2
        signal = raw_signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_025_rank(df, window=75):
        close = df['close']
        volume = df['matchingVolume']
        vol_log = np.log1p(volume)
        close_rank = close.rolling(5).rank(pct=True)
        corr = close.rolling(window).corr(vol_log)
        raw = corr * close_rank
        raw = raw / (volume.rolling(10).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_025_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        vol_log = np.log1p(volume)
        close_rank = close.rolling(5).rank(pct=True)
        corr = close.rolling(window).corr(vol_log)
        raw = corr * close_rank
        raw = raw / (volume.rolling(10).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_025_zscore(df, window=75):
        close = df['close']
        volume = df['matchingVolume']
        vol_log = np.log1p(volume)
        close_rank = close.rolling(5).rank(pct=True)
        corr = close.rolling(window).corr(vol_log)
        raw = corr * close_rank
        raw = raw / (volume.rolling(10).std() + 1e-8)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_025_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        vol_log = np.log1p(volume)
        close_rank = close.rolling(5).rank(pct=True)
        corr = close.rolling(window).corr(vol_log)
        raw = corr * close_rank
        raw = raw / (volume.rolling(10).std() + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_025_wf(df, window=80, sub_window=40):
        close = df['close']
        volume = df['matchingVolume']
        vol_log = np.log1p(volume)
        close_rank = close.rolling(5).rank(pct=True)
        corr = close.rolling(window).corr(vol_log)
        raw = corr * close_rank
        raw = raw / (volume.rolling(10).std() + 1e-8)
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_026_rank(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        raw = abs(close.rolling(window).corr(volume)) * close.rolling(5).rank(pct=True) * 2 - 1
        return raw

    @staticmethod
    def alpha_quanta_full_base_026_tanh(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        corr = abs(close.rolling(window).corr(volume))
        rank_close = close.rolling(5).rank(pct=True) * 2 - 1
        raw = corr * rank_close
        return np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(0))

    @staticmethod
    def alpha_quanta_full_base_026_zscore(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        corr = abs(close.rolling(window).corr(volume))
        rank_close = close.rolling(5).rank(pct=True) * 2 - 1
        raw = corr * rank_close
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(0)
        return ((raw - mean) / (std + 1e-8)).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_026_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        raw = abs(close.rolling(window).corr(volume)) * close.rolling(5).rank(pct=True) * 2 - 1
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_full_base_026_wf(df, window=100, sub_window=10):
        close = df['close']
        volume = df['matchingVolume']
        corr = abs(close.rolling(window).corr(volume))
        rank_close = close.rolling(5).rank(pct=True) * 2 - 1
        raw = corr * rank_close
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_027_rank(df, window=70, sub_window=10):
        vol_window = 10
        close = df['close']
        volume = df['matchingVolume'].fillna(0).replace(0, np.nan)
        volume = volume.fillna(method='ffill')
        corr_60 = close.rolling(window).corr(volume)
        sign_corr = np.sign(corr_60)
        rank_5 = close.rolling(sub_window).rank(pct=True)
        std_vol_10 = volume.rolling(vol_window).std().replace(0, np.nan).fillna(method='ffill')
        raw = sign_corr * rank_5 / (std_vol_10 + 1e-8)
        norm_window = max(window, vol_window, sub_window)
        signal = (raw.rolling(norm_window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_027_tanh(df, window=10, sub_window=40):
        vol_window = 10
        close = df['close']
        volume = df['matchingVolume'].fillna(0).replace(0, np.nan)
        volume = volume.fillna(method='ffill')
        corr_60 = close.rolling(window).corr(volume)
        sign_corr = np.sign(corr_60)
        rank_5 = close.rolling(sub_window).rank(pct=True)
        std_vol_10 = volume.rolling(vol_window).std().replace(0, np.nan).fillna(method='ffill')
        raw = sign_corr * rank_5 / (std_vol_10 + 1e-8)
        norm_window = max(window, vol_window, sub_window)
        signal = np.tanh(raw / raw.rolling(norm_window).std().replace(0, np.nan).fillna(method='ffill'))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_027_zscore(df, window=70, sub_window=5):
        vol_window = 10
        close = df['close']
        volume = df['matchingVolume'].fillna(0).replace(0, np.nan)
        volume = volume.fillna(method='ffill')
        corr_60 = close.rolling(window).corr(volume)
        sign_corr = np.sign(corr_60)
        rank_5 = close.rolling(sub_window).rank(pct=True)
        std_vol_10 = volume.rolling(vol_window).std().replace(0, np.nan).fillna(method='ffill')
        raw = sign_corr * rank_5 / (std_vol_10 + 1e-8)
        norm_window = max(window, vol_window, sub_window)
        norm_mean = raw.rolling(norm_window).mean()
        norm_std = raw.rolling(norm_window).std().replace(0, np.nan).fillna(method='ffill')
        signal = ((raw - norm_mean) / norm_std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_027_sign(df, window=100, sub_window=40):
        vol_window = 10
        close = df['close']
        volume = df['matchingVolume'].fillna(0).replace(0, np.nan)
        volume = volume.fillna(method='ffill')
        corr_60 = close.rolling(window).corr(volume)
        sign_corr = np.sign(corr_60)
        rank_5 = close.rolling(sub_window).rank(pct=True)
        std_vol_10 = volume.rolling(vol_window).std().replace(0, np.nan).fillna(method='ffill')
        raw = sign_corr * rank_5 / (std_vol_10 + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_027_wf(df, window=50, sub_window=5, p1=0.05):
        vol_window = 10
        close = df['close']
        volume = df['matchingVolume'].fillna(0).replace(0, np.nan)
        volume = volume.fillna(method='ffill')
        corr_60 = close.rolling(window).corr(volume)
        sign_corr = np.sign(corr_60)
        rank_5 = close.rolling(sub_window).rank(pct=True)
        std_vol_10 = volume.rolling(vol_window).std().replace(0, np.nan).fillna(method='ffill')
        raw = sign_corr * rank_5 / (std_vol_10 + 1e-8)
        norm_window = max(window, vol_window, sub_window)
        low = raw.rolling(norm_window).quantile(p1)
        high = raw.rolling(norm_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        numerator = numerator.clip(-0.99, 0.99)
        signal = np.arctanh(numerator)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_028_rank(df, window=20, corr_window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct = close.pct_change(window).replace([np.inf, -np.inf], np.nan)
        std_vol = volume.rolling(5).std().replace(0, np.nan)
        sign = np.sign(std_vol)
        corr = close.rolling(corr_window).corr(volume)
        raw = pct * corr * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_028_tanh(df, window=40, std_window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct = close.pct_change(window).replace([np.inf, -np.inf], np.nan)
        std_vol = volume.rolling(5).std().replace(0, np.nan)
        sign = np.sign(std_vol)
        corr = close.rolling(std_window).corr(volume)
        raw = pct * corr * sign
        normalized = np.tanh(raw / raw.rolling(std_window).std().replace(0, np.nan))
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_028_zscore(df, window=20, z_window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct = close.pct_change(window).replace([np.inf, -np.inf], np.nan)
        std_vol = volume.rolling(5).std().replace(0, np.nan)
        sign = np.sign(std_vol)
        corr = close.rolling(z_window).corr(volume)
        raw = pct * corr * sign
        mean_ = raw.rolling(z_window).mean()
        std_ = raw.rolling(z_window).std().replace(0, np.nan)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_028_sign(df, window=20, corr_window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct = close.pct_change(window).replace([np.inf, -np.inf], np.nan)
        std_vol = volume.rolling(5).std().replace(0, np.nan)
        sign = np.sign(std_vol)
        corr = close.rolling(corr_window).corr(volume)
        raw = pct * corr * sign
        normalized = np.sign(raw)
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_028_wf(df, window=10, p2=30, p1=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct = close.pct_change(window).replace([np.inf, -np.inf], np.nan)
        std_vol = volume.rolling(5).std().replace(0, np.nan)
        sign = np.sign(std_vol)
        corr = close.rolling(p2).corr(volume)
        raw = pct * corr * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_029_k(df, window=70, vol_window=3):
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else (
            df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('amount', df['close'] * df.get('matchingVolume', 1)) / df['close']
        )
        ts_corr = close.rolling(window).corr(volume)
        vol_std = volume.rolling(vol_window).std().replace(0, np.nan)
        vol_mean = volume.rolling(vol_window).mean()
        raw = ts_corr / (vol_std + vol_mean + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_029_h(df, window=10, vol_window=7):
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else (
            df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('amount', df['close'] * df.get('matchingVolume', 1)) / df['close']
        )
        ts_corr = close.rolling(window).corr(volume)
        vol_std = volume.rolling(vol_window).std().replace(0, np.nan)
        vol_mean = volume.rolling(vol_window).mean()
        raw = ts_corr / (vol_std + vol_mean + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_029_e(df, window=70, vol_window=40):
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else (
            df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('amount', df['close'] * df.get('matchingVolume', 1)) / df['close']
        )
        ts_corr = close.rolling(window).corr(volume)
        vol_std = volume.rolling(vol_window).std().replace(0, np.nan)
        vol_mean = volume.rolling(vol_window).mean()
        raw = ts_corr / (vol_std + vol_mean + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_029_y(df, window=100, vol_window=10):
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else (
            df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('amount', df['close'] * df.get('matchingVolume', 1)) / df['close']
        )
        ts_corr = close.rolling(window).corr(volume)
        vol_std = volume.rolling(vol_window).std().replace(0, np.nan)
        vol_mean = volume.rolling(vol_window).mean()
        raw = ts_corr / (vol_std + vol_mean + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = np.sign(raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_029_r(df, window=60, vol_window=7, quantile=0.05):
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else (
            df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('amount', df['close'] * df.get('matchingVolume', 1)) / df['close']
        )
        ts_corr = close.rolling(window).corr(volume)
        vol_std = volume.rolling(vol_window).std().replace(0, np.nan)
        vol_mean = volume.rolling(vol_window).mean()
        raw = ts_corr / (vol_std + vol_mean + 1e-8)
        raw = raw.ffill().fillna(0)
        p1 = quantile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_030_rank(df, window1=40, window2=3):
        pct = df['close'].pct_change(periods=window1)
        corr = df['close'].rolling(window2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)))
        std = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).rolling(5).std()
        z_pct = (pct - pct.rolling(window1).mean()) / pct.rolling(window1).std().replace(0, np.nan)
        z_corr = (corr - corr.rolling(window2).mean()) / corr.rolling(window2).std().replace(0, np.nan)
        z_std = (std - std.rolling(5).mean()) / std.rolling(5).std().replace(0, np.nan)
        raw = z_pct.fillna(0) + z_corr.fillna(0) - z_std.fillna(0)
        return ((raw.rolling(window1).rank(pct=True) * 2) - 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_030_tanh(df, window1=10, window2=3):
        pct = df['close'].pct_change(periods=window1)
        corr = df['close'].rolling(window2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)))
        std = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).rolling(5).std()
        z_pct = (pct - pct.rolling(window1).mean()) / pct.rolling(window1).std().replace(0, np.nan)
        z_corr = (corr - corr.rolling(window2).mean()) / corr.rolling(window2).std().replace(0, np.nan)
        z_std = (std - std.rolling(5).mean()) / std.rolling(5).std().replace(0, np.nan)
        raw = z_pct.fillna(0) + z_corr.fillna(0) - z_std.fillna(0)
        return np.tanh(raw / raw.rolling(window1).std().replace(0, np.nan)).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_030_zscore(df, window1=30, window2=3):
        pct = df['close'].pct_change(periods=window1)
        corr = df['close'].rolling(window2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)))
        std = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).rolling(5).std()
        z_pct = (pct - pct.rolling(window1).mean()) / pct.rolling(window1).std().replace(0, np.nan)
        z_corr = (corr - corr.rolling(window2).mean()) / corr.rolling(window2).std().replace(0, np.nan)
        z_std = (std - std.rolling(5).mean()) / std.rolling(5).std().replace(0, np.nan)
        raw = z_pct.fillna(0) + z_corr.fillna(0) - z_std.fillna(0)
        return ((raw - raw.rolling(window1).mean()) / raw.rolling(window1).std().replace(0, np.nan)).clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_030_sign(df, window1=30, window2=5):
        pct = df['close'].pct_change(periods=window1)
        corr = df['close'].rolling(window2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)))
        std = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).rolling(5).std()
        z_pct = (pct - pct.rolling(window1).mean()) / pct.rolling(window1).std().replace(0, np.nan)
        z_corr = (corr - corr.rolling(window2).mean()) / corr.rolling(window2).std().replace(0, np.nan)
        z_std = (std - std.rolling(5).mean()) / std.rolling(5).std().replace(0, np.nan)
        raw = z_pct.fillna(0) + z_corr.fillna(0) - z_std.fillna(0)
        return np.sign(raw).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_030_wf(df, window1=30, window2=5):
        pct = df['close'].pct_change(periods=window1)
        corr = df['close'].rolling(window2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)))
        std = df.get('matchingVolume', df.get('volume', df['close'] * 0 + 1)).rolling(5).std()
        z_pct = (pct - pct.rolling(window1).mean()) / pct.rolling(window1).std().replace(0, np.nan)
        z_corr = (corr - corr.rolling(window2).mean()) / corr.rolling(window2).std().replace(0, np.nan)
        z_std = (std - std.rolling(5).mean()) / std.rolling(5).std().replace(0, np.nan)
        raw = z_pct.fillna(0) + z_corr.fillna(0) - z_std.fillna(0)
        p1 = 0.05
        p2 = window1
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_031_rank(df, window=35):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_031_tanh(df, window=40):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_031_zscore(df, window=60):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_031_sign(df, window=35):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_031_wf(df, window=10, p1=0.9, p2=20):
        param_window = window
        raw = (df['high'] - df['low']) / (df['close'].rolling(param_window).mean() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_032_k(df, window=45):
        ret = df['close'].pct_change().fillna(0)
        volume = np.log1p(df['matchingVolume'].clip(lower=0))
        corr = ret.rolling(window).corr(volume).fillna(0)
        mean_ret = ret.rolling(5).mean().fillna(0)
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_032_h(df, window=30):
        ret = df['close'].pct_change().fillna(0)
        volume = np.log1p(df['matchingVolume'].clip(lower=0))
        corr = ret.rolling(window).corr(volume).fillna(0)
        mean_ret = ret.rolling(5).mean().fillna(0)
        sign = np.sign(mean_ret)
        raw = corr * sign
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-9)
        normalized = np.tanh(raw / std)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_032_p(df, window=35):
        ret = df['close'].pct_change().fillna(0)
        volume = np.log1p(df['matchingVolume'].clip(lower=0))
        corr = ret.rolling(window).corr(volume).fillna(0)
        mean_ret = ret.rolling(5).mean().fillna(0)
        sign = np.sign(mean_ret)
        raw = corr * sign
        mean = raw.rolling(window).mean().fillna(0)
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-9)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_032_y(df, window=30):
        ret = df['close'].pct_change().fillna(0)
        volume = np.log1p(df['matchingVolume'].clip(lower=0))
        corr = ret.rolling(window).corr(volume).fillna(0)
        mean_ret = ret.rolling(5).mean().fillna(0)
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_032_r(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        volume = np.log1p(df['matchingVolume'].clip(lower=0))
        corr = ret.rolling(window).corr(volume).fillna(0)
        mean_ret = ret.rolling(5).mean().fillna(0)
        sign = np.sign(mean_ret)
        raw = corr * sign
        low = raw.rolling(window).quantile(0.05)
        high = raw.rolling(window).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_033_rank(df, window_rank=75):
        raw = df['close'].rolling(5).std() / (df['close'].rolling(20).std() + df['close'].rolling(20).std().rolling(10).mean())
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_033_tanh(df, window_tanh=80):
        raw = df['close'].rolling(5).std() / (df['close'].rolling(20).std() + df['close'].rolling(20).std().rolling(10).mean())
        signal = np.tanh(raw / raw.rolling(window_tanh).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_033_zscore(df, window_zscore=35):
        raw = df['close'].rolling(5).std() / (df['close'].rolling(20).std() + df['close'].rolling(20).std().rolling(10).mean())
        signal = ((raw - raw.rolling(window_zscore).mean()) / raw.rolling(window_zscore).std()).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_033_sign(df, window_sign=30):
        raw = df['close'].rolling(5).std() / (df['close'].rolling(20).std() + df['close'].rolling(20).std().rolling(10).mean())
        signal = np.sign(raw - raw.rolling(window_sign).mean())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_033_wf(df, p1=0.3, p2=20):
        raw = df['close'].rolling(5).std() / (df['close'].rolling(20).std() + df['close'].rolling(20).std().rolling(10).mean())
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_034_rank(df, window=45):
        raw = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) * (df['close'].rolling(5).std() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal

    @staticmethod
    def alpha_quanta_full_base_034_tanh(df, window=90):
        raw = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) * (df['close'].rolling(5).std() + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal

    @staticmethod
    def alpha_quanta_full_base_034_zscore(df, window=70):
        raw = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) * (df['close'].rolling(5).std() + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_034_sign(df):
        raw = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) * (df['close'].rolling(5).std() + 1e-8))
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_full_base_034_wf(df, sub_window=100):
        raw = (df['close'] - df['open']).abs() / ((df['high'] - df['low']) * (df['close'].rolling(5).std() + 1e-8))
        p = 0.05
        low = raw.rolling(sub_window).quantile(p)
        high = raw.rolling(sub_window).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal

    @staticmethod
    def alpha_quanta_full_base_035_rank(df, window=85):
        close = df['close']
        volume = df['matchingVolume']
        mean_5 = close.rolling(5).mean()
        delay_1 = close.shift(1)
        signal_sign = np.sign(mean_5 - delay_1)
        corr = close.rolling(window).corr(volume)
        raw = signal_sign * corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_035_tanh(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        mean_5 = close.rolling(5).mean()
        delay_1 = close.shift(1)
        signal_sign = np.sign(mean_5 - delay_1)
        corr = close.rolling(window).corr(volume)
        raw = signal_sign * corr
        std_ = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std_)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_035_zscore(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        mean_5 = close.rolling(5).mean()
        delay_1 = close.shift(1)
        signal_sign = np.sign(mean_5 - delay_1)
        corr = close.rolling(window).corr(volume)
        raw = signal_sign * corr
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_035_sign(df, window=90):
        close = df['close']
        volume = df['matchingVolume']
        mean_5 = close.rolling(5).mean()
        delay_1 = close.shift(1)
        signal_sign = np.sign(mean_5 - delay_1)
        corr = close.rolling(window).corr(volume)
        raw = signal_sign * corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_035_wf(df, window=100, p1=0.9):
        close = df['close']
        volume = df['matchingVolume']
        p2 = max(window, 10)
        mean_5 = close.rolling(5).mean()
        delay_1 = close.shift(1)
        signal_sign = np.sign(mean_5 - delay_1)
        corr = close.rolling(window).corr(volume)
        raw = signal_sign * corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_036_rank(df, window=20, factor=20):
        close = df['close']
        volume = df['matchingVolume']
        ts_mean = close.rolling(window).mean()
        ts_std = close.rolling(window).std()
        zscore_volume = (volume - volume.rolling(factor).mean()) / (volume.rolling(factor).std() + 1e-9)
        raw = (close - ts_mean) / ((ts_std + 1e-8) * (np.abs(zscore_volume) + 1e-8))
        rank_raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return rank_raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_036_tanh(df, window=10, factor=5):
        close = df['close']
        volume = df['matchingVolume']
        ts_mean = close.rolling(window).mean()
        ts_std = close.rolling(window).std()
        zscore_volume = (volume - volume.rolling(factor).mean()) / (volume.rolling(factor).std() + 1e-9)
        raw = (close - ts_mean) / ((ts_std + 1e-8) * (np.abs(zscore_volume) + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_036_zscore(df, window=10, factor=10):
        close = df['close']
        volume = df['matchingVolume']
        ts_mean = close.rolling(window).mean()
        ts_std = close.rolling(window).std()
        zscore_volume = (volume - volume.rolling(factor).mean()) / (volume.rolling(factor).std() + 1e-9)
        raw = (close - ts_mean) / ((ts_std + 1e-8) * (np.abs(zscore_volume) + 1e-9))
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std()
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_036_sign(df, window=10, factor=30):
        close = df['close']
        volume = df['matchingVolume']
        ts_mean = close.rolling(window).mean()
        ts_std = close.rolling(window).std()
        zscore_volume = (volume - volume.rolling(factor).mean()) / (volume.rolling(factor).std() + 1e-9)
        raw = (close - ts_mean) / ((ts_std + 1e-8) * (np.abs(zscore_volume) + 1e-9))
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_036_wf(df, window=10, factor=5):
        close = df['close']
        volume = df['matchingVolume']
        ts_mean = close.rolling(window).mean()
        ts_std = close.rolling(window).std()
        zscore_volume = (volume - volume.rolling(factor).mean()) / (volume.rolling(factor).std() + 1e-9)
        raw = (close - ts_mean) / ((ts_std + 1e-8) * (np.abs(zscore_volume) + 1e-9))
        p1 = 0.05
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_037_rank(df, window=60):
        volume = df['matchingVolume']
        delta_vol = volume.diff(1)
        ts_mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = delta_vol / (ts_mean_vol + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_037_tanh(df, window=5):
        volume = df['matchingVolume']
        delta_vol = volume.diff(1)
        ts_mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = delta_vol / (ts_mean_vol + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_037_zscore(df, window=5):
        volume = df['matchingVolume']
        delta_vol = volume.diff(1)
        ts_mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = delta_vol / (ts_mean_vol + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_037_sign(df, window=15):
        volume = df['matchingVolume']
        delta_vol = volume.diff(1)
        ts_mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = delta_vol / (ts_mean_vol + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_037_wf(df, window=20, quantile=0.3):
        volume = df['matchingVolume']
        delta_vol = volume.diff(1)
        ts_mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = delta_vol / (ts_mean_vol + 1e-8)
        p1 = quantile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_038_rank(df, window=15):
        ret = df['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_delta = df['volume'].diff().fillna(0) / (df['volume'] + 1e-8)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        rank_corr = corr.rolling(window).rank(pct=True) * 2 - 1
        return rank_corr.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_038_tanh(df, window=15):
        ret = df['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_delta = df['volume'].diff().fillna(0) / (df['volume'] + 1e-8)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        result = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan)).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_038_zscore(df, window=15):
        ret = df['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_delta = df['volume'].diff().fillna(0) / (df['volume'] + 1e-8)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        result = ((corr - mean) / std).clip(-1, 1).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_038_sign(df, window=15):
        ret = df['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_delta = df['volume'].diff().fillna(0) / (df['volume'] + 1e-8)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        result = pd.Series(np.sign(corr), index=df.index).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_038_wf(df, window=15, p1=0.05):
        p2 = 2 * window
        ret = df['close'].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        vol_delta = df['volume'].diff().fillna(0) / (df['volume'] + 1e-8)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        low = corr.rolling(p2).quantile(p1).fillna(method='ffill').fillna(corr.min())
        high = corr.rolling(p2).quantile(1 - p1).fillna(method='ffill').fillna(corr.max())
        winsorized = corr.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = pd.Series(normalized, index=df.index).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_039_rank(df, window=85):
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        high_ratio = df['high'] / (df['high'].rolling(window).max() + 1e-8)
        low_ratio = df['low'] / (df['low'].rolling(window).min() + 1e-8)
        raw = (vol_ratio * (high_ratio + low_ratio)).rolling(window).rank(pct=True) * 2 - 1
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_039_tanh(df, window=5):
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        high_ratio = df['high'] / (df['high'].rolling(window).max() + 1e-8)
        low_ratio = df['low'] / (df['low'].rolling(window).min() + 1e-8)
        raw = vol_ratio * (high_ratio + low_ratio)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_039_zscore(df, window=95):
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        high_ratio = df['high'] / (df['high'].rolling(window).max() + 1e-8)
        low_ratio = df['low'] / (df['low'].rolling(window).min() + 1e-8)
        raw = vol_ratio * (high_ratio + low_ratio)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_039_sign(df, window=10):
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        high_ratio = df['high'] / (df['high'].rolling(window).max() + 1e-8)
        low_ratio = df['low'] / (df['low'].rolling(window).min() + 1e-8)
        raw = vol_ratio * (high_ratio + low_ratio)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_039_wf(df, window=90, p1=0.7):
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        high_ratio = df['high'] / (df['high'].rolling(window).max() + 1e-8)
        low_ratio = df['low'] / (df['low'].rolling(window).min() + 1e-8)
        raw = vol_ratio * (high_ratio + low_ratio)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_040_rank(df, window=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        vol_delta = volume.diff(1)
        corr = vol_delta.rolling(window).corr(ret)
        # Rolling Rank normalization
        raw = np.sign(vol_delta.rolling(window).mean()) * corr
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_040_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        vol_delta = volume.diff(1)
        corr = vol_delta.rolling(window).corr(ret)
        # Dynamic Tanh normalization
        raw = np.sign(vol_delta.rolling(window).mean()) * corr
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_040_zscore(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        vol_delta = volume.diff(1)
        corr = vol_delta.rolling(window).corr(ret)
        # Rolling Z-Score normalization
        raw = np.sign(vol_delta.rolling(window).mean()) * corr
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean_) / std_).clip(-1, 1)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_040_sign(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        vol_delta = volume.diff(1)
        corr = vol_delta.rolling(window).corr(ret)
        # Sign/Binary Soft normalization
        raw = np.sign(vol_delta.rolling(window).mean()) * corr
        result = np.sign(raw)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_040_wf(df, window=100, quantile=0.9):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        vol_delta = volume.diff(1)
        corr = vol_delta.rolling(window).corr(ret)
        # Winsorized Fisher normalization
        raw = np.sign(vol_delta.rolling(window).mean()) * corr
        p2 = window
        p1 = quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Avoid division by zero
        denom = (high - low).replace(0, np.nan)
        scaled = ((winsorized - low) / denom) * 1.98 - 0.99
        # Clip scaled to avoid arctanh singularities
        scaled = scaled.clip(-0.99, 0.99)
        result = np.arctanh(scaled)
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_041_rank(df, window=75):
        volume_delta = df['matchingVolume'].diff()
        close_delta = df['close'].diff()
        volume_rank = (volume_delta.rolling(window).rank(pct=True) * 2) - 1
        close_std = close_delta.rolling(window).std()
        raw = volume_rank / (close_std + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_041_tanh(df, window=5):
        volume_delta = df['matchingVolume'].diff()
        close_delta = df['close'].diff()
        volume_zscore_vol = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        close_std = close_delta.rolling(window).std()
        raw = volume_zscore_vol / (close_std + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_041_zscore(df, window=100):
        volume_delta = df['matchingVolume'].diff()
        close_delta = df['close'].diff()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        close_std = close_delta.rolling(window).std()
        raw = volume_zscore / (close_std + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_041_sign(df, window=5):
        volume_delta = df['matchingVolume'].diff()
        close_delta = df['close'].diff()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        close_std = close_delta.rolling(window).std()
        raw = volume_zscore / (close_std + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_041_wf(df, window=60, quantile_param=0.3):
        volume_delta = df['matchingVolume'].diff()
        close_delta = df['close'].diff()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        close_std = close_delta.rolling(window).std()
        raw = volume_zscore / (close_std + 1e-8)
        low = raw.rolling(window).quantile(quantile_param)
        high = raw.rolling(window).quantile(1 - quantile_param)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_042_rank(df, window=45):
        log_volume = np.log1p(df['matchingVolume'])
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope_vol = (log_volume.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        delta_close = df['close'].diff(1)
        slope_del = (delta_close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        raw = slope_vol - slope_del
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_042_tanh(df, window=45):
        log_volume = np.log1p(df['matchingVolume'])
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope_vol = (log_volume.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        delta_close = df['close'].diff(1)
        slope_del = (delta_close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        raw = slope_vol - slope_del
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_042_zscore(df, window=70):
        log_volume = np.log1p(df['matchingVolume'])
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope_vol = (log_volume.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        delta_close = df['close'].diff(1)
        slope_del = (delta_close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        raw = slope_vol - slope_del
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_042_sign(df, window=65):
        log_volume = np.log1p(df['matchingVolume'])
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope_vol = (log_volume.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        delta_close = df['close'].diff(1)
        slope_del = (delta_close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        raw = slope_vol - slope_del
        norm = np.sign(raw)
        return -pd.Series(norm, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_042_wf(df, window=70, p1=0.1):
        log_volume = np.log1p(df['matchingVolume'])
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope_vol = (log_volume.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        delta_close = df['close'].diff(1)
        slope_del = (delta_close.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan))
        raw = slope_vol - slope_del
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_043_rank(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        raw = (close - close.rolling(window).mean()) / (volume.rolling(window).mean() + 1e-8)
        # Normalize with Rolling Rank (A)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_043_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        raw = (close - close.rolling(window).mean()) / (volume.rolling(window).mean() + 1e-8)
        # Normalize with Dynamic Tanh (B)
        std = raw.rolling(window).std()
        signal = np.tanh(raw / std.replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_043_zscore(df, window=15):
        close = df['close']
        volume = df['matchingVolume']
        raw = (close - close.rolling(window).mean()) / (volume.rolling(window).mean() + 1e-8)
        # Normalize with Rolling Z-Score (C)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_043_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        raw = (close - close.rolling(window).mean()) / (volume.rolling(window).mean() + 1e-8)
        # Normalize with Sign (D)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_043_wf(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        raw = (close - close.rolling(window).mean()) / (volume.rolling(window).mean() + 1e-8)
        # Normalize with Winsorized Fisher (E)
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_044_k(df, window=100):
        hl_range = df['high'] - df['low']
        vol = df['matchingVolume']
        corr_series = hl_range.rolling(window).corr(vol)
        raw = (corr_series - corr_series.rolling(window).mean()) / corr_series.rolling(window).std()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_044_h(df, window=35):
        hl_range = df['high'] - df['low']
        vol = df['matchingVolume']
        corr_series = hl_range.rolling(window).corr(vol)
        raw = (corr_series - corr_series.rolling(window).mean()) / corr_series.rolling(window).std()
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_044_p(df, window=40):
        hl_range = df['high'] - df['low']
        vol = df['matchingVolume']
        corr_series = hl_range.rolling(window).corr(vol)
        raw = (corr_series - corr_series.rolling(window).mean()) / corr_series.rolling(window).std()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_044_t(df, window=70):
        hl_range = df['high'] - df['low']
        vol = df['matchingVolume']
        corr_series = hl_range.rolling(window).corr(vol)
        raw = (corr_series - corr_series.rolling(window).mean()) / corr_series.rolling(window).std()
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_044_r(df, window=40, p1=0.1):
        hl_range = df['high'] - df['low']
        vol = df['matchingVolume']
        corr_series = hl_range.rolling(window).corr(vol)
        raw = (corr_series - corr_series.rolling(window).mean()) / corr_series.rolling(window).std()
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_045_k(df, window=7, corr_window=100):
        ret = df['close'].pct_change()
        sign_mean = np.sign(ret.rolling(window).mean())
        delta_vol = df['matchingVolume'].diff()
        valid = ret.notna() & df['matchingVolume'].notna() & delta_vol.notna()
        raw = pd.Series(np.nan, index=df.index)
        raw[valid] = ret[valid].rolling(corr_window).corr(delta_vol[valid])
        result = sign_mean * raw
        result = (result.rolling(corr_window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_045_h(df, window=1, corr_window=90):
        ret = df['close'].pct_change()
        sign_mean = np.sign(ret.rolling(window).mean())
        delta_vol = df['matchingVolume'].diff()
        valid = ret.notna() & df['matchingVolume'].notna() & delta_vol.notna()
        raw = pd.Series(np.nan, index=df.index)
        raw[valid] = ret[valid].rolling(corr_window).corr(delta_vol[valid])
        product = sign_mean * raw
        denom = product.rolling(corr_window).std().replace(0, np.nan)
        result = np.tanh(product / denom)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_045_p(df, window=40, corr_window=30):
        ret = df['close'].pct_change()
        sign_mean = np.sign(ret.rolling(window).mean())
        delta_vol = df['matchingVolume'].diff()
        valid = ret.notna() & df['matchingVolume'].notna() & delta_vol.notna()
        raw = pd.Series(np.nan, index=df.index)
        raw[valid] = ret[valid].rolling(corr_window).corr(delta_vol[valid])
        product = sign_mean * raw
        mean_ = product.rolling(corr_window).mean()
        std_ = product.rolling(corr_window).std().replace(0, np.nan)
        result = ((product - mean_) / std_).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_045_t(df, window=1, corr_window=70):
        ret = df['close'].pct_change()
        sign_mean = np.sign(ret.rolling(window).mean())
        delta_vol = df['matchingVolume'].diff()
        valid = ret.notna() & df['matchingVolume'].notna() & delta_vol.notna()
        raw = pd.Series(np.nan, index=df.index)
        raw[valid] = ret[valid].rolling(corr_window).corr(delta_vol[valid])
        product = sign_mean * raw
        result = np.sign(product)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_045_r(df, window=30, corr_window=40, p1=0.1):
        ret = df['close'].pct_change()
        sign_mean = np.sign(ret.rolling(window).mean())
        delta_vol = df['matchingVolume'].diff()
        valid = ret.notna() & df['matchingVolume'].notna() & delta_vol.notna()
        raw = pd.Series(np.nan, index=df.index)
        raw[valid] = ret[valid].rolling(corr_window).corr(delta_vol[valid])
        product = sign_mean * raw
        low = product.rolling(corr_window).quantile(p1)
        high = product.rolling(corr_window).quantile(1 - p1)
        winsorized = product.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.clip(-0.99, 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_046_rank(df, window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = high - low
        mean_range = high_low_range.rolling(window).mean().replace(0, np.nan)
        raw1 = high_low_range / (mean_range + 1e-8)
        delta_close = close.diff()
        signed_volume = np.sign(delta_close) * volume
        corr = high_low_range.rolling(window).corr(signed_volume).fillna(0)
        raw2 = raw1 * (-corr)
        raw = raw2
        param = window
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_046_tanh(df, window=45):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = high - low
        mean_range = high_low_range.rolling(window).mean().replace(0, np.nan)
        raw1 = high_low_range / (mean_range + 1e-8)
        delta_close = close.diff()
        signed_volume = np.sign(delta_close) * volume
        corr = high_low_range.rolling(window).corr(signed_volume).fillna(0)
        raw2 = raw1 * (-corr)
        raw = raw2
        param = window
        normalized = np.tanh(raw / raw.rolling(param).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_046_zscore(df, window=5):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = high - low
        mean_range = high_low_range.rolling(window).mean().replace(0, np.nan)
        raw1 = high_low_range / (mean_range + 1e-8)
        delta_close = close.diff()
        signed_volume = np.sign(delta_close) * volume
        corr = high_low_range.rolling(window).corr(signed_volume).fillna(0)
        raw2 = raw1 * (-corr)
        raw = raw2
        param = window
        normalized = ((raw - raw.rolling(param).mean()) / raw.rolling(param).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_046_sign(df, window=5):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = high - low
        mean_range = high_low_range.rolling(window).mean().replace(0, np.nan)
        raw1 = high_low_range / (mean_range + 1e-8)
        delta_close = close.diff()
        signed_volume = np.sign(delta_close) * volume
        corr = high_low_range.rolling(window).corr(signed_volume).fillna(0)
        raw2 = raw1 * (-corr)
        raw = raw2
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_046_wf(df, window=60, p1=0.3):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = high - low
        mean_range = high_low_range.rolling(window).mean().replace(0, np.nan)
        raw1 = high_low_range / (mean_range + 1e-8)
        delta_close = close.diff()
        signed_volume = np.sign(delta_close) * volume
        corr = high_low_range.rolling(window).corr(signed_volume).fillna(0)
        raw2 = raw1 * (-corr)
        raw = raw2
        p2 = window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_047_k(df, window=10, sub_window=3):
        # Input validation
        window = int(window)
        sub_window = int(sub_window)
        # Compute raw signal: TS_ZSCORE(high - low, 15) * (SIGN(DELTA(close, 1)) * (volume / (TS_MEAN(volume, 15) + 1e-8)))
        spread = df['high'] - df['low']
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std().replace(0, np.nan)
        zscore_spread = (spread - spread_mean) / spread_std
        # delta close
        delta_close = df['close'].diff(sub_window)
        sign_delta = np.sign(delta_close)
        # volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / volume_mean
        raw = zscore_spread * (sign_delta * volume_ratio)
        # Normalize using Rolling Rank (Case A)
        raw_filled = raw.fillna(0).replace([np.inf, -np.inf], 0)
        rank = raw_filled.rolling(window).rank(pct=True)
        result = (rank * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_047_h(df, window=80, sub_window=1):
        # Input validation
        window = int(window)
        sub_window = int(sub_window)
        # Compute raw signal
        spread = df['high'] - df['low']
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std().replace(0, np.nan)
        zscore_spread = (spread - spread_mean) / spread_std
        delta_close = df['close'].diff(sub_window)
        sign_delta = np.sign(delta_close)
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / volume_mean
        raw = zscore_spread * (sign_delta * volume_ratio)
        # Normalize using Dynamic Tanh (Case B)
        raw_filled = raw.fillna(0).replace([np.inf, -np.inf], 0)
        std_raw = raw_filled.rolling(window).std().replace(0, 1)
        result = np.tanh(raw_filled / std_raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_047_e(df, window=80, sub_window=1):
        # Input validation
        window = int(window)
        sub_window = int(sub_window)
        # Compute raw signal
        spread = df['high'] - df['low']
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std().replace(0, np.nan)
        zscore_spread = (spread - spread_mean) / spread_std
        delta_close = df['close'].diff(sub_window)
        sign_delta = np.sign(delta_close)
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / volume_mean
        raw = zscore_spread * (sign_delta * volume_ratio)
        # Normalize using Rolling Z-Score (Case C)
        raw_filled = raw.fillna(0).replace([np.inf, -np.inf], 0)
        roll_mean = raw_filled.rolling(window).mean()
        roll_std = raw_filled.rolling(window).std().replace(0, 1)
        result = ((raw_filled - roll_mean) / roll_std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_047_y(df, window=10, sub_window=2):
        # Input validation
        window = int(window)
        sub_window = int(sub_window)
        # Compute raw signal
        spread = df['high'] - df['low']
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std().replace(0, np.nan)
        zscore_spread = (spread - spread_mean) / spread_std
        delta_close = df['close'].diff(sub_window)
        sign_delta = np.sign(delta_close)
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / volume_mean
        raw = zscore_spread * (sign_delta * volume_ratio)
        # Normalize using Sign/Binary Soft (Case D)
        raw_filled = raw.fillna(0).replace([np.inf, -np.inf], 0)
        result = np.sign(raw_filled)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_047_r(df, window=10, sub_window=2, p1=0.05, p2=30):
        # Input validation - p2 defaults to 2*window if not provided, p1=0.05
        window = int(window)
        sub_window = int(sub_window)
        p1 = 0.05
        p2 = int(window * 2) if p2 is None else int(p2)
        # Compute raw signal
        spread = df['high'] - df['low']
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std().replace(0, np.nan)
        zscore_spread = (spread - spread_mean) / spread_std
        delta_close = df['close'].diff(sub_window)
        sign_delta = np.sign(delta_close)
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / volume_mean
        raw = zscore_spread * (sign_delta * volume_ratio)
        # Winsorized Fisher Transform (Case E)
        raw_filled = raw.fillna(0).replace([np.inf, -np.inf], 0)
        low = raw_filled.rolling(p2).quantile(p1)
        high = raw_filled.rolling(p2).quantile(1 - p1)
        winsorized = raw_filled.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9)
        numerator_clipped = numerator.clip(1e-9, 1 - 1e-9)
        result = np.arctanh(numerator_clipped * 1.98 - 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_048_rank(df, window=30):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8)
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        delta_close = df['close'].diff(1)
        sign_delta = pd.Series(np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0)), index=df.index)
        vol_mean = df['matchingVolume'].rolling(window).mean() + 1e-8
        vol_ratio = df['matchingVolume'] / vol_mean
        corr = sign_delta.rolling(window).corr(vol_ratio)
        signal = rank * (1 - corr)
        signal = signal.replace([np.inf, -np.inf], np.nan)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_048_tanh(df, window=5):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8)
        delta_close = df['close'].diff(1)
        sign_delta = pd.Series(np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0)), index=df.index)
        vol_mean = df['matchingVolume'].rolling(window).mean() + 1e-8
        vol_ratio = df['matchingVolume'] / vol_mean
        corr = sign_delta.rolling(window).corr(vol_ratio)
        raw_signal = raw * (1 - corr)
        normalized = np.tanh(raw_signal / raw_signal.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_048_zscore(df, window=35):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8)
        delta_close = df['close'].diff(1)
        sign_delta = pd.Series(np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0)), index=df.index)
        vol_mean = df['matchingVolume'].rolling(window).mean() + 1e-8
        vol_ratio = df['matchingVolume'] / vol_mean
        corr = sign_delta.rolling(window).corr(vol_ratio)
        raw_signal = raw * (1 - corr)
        mean_ = raw_signal.rolling(window).mean()
        std_ = raw_signal.rolling(window).std()
        normalized = ((raw_signal - mean_) / std_.replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_048_sign(df, window=5):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8)
        delta_close = df['close'].diff(1)
        sign_delta = pd.Series(np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0)), index=df.index)
        vol_mean = df['matchingVolume'].rolling(window).mean() + 1e-8
        vol_ratio = df['matchingVolume'] / vol_mean
        corr = sign_delta.rolling(window).corr(vol_ratio)
        raw_signal = raw * (1 - corr)
        signal = np.sign(raw_signal)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_048_wf(df, window=3, p2=40):
        p1 = 0.1
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8)
        delta_close = df['close'].diff(1)
        sign_delta = pd.Series(np.where(delta_close > 0, 1, np.where(delta_close < 0, -1, 0)), index=df.index)
        vol_mean = df['matchingVolume'].rolling(window).mean() + 1e-8
        vol_ratio = df['matchingVolume'] / vol_mean
        corr = sign_delta.rolling(window).corr(vol_ratio)
        raw_signal = raw * (1 - corr)
        low = raw_signal.rolling(p2).quantile(p1)
        high = raw_signal.rolling(p2).quantile(1 - p1)
        winsorized = raw_signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_049_k(df, window=65):
        price_range = df['high'] - df['low']
        mid = (df['high'] + df['low']) / 2
        close_dev = (df['close'] - mid).abs()
        small_range_vol = df['matchingVolume'] * (close_dev < 0.1 * price_range).astype(int)
        raw = (df['close'] - df['open']).abs() / (price_range.rolling(5).std() + 1e-8) * (small_range_vol.rolling(5).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8))
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_049_h(df, window=5):
        price_range = df['high'] - df['low']
        mid = (df['high'] + df['low']) / 2
        close_dev = (df['close'] - mid).abs()
        small_range_vol = df['matchingVolume'] * (close_dev < 0.1 * price_range).astype(int)
        raw = (df['close'] - df['open']).abs() / (price_range.rolling(5).std() + 1e-8) * (small_range_vol.rolling(5).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_049_e(df, window=100):
        price_range = df['high'] - df['low']
        mid = (df['high'] + df['low']) / 2
        close_dev = (df['close'] - mid).abs()
        small_range_vol = df['matchingVolume'] * (close_dev < 0.1 * price_range).astype(int)
        raw = (df['close'] - df['open']).abs() / (price_range.rolling(5).std() + 1e-8) * (small_range_vol.rolling(5).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_049_t(df, window=10):
        price_range = df['high'] - df['low']
        mid = (df['high'] + df['low']) / 2
        close_dev = (df['close'] - mid).abs()
        small_range_vol = df['matchingVolume'] * (close_dev < 0.1 * price_range).astype(int)
        raw = (df['close'] - df['open']).abs() / (price_range.rolling(5).std() + 1e-8) * (small_range_vol.rolling(5).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8))
        signal = np.sign(raw.rolling(window).mean().fillna(0))
        return -signal

    @staticmethod
    def alpha_quanta_full_base_049_r(df, window=100, p1=0.1):
        price_range = df['high'] - df['low']
        mid = (df['high'] + df['low']) / 2
        close_dev = (df['close'] - mid).abs()
        small_range_vol = df['matchingVolume'] * (close_dev < 0.1 * price_range).astype(int)
        raw = (df['close'] - df['open']).abs() / (price_range.rolling(5).std() + 1e-8) * (small_range_vol.rolling(5).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8))
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_050_rank(df, window=95):
        raw = (df['close'] - df['open']).apply(np.sign) * ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)).rolling(window).corr(df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        return (raw.rolling(window).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_full_base_050_tanh(df, window=95):
        raw = (df['close'] - df['open']).apply(np.sign) * ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)).rolling(window).corr(df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        return np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_050_zscore(df, window=90):
        raw = (df['close'] - df['open']).apply(np.sign) * ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)).rolling(window).corr(df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        return ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_050_sign(df, window=90):
        raw = (df['close'] - df['open']).apply(np.sign) * ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)).rolling(window).corr(df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_full_base_050_wf(df, window=100, winsor_pct=0.1):
        raw = (df['close'] - df['open']).apply(np.sign) * ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-8)).rolling(window).corr(df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        param2 = 2 * window
        low = raw.rolling(param2).quantile(winsor_pct)
        high = raw.rolling(param2).quantile(1 - winsor_pct)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_051_rank(df, window=60):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        sign = np.sign(close - open)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        hl_range = (high - low) / ((high - low).rolling(5).mean() + 1e-8)
        slope = vol_ratio.rolling(5).cov(hl_range) / hl_range.rolling(5).var().replace(0, np.nan)
        raw = sign * slope
        result = raw.rolling(window).rank(pct=True) * 2 - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_051_tanh(df, window=5):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        sign = np.sign(close - open)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        hl_range = (high - low) / ((high - low).rolling(5).mean() + 1e-8)
        slope = vol_ratio.rolling(5).cov(hl_range) / hl_range.rolling(5).var().replace(0, np.nan)
        raw = sign * slope
        result = np.tanh(raw / raw.rolling(window).std())
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_051_zscore(df, window=50):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        sign = np.sign(close - open)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        hl_range = (high - low) / ((high - low).rolling(5).mean() + 1e-8)
        slope = vol_ratio.rolling(5).cov(hl_range) / hl_range.rolling(5).var().replace(0, np.nan)
        raw = sign * slope
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_051_sign(df):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        sign = np.sign(close - open)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        hl_range = (high - low) / ((high - low).rolling(5).mean() + 1e-8)
        slope = vol_ratio.rolling(5).cov(hl_range) / hl_range.rolling(5).var().replace(0, np.nan)
        raw = sign * slope
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_051_wf(df, window_rank=100):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        sign = np.sign(close - open)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        hl_range = (high - low) / ((high - low).rolling(5).mean() + 1e-8)
        slope = vol_ratio.rolling(5).cov(hl_range) / hl_range.rolling(5).var().replace(0, np.nan)
        raw = sign * slope
        p1 = 0.05
        low_per = raw.rolling(window_rank).quantile(p1)
        high_per = raw.rolling(window_rank).quantile(1 - p1)
        winsorized = raw.clip(lower=low_per, upper=high_per, axis=0)
        normalized = np.arctanh(((winsorized - low_per) / (high_per - low_per + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_052_k(df, window_rank=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = close.rolling(5).corr(volume)
        rank_20 = close.rolling(window_rank).rank(pct=True)
        std_vol_60 = volume.rolling(60).std()
        raw = corr_5 * rank_20 / (std_vol_60 + 1e-8)
        signal = (raw.rolling(10).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_052_h(df, smooth_window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = close.rolling(5).corr(volume)
        rank_20 = close.rolling(20).rank(pct=True)
        std_vol_60 = volume.rolling(60).std()
        raw = corr_5 * rank_20 / (std_vol_60 + 1e-8)
        signal = np.tanh(raw / raw.abs().rolling(smooth_window).mean().replace(0, np.nan).ffill())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_052_p(df, window_z=15):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = close.rolling(5).corr(volume)
        rank_20 = close.rolling(20).rank(pct=True)
        std_vol_60 = volume.rolling(60).std()
        raw = corr_5 * rank_20 / (std_vol_60 + 1e-8)
        rolling_mean = raw.rolling(window_z).mean()
        rolling_std = raw.rolling(window_z).std().replace(0, np.nan).ffill()
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_052_y(df, sign_window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = close.rolling(5).corr(volume)
        rank_20 = close.rolling(20).rank(pct=True)
        std_vol_60 = volume.rolling(60).std()
        raw = corr_5 * rank_20 / (std_vol_60 + 1e-8)
        raw_smooth = raw.rolling(sign_window).mean()
        signal = np.sign(raw_smooth)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_052_r(df, quantile_p=0.9, winsor_window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = close.rolling(5).corr(volume)
        rank_20 = close.rolling(20).rank(pct=True)
        std_vol_60 = volume.rolling(60).std()
        raw = corr_5 * rank_20 / (std_vol_60 + 1e-8)
        p = quantile_p
        w = winsor_window
        low = raw.rolling(w).quantile(p)
        high = raw.rolling(w).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_053_k(df, window=40, sub_window=30):
        # Tính rolling correlation giữa close và volume trên sub_window
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = close.rolling(sub_window).corr(volume).fillna(0)
        # SIGN(corr) -> sign_corr dùng np.sign
        sign_corr = pd.Series(np.sign(corr), index=df.index)
        # TS_RANK(close, 20)
        rank_close = close.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        # TS_STD(volume, 60) dùng volume gốc (không log) như công thức
        std_vol = volume.rolling(60).std().replace(0, np.nan).fillna(1e-8)
        raw = sign_corr * rank_close / (std_vol + 1e-8)
        # Rolling Rank normalization (Trường hợp A)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result

    @staticmethod
    def alpha_quanta_full_base_053_h(df, window=10, sub_window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = close.rolling(sub_window).corr(volume).fillna(0)
        sign_corr = pd.Series(np.sign(corr), index=df.index)
        rank_close = close.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        std_vol = volume.rolling(60).std().replace(0, np.nan).fillna(1e-8)
        raw = sign_corr * rank_close / (std_vol + 1e-8)
        # Dynamic Tanh normalization (Trường hợp B)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).fillna(1))
        return -result

    @staticmethod
    def alpha_quanta_full_base_053_e(df, window=10, sub_window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = close.rolling(sub_window).corr(volume).fillna(0)
        sign_corr = pd.Series(np.sign(corr), index=df.index)
        rank_close = close.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        std_vol = volume.rolling(60).std().replace(0, np.nan).fillna(1e-8)
        raw = sign_corr * rank_close / (std_vol + 1e-8)
        # Rolling Z-Score normalization (Trường hợp C)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).fillna(1)
        result = ((raw - mean) / std).clip(-1, 1)
        return result

    @staticmethod
    def alpha_quanta_full_base_053_y(df, window=30, sub_window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = close.rolling(sub_window).corr(volume).fillna(0)
        sign_corr = pd.Series(np.sign(corr), index=df.index)
        rank_close = close.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        std_vol = volume.rolling(60).std().replace(0, np.nan).fillna(1e-8)
        raw = sign_corr * rank_close / (std_vol + 1e-8)
        # Sign/Binary Soft normalization (Trường hợp D)
        result = pd.Series(np.sign(raw), index=df.index)
        return -result

    @staticmethod
    def alpha_quanta_full_base_053_r(df, window=30, sub_window=30, p1=0.05, p2=60):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = close.rolling(sub_window).corr(volume).fillna(0)
        sign_corr = pd.Series(np.sign(corr), index=df.index)
        rank_close = close.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        std_vol = volume.rolling(60).std().replace(0, np.nan).fillna(1e-8)
        raw = sign_corr * rank_close / (std_vol + 1e-8)
        # Winsorized Fisher normalization (Trường hợp E) với hardcode p1=0.05 , p2=60
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = winsorized - low
        denominator = (high - low + 1e-9)
        scaled = numerator / denominator * 1.98 - 0.99
        scaled = scaled.clip(-0.99, 0.99)
        result = np.arctanh(scaled)
        return -result

    @staticmethod
    def alpha_quanta_full_base_054_rank(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        corr_window = window
        corr = close.rolling(corr_window).corr(volume)
        raw = (corr / (corr.rolling(20).std() + 1e-8)) * (close.rolling(20).rank(pct=True) / (volume.rolling(60).std() + 1e-8))
        result = (raw.rolling(20).rank(pct=True) * 2) - 1
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_054_tanh(df, window=50):
        close = df['close']
        volume = df['matchingVolume']
        corr_window = window
        corr = close.rolling(corr_window).corr(volume)
        raw = (corr / (corr.rolling(20).std() + 1e-8)) * (close.rolling(20).rank(pct=True) / (volume.rolling(60).std() + 1e-8))
        result = np.tanh(raw / raw.rolling(20).std())
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_054_zscore(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        corr_window = window
        corr = close.rolling(corr_window).corr(volume)
        raw = (corr / (corr.rolling(20).std() + 1e-8)) * (close.rolling(20).rank(pct=True) / (volume.rolling(60).std() + 1e-8))
        result = ((raw - raw.rolling(20).mean()) / raw.rolling(20).std()).clip(-1, 1)
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_054_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        corr_window = window
        corr = close.rolling(corr_window).corr(volume)
        raw = (corr / (corr.rolling(20).std() + 1e-8)) * (close.rolling(20).rank(pct=True) / (volume.rolling(60).std() + 1e-8))
        raw = raw.fillna(0)
        result = np.sign(raw)
        return pd.Series(result, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_054_wf(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        corr_window = window
        corr = close.rolling(corr_window).corr(volume)
        raw = (corr / (corr.rolling(20).std() + 1e-8)) * (close.rolling(20).rank(pct=True) / (volume.rolling(60).std() + 1e-8))
        p1 = 0.05
        p2 = 20
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_055_rank(df, window=55):
        avg_range = (df['high'] - df['low']).rolling(window).mean()
        avg_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8
        raw = avg_range / avg_volume
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_055_tanh(df, window=5):
        avg_range = (df['high'] - df['low']).rolling(window).mean()
        avg_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8
        raw = avg_range / avg_volume
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_055_zscore(df, window=70):
        avg_range = (df['high'] - df['low']).rolling(window).mean()
        avg_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8
        raw = avg_range / avg_volume
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_055_sign(df, window=70):
        avg_range = (df['high'] - df['low']).rolling(window).mean()
        avg_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8
        raw = avg_range / avg_volume
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_055_wf(df, window=80, quantile_limit=0.7):
        avg_range = (df['high'] - df['low']).rolling(window).mean()
        avg_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8
        raw = avg_range / avg_volume
        low = raw.rolling(window).quantile(quantile_limit)
        high = raw.rolling(window).quantile(1 - quantile_limit)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_056_rank(df, window=85):
        ret = df['close'].pct_change(fill_method=None)
        std_5 = ret.rolling(5).std()
        delay_1 = ret.shift(1)
        corr_10 = std_5.rolling(window).corr(delay_1)
        rank_val = corr_10.rank(pct=True) * 2 - 1
        return -rank_val.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_056_tanh(df, window=45):
        ret = df['close'].pct_change(fill_method=None)
        std_5 = ret.rolling(5).std()
        delay_1 = ret.shift(1)
        corr_10 = std_5.rolling(window).corr(delay_1)
        raw = corr_10.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).fillna(1))
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_056_zscore(df, window=90):
        ret = df['close'].pct_change(fill_method=None)
        std_5 = ret.rolling(5).std()
        delay_1 = ret.shift(1)
        corr_10 = std_5.rolling(window).corr(delay_1)
        raw = corr_10.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_056_sign(df, window=45):
        ret = df['close'].pct_change(fill_method=None)
        std_5 = ret.rolling(5).std()
        delay_1 = ret.shift(1)
        corr_10 = std_5.rolling(window).corr(delay_1)
        raw = corr_10.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.sign(raw)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_056_wf(df, window=90, p1=0.1, p2=30):
        ret = df['close'].pct_change(fill_method=None)
        std_5 = ret.rolling(5).std()
        delay_1 = ret.shift(1)
        corr_10 = std_5.rolling(window).corr(delay_1)
        raw = corr_10.fillna(0).replace([np.inf, -np.inf], 0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_057_rank(df, window=95):
        ret = df['close'].pct_change()
        vol_delta = df['matchingVolume'].diff()
        abs_ret = ret.abs()
        # Compute rolling correlation using vectorized approach
        corr = abs_ret.rolling(window).corr(vol_delta)
        # Rolling rank normalization to [-1, 1]
        signal = corr.rolling(window).rank(pct=True) * 2 - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_057_tanh(df, window=35):
        ret = df['close'].pct_change()
        vol_delta = df['matchingVolume'].diff()
        abs_ret = ret.abs()
        corr = abs_ret.rolling(window).corr(vol_delta)
        # Dynamic tanh normalization
        std_corr = corr.rolling(window).std()
        signal = np.tanh(corr / std_corr.replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_057_zscore(df, window=95):
        ret = df['close'].pct_change()
        vol_delta = df['matchingVolume'].diff()
        abs_ret = ret.abs()
        corr = abs_ret.rolling(window).corr(vol_delta)
        # Rolling Z-Score normalization
        corr_mean = corr.rolling(window).mean()
        corr_std = corr.rolling(window).std()
        signal = ((corr - corr_mean) / corr_std.replace(0, np.nan)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_057_sign(df, window=25):
        ret = df['close'].pct_change()
        vol_delta = df['matchingVolume'].diff()
        abs_ret = ret.abs()
        corr = abs_ret.rolling(window).corr(vol_delta)
        # Binary soft normalization
        signal = np.sign(corr)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_057_wf(df, window=50, p1=0.3):
        ret = df['close'].pct_change()
        vol_delta = df['matchingVolume'].diff()
        abs_ret = ret.abs()
        corr = abs_ret.rolling(window).corr(vol_delta)
        p2 = window
        # Winsorized Fisher normalization
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        scaled = scaled.clip(-0.9999, 0.9999)
        signal = np.arctanh(scaled)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_058_rank(df, window=65):
        volume_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        rank_vol = (volume_ratio.rolling(window).rank(pct=True) * 2) - 1
        half_range = (df['high'] - df['low']) / 2
        corr = half_range.rolling(5).corr(df['matchingVolume'])
        sign_corr = np.sign(corr)
        signal = rank_vol * sign_corr
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_058_tanh(df, window=10):
        volume_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        tanh_vol = np.tanh(volume_ratio / volume_ratio.rolling(window).std())
        half_range = (df['high'] - df['low']) / 2
        corr = half_range.rolling(5).corr(df['matchingVolume'])
        sign_corr = np.sign(corr)
        signal = tanh_vol * sign_corr
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_058_zscore(df, window=100):
        volume_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        zscore_vol = ((volume_ratio - volume_ratio.rolling(window).mean()) / volume_ratio.rolling(window).std()).clip(-1, 1)
        half_range = (df['high'] - df['low']) / 2
        corr = half_range.rolling(5).corr(df['matchingVolume'])
        sign_corr = np.sign(corr)
        signal = zscore_vol * sign_corr
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_058_sign(df, window=80):
        volume_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        rank_vol = (volume_ratio.rolling(window).rank(pct=True) * 2) - 1
        sign_vol = np.sign(volume_ratio)
        half_range = (df['high'] - df['low']) / 2
        corr = half_range.rolling(5).corr(df['matchingVolume'])
        sign_corr = np.sign(corr)
        signal = sign_vol * sign_corr
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_058_wf(df, window=100, p1=0.1):
        volume_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        p2 = window
        low = volume_ratio.rolling(p2).quantile(p1)
        high = volume_ratio.rolling(p2).quantile(1 - p1)
        winsorized = volume_ratio.clip(lower=low, upper=high, axis=0)
        fisher = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        half_range = (df['high'] - df['low']) / 2
        corr = half_range.rolling(5).corr(df['matchingVolume'])
        sign_corr = np.sign(corr)
        signal = fisher * sign_corr
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_059_rank(df, window=50):
        return_val = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = return_val.rolling(window).std()
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        rolling_mean_log_vol = log_vol.rolling(window).mean()
        raw = rolling_std / (rolling_mean_log_vol + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_059_tanh(df, window=70):
        return_val = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = return_val.rolling(window).std()
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        rolling_mean_log_vol = log_vol.rolling(window).mean()
        raw = rolling_std / (rolling_mean_log_vol + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_059_zscore(df, window=50):
        return_val = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = return_val.rolling(window).std()
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        rolling_mean_log_vol = log_vol.rolling(window).mean()
        raw = rolling_std / (rolling_mean_log_vol + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_059_sign(df, window=15):
        return_val = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = return_val.rolling(window).std()
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        rolling_mean_log_vol = log_vol.rolling(window).mean()
        raw = rolling_std / (rolling_mean_log_vol + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_059_wf(df, window=10, p1=0.1):
        return_val = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        rolling_std = return_val.rolling(window).std()
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        rolling_mean_log_vol = log_vol.rolling(window).mean()
        raw = rolling_std / (rolling_mean_log_vol + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        rng = high - low + 1e-9
        normalized = np.arctanh(((winsorized - low) / rng) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_060_k(df, window=100):
        high_low = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        rolling_std_hl = high_low.rolling(window).std()
        rolling_std_v = volume.rolling(window).std()
        corr = (high_low.rolling(window).cov(volume) / (rolling_std_hl * rolling_std_v)).replace([np.inf, -np.inf], np.nan)
        rank_corr = corr.rolling(window).rank(pct=True) * 2 - 1
        vol_mean = volume.rolling(window).mean().replace(0, np.nan)
        ratio = volume / (vol_mean + 1e-8) - 1
        signal = rank_corr * np.sign(ratio)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_060_h(df, window=30):
        high_low = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        rolling_std_hl = high_low.rolling(window).std()
        rolling_std_v = volume.rolling(window).std()
        corr = (high_low.rolling(window).cov(volume) / (rolling_std_hl * rolling_std_v)).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean().replace(0, np.nan)
        ratio = volume / (vol_mean + 1e-8) - 1
        raw = corr * np.sign(ratio)
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -norm.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_060_e(df, window=35):
        high_low = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        rolling_std_hl = high_low.rolling(window).std()
        rolling_std_v = volume.rolling(window).std()
        corr = (high_low.rolling(window).cov(volume) / (rolling_std_hl * rolling_std_v)).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean().replace(0, np.nan)
        ratio = volume / (vol_mean + 1e-8) - 1
        raw = corr * np.sign(ratio)
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -norm.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_060_y(df, window=85):
        high_low = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        rolling_std_hl = high_low.rolling(window).std()
        rolling_std_v = volume.rolling(window).std()
        corr = (high_low.rolling(window).cov(volume) / (rolling_std_hl * rolling_std_v)).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean().replace(0, np.nan)
        ratio = volume / (vol_mean + 1e-8) - 1
        raw = corr * np.sign(ratio)
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_060_r(df, window=20, p1=0.9):
        p2 = window
        high_low = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        rolling_std_hl = high_low.rolling(window).std()
        rolling_std_v = volume.rolling(window).std()
        corr = (high_low.rolling(window).cov(volume) / (rolling_std_hl * rolling_std_v)).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean().replace(0, np.nan)
        ratio = volume / (vol_mean + 1e-8) - 1
        raw = corr * np.sign(ratio)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_061_rank(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1)
        vol_log = np.log1p(volume)
        raw = delta_close.rolling(window).corr(vol_log).rank(pct=True) * 2 - 1
        signal = raw.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_061_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1)
        vol_log = np.log1p(volume)
        corr_val = delta_close.rolling(window).corr(vol_log)
        std_corr = corr_val.rolling(window).std()
        std_corr = std_corr.replace(0, np.nan).ffill()
        raw = np.tanh(corr_val / std_corr)
        signal = raw.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_061_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1)
        vol_log = np.log1p(volume)
        corr_val = delta_close.rolling(window).corr(vol_log)
        mean_corr = corr_val.rolling(window).mean()
        std_corr = corr_val.rolling(window).std().replace(0, np.nan)
        raw = ((corr_val - mean_corr) / std_corr).clip(-1, 1)
        signal = raw.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_061_sign(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1)
        vol_log = np.log1p(volume)
        corr_val = delta_close.rolling(window).corr(vol_log)
        raw = np.sign(corr_val)
        signal = pd.Series(raw, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_061_wf(df, window=10, percentile=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1)
        vol_log = np.log1p(volume)
        corr_val = delta_close.rolling(window).corr(vol_log)
        low = corr_val.rolling(window).quantile(percentile)
        high = corr_val.rolling(window).quantile(1 - percentile)
        winsorized = corr_val.clip(lower=low, upper=high, axis=0)
        range_wh = high - low
        range_wh = range_wh.replace(0, np.nan)
        normalized = np.arctanh(((winsorized - low) / (range_wh + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_062_rank(df, rank_window=60):
        # Tính raw signal
        low_5min = df['low'].rolling(window=5).min()
        low_10std = df['low'].rolling(window=10).std()
        raw = (df['low'] - low_5min) / (low_10std + 1e-8)
        # Rolling rank normalization
        normalized = (raw.rolling(window=rank_window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_062_tanh(df, norm_window=15):
        # Tính raw signal
        low_5min = df['low'].rolling(window=5).min()
        low_10std = df['low'].rolling(window=10).std()
        raw = (df['low'] - low_5min) / (low_10std + 1e-8)
        # Dynamic tanh normalization
        normalized = np.tanh(raw / raw.rolling(window=norm_window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_062_zscore(df, window=45):
        # Tính raw signal
        low_5min = df['low'].rolling(window=5).min()
        low_10std = df['low'].rolling(window=10).std()
        raw = (df['low'] - low_5min) / (low_10std + 1e-8)
        # Rolling z-score normalization
        mean = raw.rolling(window=window).mean()
        std = raw.rolling(window=window).std()
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_062_sign(df):
        # Tính raw signal
        low_5min = df['low'].rolling(window=5).min()
        low_10std = df['low'].rolling(window=10).std()
        raw = (df['low'] - low_5min) / (low_10std + 1e-8)
        # Sign/binary soft normalization
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_062_wf(df, p1=0.1, p2=20):
        # Tính raw signal
        low_5min = df['low'].rolling(window=5).min()
        low_10std = df['low'].rolling(window=10).std()
        raw = (df['low'] - low_5min) / (low_10std + 1e-8)
        # Winsorized Fisher normalization
        low_q = raw.rolling(window=p2).quantile(p1)
        high_q = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_063_rank(df, window=10):
        close = df['close']
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-8)
        raw = np.sign(close - ma) * (close - ma).abs() / (std + 1e-8)
        signal = raw.rolling(window).rank(pct=True).fillna(0) * 2 - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_063_tanh(df, window=5):
        close = df['close']
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-8)
        raw = np.sign(close - ma) * (close - ma).abs() / (std + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().fillna(1))
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_063_zscore(df, window=10):
        close = df['close']
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-8)
        raw = np.sign(close - ma) * (close - ma).abs() / (std + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1).fillna(0)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_063_sign(df, window=5):
        close = df['close']
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-8)
        raw = np.sign(close - ma) * (close - ma).abs() / (std + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_063_wf(df, window=10, p1=0.1):
        close = df['close']
        ma = close.rolling(window, min_periods=1).mean()
        std = close.rolling(window, min_periods=1).std().replace(0, np.nan).fillna(1e-8)
        raw = np.sign(close - ma) * (close - ma).abs() / (std + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_064_rank(df, window_rank=35):
        ret = df['close'].pct_change()
        mean5 = ret.rolling(5).mean()
        std5 = ret.rolling(5).std()
        ratio = mean5 / (std5 + 1e-8)
        sign3 = np.sign(ret.rolling(3).mean())
        raw = ratio * sign3
        # Rolling Rank normalization (Case A)
        norm = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return norm.fillna(0).ffill()

    @staticmethod
    def alpha_quanta_full_base_064_tanh(df, window_std=95):
        ret = df['close'].pct_change()
        mean5 = ret.rolling(5).mean()
        std5 = ret.rolling(5).std()
        ratio = mean5 / (std5 + 1e-8)
        sign3 = np.sign(ret.rolling(3).mean())
        raw = ratio * sign3
        # Dynamic Tanh normalization (Case B)
        norm = np.tanh(raw / raw.rolling(window_std).std().replace(0, np.nan))
        return norm.fillna(0).ffill()

    @staticmethod
    def alpha_quanta_full_base_064_zscore(df, window_z=45):
        ret = df['close'].pct_change()
        mean5 = ret.rolling(5).mean()
        std5 = ret.rolling(5).std()
        ratio = mean5 / (std5 + 1e-8)
        sign3 = np.sign(ret.rolling(3).mean())
        raw = ratio * sign3
        # Rolling Z-Score/Clip normalization (Case C)
        norm = ((raw - raw.rolling(window_z).mean()) / raw.rolling(window_z).std().replace(0, np.nan)).clip(-1, 1)
        return norm.fillna(0).ffill()

    @staticmethod
    def alpha_quanta_full_base_064_sign(df):
        ret = df['close'].pct_change()
        mean5 = ret.rolling(5).mean()
        std5 = ret.rolling(5).std()
        ratio = mean5 / (std5 + 1e-8)
        sign3 = np.sign(ret.rolling(3).mean())
        raw = ratio * sign3
        # Sign/Binary Soft normalization (Case D)
        norm = np.sign(raw)
        return norm.fillna(0).ffill()

    @staticmethod
    def alpha_quanta_full_base_064_wf(df, p2=80, p1=0.1):
        ret = df['close'].pct_change()
        mean5 = ret.rolling(5).mean()
        std5 = ret.rolling(5).std()
        ratio = mean5 / (std5 + 1e-8)
        sign3 = np.sign(ret.rolling(3).mean())
        raw = ratio * sign3
        # Winsorized Fisher normalization (Case E)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0).ffill()

    @staticmethod
    def alpha_quanta_full_base_065_rank(df, window=35):
        try:
            close = df['close']
            low = df['low']
            ts_min_low = low.rolling(window=window).min()
            ts_std_close = close.rolling(window=window).std()
            delay_close = close.shift(1)
            raw = (close - ts_min_low) / (ts_std_close + 1e-8) * (close / delay_close)
            raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            # Rolling rank normalization
            rank_raw = raw.rolling(window=window).rank(pct=True) * 2 - 1
            signal = rank_raw.fillna(0).clip(-1, 1)
            return signal
        except Exception as e:
            return pd.Series(index=df.index, dtype=float).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_065_tanh(df, window=5):
        try:
            close = df['close']
            low = df['low']
            ts_min_low = low.rolling(window=window).min()
            ts_std_close = close.rolling(window=window).std()
            delay_close = close.shift(1)
            raw = (close - ts_min_low) / (ts_std_close + 1e-8) * (close / delay_close)
            raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            # Dynamic tanh normalization
            raw_std = raw.rolling(window=window).std().replace(0, np.nan).fillna(1)
            signal = np.tanh(raw / (raw_std + 1e-9))
            return signal.fillna(0)
        except Exception as e:
            return pd.Series(index=df.index, dtype=float).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_065_zscore(df, window=10):
        try:
            close = df['close']
            low = df['low']
            ts_min_low = low.rolling(window=window).min()
            ts_std_close = close.rolling(window=window).std()
            delay_close = close.shift(1)
            raw = (close - ts_min_low) / (ts_std_close + 1e-8) * (close / delay_close)
            raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            # Rolling z-score normalization
            mean_raw = raw.rolling(window=window).mean()
            std_raw = raw.rolling(window=window).std().replace(0, np.nan).fillna(1)
            signal = ((raw - mean_raw) / (std_raw + 1e-9)).clip(-1, 1)
            return signal.fillna(0)
        except Exception as e:
            return pd.Series(index=df.index, dtype=float).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_065_sign(df, window=55):
        try:
            close = df['close']
            low = df['low']
            ts_min_low = low.rolling(window=window).min()
            ts_std_close = close.rolling(window=window).std()
            delay_close = close.shift(1)
            raw = (close - ts_min_low) / (ts_std_close + 1e-8) * (close / delay_close)
            raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            # Sign normalization
            signal = np.sign(raw).astype(float)
            return signal.fillna(0)
        except Exception as e:
            return pd.Series(index=df.index, dtype=float).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_065_wf(df, window=40):
        try:
            close = df['close']
            low = df['low']
            ts_min_low = low.rolling(window=window).min()
            ts_std_close = close.rolling(window=window).std()
            delay_close = close.shift(1)
            raw = (close - ts_min_low) / (ts_std_close + 1e-8) * (close / delay_close)
            raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            # Winsorized Fisher normalization
            p1 = 0.05
            low_quantile = raw.rolling(window=window).quantile(p1)
            high_quantile = raw.rolling(window=window).quantile(1 - p1)
            winsorized = raw.clip(lower=low_quantile, upper=high_quantile, axis=0)
            normalized = np.arctanh(((winsorized - low_quantile) / (high_quantile - low_quantile + 1e-9)) * 1.98 - 0.99)
            signal = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            return signal.clip(-1, 1)
        except Exception as e:
            return pd.Series(index=df.index, dtype=float).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_066_rank(df, window=25):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        delta_close = close.diff(1)
        corr1 = delta_close.rolling(window).corr(volume)
        hl_diff = high - low
        corr2 = hl_diff.rolling(window).corr(volume)
        raw = corr1 - corr2
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_066_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        delta_close = close.diff(1)
        corr1 = delta_close.rolling(window).corr(volume)
        hl_diff = high - low
        corr2 = hl_diff.rolling(window).corr(volume)
        raw = corr1 - corr2
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_066_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        delta_close = close.diff(1)
        corr1 = delta_close.rolling(window).corr(volume)
        hl_diff = high - low
        corr2 = hl_diff.rolling(window).corr(volume)
        raw = corr1 - corr2
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        signal = signal.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_066_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        delta_close = close.diff(1)
        corr1 = delta_close.rolling(window).corr(volume)
        hl_diff = high - low
        corr2 = hl_diff.rolling(window).corr(volume)
        raw = corr1 - corr2
        signal = np.sign(raw)
        signal = signal.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_066_wf(df, window=20, p1=0.3):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        delta_close = close.diff(1)
        corr1 = delta_close.rolling(window).corr(volume)
        hl_diff = high - low
        corr2 = hl_diff.rolling(window).corr(volume)
        raw = corr1 - corr2
        p2 = window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_067_rank(df, window=100):
        close = df['close']
        open_ = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay = close.shift(5)
        sign = np.sign((close - delay) / (close + 1e-8))
        raw = sign * (open_ - low) / (open_ + 1e-8) * volume.rolling(5).mean() / (volume.rolling(10).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_067_tanh(df, window=85):
        close = df['close']
        open_ = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay = close.shift(5)
        sign = np.sign((close - delay) / (close + 1e-8))
        raw = sign * (open_ - low) / (open_ + 1e-8) * volume.rolling(5).mean() / (volume.rolling(10).mean() + 1e-8)
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_067_zscore(df, window=5):
        close = df['close']
        open_ = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay = close.shift(5)
        sign = np.sign((close - delay) / (close + 1e-8))
        raw = sign * (open_ - low) / (open_ + 1e-8) * volume.rolling(5).mean() / (volume.rolling(10).mean() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / (raw.rolling(window).std() + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_067_sign(df):
        close = df['close']
        open_ = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay = close.shift(5)
        sign = np.sign((close - delay) / (close + 1e-8))
        raw = sign * (open_ - low) / (open_ + 1e-8) * volume.rolling(5).mean() / (volume.rolling(10).mean() + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_067_wf(df, window=20, p2=30):
        close = df['close']
        open_ = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay = close.shift(5)
        sign = np.sign((close - delay) / (close + 1e-8))
        raw = sign * (open_ - low) / (open_ + 1e-8) * volume.rolling(5).mean() / (volume.rolling(10).mean() + 1e-8)
        low_val = raw.rolling(p2).quantile(0.05)
        high_val = raw.rolling(p2).quantile(0.95)
        winsorized = raw.clip(lower=low_val, upper=high_val, axis=0)
        normalized = np.arctanh(((winsorized - low_val) / (high_val - low_val + 1e-9)) * 1.98 - 0.99)
        signal = (normalized.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_068_zscore(df, window=25):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('volume', pd.Series(1, index=df.index))
        raw = close - open_
        raw = raw / (open_ + 1e-8)
        ts_corr = raw.rolling(10).corr(volume)
        ts_mean_vol = volume.rolling(window).mean()
        vol_ratio = 1 - (volume / (ts_mean_vol + 1e-8))
        delay_5 = close.shift(5)
        sign = np.sign((close - delay_5) / (close + 1e-8))
        raw = ts_corr * vol_ratio * sign
        raw = raw.replace([np.inf, -np.inf], np.nan)
        normalized = ((raw - raw.rolling(20).mean()) / raw.rolling(20).std()).clip(-1, 1)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_068_rank(df, window=15):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('volume', pd.Series(1, index=df.index))
        raw = close - open_
        raw = raw / (open_ + 1e-8)
        ts_corr = raw.rolling(10).corr(volume)
        ts_mean_vol = volume.rolling(window).mean()
        vol_ratio = 1 - (volume / (ts_mean_vol + 1e-8))
        delay_5 = close.shift(5)
        sign = np.sign((close - delay_5) / (close + 1e-8))
        raw = ts_corr * vol_ratio * sign
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        normalized = (raw.rolling(20).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_068_tanh(df, window=85):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('volume', pd.Series(1, index=df.index))
        raw = close - open_
        raw = raw / (open_ + 1e-8)
        ts_corr = raw.rolling(10).corr(volume)
        ts_mean_vol = volume.rolling(window).mean()
        vol_ratio = 1 - (volume / (ts_mean_vol + 1e-8))
        delay_5 = close.shift(5)
        sign = np.sign((close - delay_5) / (close + 1e-8))
        raw = ts_corr * vol_ratio * sign
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        std = raw.rolling(20).std().replace(0, np.nan)
        normalized = np.tanh(raw / std)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_068_sign(df, window=100):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('volume', pd.Series(1, index=df.index))
        raw = close - open_
        raw = raw / (open_ + 1e-8)
        ts_corr = raw.rolling(10).corr(volume)
        ts_mean_vol = volume.rolling(window).mean()
        vol_ratio = 1 - (volume / (ts_mean_vol + 1e-8))
        delay_5 = close.shift(5)
        sign = np.sign((close - delay_5) / (close + 1e-8))
        raw = ts_corr * vol_ratio * sign
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        normalized = np.sign(raw)
        normalized = normalized * 1.0
        return normalized

    @staticmethod
    def alpha_quanta_full_base_068_wf(df, window_rank_=100, p1=0.3):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df.get('volume', pd.Series(1, index=df.index))
        raw = close - open_
        raw = raw / (open_ + 1e-8)
        ts_corr = raw.rolling(10).corr(volume)
        ts_mean_vol = volume.rolling(window_rank_).mean()
        vol_ratio = 1 - (volume / (ts_mean_vol + 1e-8))
        delay_5 = close.shift(5)
        sign = np.sign((close - delay_5) / (close + 1e-8))
        raw = ts_corr * vol_ratio * sign
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0)
        low = raw.rolling(window_rank_).quantile(p1)
        high = raw.rolling(window_rank_).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_069_rank(df, window=85):
        close = df['close']
        open = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay_close = close.shift(window)
        raw_inner = (close - delay_close) / (close.rolling(10).std() + 1e-8)
        condition = raw_inner > 0.5
        numerator = (open - low) / (open + 1e-8) * volume
        denominator = volume.rolling(10).mean() + 1e-8
        raw = numerator / denominator
        raw = pd.Series(np.where(condition, raw, 0), index=df.index)
        signal = (raw.rolling(20).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_069_tanh(df, window=100):
        close = df['close']
        open = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay_close = close.shift(window)
        raw_inner = (close - delay_close) / (close.rolling(10).std() + 1e-8)
        condition = raw_inner > 0.5
        numerator = (open - low) / (open + 1e-8) * volume
        denominator = volume.rolling(10).mean() + 1e-8
        raw = numerator / denominator
        raw = pd.Series(np.where(condition, raw, 0), index=df.index)
        signal = np.tanh(raw / (raw.rolling(20).std() + 1e-8))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_069_zscore(df, window=85):
        close = df['close']
        open = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay_close = close.shift(window)
        raw_inner = (close - delay_close) / (close.rolling(10).std() + 1e-8)
        condition = raw_inner > 0.5
        numerator = (open - low) / (open + 1e-8) * volume
        denominator = volume.rolling(10).mean() + 1e-8
        raw = numerator / denominator
        raw = pd.Series(np.where(condition, raw, 0), index=df.index)
        signal = ((raw - raw.rolling(20).mean()) / (raw.rolling(20).std() + 1e-8)).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_069_sign(df, window=5):
        close = df['close']
        open = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay_close = close.shift(window)
        raw_inner = (close - delay_close) / (close.rolling(10).std() + 1e-8)
        condition = raw_inner > 0.5
        numerator = (open - low) / (open + 1e-8) * volume
        denominator = volume.rolling(10).mean() + 1e-8
        raw = numerator / denominator
        raw = pd.Series(np.where(condition, raw, 0), index=df.index)
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_069_wf(df, window=80, p1=0.3, p2=20):
        close = df['close']
        open = df['open']
        low = df['low']
        volume = df['matchingVolume']
        delay_close = close.shift(window)
        raw_inner = (close - delay_close) / (close.rolling(10).std() + 1e-8)
        condition = raw_inner > 0.5
        numerator = (open - low) / (open + 1e-8) * volume
        denominator = volume.rolling(10).mean() + 1e-8
        raw = numerator / denominator
        raw = pd.Series(np.where(condition, raw, 0), index=df.index)
        p1_val = p1
        p2_val = p2
        low_q = raw.rolling(p2_val).quantile(p1_val)
        high_q = raw.rolling(p2_val).quantile(1 - p1_val)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_070_rank(df, window=10):
        close = df['close']
        low = df['low']
        ts_mean_low = low.rolling(window=window).mean()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_mean_low) / (ts_max_close - ts_mean_low + 1e-8)
        signal = (raw.rolling(window=window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_070_tanh(df, window=5):
        close = df['close']
        low = df['low']
        ts_mean_low = low.rolling(window=window).mean()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_mean_low) / (ts_max_close - ts_mean_low + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_070_zscore(df, window=10):
        close = df['close']
        low = df['low']
        ts_mean_low = low.rolling(window=window).mean()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_mean_low) / (ts_max_close - ts_mean_low + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_070_sign(df, window=5):
        close = df['close']
        low = df['low']
        ts_mean_low = low.rolling(window=window).mean()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_mean_low) / (ts_max_close - ts_mean_low + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_070_wf(df, window=10, p1=0.1):
        close = df['close']
        low = df['low']
        ts_mean_low = low.rolling(window=window).mean()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_mean_low) / (ts_max_close - ts_mean_low + 1e-8)
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        signal = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_071_k(df, window=35):
        _high = df['high']
        _low = df['low']
        _range = _high - _low
        _mean_range = _range.rolling(window=window).mean()
        _raw_1 = _range / _mean_range
        _volume = df.get('matchingVolume', df.get('volume', 1))
        _delta_vol = _volume.diff()
        _std_delta = _delta_vol.rolling(window=window).std() + 1e-8
        _raw = _raw_1 / _std_delta
        _ranked = _raw.rolling(window=window).rank(pct=True) * 2 - 1
        return _ranked.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_071_h(df, window=40):
        _high = df['high']
        _low = df['low']
        _range = _high - _low
        _mean_range = _range.rolling(window=window).mean()
        _raw_1 = _range / _mean_range
        _volume = df.get('matchingVolume', df.get('volume', 1))
        _delta_vol = _volume.diff()
        _std_delta = _delta_vol.rolling(window=window).std() + 1e-8
        _raw = _raw_1 / _std_delta
        _std_raw = _raw.rolling(window=window).std() + 1e-8
        _norm = np.tanh(_raw / _std_raw)
        return -_norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_071_e(df, window=55):
        _high = df['high']
        _low = df['low']
        _range = _high - _low
        _mean_range = _range.rolling(window=window).mean()
        _raw_1 = _range / _mean_range
        _volume = df.get('matchingVolume', df.get('volume', 1))
        _delta_vol = _volume.diff()
        _std_delta = _delta_vol.rolling(window=window).std() + 1e-8
        _raw = _raw_1 / _std_delta
        _mean_raw = _raw.rolling(window=window).mean()
        _std_raw = _raw.rolling(window=window).std() + 1e-8
        _z = ((_raw - _mean_raw) / _std_raw).clip(-1, 1)
        return _z.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_071_y(df, window=75):
        _high = df['high']
        _low = df['low']
        _range = _high - _low
        _mean_range = _range.rolling(window=window).mean()
        _raw_1 = _range / _mean_range
        _volume = df.get('matchingVolume', df.get('volume', 1))
        _delta_vol = _volume.diff()
        _std_delta = _delta_vol.rolling(window=window).std() + 1e-8
        _raw = _raw_1 / _std_delta
        _sign = np.sign(_raw)
        return pd.Series(_sign, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_071_r(df, window=90, p1=0.7):
        _high = df['high']
        _low = df['low']
        _range = _high - _low
        _mean_range = _range.rolling(window=window).mean()
        _raw_1 = _range / _mean_range
        _volume = df.get('matchingVolume', df.get('volume', 1))
        _delta_vol = _volume.diff()
        _std_delta = _delta_vol.rolling(window=window).std() + 1e-8
        _raw = _raw_1 / _std_delta
        p2 = window
        low = _raw.rolling(p2).quantile(p1)
        high = _raw.rolling(p2).quantile(1 - p1)
        winsorized = _raw.clip(lower=low, upper=high, axis=0)
        norm_raw = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        norm_final = norm_raw.clip(-0.99, 0.99)
        norm_final = np.arctanh(norm_final)
        return -norm_final.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_072_rank(df, window=75):
        ret = df['close'].pct_change()
        vol = df.get('volume', df['matchingVolume'])
        vol_delta = vol.diff()
        raw = pd.Series(np.sign(ret.rolling(window).mean()), index=df.index) * ret.rolling(window).corr(vol_delta)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_072_tanh(df, window=35):
        ret = df['close'].pct_change()
        vol = df.get('volume', df['matchingVolume'])
        vol_delta = vol.diff()
        raw = pd.Series(np.sign(ret.rolling(window).mean()), index=df.index) * ret.rolling(window).corr(vol_delta)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_072_zscore(df, window=55):
        ret = df['close'].pct_change()
        vol = df.get('volume', df['matchingVolume'])
        vol_delta = vol.diff()
        raw = pd.Series(np.sign(ret.rolling(window).mean()), index=df.index) * ret.rolling(window).corr(vol_delta)
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_072_sign(df, window=40):
        ret = df['close'].pct_change()
        vol = df.get('volume', df['matchingVolume'])
        vol_delta = vol.diff()
        raw = pd.Series(np.sign(ret.rolling(window).mean()), index=df.index) * ret.rolling(window).corr(vol_delta)
        result = np.sign(raw)
        return pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_072_wf(df, window=80, p1=0.3):
        ret = df['close'].pct_change()
        vol = df.get('volume', df['matchingVolume'])
        vol_delta = vol.diff()
        raw = pd.Series(np.sign(ret.rolling(window).mean()), index=df.index) * ret.rolling(window).corr(vol_delta)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_073_rank(df, window=15):
        raw = (df['low'] - df['low'].rolling(window=5).min()) / (df['close'].pct_change().rolling(window=5).std() + 1e-8)
        signal = (raw.rolling(window=window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_073_tanh(df, window=5):
        raw = (df['low'] - df['low'].rolling(window=5).min()) / (df['close'].pct_change().rolling(window=5).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window=window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_073_zscore(df, window=10):
        raw = (df['low'] - df['low'].rolling(window=5).min()) / (df['close'].pct_change().rolling(window=5).std() + 1e-8)
        signal = ((raw - raw.rolling(window=window).mean()) / raw.rolling(window=window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_073_sign(df):
        raw = (df['low'] - df['low'].rolling(window=5).min()) / (df['close'].pct_change().rolling(window=5).std() + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_073_wf(df, window=20, p1=0.1):
        raw = (df['low'] - df['low'].rolling(window=5).min()) / (df['close'].pct_change().rolling(window=5).std() + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_074_rank(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).corr(df['volume'].diff(1).fillna(0)).rank(pct=True) * 2 - 1
        return raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_074_tanh(df, window=5):
        numerator = (df['high'] - df['low']).rolling(window).corr(df['volume'].diff(1).fillna(0))
        denom = numerator.rolling(window).std()
        result = np.tanh(numerator / denom.replace(0, np.nan))
        return result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_074_zscore(df, window=5):
        numerator = (df['high'] - df['low']).rolling(window).corr(df['volume'].diff(1).fillna(0))
        result = ((numerator - numerator.rolling(window).mean()) / numerator.rolling(window).std()).clip(-1, 1)
        return result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_074_sign(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).corr(df['volume'].diff(1).fillna(0))
        result = -np.sign(raw)
        return result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_074_wf(df, window=5, p1=0.25, p2=10):
        numerator = (df['high'] - df['low']).rolling(window).corr(df['volume'].diff(1).fillna(0))
        low = numerator.rolling(p2).quantile(p1)
        high = numerator.rolling(p2).quantile(1 - p1)
        winsorized = numerator.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_075_rank(df, window=60):
        close = df['close']
        sma_close = close.rolling(window).mean()
        raw = (close - sma_close) / (sma_close + 1e-8)
        ret = close.pct_change()
        vol = df['matchingVolume']
        del_vol = vol.diff()
        corr = ret.rolling(window).corr(del_vol).fillna(0)
        sign = np.sign(corr)
        raw = raw * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_075_tanh(df, window=40):
        close = df['close']
        sma_close = close.rolling(window).mean()
        raw = (close - sma_close) / (sma_close + 1e-8)
        ret = close.pct_change()
        vol = df['matchingVolume']
        del_vol = vol.diff()
        corr = ret.rolling(window).corr(del_vol).fillna(0)
        sign = np.sign(corr)
        raw = raw * sign
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_075_zscore(df, window=45):
        close = df['close']
        sma_close = close.rolling(window).mean()
        raw = (close - sma_close) / (sma_close + 1e-8)
        ret = close.pct_change()
        vol = df['matchingVolume']
        del_vol = vol.diff()
        corr = ret.rolling(window).corr(del_vol).fillna(0)
        sign = np.sign(corr)
        raw = raw * sign
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_075_sign(df, window=90):
        close = df['close']
        sma_close = close.rolling(window).mean()
        raw = (close - sma_close) / (sma_close + 1e-8)
        ret = close.pct_change()
        vol = df['matchingVolume']
        del_vol = vol.diff()
        corr = ret.rolling(window).corr(del_vol).fillna(0)
        sign = np.sign(corr)
        raw = raw * sign
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_075_wf(df, window=90, p1=0.1, p2=20):
        p1 = 0.05
        p2 = 20
        close = df['close']
        sma_close = close.rolling(window).mean()
        raw = (close - sma_close) / (sma_close + 1e-8)
        ret = close.pct_change()
        vol = df['matchingVolume']
        del_vol = vol.diff()
        corr = ret.rolling(window).corr(del_vol).fillna(0)
        sign = np.sign(corr)
        raw = raw * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_076_k(df, window=60):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        range_ratio = (high - low) / (close + 1e-8)
        vol_std = ret.rolling(5).std().fillna(0)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        raw = range_ratio * vol_std * vol_ratio
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_076_h(df, window=70):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        range_ratio = (high - low) / (close + 1e-8)
        vol_std = ret.rolling(5).std().fillna(0)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        raw = range_ratio * vol_std * vol_ratio
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_076_e(df, window=90):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        range_ratio = (high - low) / (close + 1e-8)
        vol_std = ret.rolling(5).std().fillna(0)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        raw = range_ratio * vol_std * vol_ratio
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_076_y(df):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        range_ratio = (high - low) / (close + 1e-8)
        vol_std = ret.rolling(5).std().fillna(0)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        raw = range_ratio * vol_std * vol_ratio
        signal = np.sign(raw).fillna(0) * 1.0
        return signal

    @staticmethod
    def alpha_quanta_full_base_076_r(df, window=30, p1=0.3):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        range_ratio = (high - low) / (close + 1e-8)
        vol_std = ret.rolling(5).std().fillna(0)
        vol_ratio = volume / (volume.rolling(5).mean() + 1e-8)
        raw = range_ratio * vol_std * vol_ratio
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_077_k(df, window=25):
        open = df['open']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = ((close - open) / (open + 1e-8)) * (vol / (vol.rolling(5).mean() + 1e-8))
        return (raw.rolling(window).rank(pct=True) * 2 - 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_077_h(df, window=5):
        open = df['open']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = ((close - open) / (open + 1e-8)) * (vol / (vol.rolling(5).mean() + 1e-8))
        return np.tanh(raw / (raw.rolling(window).std() + 1e-8)).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_077_e(df, window=15):
        open = df['open']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = ((close - open) / (open + 1e-8)) * (vol / (vol.rolling(5).mean() + 1e-8))
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        return ((raw - mean) / (std + 1e-8)).clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_077_y(df):
        open = df['open']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = ((close - open) / (open + 1e-8)) * (vol / (vol.rolling(5).mean() + 1e-8))
        return pd.Series(np.sign(raw), index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_077_r(df, window=60, p1=0.1):
        open = df['open']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = ((close - open) / (open + 1e-8)) * (vol / (vol.rolling(5).mean() + 1e-8))
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_078_rank(df, window=20):
        high = df['high']
        low = df['low']
        open_ = df['open']
        # TS_ARGMAX and TS_ARGMIN: find indices of max/min in rolling window, use vectorized rolling apply via shift
        argmax = high.rolling(5).apply(lambda x: x.idxmax() if len(x) == 5 else 0, raw=False)
        argmin = low.rolling(5).apply(lambda x: x.idxmin() if len(x) == 5 else 0, raw=False)
        # Convert index differences to numeric values relative to window end
        # Since rolling index gives position, we approximate as: (argmax - argmin) / 5
        # Shift to align: use last value in window
        # Simpler: use pandas rolling with custom function for difference of positions
        # Use a different approach: compute rolling max/min positions using shift and cumcount
        idx = df.index.to_series()
        max_idx = high.rolling(5).apply(lambda x: (x == x.max()).idxmax() if len(x) == 5 else 0, raw=False)
        min_idx = low.rolling(5).apply(lambda x: (x == x.min()).idxmax() if len(x) == 5 else 0, raw=False)
        # Convert to numeric ranks: position from end (0=current, 4=oldest)
        # Actually simpler: use rank or directly compute difference in integer indices
        # For simplicity, use rolling with numpy
        # Vectorized: compute max/min positions using rolling window
        # Use convolution to find positions
        # Better: use rolling windows with indices
        # Let's use a simpler method: compute rolling max/min and then find lag
        # Use pandas rolling with index
        # Alternative: compute TS_ARGMAX as number of periods since the max occurred within window
        # Use rolling with argmax via expanding window
        # To avoid apply, we can use numpy sliding_window_view, but we stick to pandas
        # We'll use rolling().apply but with raw=True for speed, but rule prohibits? Okay, use vectorized custom
        # Actually, let's use a vectorized approach: compute rolling max and track index via cumcount
        arr_high = high.values
        arr_low = low.values
        arr_open = open_.values
        n = len(df)
        argmax_result = np.full(n, np.nan)
        argmin_result = np.full(n, np.nan)
        for i in range(4, n):
            win_high = arr_high[i-4:i+1]
            win_low = arr_low[i-4:i+1]
            argmax_result[i] = np.argmax(win_high)
            argmin_result[i] = np.argmin(win_low)
        argmax_series = pd.Series(argmax_result, index=df.index)
        argmin_series = pd.Series(argmin_result, index=df.index)
        a = (argmax_series - argmin_series) / 5.0
        b = (low - open_) / (open_.rolling(5).mean() + 1e-8)
        raw = a * b
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        result = ((raw - mean) / std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_078_tanh(df, window=95):
        high = df['high']
        low = df['low']
        open_ = df['open']
        arr_high = high.values
        arr_low = low.values
        arr_open = open_.values
        n = len(df)
        argmax_result = np.full(n, np.nan)
        argmin_result = np.full(n, np.nan)
        for i in range(4, n):
            win_high = arr_high[i-4:i+1]
            win_low = arr_low[i-4:i+1]
            argmax_result[i] = np.argmax(win_high)
            argmin_result[i] = np.argmin(win_low)
        argmax_series = pd.Series(argmax_result, index=df.index)
        argmin_series = pd.Series(argmin_result, index=df.index)
        a = (argmax_series - argmin_series) / 5.0
        b = (low - open_) / (open_.rolling(5).mean() + 1e-8)
        raw = a * b
        result = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_078_zscore(df, window=5):
        high = df['high']
        low = df['low']
        open_ = df['open']
        arr_high = high.values
        arr_low = low.values
        arr_open = open_.values
        n = len(df)
        argmax_result = np.full(n, np.nan)
        argmin_result = np.full(n, np.nan)
        for i in range(4, n):
            win_high = arr_high[i-4:i+1]
            win_low = arr_low[i-4:i+1]
            argmax_result[i] = np.argmax(win_high)
            argmin_result[i] = np.argmin(win_low)
        argmax_series = pd.Series(argmax_result, index=df.index)
        argmin_series = pd.Series(argmin_result, index=df.index)
        a = (argmax_series - argmin_series) / 5.0
        b = (low - open_) / (open_.rolling(5).mean() + 1e-8)
        raw = a * b
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        result = ((raw - mean) / std).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_078_sign(df):
        high = df['high']
        low = df['low']
        open_ = df['open']
        arr_high = high.values
        arr_low = low.values
        arr_open = open_.values
        n = len(df)
        argmax_result = np.full(n, np.nan)
        argmin_result = np.full(n, np.nan)
        for i in range(4, n):
            win_high = arr_high[i-4:i+1]
            win_low = arr_low[i-4:i+1]
            argmax_result[i] = np.argmax(win_high)
            argmin_result[i] = np.argmin(win_low)
        argmax_series = pd.Series(argmax_result, index=df.index)
        argmin_series = pd.Series(argmin_result, index=df.index)
        a = (argmax_series - argmin_series) / 5.0
        b = (low - open_) / (open_.rolling(5).mean() + 1e-8)
        raw = a * b
        result = np.sign(raw)
        return -pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_078_wf(df, window=10, p1=0.1):
        high = df['high']
        low = df['low']
        open_ = df['open']
        arr_high = high.values
        arr_low = low.values
        arr_open = open_.values
        n = len(df)
        argmax_result = np.full(n, np.nan)
        argmin_result = np.full(n, np.nan)
        for i in range(4, n):
            win_high = arr_high[i-4:i+1]
            win_low = arr_low[i-4:i+1]
            argmax_result[i] = np.argmax(win_high)
            argmin_result[i] = np.argmin(win_low)
        argmax_series = pd.Series(argmax_result, index=df.index)
        argmin_series = pd.Series(argmin_result, index=df.index)
        a = (argmax_series - argmin_series) / 5.0
        b = (low - open_) / (open_.rolling(5).mean() + 1e-8)
        raw = a * b
        low_q = raw.rolling(window).quantile(p1)
        high_q = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        norm = ((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99
        norm = norm.clip(-0.99, 0.99)
        result = np.arctanh(norm)
        result = result.clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_079_rank(df, window=25):
        # Trường hợp A: Rolling Rank
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        amount = df.get('amount', df['close'] * df.get('matchingVolume', df.get('volume', 1)))
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df['volume']
        # Tính TVOL_SQRT
        tvol_sqrt = np.sqrt(amount / (df['high'] - df['low'] + 1e-8))
        # Thành phần từ công thức gốc
        ts_mean_vol_5 = volume.rolling(5).mean()
        ts_mean_vol_20 = volume.rolling(20).mean()
        ratio_vol = ts_mean_vol_5 / (ts_mean_vol_20 + 1e-8)
        delta_close = df['close'].diff(1)
        corr_part = delta_close.rolling(10).corr(volume) * (-1)
        hl_range = df['high'] - df['low']
        ts_mean_hl = hl_range.rolling(10).mean()
        hl_norm = hl_range / (ts_mean_hl + 1e-8)
        raw = ratio_vol * corr_part * hl_norm
        # Chuẩn hóa Rolling Rank
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_079_tanh(df, window=75):
        # Trường hợp B: Dynamic Tanh
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        amount = df.get('amount', df['close'] * df.get('matchingVolume', df.get('volume', 1)))
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df['volume']
        tvol_sqrt = np.sqrt(amount / (df['high'] - df['low'] + 1e-8))
        ts_mean_vol_5 = volume.rolling(5).mean()
        ts_mean_vol_20 = volume.rolling(20).mean()
        ratio_vol = ts_mean_vol_5 / (ts_mean_vol_20 + 1e-8)
        delta_close = df['close'].diff(1)
        corr_part = delta_close.rolling(10).corr(volume) * (-1)
        hl_range = df['high'] - df['low']
        ts_mean_hl = hl_range.rolling(10).mean()
        hl_norm = hl_range / (ts_mean_hl + 1e-8)
        raw = ratio_vol * corr_part * hl_norm
        # Chuẩn hóa Dynamic Tanh
        std = raw.rolling(window).std() + 1e-8
        result = np.tanh(raw / std)
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_079_zscore(df, window=10):
        # Trường hợp C: Rolling Z-Score/Clip
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        amount = df.get('amount', df['close'] * df.get('matchingVolume', df.get('volume', 1)))
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df['volume']
        tvol_sqrt = np.sqrt(amount / (df['high'] - df['low'] + 1e-8))
        ts_mean_vol_5 = volume.rolling(5).mean()
        ts_mean_vol_20 = volume.rolling(20).mean()
        ratio_vol = ts_mean_vol_5 / (ts_mean_vol_20 + 1e-8)
        delta_close = df['close'].diff(1)
        corr_part = delta_close.rolling(10).corr(volume) * (-1)
        hl_range = df['high'] - df['low']
        ts_mean_hl = hl_range.rolling(10).mean()
        hl_norm = hl_range / (ts_mean_hl + 1e-8)
        raw = ratio_vol * corr_part * hl_norm
        # Chuẩn hóa Z-Score
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std() + 1e-8
        result = ((raw - mean) / std).clip(-1, 1)
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_079_sign(df):
        # Trường hợp D: Sign/Binary Soft
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        amount = df.get('amount', df['close'] * df.get('matchingVolume', df.get('volume', 1)))
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df['volume']
        tvol_sqrt = np.sqrt(amount / (df['high'] - df['low'] + 1e-8))
        ts_mean_vol_5 = volume.rolling(5).mean()
        ts_mean_vol_20 = volume.rolling(20).mean()
        ratio_vol = ts_mean_vol_5 / (ts_mean_vol_20 + 1e-8)
        delta_close = df['close'].diff(1)
        corr_part = delta_close.rolling(10).corr(volume) * (-1)
        hl_range = df['high'] - df['low']
        ts_mean_hl = hl_range.rolling(10).mean()
        hl_norm = hl_range / (ts_mean_hl + 1e-8)
        raw = ratio_vol * corr_part * hl_norm
        # Chuẩn hóa Sign
        result = np.sign(raw)
        result = pd.Series(result, index=df.index).ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_079_wf(df, window=10, p1=0.1):
        # Trường hợp E: Winsorized Fisher
        vwap = df.get('vwap', (df['high'] + df['low'] + df['close']) / 3)
        amount = df.get('amount', df['close'] * df.get('matchingVolume', df.get('volume', 1)))
        volume = df['matchingVolume'] if 'matchingVolume' in df.columns else df['volume']
        tvol_sqrt = np.sqrt(amount / (df['high'] - df['low'] + 1e-8))
        ts_mean_vol_5 = volume.rolling(5).mean()
        ts_mean_vol_20 = volume.rolling(20).mean()
        ratio_vol = ts_mean_vol_5 / (ts_mean_vol_20 + 1e-8)
        delta_close = df['close'].diff(1)
        corr_part = delta_close.rolling(10).corr(volume) * (-1)
        hl_range = df['high'] - df['low']
        ts_mean_hl = hl_range.rolling(10).mean()
        hl_norm = hl_range / (ts_mean_hl + 1e-8)
        raw = ratio_vol * corr_part * hl_norm
        # Winsorized Fisher
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_080_rank(df, window=15):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        vol = df['matchingVolume']
        argmax_5 = high.rolling(window).apply(np.argmax, raw=True)
        argmin_5 = low.rolling(window).apply(np.argmin, raw=True)
        ts_argmax = high.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        ts_argmin = low.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        diff_val = (ts_argmax - ts_argmin) / window
        mean_open_5 = open_.rolling(window).mean() + 1e-8
        part2 = (low - open_) / mean_open_5
        mean_vol_5 = vol.rolling(window).mean()
        mean_vol_20 = vol.rolling(20).mean() + 1e-8
        part3 = mean_vol_5 / mean_vol_20
        delta_close = close.diff()
        corr_delta_vol_5 = delta_close.rolling(window).corr(vol)
        part4 = -corr_delta_vol_5
        raw = diff_val * part2 * part3 * part4
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_080_tanh(df, window=10):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        vol = df['matchingVolume']
        argmax_5 = high.rolling(window).apply(np.argmax, raw=True)
        argmin_5 = low.rolling(window).apply(np.argmin, raw=True)
        ts_argmax = high.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        ts_argmin = low.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        diff_val = (ts_argmax - ts_argmin) / window
        mean_open_5 = open_.rolling(window).mean() + 1e-8
        part2 = (low - open_) / mean_open_5
        mean_vol_5 = vol.rolling(window).mean()
        mean_vol_20 = vol.rolling(20).mean() + 1e-8
        part3 = mean_vol_5 / mean_vol_20
        delta_close = close.diff()
        corr_delta_vol_5 = delta_close.rolling(window).corr(vol)
        part4 = -corr_delta_vol_5
        raw = diff_val * part2 * part3 * part4
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_080_zscore(df, window=15):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        vol = df['matchingVolume']
        argmax_5 = high.rolling(window).apply(np.argmax, raw=True)
        argmin_5 = low.rolling(window).apply(np.argmin, raw=True)
        ts_argmax = high.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        ts_argmin = low.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        diff_val = (ts_argmax - ts_argmin) / window
        mean_open_5 = open_.rolling(window).mean() + 1e-8
        part2 = (low - open_) / mean_open_5
        mean_vol_5 = vol.rolling(window).mean()
        mean_vol_20 = vol.rolling(20).mean() + 1e-8
        part3 = mean_vol_5 / mean_vol_20
        delta_close = close.diff()
        corr_delta_vol_5 = delta_close.rolling(window).corr(vol)
        part4 = -corr_delta_vol_5
        raw = diff_val * part2 * part3 * part4
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std()
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_080_sign(df, window=35):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        vol = df['matchingVolume']
        argmax_5 = high.rolling(window).apply(np.argmax, raw=True)
        argmin_5 = low.rolling(window).apply(np.argmin, raw=True)
        ts_argmax = high.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        ts_argmin = low.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        diff_val = (ts_argmax - ts_argmin) / window
        mean_open_5 = open_.rolling(window).mean() + 1e-8
        part2 = (low - open_) / mean_open_5
        mean_vol_5 = vol.rolling(window).mean()
        mean_vol_20 = vol.rolling(20).mean() + 1e-8
        part3 = mean_vol_5 / mean_vol_20
        delta_close = close.diff()
        corr_delta_vol_5 = delta_close.rolling(window).corr(vol)
        part4 = -corr_delta_vol_5
        raw = diff_val * part2 * part3 * part4
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_080_wf(df, window=10, p1=0.3):
        high = df['high']
        low = df['low']
        open_ = df['open']
        close = df['close']
        vol = df['matchingVolume']
        argmax_5 = high.rolling(window).apply(np.argmax, raw=True)
        argmin_5 = low.rolling(window).apply(np.argmin, raw=True)
        ts_argmax = high.rolling(window).apply(lambda x: np.argmax(x), raw=True)
        ts_argmin = low.rolling(window).apply(lambda x: np.argmin(x), raw=True)
        diff_val = (ts_argmax - ts_argmin) / window
        mean_open_5 = open_.rolling(window).mean() + 1e-8
        part2 = (low - open_) / mean_open_5
        mean_vol_5 = vol.rolling(window).mean()
        mean_vol_20 = vol.rolling(20).mean() + 1e-8
        part3 = mean_vol_5 / mean_vol_20
        delta_close = close.diff()
        corr_delta_vol_5 = delta_close.rolling(window).corr(vol)
        part4 = -corr_delta_vol_5
        raw = diff_val * part2 * part3 * part4
        p2 = window * 2
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q)
        scaled = ((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_081_rank(df, window=15):
        close = df['close']
        ret = close.pct_change()
        ra = ret.rolling(5).mean()
        rs = ret.rolling(5).std()
        raw = (ra / rs.replace(0, np.nan))
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_081_tanh(df, window=70):
        close = df['close']
        ret = close.pct_change()
        ra = ret.rolling(5).mean()
        rs = ret.rolling(5).std()
        raw = (ra / rs.replace(0, np.nan))
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_081_zscore(df, window=10):
        close = df['close']
        ret = close.pct_change()
        ra = ret.rolling(5).mean()
        rs = ret.rolling(5).std()
        raw = (ra / rs.replace(0, np.nan))
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_081_sign(df):
        close = df['close']
        ret = close.pct_change()
        ra = ret.rolling(5).mean()
        rs = ret.rolling(5).std()
        raw = (ra / rs.replace(0, np.nan))
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_081_wf(df, p1=0.1, p2=10):
        close = df['close']
        ret = close.pct_change()
        ra = ret.rolling(5).mean()
        rs = ret.rolling(5).std()
        raw = (ra / rs.replace(0, np.nan))
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsor = raw.clip(lower=low, upper=high, axis=0)
        denom = (high - low + 1e-9)
        scaled = ((winsor - low) / denom) * 1.98 - 0.99
        scaled = scaled.clip(-0.99 + 1e-9, 0.99 - 1e-9)
        norm = np.arctanh(scaled)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_082_k(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        raw = -corr
        return -(raw.rolling(window).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_full_base_082_h(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        raw = -corr
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        return -np.tanh(raw / std)

    @staticmethod
    def alpha_quanta_full_base_082_p(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        raw = -corr
        mean = raw.rolling(window).mean().fillna(0)
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        return -((raw - mean) / std).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_082_y(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        raw = -corr
        return -np.sign(raw)

    @staticmethod
    def alpha_quanta_full_base_082_r(df, window=10, p1=0.7):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        corr = ret.rolling(window).corr(vol_delta).fillna(0)
        raw = -corr
        p2 = window
        low = raw.rolling(p2).quantile(p1).fillna(raw.min())
        high = raw.rolling(p2).quantile(1 - p1).fillna(raw.max())
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_083_rank(df, window=50):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = df['high'] - df['low']
        corr = high_low_range.rolling(window).corr(volume).replace(0, np.nan)
        z = -((corr - corr.rolling(window).mean()) / corr.rolling(window).std())
        normalized = (z.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_083_tanh(df, window=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = df['high'] - df['low']
        corr = high_low_range.rolling(window).corr(volume).replace(0, np.nan)
        z = -((corr - corr.rolling(window).mean()) / corr.rolling(window).std())
        normalized = np.tanh(z / z.rolling(window).std())
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_083_zscore(df, window=40):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = df['high'] - df['low']
        corr = high_low_range.rolling(window).corr(volume).replace(0, np.nan)
        z = -((corr - corr.rolling(window).mean()) / corr.rolling(window).std())
        normalized = ((z - z.rolling(window).mean()) / z.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_083_sign(df, window=70):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = df['high'] - df['low']
        corr = high_low_range.rolling(window).corr(volume).replace(0, np.nan)
        z = -((corr - corr.rolling(window).mean()) / corr.rolling(window).std())
        normalized = np.sign(z)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_083_wf(df, window=50, p1=0.7):
        p2 = window
        volume = df.get('matchingVolume', df.get('volume', 1))
        high_low_range = df['high'] - df['low']
        corr = high_low_range.rolling(window).corr(volume).replace(0, np.nan)
        z = -((corr - corr.rolling(window).mean()) / corr.rolling(window).std())
        low = z.rolling(p2).quantile(p1)
        high = z.rolling(p2).quantile(1 - p1)
        winsorized = z.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_084_rank(df, window=40):
        close = df['close']
        high = df['high']
        low = df['low']

        delta_close = close.diff(window)
        rolling_std = close.rolling(window).std()

        raw_part1 = delta_close / (rolling_std + 1e-8)

        ts_min_low = low.rolling(window).min()
        ts_mean_range = (high - low).rolling(window).mean()

        raw_part2 = 1 - (close - ts_min_low) / (ts_mean_range + 1e-8)

        raw = raw_part1 * raw_part2

        signal = (raw.rolling(window).rank(pct=True) * 2) - 1

        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_084_tanh(df, window=85):
        close = df['close']
        high = df['high']
        low = df['low']

        delta_close = close.diff(window)
        rolling_std = close.rolling(window).std()

        raw_part1 = delta_close / (rolling_std + 1e-8)

        ts_min_low = low.rolling(window).min()
        ts_mean_range = (high - low).rolling(window).mean()

        raw_part2 = 1 - (close - ts_min_low) / (ts_mean_range + 1e-8)

        raw = raw_part1 * raw_part2

        signal = np.tanh(raw / raw.rolling(window).std())

        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_084_zscore(df, window=100):
        close = df['close']
        high = df['high']
        low = df['low']

        delta_close = close.diff(window)
        rolling_std = close.rolling(window).std()

        raw_part1 = delta_close / (rolling_std + 1e-8)

        ts_min_low = low.rolling(window).min()
        ts_mean_range = (high - low).rolling(window).mean()

        raw_part2 = 1 - (close - ts_min_low) / (ts_mean_range + 1e-8)

        raw = raw_part1 * raw_part2

        rolling_mean = raw.rolling(window).mean()
        rolling_std_local = raw.rolling(window).std()
        signal = ((raw - rolling_mean) / rolling_std_local).clip(-1, 1)

        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_084_sign(df, window=85):
        close = df['close']
        high = df['high']
        low = df['low']

        delta_close = close.diff(window)
        rolling_std = close.rolling(window).std()

        raw_part1 = delta_close / (rolling_std + 1e-8)

        ts_min_low = low.rolling(window).min()
        ts_mean_range = (high - low).rolling(window).mean()

        raw_part2 = 1 - (close - ts_min_low) / (ts_mean_range + 1e-8)

        raw = raw_part1 * raw_part2

        signal = np.sign(raw)

        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_084_wf(df, window=40, p1=0.9):
        close = df['close']
        high = df['high']
        low = df['low']

        delta_close = close.diff(window)
        rolling_std = close.rolling(window).std()

        raw_part1 = delta_close / (rolling_std + 1e-8)

        ts_min_low = low.rolling(window).min()
        ts_mean_range = (high - low).rolling(window).mean()

        raw_part2 = 1 - (close - ts_min_low) / (ts_mean_range + 1e-8)

        raw = raw_part1 * raw_part2

        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)

        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)

        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)

        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_085_rank(df, window=45):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_delta = volume.diff().fillna(0) / (volume + 1e-8)
        high_low_ratio = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        corr = volume_delta.rolling(window).corr(high_low_ratio)
        ret = df['close'].pct_change().fillna(0)
        sign = np.sign(ret.rolling(5).mean().fillna(0))
        raw = corr * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_085_tanh(df, window=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_delta = volume.diff().fillna(0) / (volume + 1e-8)
        high_low_ratio = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        corr = volume_delta.rolling(window).corr(high_low_ratio)
        ret = df['close'].pct_change().fillna(0)
        sign = np.sign(ret.rolling(5).mean().fillna(0))
        raw = corr * sign
        normalized = np.tanh(raw / (raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1) + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_085_zscore(df, window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_delta = volume.diff().fillna(0) / (volume + 1e-8)
        high_low_ratio = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        corr = volume_delta.rolling(window).corr(high_low_ratio)
        ret = df['close'].pct_change().fillna(0)
        sign = np.sign(ret.rolling(5).mean().fillna(0))
        raw = corr * sign
        mean = raw.rolling(window).mean().fillna(0)
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_085_sign(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_delta = volume.diff().fillna(0) / (volume + 1e-8)
        high_low_ratio = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        corr = volume_delta.rolling(window).corr(high_low_ratio)
        ret = df['close'].pct_change().fillna(0)
        sign = np.sign(ret.rolling(5).mean().fillna(0))
        raw = corr * sign
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_085_wf(df, window=30, quantile_factor=0.3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_delta = volume.diff().fillna(0) / (volume + 1e-8)
        high_low_ratio = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + 1e-8)
        corr = volume_delta.rolling(window).corr(high_low_ratio)
        ret = df['close'].pct_change().fillna(0)
        sign = np.sign(ret.rolling(5).mean().fillna(0))
        raw = corr * sign
        p2 = window
        low = raw.rolling(p2).quantile(quantile_factor)
        high = raw.rolling(p2).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_086_rank(df, window=70, sub_window=10):
        close = df['close']
        ret = close.pct_change()
        ret_vol = ret.rolling(sub_window).std()
        vol_ratio = ret_vol / (ret_vol.rolling(window).mean() + 1e-8)
        ts_zscore_delta = ((close.diff(15) - close.diff(15).rolling(window).mean()) / close.diff(15).rolling(window).std()).clip(-1, 1)
        vol_regime = 1 - vol_ratio
        sma_close = close.rolling(15).mean()
        norm_pos = (close - sma_close) / (close.rolling(15).std() + 1e-8)
        norm_pos_zscore = ((norm_pos - norm_pos.rolling(window).mean()) / norm_pos.rolling(window).std()).clip(-1, 1)
        raw = vol_regime * ts_zscore_delta + vol_ratio * norm_pos_zscore
        result = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_086_tanh(df, window=10, sub_window=20):
        close = df['close']
        ret = close.pct_change()
        ret_vol = ret.rolling(sub_window).std()
        vol_ratio = ret_vol / (ret_vol.rolling(window).mean() + 1e-8)
        ts_zscore_delta = ((close.diff(15) - close.diff(15).rolling(window).mean()) / close.diff(15).rolling(window).std()).clip(-1, 1)
        vol_regime = 1 - vol_ratio
        sma_close = close.rolling(15).mean()
        norm_pos = (close - sma_close) / (close.rolling(15).std() + 1e-8)
        norm_pos_zscore = ((norm_pos - norm_pos.rolling(window).mean()) / norm_pos.rolling(window).std()).clip(-1, 1)
        raw = vol_regime * ts_zscore_delta + vol_ratio * norm_pos_zscore
        result = np.tanh(raw / raw.rolling(sub_window).std())
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_086_zscore(df, window=100, sub_window=5):
        close = df['close']
        ret = close.pct_change()
        ret_vol = ret.rolling(sub_window).std()
        vol_ratio = ret_vol / (ret_vol.rolling(window).mean() + 1e-8)
        ts_zscore_delta = ((close.diff(15) - close.diff(15).rolling(window).mean()) / close.diff(15).rolling(window).std()).clip(-1, 1)
        vol_regime = 1 - vol_ratio
        sma_close = close.rolling(15).mean()
        norm_pos = (close - sma_close) / (close.rolling(15).std() + 1e-8)
        norm_pos_zscore = ((norm_pos - norm_pos.rolling(window).mean()) / norm_pos.rolling(window).std()).clip(-1, 1)
        raw = vol_regime * ts_zscore_delta + vol_ratio * norm_pos_zscore
        result = ((raw - raw.rolling(sub_window).mean()) / raw.rolling(sub_window).std()).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_086_sign(df, window=10, sub_window=10):
        close = df['close']
        ret = close.pct_change()
        ret_vol = ret.rolling(sub_window).std()
        vol_ratio = ret_vol / (ret_vol.rolling(window).mean() + 1e-8)
        ts_zscore_delta = ((close.diff(15) - close.diff(15).rolling(window).mean()) / close.diff(15).rolling(window).std()).clip(-1, 1)
        vol_regime = 1 - vol_ratio
        sma_close = close.rolling(15).mean()
        norm_pos = (close - sma_close) / (close.rolling(15).std() + 1e-8)
        norm_pos_zscore = ((norm_pos - norm_pos.rolling(window).mean()) / norm_pos.rolling(window).std()).clip(-1, 1)
        raw = vol_regime * ts_zscore_delta + vol_ratio * norm_pos_zscore
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_086_wf(df, window=50, sub_window=7):
        close = df['close']
        ret = close.pct_change()
        ret_vol = ret.rolling(sub_window).std()
        vol_ratio = ret_vol / (ret_vol.rolling(window).mean() + 1e-8)
        ts_zscore_delta = ((close.diff(15) - close.diff(15).rolling(window).mean()) / close.diff(15).rolling(window).std()).clip(-1, 1)
        vol_regime = 1 - vol_ratio
        sma_close = close.rolling(15).mean()
        norm_pos = (close - sma_close) / (close.rolling(15).std() + 1e-8)
        norm_pos_zscore = ((norm_pos - norm_pos.rolling(window).mean()) / norm_pos.rolling(window).std()).clip(-1, 1)
        raw = vol_regime * ts_zscore_delta + vol_ratio * norm_pos_zscore
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_087_rank(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        return_ = df['close'].pct_change().fillna(0)
        mean_10 = close.rolling(window).mean()
        std_10 = close.rolling(window).std()
        z_score = (close - mean_10) / (std_10 + 1e-8)
        vol_mean_10 = volume.rolling(window).mean()
        vol_ratio = volume / (vol_mean_10 + 1e-8)
        sign_return -= np.sign(return_)
        product = sign_return * vol_ratio
        corr = close.rolling(window).corr(product)
        rank = corr.rolling(window).rank(pct=True) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_087_tanh(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        return_ = df['close'].pct_change().fillna(0)
        mean_10 = close.rolling(window).mean()
        std_10 = close.rolling(window).std()
        z_score = (close - mean_10) / (std_10 + 1e-8)
        vol_mean_10 = volume.rolling(window).mean()
        vol_ratio = volume / (vol_mean_10 + 1e-8)
        sign_return -= np.sign(return_)
        product = sign_return * vol_ratio
        corr = close.rolling(window).corr(product)
        normalized = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_087_zscore(df, window=75):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        return_ = df['close'].pct_change().fillna(0)
        mean_10 = close.rolling(window).mean()
        std_10 = close.rolling(window).std()
        z_score = (close - mean_10) / (std_10 + 1e-8)
        vol_mean_10 = volume.rolling(window).mean()
        vol_ratio = volume / (vol_mean_10 + 1e-8)
        sign_return = np.sign(return_)
        product = sign_return * vol_ratio
        corr = close.rolling(window).corr(product)
        normalized = ((corr - corr.rolling(window).mean()) / corr.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_087_sign(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        return_ = df['close'].pct_change().fillna(0)
        mean_10 = close.rolling(window).mean()
        std_10 = close.rolling(window).std()
        z_score = (close - mean_10) / (std_10 + 1e-8)
        vol_mean_10 = volume.rolling(window).mean()
        vol_ratio = volume / (vol_mean_10 + 1e-8)
        sign_return -= np.sign(return_)
        product = sign_return * vol_ratio
        corr = close.rolling(window).corr(product)
        normalized = np.sign(corr)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_087_wf(df, window=20, sub_window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        return_ = df['close'].pct_change().fillna(0)
        mean_10 = close.rolling(window).mean()
        std_10 = close.rolling(window).std()
        z_score = (close - mean_10) / (std_10 + 1e-8)
        vol_mean_10 = volume.rolling(window).mean()
        vol_ratio = volume / (vol_mean_10 + 1e-8)
        sign_return -= np.sign(return_)
        product = sign_return * vol_ratio
        corr = close.rolling(window).corr(product)
        p1 = 0.05
        low = corr.rolling(sub_window).quantile(p1)
        high = corr.rolling(sub_window).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_088_rank(df, window=15):
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_mean_15 = volume.rolling(window).mean().replace(0, np.nan)
        price_ratio = (close - open_price) / (vol_mean_15 + 1e-8)
        hl_spread = (high - low) / ((high - low).rolling(window).mean() + 1e-8)
        vol_delta = volume.diff(1) / (vol_mean_15 + 1e-8)
        corr_15 = hl_spread.rolling(window).corr(vol_delta)
        raw = price_ratio * corr_15
        raw_rank = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = raw_rank.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_088_tanh(df, window=60):
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_mean_15 = volume.rolling(window).mean().replace(0, np.nan)
        price_ratio = (close - open_price) / (vol_mean_15 + 1e-8)
        hl_spread = (high - low) / ((high - low).rolling(window).mean() + 1e-8)
        vol_delta = volume.diff(1) / (vol_mean_15 + 1e-8)
        corr_15 = hl_spread.rolling(window).corr(vol_delta)
        raw = price_ratio * corr_15
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std_raw)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_088_zscore(df, window=15):
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_mean_15 = volume.rolling(window).mean().replace(0, np.nan)
        price_ratio = (close - open_price) / (vol_mean_15 + 1e-8)
        hl_spread = (high - low) / ((high - low).rolling(window).mean() + 1e-8)
        vol_delta = volume.diff(1) / (vol_mean_15 + 1e-8)
        corr_15 = hl_spread.rolling(window).corr(vol_delta)
        raw = price_ratio * corr_15
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_088_sign(df, window=85):
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_mean_15 = volume.rolling(window).mean().replace(0, np.nan)
        price_ratio = (close - open_price) / (vol_mean_15 + 1e-8)
        hl_spread = (high - low) / ((high - low).rolling(window).mean() + 1e-8)
        vol_delta = volume.diff(1) / (vol_mean_15 + 1e-8)
        corr_15 = hl_spread.rolling(window).corr(vol_delta)
        raw = price_ratio * corr_15
        signal = np.sign(raw)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_088_wf(df, window=90, p1=0.1, p2=30):
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_mean_15 = volume.rolling(window).mean().replace(0, np.nan)
        price_ratio = (close - open_price) / (vol_mean_15 + 1e-8)
        hl_spread = (high - low) / ((high - low).rolling(window).mean() + 1e-8)
        vol_delta = volume.diff(1) / (vol_mean_15 + 1e-8)
        corr_15 = hl_spread.rolling(window).corr(vol_delta)
        raw = price_ratio * corr_15
        low_perc = raw.rolling(p2).quantile(p1)
        high_perc = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_perc, upper=high_perc, axis=0)
        denominator = high_perc - low_perc
        denominator = denominator.replace(0, np.nan)
        normalized = np.arctanh(((winsorized - low_perc) / (denominator + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_089_rank(df, window=80):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sma_close = close.rolling(window).mean()
        std_close = close.rolling(window).std().replace(0, np.nan)
        zscore_close = (close - sma_close) / (std_close + 1e-8)
        sma_z = zscore_close.rolling(window).mean()
        std_z = zscore_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (zscore_close - sma_z) / (std_z + 1e-8)
        ret = close.diff(1) / close.shift(1).replace(0, np.nan)
        sign_ret = np.sign(ret)
        volume_ma = volume.rolling(window).mean().replace(0, np.nan)
        hl_ratio = (high - low) / (volume_ma + 1e-8)
        delta_vol = volume.diff(1) / (volume_ma + 1e-8)
        corr_hl_dv = hl_ratio.rolling(window).corr(delta_vol).replace(0, np.nan)
        raw = ts_zscore * sign_ret * corr_hl_dv
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = rank.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_089_tanh(df, window=60):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sma_close = close.rolling(window).mean()
        std_close = close.rolling(window).std().replace(0, np.nan)
        zscore_close = (close - sma_close) / (std_close + 1e-8)
        sma_z = zscore_close.rolling(window).mean()
        std_z = zscore_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (zscore_close - sma_z) / (std_z + 1e-8)
        ret = close.diff(1) / close.shift(1).replace(0, np.nan)
        sign_ret = np.sign(ret)
        volume_ma = volume.rolling(window).mean().replace(0, np.nan)
        hl_ratio = (high - low) / (volume_ma + 1e-8)
        delta_vol = volume.diff(1) / (volume_ma + 1e-8)
        corr_hl_dv = hl_ratio.rolling(window).corr(delta_vol).replace(0, np.nan)
        raw = ts_zscore * sign_ret * corr_hl_dv
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_089_zscore(df, window=55):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sma_close = close.rolling(window).mean()
        std_close = close.rolling(window).std().replace(0, np.nan)
        zscore_close = (close - sma_close) / (std_close + 1e-8)
        sma_z = zscore_close.rolling(window).mean()
        std_z = zscore_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (zscore_close - sma_z) / (std_z + 1e-8)
        ret = close.diff(1) / close.shift(1).replace(0, np.nan)
        sign_ret = np.sign(ret)
        volume_ma = volume.rolling(window).mean().replace(0, np.nan)
        hl_ratio = (high - low) / (volume_ma + 1e-8)
        delta_vol = volume.diff(1) / (volume_ma + 1e-8)
        corr_hl_dv = hl_ratio.rolling(window).corr(delta_vol).replace(0, np.nan)
        raw = ts_zscore * sign_ret * corr_hl_dv
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std().replace(0, np.nan)
        zscore_raw = (raw - raw_mean) / raw_std
        signal = zscore_raw.clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_089_sign(df, window=50):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sma_close = close.rolling(window).mean()
        std_close = close.rolling(window).std().replace(0, np.nan)
        zscore_close = (close - sma_close) / (std_close + 1e-8)
        sma_z = zscore_close.rolling(window).mean()
        std_z = zscore_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (zscore_close - sma_z) / (std_z + 1e-8)
        ret = close.diff(1) / close.shift(1).replace(0, np.nan)
        sign_ret = np.sign(ret)
        volume_ma = volume.rolling(window).mean().replace(0, np.nan)
        hl_ratio = (high - low) / (volume_ma + 1e-8)
        delta_vol = volume.diff(1) / (volume_ma + 1e-8)
        corr_hl_dv = hl_ratio.rolling(window).corr(delta_vol).replace(0, np.nan)
        raw = ts_zscore * sign_ret * corr_hl_dv
        signal = np.sign(raw)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_089_wf(df, window=70, quantile_range=0.9):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        sma_close = close.rolling(window).mean()
        std_close = close.rolling(window).std().replace(0, np.nan)
        zscore_close = (close - sma_close) / (std_close + 1e-8)
        sma_z = zscore_close.rolling(window).mean()
        std_z = zscore_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (zscore_close - sma_z) / (std_z + 1e-8)
        ret = close.diff(1) / close.shift(1).replace(0, np.nan)
        sign_ret = np.sign(ret)
        volume_ma = volume.rolling(window).mean().replace(0, np.nan)
        hl_ratio = (high - low) / (volume_ma + 1e-8)
        delta_vol = volume.diff(1) / (volume_ma + 1e-8)
        corr_hl_dv = hl_ratio.rolling(window).corr(delta_vol).replace(0, np.nan)
        raw = ts_zscore * sign_ret * corr_hl_dv
        p1 = quantile_range
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_090_rank(df, window=15):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', np.ones(len(df))))
        raw = close.rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal

    @staticmethod
    def alpha_quanta_full_base_090_tanh(df, window=35):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', np.ones(len(df))))
        raw = close.rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_090_zscore(df, window=15):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', np.ones(len(df))))
        raw = close.rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan).ffill()
        signal = ((raw - mean_) / std_).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_090_sign(df, window=25):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', np.ones(len(df))))
        raw = close.rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        signal = np.sign(raw - raw.rolling(window).median())
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_090_wf(df, window=10, p1=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', np.ones(len(df))))
        raw = close.rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_091_rank(df, window=100):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change().fillna(0)
        vol_ret = volume.pct_change().fillna(0)
        raw = ret.rolling(window).corr(vol_ret)
        sign = np.sign(raw)
        avg_ret = ret.rolling(5).mean()
        alpha = sign * avg_ret
        normalized = alpha.rolling(window).rank(pct=True) * 2 - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_091_tanh(df, window=50):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change().fillna(0)
        vol_ret = volume.pct_change().fillna(0)
        raw = ret.rolling(window).corr(vol_ret)
        sign = np.sign(raw)
        avg_ret = ret.rolling(5).mean()
        alpha = sign * avg_ret
        normalized = np.tanh(alpha / alpha.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_091_zscore(df, window=50):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change().fillna(0)
        vol_ret = volume.pct_change().fillna(0)
        raw = ret.rolling(window).corr(vol_ret)
        sign = np.sign(raw)
        avg_ret = ret.rolling(5).mean()
        alpha = sign * avg_ret
        mean = alpha.rolling(window).mean()
        std = alpha.rolling(window).std().replace(0, np.nan)
        normalized = ((alpha - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_091_sign(df, window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change().fillna(0)
        vol_ret = volume.pct_change().fillna(0)
        raw = ret.rolling(window).corr(vol_ret)
        sign = np.sign(raw)
        avg_ret = ret.rolling(5).mean()
        alpha = sign * avg_ret
        normalized = np.sign(alpha)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_091_wf(df, window=100):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change().fillna(0)
        vol_ret = volume.pct_change().fillna(0)
        raw = ret.rolling(window).corr(vol_ret)
        sign = np.sign(raw)
        avg_ret = ret.rolling(5).mean()
        alpha = sign * avg_ret
        p1 = 0.05
        p2 = window
        low = alpha.rolling(p2).quantile(p1)
        high = alpha.rolling(p2).quantile(1 - p1)
        winsorized = alpha.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_092_rank(df, window=60):
        raw = df['close'] / (df['matchingVolume'] + 1e-8)
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        inv_ratio = 1.0 / ((std / (mean + 1e-8)) + 1e-8)
        normalized = (inv_ratio.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_092_tanh(df, window=30):
        raw = df['close'] / (df['matchingVolume'] + 1e-8)
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        inv_ratio = 1.0 / ((std / (mean + 1e-8)) + 1e-8)
        normalized = np.tanh(inv_ratio / inv_ratio.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_092_zscore(df, window=85):
        raw = df['close'] / (df['matchingVolume'] + 1e-8)
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        inv_ratio = 1.0 / ((std / (mean + 1e-8)) + 1e-8)
        normalized = ((inv_ratio - inv_ratio.rolling(window).mean()) / inv_ratio.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_092_sign(df, window=30):
        raw = df['close'] / (df['matchingVolume'] + 1e-8)
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        inv_ratio = 1.0 / ((std / (mean + 1e-8)) + 1e-8)
        normalized = np.sign(inv_ratio)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_092_wf(df, window=80, p1=0.1):
        raw = df['close'] / (df['matchingVolume'] + 1e-8)
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        inv_ratio = 1.0 / ((std / (mean + 1e-8)) + 1e-8)
        low = inv_ratio.rolling(window).quantile(p1)
        high = inv_ratio.rolling(window).quantile(1 - p1)
        winsorized = inv_ratio.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_093_rank(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).sum()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_093_tanh(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).sum()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_093_zscore(df, window=25):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).sum()
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_093_sign(df, window=30):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).sum()
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_093_wf(df, window=40, p1=0.3):
        p2 = window
        ret = df['close'].pct_change()
        raw = ret.rolling(window).sum()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_094_rank(df, window=90):
        volume = df['matchingVolume']
        delta_vol = volume.diff(5)
        mean_vol = volume.rolling(40).mean()
        raw = delta_vol / (mean_vol + 1e-8)
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_094_tanh(df, window=10):
        volume = df['matchingVolume']
        delta_vol = volume.diff(5)
        mean_vol = volume.rolling(40).mean()
        raw = delta_vol / (mean_vol + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_094_zscore(df, window=80):
        volume = df['matchingVolume']
        delta_vol = volume.diff(5)
        mean_vol = volume.rolling(40).mean()
        raw = delta_vol / (mean_vol + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_094_sign(df):
        volume = df['matchingVolume']
        delta_vol = volume.diff(5)
        mean_vol = volume.rolling(40).mean()
        raw = delta_vol / (mean_vol + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_094_wf(df, window=100, quantile=0.1):
        volume = df['matchingVolume']
        delta_vol = volume.diff(5)
        mean_vol = volume.rolling(40).mean()
        raw = delta_vol / (mean_vol + 1e-8)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_095_rank(df, window=85):
        raw = (df['high'] - df['low']).rolling(window).corr(df['close'].pct_change())
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_095_tanh(df, window=45):
        raw = (df['high'] - df['low']).rolling(window).corr(df['close'].pct_change())
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_095_zscore(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).corr(df['close'].pct_change())
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_095_sign(df, window=75):
        raw = (df['high'] - df['low']).rolling(window).corr(df['close'].pct_change())
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_095_wf(df, window=80, p1=0.1, p2=60):
        raw = (df['high'] - df['low']).rolling(window).corr(df['close'].pct_change())
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_096_rank(df, window=15):
        ret = df['close'].pct_change()
        pos_ratio = ret.gt(0).rolling(window).sum() / window
        positive_mean = ret.clip(0).rolling(window).mean()
        vol = ret.rolling(window).std().replace(0, np.nan)
        raw = pos_ratio * (positive_mean / (vol + 1e-8))
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_096_tanh(df, window=5):
        ret = df['close'].pct_change()
        pos_ratio = ret.gt(0).rolling(window).sum() / window
        positive_mean = ret.clip(0).rolling(window).mean()
        vol = ret.rolling(window).std().replace(0, np.nan)
        raw = pos_ratio * (positive_mean / (vol + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_096_zscore(df, window=15):
        ret = df['close'].pct_change()
        pos_ratio = ret.gt(0).rolling(window).sum() / window
        positive_mean = ret.clip(0).rolling(window).mean()
        vol = ret.rolling(window).std().replace(0, np.nan)
        raw = pos_ratio * (positive_mean / (vol + 1e-8))
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_096_sign(df, window=55):
        ret = df['close'].pct_change()
        pos_ratio = ret.gt(0).rolling(window).sum() / window
        positive_mean = ret.clip(0).rolling(window).mean()
        vol = ret.rolling(window).std().replace(0, np.nan)
        raw = pos_ratio * (positive_mean / (vol + 1e-8))
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_096_wf(df, window=30, p1=0.1, p2=40):
        ret = df['close'].pct_change()
        pos_ratio = ret.gt(0).rolling(window).sum() / window
        positive_mean = ret.clip(0).rolling(window).mean()
        vol = ret.rolling(window).std().replace(0, np.nan)
        raw = pos_ratio * (positive_mean / (vol + 1e-8))
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.sub(normalized.rolling(window).mean()).div(normalized.rolling(window).std().replace(0, np.nan)).clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_097_rank(df, window=50):
        close_delta = df['close'].diff(1).fillna(0)
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1).fillna(0)
        abs_close_delta = close_delta.abs()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = abs_close_delta.rolling(window).corr(volume).fillna(0)
        sign_close = np.sign(close_delta)
        sign_volume = np.sign(volume_delta)
        interaction = pd.Series(sign_close * sign_volume, index=df.index).fillna(0)
        mean_interaction = interaction.rolling(window).mean().fillna(0)
        raw = corr * mean_interaction
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_097_tanh(df, window=80):
        close_delta = df['close'].diff(1).fillna(0)
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1).fillna(0)
        abs_close_delta = close_delta.abs()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = abs_close_delta.rolling(window).corr(volume).fillna(0)
        sign_close = np.sign(close_delta)
        sign_volume = np.sign(volume_delta)
        interaction = pd.Series(sign_close * sign_volume, index=df.index).fillna(0)
        mean_interaction = interaction.rolling(window).mean().fillna(0)
        raw = corr * mean_interaction
        std_raw = raw.rolling(window).std().replace(0, np.nan).fillna(1e-9)
        normalized = np.tanh(raw / std_raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_097_zscore(df, window=15):
        close_delta = df['close'].diff(1).fillna(0)
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1).fillna(0)
        abs_close_delta = close_delta.abs()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = abs_close_delta.rolling(window).corr(volume).fillna(0)
        sign_close = np.sign(close_delta)
        sign_volume = np.sign(volume_delta)
        interaction = pd.Series(sign_close * sign_volume, index=df.index).fillna(0)
        mean_interaction = interaction.rolling(window).mean().fillna(0)
        raw = corr * mean_interaction
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan).fillna(1e-9)
        zscore = (raw - mean_raw) / std_raw
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_097_sign(df, window=75):
        close_delta = df['close'].diff(1).fillna(0)
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1).fillna(0)
        abs_close_delta = close_delta.abs()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = abs_close_delta.rolling(window).corr(volume).fillna(0)
        sign_close = np.sign(close_delta)
        sign_volume = np.sign(volume_delta)
        interaction = pd.Series(sign_close * sign_volume, index=df.index).fillna(0)
        mean_interaction = interaction.rolling(window).mean().fillna(0)
        raw = corr * mean_interaction
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_097_wf(df, window=40, p1=0.1):
        close_delta = df['close'].diff(1).fillna(0)
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1).fillna(0)
        abs_close_delta = close_delta.abs()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = abs_close_delta.rolling(window).corr(volume).fillna(0)
        sign_close = np.sign(close_delta)
        sign_volume = np.sign(volume_delta)
        interaction = pd.Series(sign_close * sign_volume, index=df.index).fillna(0)
        mean_interaction = interaction.rolling(window).mean().fillna(0)
        raw = corr * mean_interaction
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_098_rank(df, window=75):
        raw = df['close'] - df['open']
        high_low = df['high'] - df['low'] + 1e-8
        mean_abs_ratio = (raw.abs() / high_low).rolling(window).mean()
        std_open_ratio = (raw / (df['open'] + 1e-8)).rolling(window).std() + 1e-8
        signal = mean_abs_ratio / std_open_ratio
        signal = signal.ffill().fillna(0)
        normalized = (signal.rolling(window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_full_base_098_tanh(df, window=45):
        raw = df['close'] - df['open']
        high_low = df['high'] - df['low'] + 1e-8
        mean_abs_ratio = (raw.abs() / high_low).rolling(window).mean()
        std_open_ratio = (raw / (df['open'] + 1e-8)).rolling(window).std() + 1e-8
        signal = mean_abs_ratio / std_open_ratio
        signal = signal.ffill().fillna(0)
        normalized = np.tanh(signal / signal.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8))
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_098_zscore(df, window=30):
        raw = df['close'] - df['open']
        high_low = df['high'] - df['low'] + 1e-8
        mean_abs_ratio = (raw.abs() / high_low).rolling(window).mean()
        std_open_ratio = (raw / (df['open'] + 1e-8)).rolling(window).std() + 1e-8
        signal = mean_abs_ratio / std_open_ratio
        signal = signal.ffill().fillna(0)
        normalized = ((signal - signal.rolling(window).mean()) / signal.rolling(window).std()).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_098_sign(df, window=75):
        raw = df['close'] - df['open']
        high_low = df['high'] - df['low'] + 1e-8
        mean_abs_ratio = (raw.abs() / high_low).rolling(window).mean()
        std_open_ratio = (raw / (df['open'] + 1e-8)).rolling(window).std() + 1e-8
        signal = mean_abs_ratio / std_open_ratio
        signal = signal.ffill().fillna(0)
        normalized = np.sign(signal)
        return pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_098_wf(df, window=7, p2=70):
        raw = df['close'] - df['open']
        high_low = df['high'] - df['low'] + 1e-8
        mean_abs_ratio = (raw.abs() / high_low).rolling(window).mean()
        std_open_ratio = (raw / (df['open'] + 1e-8)).rolling(window).std() + 1e-8
        signal = mean_abs_ratio / std_open_ratio
        signal = signal.ffill().fillna(0)
        p1 = 0.05
        low = signal.rolling(p2).quantile(p1)
        high = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_099_rank(df, window=100):
        close = df['close']
        ts_mean_60 = close.rolling(window=60).mean()
        ts_rank_60 = (close / (ts_mean_60 + 1e-8)).rolling(window=60).rank(pct=True)
        ts_mean_5 = close.rolling(window=5).mean()
        ts_std_20 = close.rolling(window=20).std()
        ts_mean_60_2 = ((close - ts_mean_5) / (ts_std_20 + 1e-8)).rolling(window=60).mean()
        raw = ts_rank_60 * ts_mean_60_2
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_099_tanh(df, window=10):
        close = df['close']
        ts_mean_60 = close.rolling(window=60).mean()
        ts_rank_60 = (close / (ts_mean_60 + 1e-8)).rolling(window=60).rank(pct=True)
        ts_mean_5 = close.rolling(window=5).mean()
        ts_std_20 = close.rolling(window=20).std()
        ts_mean_60_2 = ((close - ts_mean_5) / (ts_std_20 + 1e-8)).rolling(window=60).mean()
        raw = ts_rank_60 * ts_mean_60_2
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_099_zscore(df, window=5):
        close = df['close']
        ts_mean_60 = close.rolling(window=60).mean()
        ts_rank_60 = (close / (ts_mean_60 + 1e-8)).rolling(window=60).rank(pct=True)
        ts_mean_5 = close.rolling(window=5).mean()
        ts_std_20 = close.rolling(window=20).std()
        ts_mean_60_2 = ((close - ts_mean_5) / (ts_std_20 + 1e-8)).rolling(window=60).mean()
        raw = ts_rank_60 * ts_mean_60_2
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_099_sign(df):
        close = df['close']
        ts_mean_60 = close.rolling(window=60).mean()
        ts_rank_60 = (close / (ts_mean_60 + 1e-8)).rolling(window=60).rank(pct=True)
        ts_mean_5 = close.rolling(window=5).mean()
        ts_std_20 = close.rolling(window=20).std()
        ts_mean_60_2 = ((close - ts_mean_5) / (ts_std_20 + 1e-8)).rolling(window=60).mean()
        raw = ts_rank_60 * ts_mean_60_2
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_099_wf(df, p1=0.3, p2=100):
        close = df['close']
        ts_mean_60 = close.rolling(window=60).mean()
        ts_rank_60 = (close / (ts_mean_60 + 1e-8)).rolling(window=60).rank(pct=True)
        ts_mean_5 = close.rolling(window=5).mean()
        ts_std_20 = close.rolling(window=20).std()
        ts_mean_60_2 = ((close - ts_mean_5) / (ts_std_20 + 1e-8)).rolling(window=60).mean()
        raw = ts_rank_60 * ts_mean_60_2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_100_rank(df: pd.DataFrame, window: int = 40, sub_window: int = 20) -> pd.Series:
        close = df['close']
        ret = close.pct_change()
        ts_mean_close = close.rolling(window).mean()
        raw_ratio = close / (ts_mean_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw_ratio.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        regressi = slope * sub_window
        mean_ret_20 = ret.rolling(sub_window).mean()
        sign = np.sign(mean_ret_20 - ret.rolling(window).std())
        raw = regressi * sign
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_100_tanh(df: pd.DataFrame, window: int = 40, sub_window: int = 20) -> pd.Series:
        close = df['close']
        ret = close.pct_change()
        ts_mean_close = close.rolling(window).mean()
        raw_ratio = close / (ts_mean_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw_ratio.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        regressi = slope * sub_window
        mean_ret_20 = ret.rolling(sub_window).mean()
        sign = np.sign(mean_ret_20 - ret.rolling(window).std())
        raw = regressi * sign
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_100_zscore(df: pd.DataFrame, window: int = 40, sub_window: int = 20) -> pd.Series:
        close = df['close']
        ret = close.pct_change()
        ts_mean_close = close.rolling(window).mean()
        raw_ratio = close / (ts_mean_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw_ratio.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        regressi = slope * sub_window
        mean_ret_20 = ret.rolling(sub_window).mean()
        sign = np.sign(mean_ret_20 - ret.rolling(window).std())
        raw = regressi * sign
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_100_sign(df: pd.DataFrame, window: int = 40, sub_window: int = 20) -> pd.Series:
        close = df['close']
        ret = close.pct_change()
        ts_mean_close = close.rolling(window).mean()
        raw_ratio = close / (ts_mean_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw_ratio.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        regressi = slope * sub_window
        mean_ret_20 = ret.rolling(sub_window).mean()
        sign = np.sign(mean_ret_20 - ret.rolling(window).std())
        raw = regressi * sign
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_100_wf(df: pd.DataFrame, window: int = 40, sub_window: int = 20, p1: float = 0.05, p2: int = 60) -> pd.Series:
        close = df['close']
        ret = close.pct_change()
        ts_mean_close = close.rolling(window).mean()
        raw_ratio = close / (ts_mean_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw_ratio.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        regressi = slope * sub_window
        mean_ret_20 = ret.rolling(sub_window).mean()
        sign = np.sign(mean_ret_20 - ret.rolling(window).std())
        raw = regressi * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_101_k(df, window=40):
        close = df['close']
        volume = df['matchingVolume']
        mean_close_50 = close.rolling(window).mean().replace(0, np.nan)
        mean_close_10 = close.rolling(10).mean().replace(0, np.nan)
        mean_volume_10 = volume.rolling(10).mean().replace(0, np.nan)
        raw = (1 - (close / mean_close_50).rolling(window).quantile(0.5)) * (close / mean_close_10).rolling(20).corr(volume / mean_volume_10).rank(pct=True)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_101_h(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        mean_close_50 = close.rolling(window).mean().replace(0, np.nan)
        mean_close_10 = close.rolling(10).mean().replace(0, np.nan)
        mean_volume_10 = volume.rolling(10).mean().replace(0, np.nan)
        raw = (1 - (close / mean_close_50).rolling(window).quantile(0.5)) * (close / mean_close_10).rolling(20).corr(volume / mean_volume_10).rank(pct=True)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_101_e(df, window=25):
        close = df['close']
        volume = df['matchingVolume']
        mean_close_50 = close.rolling(window).mean().replace(0, np.nan)
        mean_close_10 = close.rolling(10).mean().replace(0, np.nan)
        mean_volume_10 = volume.rolling(10).mean().replace(0, np.nan)
        raw = (1 - (close / mean_close_50).rolling(window).quantile(0.5)) * (close / mean_close_10).rolling(20).corr(volume / mean_volume_10).rank(pct=True)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_101_y(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        mean_close_50 = close.rolling(window).mean().replace(0, np.nan)
        mean_close_10 = close.rolling(10).mean().replace(0, np.nan)
        mean_volume_10 = volume.rolling(10).mean().replace(0, np.nan)
        raw = (1 - (close / mean_close_50).rolling(window).quantile(0.5)) * (close / mean_close_10).rolling(20).corr(volume / mean_volume_10).rank(pct=True)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_101_r(df, window=20, p1=0.3):
        close = df['close']
        volume = df['matchingVolume']
        mean_close_50 = close.rolling(window).mean().replace(0, np.nan)
        mean_close_10 = close.rolling(10).mean().replace(0, np.nan)
        mean_volume_10 = volume.rolling(10).mean().replace(0, np.nan)
        raw = (1 - (close / mean_close_50).rolling(window).quantile(0.5)) * (close / mean_close_10).rolling(20).corr(volume / mean_volume_10).rank(pct=True)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_102_rank(df, window=20):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff(1)
        volume_sign = np.sign(volume_delta)
        rolling_corr = high_low.rolling(window).corr(volume_sign)
        result = (rolling_corr.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_102_tanh(df, window=5):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff(1)
        volume_sign = np.sign(volume_delta)
        rolling_corr = high_low.rolling(window).corr(volume_sign)
        std = rolling_corr.rolling(window).std()
        result = np.tanh(rolling_corr / std.replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_102_zscore(df, window=15):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff(1)
        volume_sign = np.sign(volume_delta)
        rolling_corr = high_low.rolling(window).corr(volume_sign)
        mean = rolling_corr.rolling(window).mean()
        std = rolling_corr.rolling(window).std()
        result = ((rolling_corr - mean) / std.replace(0, np.nan)).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_102_sign(df, window=5):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff(1)
        volume_sign = np.sign(volume_delta)
        rolling_corr = high_low.rolling(window).corr(volume_sign)
        result = np.sign(rolling_corr)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_102_wf(df, window_rank=50, p1=0.1):
        p2 = window_rank
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff(1)
        volume_sign = np.sign(volume_delta)
        rolling_corr = high_low.rolling(window_rank).corr(volume_sign)
        low = rolling_corr.rolling(p2).quantile(p1)
        high = rolling_corr.rolling(p2).quantile(1 - p1)
        winsorized = rolling_corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_103_zscore(df, window=60):
        # Raw signal: rolling correlation between (high-low)/mean(close) and zscore(volume)
        epsilon = 1e-8
        hl_range = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + epsilon)
        volume_ma = df['matchingVolume'].rolling(window).mean()
        volume_std = df['matchingVolume'].rolling(window).std()
        volume_z = (df['matchingVolume'] - volume_ma) / (volume_std + epsilon)
        # Rolling correlation between two series
        x = hl_range
        y = volume_z
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window).sum()
        std_x = ((x - x_mean) ** 2).rolling(window).sum() ** 0.5
        std_y = ((y - y_mean) ** 2).rolling(window).sum() ** 0.5
        raw = cov / (std_x * std_y + epsilon)
        # Normalization using Method C: Rolling Z-Score/Clip
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_103_rank(df, window=85):
        epsilon = 1e-8
        hl_range = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + epsilon)
        volume_ma = df['matchingVolume'].rolling(window).mean()
        volume_std = df['matchingVolume'].rolling(window).std()
        volume_z = (df['matchingVolume'] - volume_ma) / (volume_std + epsilon)
        x = hl_range
        y = volume_z
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window).sum()
        std_x = ((x - x_mean) ** 2).rolling(window).sum() ** 0.5
        std_y = ((y - y_mean) ** 2).rolling(window).sum() ** 0.5
        raw = cov / (std_x * std_y + epsilon)
        # Normalization using Method A: Rolling Rank (uniform, robust)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_103_tanh(df, window=45):
        epsilon = 1e-8
        hl_range = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + epsilon)
        volume_ma = df['matchingVolume'].rolling(window).mean()
        volume_std = df['matchingVolume'].rolling(window).std()
        volume_z = (df['matchingVolume'] - volume_ma) / (volume_std + epsilon)
        x = hl_range
        y = volume_z
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window).sum()
        std_x = ((x - x_mean) ** 2).rolling(window).sum() ** 0.5
        std_y = ((y - y_mean) ** 2).rolling(window).sum() ** 0.5
        raw = cov / (std_x * std_y + epsilon)
        # Normalization using Method B: Dynamic Tanh (preserve magnitude)
        std_raw = raw.rolling(window).std()
        normalized = np.tanh(raw / (std_raw + epsilon))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_103_sign(df, window=60):
        epsilon = 1e-8
        hl_range = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + epsilon)
        volume_ma = df['matchingVolume'].rolling(window).mean()
        volume_std = df['matchingVolume'].rolling(window).std()
        volume_z = (df['matchingVolume'] - volume_ma) / (volume_std + epsilon)
        x = hl_range
        y = volume_z
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window).sum()
        std_x = ((x - x_mean) ** 2).rolling(window).sum() ** 0.5
        std_y = ((y - y_mean) ** 2).rolling(window).sum() ** 0.5
        raw = cov / (std_x * std_y + epsilon)
        # Normalization using Method D: Sign/Binary Soft for extreme correlation
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_103_wf(df, window=90, p1=0.9):
        epsilon = 1e-8
        hl_range = (df['high'] - df['low']) / (df['close'].rolling(window).mean() + epsilon)
        volume_ma = df['matchingVolume'].rolling(window).mean()
        volume_std = df['matchingVolume'].rolling(window).std()
        volume_z = (df['matchingVolume'] - volume_ma) / (volume_std + epsilon)
        x = hl_range
        y = volume_z
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window).sum()
        std_x = ((x - x_mean) ** 2).rolling(window).sum() ** 0.5
        std_y = ((y - y_mean) ** 2).rolling(window).sum() ** 0.5
        raw = cov / (std_x * std_y + epsilon)
        # Normalization using Method E: Winsorized Fisher Transport
        p2 = 2 * window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + epsilon)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_104_rank(df, window_rank=15):
        log_range = np.log(df['high'] - df['low'] + 1e-8)
        log_vol_ratio = np.log(df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8))
        corr = log_range.rolling(window_rank).corr(log_vol_ratio)
        raw = corr
        normalized = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_104_tanh(df, window_tanh=10):
        log_range = np.log(df['high'] - df['low'] + 1e-8)
        log_vol_ratio = np.log(df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8))
        corr = log_range.rolling(window_tanh).corr(log_vol_ratio)
        raw = corr
        normalized = np.tanh(raw / raw.rolling(window_tanh).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_104_zscore(df, window_z=20):
        log_range = np.log(df['high'] - df['low'] + 1e-8)
        log_vol_ratio = np.log(df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8))
        corr = log_range.rolling(window_z).corr(log_vol_ratio)
        raw = corr
        mean_ = raw.rolling(window_z).mean()
        std_ = raw.rolling(window_z).std().replace(0, np.nan)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_104_sign(df, window_sign=5):
        log_range = np.log(df['high'] - df['low'] + 1e-8)
        log_vol_ratio = np.log(df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8))
        corr = log_range.rolling(window_sign).corr(log_vol_ratio)
        raw = corr
        normalized = np.sign(raw)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_104_wf(df, p1=0.1, p2=20):
        log_range = np.log(df['high'] - df['low'] + 1e-8)
        log_vol_ratio = np.log(df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8))
        corr = log_range.rolling(p2).corr(log_vol_ratio)
        raw = corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_105_rank(df, window=25):
        close = df['close']
        pct_5 = close.pct_change(5)
        raw = np.sign(pct_5 - pct_5.rolling(window).mean()) * pct_5.abs()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_105_tanh(df, window=25):
        close = df['close']
        pct_5 = close.pct_change(5)
        raw = np.sign(pct_5 - pct_5.rolling(window).mean()) * pct_5.abs()
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_105_zscore(df, window=5):
        close = df['close']
        pct_5 = close.pct_change(5)
        raw = np.sign(pct_5 - pct_5.rolling(window).mean()) * pct_5.abs()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_105_sign(df, window=5):
        close = df['close']
        pct_5 = close.pct_change(5)
        raw = np.sign(pct_5 - pct_5.rolling(window).mean()) * pct_5.abs()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_105_wf(df, window=40, p1=0.1):
        close = df['close']
        pct_5 = close.pct_change(5)
        raw = np.sign(pct_5 - pct_5.rolling(window).mean()) * pct_5.abs()
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_106_rank(df, window=95):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = delta_volume.rolling(window).corr(seq)
        ret = df['close'].pct_change()
        return_mean = ret.rolling(window).mean()
        raw = corr - return_mean
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_106_tanh(df, window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = delta_volume.rolling(window).corr(seq)
        ret = df['close'].pct_change()
        return_mean = ret.rolling(window).mean()
        raw = corr - return_mean
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_106_zscore(df, window=75):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = delta_volume.rolling(window).corr(seq)
        ret = df['close'].pct_change()
        return_mean = ret.rolling(window).mean()
        raw = corr - return_mean
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_106_sign(df, window=70):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = delta_volume.rolling(window).corr(seq)
        ret = df['close'].pct_change()
        return_mean = ret.rolling(window).mean()
        raw = corr - return_mean
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_106_wf(df, window=10, p1=0.1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = delta_volume.rolling(window).corr(seq)
        ret = df['close'].pct_change()
        return_mean = ret.rolling(window).mean()
        raw = corr - return_mean
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_107_rank(df, window=20):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(15).std() + 1e-8)
        x = pd.Series(np.arange(1, 16), index=df.index[-15:])
        corr = df['matchingVolume'].rolling(15).corr(x)
        raw = raw * corr
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_107_tanh(df, window=70):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(15).std() + 1e-8)
        x = pd.Series(np.arange(1, 16), index=df.index[-15:])
        corr = df['matchingVolume'].rolling(15).corr(x)
        raw = raw * corr
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_107_zscore(df, window=35):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(15).std() + 1e-8)
        x = pd.Series(np.arange(1, 16), index=df.index[-15:])
        corr = df['matchingVolume'].rolling(15).corr(x)
        raw = raw * corr
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_107_sign(df):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(15).std() + 1e-8)
        x = pd.Series(np.arange(1, 16), index=df.index[-15:])
        corr = df['matchingVolume'].rolling(15).corr(x)
        raw = raw * corr
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_107_wf(df, p1=0.1, p2=30):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(15).std() + 1e-8)
        x = pd.Series(np.arange(1, 16), index=df.index[-15:])
        corr = df['matchingVolume'].rolling(15).corr(x)
        raw = raw * corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_108_rank(df, window=10):
        ret = df['close'].pct_change().fillna(0)
        vol_delta = df['close'].diff() / (df['close'].shift(1) + 1e-8)
        vol_delta = vol_delta.diff() / (df.get('matchingVolume', df['close'] * 1).diff().abs().fillna(0) + 1e-8)
        # Handle missing volume: use volume from df if available, else use 0
        volume = df.get('matchingVolume', df.get('volume', pd.Series(0, index=df.index)))
        delta_vol = volume.diff().fillna(0) / (volume.shift(1).fillna(0) + 1e-8)
        # Ensure same length
        min_len = min(len(ret), len(delta_vol))
        ret = ret.iloc[-min_len:]
        delta_vol = delta_vol.iloc[-min_len:]
        # Calculate rolling correlation
        rolling_corr = ret.rolling(window).corr(delta_vol).fillna(0)
        # Calculate rolling std
        std_ret = ret.rolling(window).std().fillna(0)
        # Calculate mean of rolling std over longer window
        mean_std_ret = std_ret.rolling(window*3).mean().fillna(0)
        # Combine
        raw = rolling_corr * std_ret / (mean_std_ret + 1e-8)
        # Normalization - Rolling Rank
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_108_tanh(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        volume = df.get('matchingVolume', df.get('volume', pd.Series(0, index=df.index)))
        delta_vol = volume.diff().fillna(0) / (volume.shift(1).fillna(0) + 1e-8)
        min_len = min(len(ret), len(delta_vol))
        ret = ret.iloc[-min_len:]
        delta_vol = delta_vol.iloc[-min_len:]
        rolling_corr = ret.rolling(window).corr(delta_vol).fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        mean_std_ret = std_ret.rolling(window*3).mean().fillna(0)
        raw = rolling_corr * std_ret / (mean_std_ret + 1e-8)
        # Normalization - Dynamic Tanh
        raw_std = raw.rolling(window).std().fillna(1e-8)
        signal = np.tanh(raw / raw_std)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_108_zscore(df, window=10):
        ret = df['close'].pct_change().fillna(0)
        volume = df.get('matchingVolume', df.get('volume', pd.Series(0, index=df.index)))
        delta_vol = volume.diff().fillna(0) / (volume.shift(1).fillna(0) + 1e-8)
        min_len = min(len(ret), len(delta_vol))
        ret = ret.iloc[-min_len:]
        delta_vol = delta_vol.iloc[-min_len:]
        rolling_corr = ret.rolling(window).corr(delta_vol).fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        mean_std_ret = std_ret.rolling(window*3).mean().fillna(0)
        raw = rolling_corr * std_ret / (mean_std_ret + 1e-8)
        # Normalization - Rolling Z-Score Clip
        mean_raw = raw.rolling(window).mean().fillna(0)
        std_raw = raw.rolling(window).std().fillna(1e-8)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_108_sign(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        volume = df.get('matchingVolume', df.get('volume', pd.Series(0, index=df.index)))
        delta_vol = volume.diff().fillna(0) / (volume.shift(1).fillna(0) + 1e-8)
        min_len = min(len(ret), len(delta_vol))
        ret = ret.iloc[-min_len:]
        delta_vol = delta_vol.iloc[-min_len:]
        rolling_corr = ret.rolling(window).corr(delta_vol).fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        mean_std_ret = std_ret.rolling(window*3).mean().fillna(0)
        raw = rolling_corr * std_ret / (mean_std_ret + 1e-8)
        # Normalization - Sign/Binary Soft
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index[-len(signal):]).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_108_wf(df, window=5, p2=40, p1=0.05):
        ret = df['close'].pct_change().fillna(0)
        volume = df.get('matchingVolume', df.get('volume', pd.Series(0, index=df.index)))
        delta_vol = volume.diff().fillna(0) / (volume.shift(1).fillna(0) + 1e-8)
        min_len = min(len(ret), len(delta_vol))
        ret = ret.iloc[-min_len:]
        delta_vol = delta_vol.iloc[-min_len:]
        rolling_corr = ret.rolling(window).corr(delta_vol).fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        mean_std_ret = std_ret.rolling(window*3).mean().fillna(0)
        raw = rolling_corr * std_ret / (mean_std_ret + 1e-8)
        # Winsorized Fisher Transform
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1-p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Avoid division by zero
        denom = (high - low + 1e-9)
        # Normalize to [-0.99, 0.99] safely
        ratio = ((winsorized - low) / denom) * 1.98 - 0.99
        ratio = ratio.clip(-0.99, 0.99)
        signal = np.arctanh(ratio)
        signal = pd.Series(signal, index=raw.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_109_k(df, window_rank=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ema_close = close.ewm(span=window_rank, adjust=False).mean()
        std_close = close.rolling(window_rank).std()
        raw = (close - ema_close) / (std_close + 1e-8)
        vol_delta = volume.diff(1)
        sign_vol = np.sign(vol_delta / (volume + 1e-8))
        signal = raw * sign_vol
        normalized = (signal.rolling(window_rank).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_109_h(df, window_tanh=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ema_close = close.ewm(span=window_tanh, adjust=False).mean()
        std_close = close.rolling(window_tanh).std()
        raw = (close - ema_close) / (std_close + 1e-8)
        vol_delta = volume.diff(1)
        sign_vol = np.sign(vol_delta / (volume + 1e-8))
        signal = raw * sign_vol
        normalized = np.tanh(signal / signal.rolling(window_tanh).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_109_p(df, window_z=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ema_close = close.ewm(span=window_z, adjust=False).mean()
        std_close = close.rolling(window_z).std()
        raw = (close - ema_close) / (std_close + 1e-8)
        vol_delta = volume.diff(1)
        sign_vol = np.sign(vol_delta / (volume + 1e-8))
        signal = raw * sign_vol
        normalized = ((signal - signal.rolling(window_z).mean()) / signal.rolling(window_z).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_109_t(df, window_sign=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ema_close = close.ewm(span=window_sign, adjust=False).mean()
        std_close = close.rolling(window_sign).std()
        raw = (close - ema_close) / (std_close + 1e-8)
        vol_delta = volume.diff(1)
        sign_vol = np.sign(vol_delta / (volume + 1e-8))
        signal = raw * sign_vol
        normalized = np.sign(signal)
        return pd.Series(normalized, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_109_r(df, p1=0.1, p2=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ema_close = close.ewm(span=p2, adjust=False).mean()
        std_close = close.rolling(p2).std()
        raw = (close - ema_close) / (std_close + 1e-8)
        vol_delta = volume.diff(1)
        sign_vol = np.sign(vol_delta / (volume + 1e-8))
        signal = raw * sign_vol
        low = signal.rolling(p2).quantile(p1)
        high = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_110_rank(df, window_rank=45):
        close = df['close']
        low = df['low']
        high = df['high']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = ((rsi / 50) - 1) * ((close - low.rolling(window=5).mean()) / (high - low + 1e-8))
        signal = raw.rolling(window_rank).rank(pct=True) * 2 - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_110_tanh(df, window_std=40):
        close = df['close']
        low = df['low']
        high = df['high']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = ((rsi / 50) - 1) * ((close - low.rolling(window=5).mean()) / (high - low + 1e-8))
        signal = np.tanh(raw / raw.rolling(window_std).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_110_zscore(df, window_zscore=70):
        close = df['close']
        low = df['low']
        high = df['high']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = ((rsi / 50) - 1) * ((close - low.rolling(window=5).mean()) / (high - low + 1e-8))
        mean = raw.rolling(window_zscore).mean()
        std = raw.rolling(window_zscore).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_110_sign(df):
        close = df['close']
        low = df['low']
        high = df['high']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = ((rsi / 50) - 1) * ((close - low.rolling(window=5).mean()) / (high - low + 1e-8))
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_110_wf(df, window_quantile=40, p_value=0.1):
        close = df['close']
        low = df['low']
        high = df['high']
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=10).mean()
        avg_loss = loss.rolling(window=10).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        raw = ((rsi / 50) - 1) * ((close - low.rolling(window=5).mean()) / (high - low + 1e-8))
        low_quant = raw.rolling(window_quantile).quantile(p_value)
        high_quant = raw.rolling(window_quantile).quantile(1 - p_value)
        winsorized = raw.clip(lower=low_quant, upper=high_quant, axis=0)
        scale = (winsorized - low_quant) / (high_quant - low_quant + 1e-9)
        fisher = np.arctanh(scale * 1.98 - 0.99)
        signal = fisher.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_111_rank(df, window=5, sub_window=70):
        ret = df['close'].pct_change()
        ret_short = ret.rolling(window).mean()
        ret_long = ret.rolling(sub_window).mean()
        signal = (ret_short - ret_long).rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_111_tanh(df, window=40, sub_window=50):
        ret = df['close'].pct_change()
        ret_short = ret.rolling(window).mean()
        ret_long = ret.rolling(sub_window).mean()
        raw = (ret_short - ret_long) / (ret.rolling(10).std() + 0.01)
        return np.tanh(raw).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_111_zscore(df, window=1, sub_window=80):
        ret = df['close'].pct_change()
        ret_short = ret.rolling(window).mean()
        ret_long = ret.rolling(sub_window).mean()
        raw = (ret_short - ret_long) / (ret.rolling(10).std() + 0.01)
        normalized = ((raw - raw.rolling(100).mean()) / raw.rolling(100).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_111_sign(df, window=3, sub_window=80):
        ret = df['close'].pct_change()
        ret_short = ret.rolling(window).mean()
        ret_long = ret.rolling(sub_window).mean()
        stream = ret_short - ret_long
        signal = np.sign(stream)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_111_wf(df, window=1, sub_window=50):
        ret = df['close'].pct_change()
        ret_short = ret.rolling(window).mean()
        ret_long = ret.rolling(sub_window).mean()
        raw = (ret_short - ret_long) / (ret.rolling(10).std() + 0.01)
        p1 = 0.05
        p2 = 200
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_112_rank(df, window=60):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw1 = (high - low) / (high + low + 1e-8)
        pct_chg = volume.pct_change(periods=5)
        valid = raw1.notna() & pct_chg.notna()
        r1 = raw1.where(valid).rolling(window).rank(pct=True)
        r2 = pct_chg.where(valid).rolling(window).rank(pct=True)
        n = window
        mean_r1 = r1.rolling(window).mean()
        mean_r2 = r2.rolling(window).mean()
        diff1 = r1 - mean_r1
        diff2 = r2 - mean_r2
        cov = (diff1 * diff2).rolling(window).mean()
        std1 = r1.rolling(window).std()
        std2 = r2.rolling(window).std()
        corr = cov / (std1 * std2 + 1e-9)
        signal = (corr.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_112_tanh(df, window=5):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw1 = (high - low) / (high + low + 1e-8)
        pct_chg = volume.pct_change(periods=5)
        valid = raw1.notna() & pct_chg.notna()
        r1 = raw1.where(valid).rolling(window).rank(pct=True)
        r2 = pct_chg.where(valid).rolling(window).rank(pct=True)
        n = window
        mean_r1 = r1.rolling(window).mean()
        mean_r2 = r2.rolling(window).mean()
        diff1 = r1 - mean_r1
        diff2 = r2 - mean_r2
        cov = (diff1 * diff2).rolling(window).mean()
        std1 = r1.rolling(window).std()
        std2 = r2.rolling(window).std()
        corr = cov / (std1 * std2 + 1e-9)
        signal = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan).ffill())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_112_zscore(df, window=60):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw1 = (high - low) / (high + low + 1e-8)
        pct_chg = volume.pct_change(periods=5)
        valid = raw1.notna() & pct_chg.notna()
        r1 = raw1.where(valid).rolling(window).rank(pct=True)
        r2 = pct_chg.where(valid).rolling(window).rank(pct=True)
        n = window
        mean_r1 = r1.rolling(window).mean()
        mean_r2 = r2.rolling(window).mean()
        diff1 = r1 - mean_r1
        diff2 = r2 - mean_r2
        cov = (diff1 * diff2).rolling(window).mean()
        std1 = r1.rolling(window).std()
        std2 = r2.rolling(window).std()
        corr = cov / (std1 * std2 + 1e-9)
        signal = ((corr - corr.rolling(window).mean()) / corr.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_112_sign(df, window=15):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw1 = (high - low) / (high + low + 1e-8)
        pct_chg = volume.pct_change(periods=5)
        valid = raw1.notna() & pct_chg.notna()
        r1 = raw1.where(valid).rolling(window).rank(pct=True)
        r2 = pct_chg.where(valid).rolling(window).rank(pct=True)
        n = window
        mean_r1 = r1.rolling(window).mean()
        mean_r2 = r2.rolling(window).mean()
        diff1 = r1 - mean_r1
        diff2 = r2 - mean_r2
        cov = (diff1 * diff2).rolling(window).mean()
        std1 = r1.rolling(window).std()
        std2 = r2.rolling(window).std()
        corr = cov / (std1 * std2 + 1e-9)
        signal = np.sign(corr)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_112_wf(df, window=60, p1=0.1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw1 = (high - low) / (high + low + 1e-8)
        pct_chg = volume.pct_change(periods=5)
        valid = raw1.notna() & pct_chg.notna()
        r1 = raw1.where(valid).rolling(window).rank(pct=True)
        r2 = pct_chg.where(valid).rolling(window).rank(pct=True)
        n = window
        mean_r1 = r1.rolling(window).mean()
        mean_r2 = r2.rolling(window).mean()
        diff1 = r1 - mean_r1
        diff2 = r2 - mean_r2
        cov = (diff1 * diff2).rolling(window).mean()
        std1 = r1.rolling(window).std()
        std2 = r2.rolling(window).std()
        corr = cov / (std1 * std2 + 1e-9)
        p2 = window
        low_q = corr.rolling(p2).quantile(p1)
        high_q = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_113_k(df, window=25):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1).fillna(0)
        buy_volume = np.where(delta_close > 0, delta_close * volume, 0)
        sell_volume = np.where(delta_close < 0, -delta_close * volume, 0)
        raw = (pd.Series(buy_volume, index=df.index).rolling(window).sum() - pd.Series(sell_volume, index=df.index).rolling(window).sum()) / (volume.rolling(window).sum() + 1)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_113_h(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1).fillna(0)
        buy_volume = np.where(delta_close > 0, delta_close * volume, 0)
        sell_volume = np.where(delta_close < 0, -delta_close * volume, 0)
        raw = (pd.Series(buy_volume, index=df.index).rolling(window).sum() - pd.Series(sell_volume, index=df.index).rolling(window).sum()) / (volume.rolling(window).sum() + 1)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_113_e(df, window=25):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1).fillna(0)
        buy_volume = np.where(delta_close > 0, delta_close * volume, 0)
        sell_volume = np.where(delta_close < 0, -delta_close * volume, 0)
        raw = (pd.Series(buy_volume, index=df.index).rolling(window).sum() - pd.Series(sell_volume, index=df.index).rolling(window).sum()) / (volume.rolling(window).sum() + 1)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_113_t(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1).fillna(0)
        buy_volume = np.where(delta_close > 0, delta_close * volume, 0)
        sell_volume = np.where(delta_close < 0, -delta_close * volume, 0)
        raw = (pd.Series(buy_volume, index=df.index).rolling(window).sum() - pd.Series(sell_volume, index=df.index).rolling(window).sum()) / (volume.rolling(window).sum() + 1)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_113_r(df, window=30, p1=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close = close.diff(1).fillna(0)
        buy_volume = np.where(delta_close > 0, delta_close * volume, 0)
        sell_volume = np.where(delta_close < 0, -delta_close * volume, 0)
        raw = (pd.Series(buy_volume, index=df.index).rolling(window).sum() - pd.Series(sell_volume, index=df.index).rolling(window).sum()) / (volume.rolling(window).sum() + 1)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_114_rank(df, window=5):
        raw = df['close'].pct_change(10)
        rank_raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return rank_raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_114_tanh(df, window=25):
        raw = df['close'].pct_change(10)
        vol = df['close'].pct_change(1).rolling(window).std()
        normalized = np.tanh(raw / (vol + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_114_zscore(df, window=5):
        raw = df['close'].pct_change(10)
        zscore = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_114_sign(df, window=45):
        raw = df['close'].pct_change(10)
        vol = df['close'].pct_change(1).rolling(window).std()
        signal = np.sign(raw / (vol + 1e-8))
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_114_wf(df, window=40, quantile=0.1):
        raw = df['close'].pct_change(10)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_115_rank(df, window=25):
        raw = (df['close'].diff(window) * ((df['close'] > 0.8 * df['high'] + 0.2 * df['low']).rolling(window).sum() / window))
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_115_tanh(df, window=10):
        raw = (df['close'].diff(window) * ((df['close'] > 0.8 * df['high'] + 0.2 * df['low']).rolling(window).sum() / window))
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_115_zscore(df, window=25):
        raw = (df['close'].diff(window) * ((df['close'] > 0.8 * df['high'] + 0.2 * df['low']).rolling(window).sum() / window))
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_115_sign(df, window=10):
        raw = (df['close'].diff(window) * ((df['close'] > 0.8 * df['high'] + 0.2 * df['low']).rolling(window).sum() / window))
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_115_wf(df, window_rank=20, quantile_factor=0.3):
        raw = (df['close'].diff(window_rank) * ((df['close'] > 0.8 * df['high'] + 0.2 * df['low']).rolling(window_rank).sum() / window_rank))
        low = raw.rolling(window_rank).quantile(quantile_factor)
        high = raw.rolling(window_rank).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_116_6(df, window=80, delta=7):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = np.sign(close - open_) * volume
        raw_ffill = raw.ffill().fillna(0)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = close.diff(delta).ffill().fillna(0)
        cov = raw_ffill.rolling(window).cov(y)
        var_y = y.rolling(window).var().replace(0, np.nan).ffill().fillna(1e-9)
        var_raw = raw_ffill.rolling(window).var().replace(0, np.nan).ffill().fillna(1e-9)
        corr = cov / (np.sqrt(var_raw * var_y) + 1e-9)
        corr = corr.ffill().fillna(0)
        normalized = ((corr.rolling(window).rank(pct=True) * 2) - 1).ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_116_rank(df, window_rank=95):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close_5 = close.diff(5)
        raw = np.sign(close - open_) * volume
        corr_series = raw.rolling(window_rank).corr(delta_close_5)
        normalized = (corr_series.rolling(window_rank).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_116_tanh(df, window_tanh=10):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close_5 = close.diff(5)
        raw = np.sign(close - open_) * volume
        corr_series = raw.rolling(window_tanh).corr(delta_close_5)
        normalized = np.tanh(corr_series / corr_series.rolling(window_tanh).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_116_zscore(df, window_zscore=10):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close_5 = close.diff(5)
        raw = np.sign(close - open_) * volume
        corr_series = raw.rolling(window_zscore).corr(delta_close_5)
        normalized = ((corr_series - corr_series.rolling(window_zscore).mean()) / corr_series.rolling(window_zscore).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_116_sign(df, window_sign=50):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close_5 = close.diff(5)
        raw = np.sign(close - open_) * volume
        corr_series = raw.rolling(window_sign).corr(delta_close_5)
        normalized = np.sign(corr_series)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_116_wf(df, p1=0.1, p2=10):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_close_5 = close.diff(5)
        raw = np.sign(close - open_) * volume
        corr_series = raw.rolling(p2).corr(delta_close_5).fillna(0)
        low = corr_series.rolling(p2).quantile(p1)
        high = corr_series.rolling(p2).quantile(1 - p1)
        winsorized = corr_series.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_117_rank(df, window=50):
        # TRƯỜNG HỢP A (Rolling Rank)
        hl_diff = df['high'] - df['low']
        ts_corr = hl_diff.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std_close = df['close'].rolling(window*2).std() + 1e-8
        raw = ts_corr / std_close
        # Xử lý NaN: forward fill, còn lại fill 0 (neutral)
        raw = raw.ffill().fillna(0)
        signal = (raw.rolling(window*2).rank(pct=True) * 2) - 1
        return signal

    @staticmethod
    def alpha_quanta_full_base_117_tanh(df, window=15):
        # TRƯỜNG HỢP B (Dynamic Tanh)
        hl_diff = df['high'] - df['low']
        ts_corr = hl_diff.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std_close = df['close'].rolling(window*2).std() + 1e-8
        raw = ts_corr / std_close
        raw = raw.ffill().fillna(0)
        signal = np.tanh(raw / raw.rolling(window*2).std().replace(0, np.nan).ffill().fillna(1))
        return -signal

    @staticmethod
    def alpha_quanta_full_base_117_zscore(df, window=40):
        # TRƯỜNG HỢP C (Rolling Z-Score/Clip)
        hl_diff = df['high'] - df['low']
        ts_corr = hl_diff.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std_close = df['close'].rolling(window*2).std() + 1e-8
        raw = ts_corr / std_close
        raw = raw.ffill().fillna(0)
        signal = ((raw - raw.rolling(window*2).mean()) / raw.rolling(window*2).std().replace(0, np.nan).ffill().fillna(1)).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_117_sign(df, window=5):
        # TRƯỜNG HỢP D (Sign/Binary Soft)
        hl_diff = df['high'] - df['low']
        ts_corr = hl_diff.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std_close = df['close'].rolling(window*2).std() + 1e-8
        raw = ts_corr / std_close
        raw = raw.ffill().fillna(0)
        signal = np.sign(raw)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_117_wf(df, window=10, quantile=0.7):
        # TRƯỜNG HỢP E (Winsorized Fisher)
        hl_diff = df['high'] - df['low']
        ts_corr = hl_diff.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std_close = df['close'].rolling(window*2).std() + 1e-8
        raw = ts_corr / std_close
        raw = raw.ffill().fillna(0)
        low = raw.rolling(window*2).quantile(quantile)
        high = raw.rolling(window*2).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform xấp xỉ [-1,1]
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_118_rank(df, window=10):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume']
        sign_series = np.sign(close - open_)
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        raw = sign_series.rolling(window).corr(volume_ratio)
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_118_tanh(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume']
        sign_series = np.sign(close - open_)
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        raw = sign_series.rolling(window).corr(volume_ratio)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_118_zscore(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume']
        sign_series = np.sign(close - open_)
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        raw = sign_series.rolling(window).corr(volume_ratio)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_118_sign(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume']
        sign_series = np.sign(close - open_)
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        raw = sign_series.rolling(window).corr(volume_ratio)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_118_wf(df, window=90, p1=0.1, p2=20):
        close = df['close']
        open_ = df['open']
        volume = df['matchingVolume']
        sign_series = np.sign(close - open_)
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        raw = sign_series.rolling(window).corr(volume_ratio)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_119_k(df, window_rank_mean=45):
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        raw = (df['close'] - mean_20) / (std_20 + 1e-8)
        signal = (raw.rolling(window_rank_mean).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_119_h(df, window_std=75):
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        raw = (df['close'] - mean_20) / (std_20 + 1e-8)
        signal = np.tanh(raw / raw.rolling(window_std).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_119_e(df, window_norm=5):
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        raw = (df['close'] - mean_20) / (std_20 + 1e-8)
        signal = ((raw - raw.rolling(window_norm).mean()) / raw.rolling(window_norm).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_119_y(df):
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        raw = (df['close'] - mean_20) / (std_20 + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_119_r(df, p1=0.3, p2=10):
        mean_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        raw = (df['close'] - mean_20) / (std_20 + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_120_rank(df, window=80):
        raw = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        corr = raw.rolling(window).corr(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = pd.Series(np.sign(corr) * np.abs(corr), index=df.index)
        normalized = (result.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_120_tanh(df, window=5):
        raw = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        corr = raw.rolling(window).corr(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = pd.Series(np.sign(corr) * np.abs(corr), index=df.index)
        normalized = np.tanh(result / result.rolling(window).std().replace(0, np.nan).fillna(0))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_120_zscore(df, window=80):
        raw = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        corr = raw.rolling(window).corr(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = pd.Series(np.sign(corr) * np.abs(corr), index=df.index)
        mean = result.rolling(window).mean()
        std = result.rolling(window).std().replace(0, np.nan).fillna(0)
        normalized = ((result - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_120_sign(df, window=5):
        raw = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        corr = raw.rolling(window).corr(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = pd.Series(np.sign(corr) * np.abs(corr), index=df.index)
        normalized = np.sign(result)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_120_wf(df, window=80, quantile=0.1):
        raw = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        corr = raw.rolling(window).corr(vol).replace([np.inf, -np.inf], np.nan).fillna(0)
        result = pd.Series(np.sign(corr) * np.abs(corr), index=df.index)
        low = result.rolling(window).quantile(quantile)
        high = result.rolling(window).quantile(1 - quantile)
        winsorized = result.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_121_rank(df, window=5):
        close = df['close']
        low = df['low']
        raw = (close - low.rolling(window).min()) / (close.rolling(window).mean() + 1e-8) * np.sign(close.diff(window))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_121_tanh(df, window=5):
        close = df['close']
        low = df['low']
        raw = (close - low.rolling(window).min()) / (close.rolling(window).mean() + 1e-8) * np.sign(close.diff(window))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_121_zscore(df, window=5):
        close = df['close']
        low = df['low']
        raw = (close - low.rolling(window).min()) / (close.rolling(window).mean() + 1e-8) * np.sign(close.diff(window))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_121_sign(df, window=30):
        close = df['close']
        low = df['low']
        raw = (close - low.rolling(window).min()) / (close.rolling(window).mean() + 1e-8) * np.sign(close.diff(window))
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_121_wf(df, window=10, p2=40):
        close = df['close']
        low = df['low']
        raw = (close - low.rolling(window).min()) / (close.rolling(window).mean() + 1e-8) * np.sign(close.diff(window))
        p1 = 0.05
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_122_rank(df, window=60):
        close_std = df['close'].rolling(window).std()
        delta_std = close_std.diff(1)
        term1 = delta_std / (close_std + 1e-8)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        delta_vol = vol_mean.diff(1)
        term2 = delta_vol / (vol_mean + 1e-8)
        raw = term1 - term2
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_122_tanh(df, window=60):
        close_std = df['close'].rolling(window).std()
        delta_std = close_std.diff(1)
        term1 = delta_std / (close_std + 1e-8)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        delta_vol = vol_mean.diff(1)
        term2 = delta_vol / (vol_mean + 1e-8)
        raw = term1 - term2
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_122_zscore(df, window=75):
        close_std = df['close'].rolling(window).std()
        delta_std = close_std.diff(1)
        term1 = delta_std / (close_std + 1e-8)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        delta_vol = vol_mean.diff(1)
        term2 = delta_vol / (vol_mean + 1e-8)
        raw = term1 - term2
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_122_sign(df, window=90):
        close_std = df['close'].rolling(window).std()
        delta_std = close_std.diff(1)
        term1 = delta_std / (close_std + 1e-8)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        delta_vol = vol_mean.diff(1)
        term2 = delta_vol / (vol_mean + 1e-8)
        raw = term1 - term2
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_122_wf(df, window=60, percentile=0.1):
        close_std = df['close'].rolling(window).std()
        delta_std = close_std.diff(1)
        term1 = delta_std / (close_std + 1e-8)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        delta_vol = vol_mean.diff(1)
        term2 = delta_vol / (vol_mean + 1e-8)
        raw = term1 - term2
        low = raw.rolling(window).quantile(percentile)
        high = raw.rolling(window).quantile(1 - percentile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        ratio = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(ratio * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_123_rank(df, window_rank=5):
        close_pct = df['close'].pct_change(5)
        volume_pct = df['matchingVolume'].pct_change(5)
        raw = close_pct - volume_pct
        normalized = ((raw.rolling(window_rank).rank(pct=True) * 2) - 1).fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_123_tanh(df, window_std=100):
        close_pct = df['close'].pct_change(5)
        volume_pct = df['matchingVolume'].pct_change(5)
        raw = close_pct - volume_pct
        std = raw.rolling(window_std).std()
        normalized = np.tanh(raw / std.replace(0, np.nan)).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_123_zscore(df, window_z=80):
        close_pct = df['close'].pct_change(5)
        volume_pct = df['matchingVolume'].pct_change(5)
        raw = close_pct - volume_pct
        mean_ = raw.rolling(window_z).mean()
        std_ = raw.rolling(window_z).std()
        normalized = ((raw - mean_) / std_.replace(0, np.nan)).clip(-1, 1).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_123_sign(df):
        close_pct = df['close'].pct_change(5)
        volume_pct = df['matchingVolume'].pct_change(5)
        raw = close_pct - volume_pct
        normalized = np.sign(raw).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_123_wf(df, quantile=0.3, window_winsor=80):
        close_pct = df['close'].pct_change(5)
        volume_pct = df['matchingVolume'].pct_change(5)
        raw = close_pct - volume_pct
        low = raw.rolling(window_winsor).quantile(quantile)
        high = raw.rolling(window_winsor).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_124_rank(df, window=5, sub_window=50):
        # Case A: Rolling Rank normalization
        # Set default values if needed
        close_minus_open = df['close'] - df['open']
        high_minus_low = df['high'] - df['low']
        volume = df['matchingVolume']
        # Compute rolling correlations
        corr1 = close_minus_open.rolling(window).corr(volume)
        corr2 = high_minus_low.rolling(window).corr(volume)
        # Calculate raw spread
        raw = corr1 - corr2
        # Normalize with Rolling Rank
        rank_normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        signal = rank_normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_124_tanh(df, window=3, sub_window=20):
        # Case B: Dynamic Tanh normalization
        close_minus_open = df['close'] - df['open']
        high_minus_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr1 = close_minus_open.rolling(window).corr(volume)
        corr2 = high_minus_low.rolling(window).corr(volume)
        raw = corr1 - corr2
        # Dynamic Tanh
        std = raw.rolling(sub_window).std().replace(0, np.nan).fillna(1e-9)
        normalized = np.tanh(raw / std)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_124_zscore(df, window=3, sub_window=40):
        # Case C: Rolling Z-Score / Clip normalization
        close_minus_open = df['close'] - df['open']
        high_minus_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr1 = close_minus_open.rolling(window).corr(volume)
        corr2 = high_minus_low.rolling(window).corr(volume)
        raw = corr1 - corr2
        # Rolling Z-Score
        mean = raw.rolling(sub_window).mean()
        std = raw.rolling(sub_window).std().replace(0, np.nan).fillna(1)
        zscore = ((raw - mean) / std).clip(-1, 1)
        signal = zscore.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_124_sign(df, window=5):
        # Case D: Sign / Binary Soft normalization
        close_minus_open = df['close'] - df['open']
        high_minus_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr1 = close_minus_open.rolling(window).corr(volume)
        corr2 = high_minus_low.rolling(window).corr(volume)
        raw = corr1 - corr2
        # Sign normalization
        signal = np.sign(raw).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_124_wf(df, window=10, p1=0.1, p2=30):
        # Case E: Winsorized Fisher normalization
        close_minus_open = df['close'] - df['open']
        high_minus_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr1 = close_minus_open.rolling(window).corr(volume)
        corr2 = high_minus_low.rolling(window).corr(volume)
        raw = corr1 - corr2
        # Winsorized Fisher
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_125_rank(df, window=80):
        ret = df['close'].pct_change()
        mean_5 = ret.rolling(5).mean()
        std_20 = ret.rolling(20).std().replace(0, np.nan)
        ratio = mean_5 / (std_20 + 1e-8)
        mean_15 = ret.rolling(15).mean()
        sign_15 = np.sign(mean_15)
        raw = ratio * sign_15
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_125_tanh(df, window=5):
        ret = df['close'].pct_change()
        mean_5 = ret.rolling(5).mean()
        std_20 = ret.rolling(20).std().replace(0, np.nan)
        ratio = mean_5 / (std_20 + 1e-8)
        mean_15 = ret.rolling(15).mean()
        sign_15 = np.sign(mean_15)
        raw = ratio * sign_15
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_125_zscore(df, window=80):
        ret = df['close'].pct_change()
        mean_5 = ret.rolling(5).mean()
        std_20 = ret.rolling(20).std().replace(0, np.nan)
        ratio = mean_5 / (std_20 + 1e-8)
        mean_15 = ret.rolling(15).mean()
        sign_15 = np.sign(mean_15)
        raw = ratio * sign_15
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_125_sign(df):
        ret = df['close'].pct_change()
        mean_5 = ret.rolling(5).mean()
        std_20 = ret.rolling(20).std().replace(0, np.nan)
        ratio = mean_5 / (std_20 + 1e-8)
        mean_15 = ret.rolling(15).mean()
        sign_15 = np.sign(mean_15)
        raw = ratio * sign_15
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_125_wf(df, window=90, p1=0.3):
        p2 = window // 2
        ret = df['close'].pct_change()
        mean_5 = ret.rolling(5).mean()
        std_20 = ret.rolling(20).std().replace(0, np.nan)
        ratio = mean_5 / (std_20 + 1e-8)
        mean_15 = ret.rolling(15).mean()
        sign_15 = np.sign(mean_15)
        raw = ratio * sign_15
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_126_rank(df, window=60):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        sign_delta_close = np.sign(delta_close)
        raw = sign_delta_close * delta_close / (volume.rolling(window).mean() + 1e-8) - volume.diff(1) / (volume.rolling(window).std() + 1e-8)
        signal = raw.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False) * 2 - 1
        return -signal

    @staticmethod
    def alpha_quanta_full_base_126_tanh(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        sign_delta_close = np.sign(delta_close)
        raw = sign_delta_close * delta_close / (volume.rolling(window).mean() + 1e-8) - volume.diff(1) / (volume.rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal

    @staticmethod
    def alpha_quanta_full_base_126_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        sign_delta_close = np.sign(delta_close)
        raw = sign_delta_close * delta_close / (volume.rolling(window).mean() + 1e-8) - volume.diff(1) / (volume.rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_126_sign(df, window=45):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        sign_delta_close = np.sign(delta_close)
        raw = sign_delta_close * delta_close / (volume.rolling(window).mean() + 1e-8) - volume.diff(1) / (volume.rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_full_base_126_wf(df, window=60, p1=0.3):
        p2 = window
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        sign_delta_close = np.sign(delta_close)
        raw = sign_delta_close * delta_close / (volume.rolling(window).mean() + 1e-8) - volume.diff(1) / (volume.rolling(window).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_127_rank(df, window=35):
        ret = df['close'].pct_change()
        ts_mean_5 = ret.rolling(5).mean()
        delta_ts_mean = ts_mean_5.diff(1)
        ts_std_20 = ret.rolling(window).std()
        raw = delta_ts_mean / (ts_std_20 + 1e-8)
        demean = raw - raw.rolling(window).mean()
        signal = demean.rolling(window).rank(pct=True) * 2 - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_127_tanh(df, window=85):
        ret = df['close'].pct_change()
        ts_mean_5 = ret.rolling(5).mean()
        delta_ts_mean = ts_mean_5.diff(1)
        ts_std_20 = ret.rolling(window).std()
        raw = delta_ts_mean / (ts_std_20 + 1e-8)
        demean = raw - raw.rolling(window).mean()
        signal = np.tanh(demean / demean.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_127_zscore(df, window=35):
        ret = df['close'].pct_change()
        ts_mean_5 = ret.rolling(5).mean()
        delta_ts_mean = ts_mean_5.diff(1)
        ts_std_20 = ret.rolling(window).std()
        raw = delta_ts_mean / (ts_std_20 + 1e-8)
        demean = raw - raw.rolling(window).mean()
        signal = ((demean - demean.rolling(window).mean()) / demean.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_127_sign(df, window=90):
        ret = df['close'].pct_change()
        ts_mean_5 = ret.rolling(5).mean()
        delta_ts_mean = ts_mean_5.diff(1)
        ts_std_20 = ret.rolling(window).std()
        raw = delta_ts_mean / (ts_std_20 + 1e-8)
        demean = raw - raw.rolling(window).mean()
        signal = np.sign(demean)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_127_wf(df, window=80, p1=0.1):
        ret = df['close'].pct_change()
        ts_mean_5 = ret.rolling(5).mean()
        delta_ts_mean = ts_mean_5.diff(1)
        ts_std_20 = ret.rolling(window).std()
        raw = delta_ts_mean / (ts_std_20 + 1e-8)
        demean = raw - raw.rolling(window).mean()
        low = demean.rolling(window).quantile(p1)
        high = demean.rolling(window).quantile(1 - p1)
        winsorized = demean.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_128_rank(df, window=1, window_vol=20):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff()
        sign_vol = np.sign(delta_vol / (vol + 1e-8))
        std_ret = ret.rolling(window_vol).std()
        median_std = std_ret.rolling(60).median()
        factor = pd.Series(np.where(std_ret > median_std, 0.7, 1.3), index=df.index)
        raw = mean_ret * sign_vol * factor
        norm = (raw.rolling(window_vol).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_128_tanh(df, window=1, window_vol=20):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff()
        sign_vol = np.sign(delta_vol / (vol + 1e-8))
        std_ret = ret.rolling(window_vol).std()
        median_std = std_ret.rolling(60).median()
        factor = pd.Series(np.where(std_ret > median_std, 0.7, 1.3), index=df.index)
        raw = mean_ret * sign_vol * factor
        norm = np.tanh(raw / raw.rolling(window_vol).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_128_zscore(df, window=1, window_vol=50):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff()
        sign_vol = np.sign(delta_vol / (vol + 1e-8))
        std_ret = ret.rolling(window_vol).std()
        median_std = std_ret.rolling(60).median()
        factor = pd.Series(np.where(std_ret > median_std, 0.7, 1.3), index=df.index)
        raw = mean_ret * sign_vol * factor
        norm = ((raw - raw.rolling(window_vol).mean()) / raw.rolling(window_vol).std().replace(0, np.nan)).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_128_sign(df, window=10):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff()
        sign_vol = np.sign(delta_vol / (vol + 1e-8))
        std_ret = ret.rolling(20).std()
        median_std = std_ret.rolling(60).median()
        factor = pd.Series(np.where(std_ret > median_std, 0.7, 1.3), index=df.index)
        raw = mean_ret * sign_vol * factor
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_128_wf(df, window=1, window_vol=50):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff()
        sign_vol = np.sign(delta_vol / (vol + 1e-8))
        std_ret = ret.rolling(window_vol).std()
        median_std = std_ret.rolling(60).median()
        factor = pd.Series(np.where(std_ret > median_std, 0.7, 1.3), index=df.index)
        raw = mean_ret * sign_vol * factor
        p1 = 0.05
        p2 = 60
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        epsilon = 1e-9
        norm = np.arctanh(((winsorized - low) / (high - low + epsilon)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_129_rank(df, window=15):
        close = df['close']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_min_10 = low.rolling(window).min()
        raw_ratio = (close - ts_min_10) / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        delta_volume = delta_volume.replace([np.inf, -np.inf], np.nan)
        close_open_diff = close - open_price
        corr_10 = close_open_diff.rolling(window).corr(delta_volume)
        raw_z = (raw_ratio - raw_ratio.rolling(window).mean()) / raw_ratio.rolling(window).std().replace(0, np.nan)
        normalized = (raw_z.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_129_tanh(df, window=5):
        close = df['close']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_min_10 = low.rolling(window).min()
        raw_ratio = (close - ts_min_10) / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        delta_volume = delta_volume.replace([np.inf, -np.inf], np.nan)
        close_open_diff = close - open_price
        corr_10 = close_open_diff.rolling(window).corr(delta_volume)
        raw_combined = raw_ratio * corr_10
        std = raw_combined.rolling(window).std().replace(0, np.nan).ffill()
        normalized = np.tanh(raw_combined / std)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_129_zscore(df, window=5):
        close = df['close']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_min_10 = low.rolling(window).min()
        raw_ratio = (close - ts_min_10) / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        delta_volume = delta_volume.replace([np.inf, -np.inf], np.nan)
        close_open_diff = close - open_price
        corr_10 = close_open_diff.rolling(window).corr(delta_volume)
        raw = raw_ratio * corr_10
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).ffill()
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_129_sign(df, window=5):
        close = df['close']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_min_10 = low.rolling(window).min()
        raw_ratio = (close - ts_min_10) / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        delta_volume = delta_volume.replace([np.inf, -np.inf], np.nan)
        close_open_diff = close - open_price
        corr_10 = close_open_diff.rolling(window).corr(delta_volume)
        raw = raw_ratio * corr_10
        normalized = np.sign(raw).fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_129_wf(df, window=10, quantile_param=0.1):
        window2 = window
        close = df['close']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_min_10 = low.rolling(window).min()
        raw_ratio = (close - ts_min_10) / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        delta_volume = delta_volume.replace([np.inf, -np.inf], np.nan)
        close_open_diff = close - open_price
        corr_10 = close_open_diff.rolling(window).corr(delta_volume)
        raw = raw_ratio * corr_10
        low_b = raw.rolling(window2).quantile(quantile_param)
        high_b = raw.rolling(window2).quantile(1 - quantile_param)
        winsorized = raw.clip(lower=low_b, upper=high_b, axis=0)
        normalized = np.arctanh(((winsorized - low_b) / (high_b - low_b + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_130_rank(df, window=100):
        series_close = df['close'].values.astype(float)
        n = len(series_close)
        ret_5 = np.full(n, np.nan)
        ret_20 = np.full(n, np.nan)
        if n >= 5:
            ret_5_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(5), mode='valid') / 5
            ret_5[-len(ret_5_val):] = ret_5_val
        if n >= 20:
            ret_20_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(20), mode='valid') / 20
            ret_20[-len(ret_20_val):] = ret_20_val
        ret_5 = pd.Series(ret_5, index=df.index)
        ret_20 = pd.Series(ret_20, index=df.index)
        ret_short = ret_5 - ret_20
        std_ret = df['close'].pct_change().rolling(window).std() + 1e-8
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        raw = ret_short / std_ret * zscore_close
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_130_tanh(df, window=95):
        series_close = df['close'].values.astype(float)
        n = len(series_close)
        ret_5 = np.full(n, np.nan)
        ret_20 = np.full(n, np.nan)
        if n >= 5:
            ret_5_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(5), mode='valid') / 5
            ret_5[-len(ret_5_val):] = ret_5_val
        if n >= 20:
            ret_20_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(20), mode='valid') / 20
            ret_20[-len(ret_20_val):] = ret_20_val
        ret_5 = pd.Series(ret_5, index=df.index)
        ret_20 = pd.Series(ret_20, index=df.index)
        ret_short = ret_5 - ret_20
        std_ret = df['close'].pct_change().rolling(window).std() + 1e-8
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        raw = ret_short / std_ret * zscore_close
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_130_zscore(df, window=100):
        series_close = df['close'].values.astype(float)
        n = len(series_close)
        ret_5 = np.full(n, np.nan)
        ret_20 = np.full(n, np.nan)
        if n >= 5:
            ret_5_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(5), mode='valid') / 5
            ret_5[-len(ret_5_val):] = ret_5_val
        if n >= 20:
            ret_20_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(20), mode='valid') / 20
            ret_20[-len(ret_20_val):] = ret_20_val
        ret_5 = pd.Series(ret_5, index=df.index)
        ret_20 = pd.Series(ret_20, index=df.index)
        ret_short = ret_5 - ret_20
        std_ret = df['close'].pct_change().rolling(window).std() + 1e-8
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        raw = ret_short / std_ret * zscore_close
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_130_sign(df, window=35):
        series_close = df['close'].values.astype(float)
        n = len(series_close)
        ret_5 = np.full(n, np.nan)
        ret_20 = np.full(n, np.nan)
        if n >= 5:
            ret_5_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(5), mode='valid') / 5
            ret_5[-len(ret_5_val):] = ret_5_val
        if n >= 20:
            ret_20_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(20), mode='valid') / 20
            ret_20[-len(ret_20_val):] = ret_20_val
        ret_5 = pd.Series(ret_5, index=df.index)
        ret_20 = pd.Series(ret_20, index=df.index)
        ret_short = ret_5 - ret_20
        std_ret = df['close'].pct_change().rolling(window).std() + 1e-8
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        raw = ret_short / std_ret * zscore_close
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_130_wf(df, window=40, p1=0.1):
        series_close = df['close'].values.astype(float)
        n = len(series_close)
        ret_5 = np.full(n, np.nan)
        ret_20 = np.full(n, np.nan)
        if n >= 5:
            ret_5_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(5), mode='valid') / 5
            ret_5[-len(ret_5_val):] = ret_5_val
        if n >= 20:
            ret_20_val = np.convolve(np.diff(series_close, 1, prepend=series_close[0]) / series_close, np.ones(20), mode='valid') / 20
            ret_20[-len(ret_20_val):] = ret_20_val
        ret_5 = pd.Series(ret_5, index=df.index)
        ret_20 = pd.Series(ret_20, index=df.index)
        ret_short = ret_5 - ret_20
        std_ret = df['close'].pct_change().rolling(window).std() + 1e-8
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std()
        raw = ret_short / std_ret * zscore_close
        winsor_lo = raw.rolling(window).quantile(p1)
        winsor_hi = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=winsor_lo, upper=winsor_hi, axis=0)
        normalized = np.arctanh(((winsorized - winsor_lo) / (winsor_hi - winsor_lo + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_131_rank(df, window=35):
        raw = df['high'] - df['low']
        raw = raw / (df['close'].rolling(window).std() + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw.rolling(window).corr(vol_delta)
        raw = corr * corr.abs()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_131_tanh(df, window=15):
        raw = df['high'] - df['low']
        raw = raw / (df['close'].rolling(window).std() + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw.rolling(window).corr(vol_delta)
        raw = corr * corr.abs()
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_131_zscore(df, window=50):
        raw = df['high'] - df['low']
        raw = raw / (df['close'].rolling(window).std() + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw.rolling(window).corr(vol_delta)
        raw = corr * corr.abs()
        nmean = raw.rolling(window).mean()
        nstd = raw.rolling(window).std()
        normalized = ((raw - nmean) / nstd).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_131_sign(df, window=5):
        raw = df['high'] - df['low']
        raw = raw / (df['close'].rolling(window).std() + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw.rolling(window).corr(vol_delta)
        raw = corr * corr.abs()
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_131_wf(df, window=10, p2=80):
        raw = df['high'] - df['low']
        raw = raw / (df['close'].rolling(window).std() + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw.rolling(window).corr(vol_delta)
        raw_series = corr * corr.abs()
        p1 = 0.05
        low = raw_series.rolling(p2).quantile(p1)
        high = raw_series.rolling(p2).quantile(1 - p1)
        winsorized = raw_series.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_132_k(df, window=45):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        ts_mean_std = mean_ret / (std_ret + 1e-8)
        delta_close = df['close'].diff(1).fillna(0)
        std_close = df['close'].rolling(window).std().replace(0, np.nan)
        delta_close_std = -delta_close / (std_close + 1e-8)
        raw = ts_mean_std + delta_close_std
        raw_ffill = raw.ffill().fillna(0)
        normalized = (raw_ffill.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_132_h(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        ts_mean_std = mean_ret / (std_ret + 1e-8)
        delta_close = df['close'].diff(1).fillna(0)
        std_close = df['close'].rolling(window).std().replace(0, np.nan)
        delta_close_std = -delta_close / (std_close + 1e-8)
        raw = ts_mean_std + delta_close_std
        raw_ffill = raw.ffill().fillna(0)
        normalized = np.tanh(raw_ffill / raw_ffill.rolling(window).std().replace(0, np.nan))
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_132_e(df, window=20):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        ts_mean_std = mean_ret / (std_ret + 1e-8)
        delta_close = df['close'].diff(1).fillna(0)
        std_close = df['close'].rolling(window).std().replace(0, np.nan)
        delta_close_std = -delta_close / (std_close + 1e-8)
        raw = ts_mean_std + delta_close_std
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        zscore = (raw - rolling_mean) / (rolling_std + 1e-8)
        raw_ffill = zscore.ffill().fillna(0)
        normalized = raw_ffill.clip(-1, 1)
        signal = normalized.replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_132_y(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        ts_mean_std = mean_ret / (std_ret + 1e-8)
        delta_close = df['close'].diff(1).fillna(0)
        std_close = df['close'].rolling(window).std().replace(0, np.nan)
        delta_close_std = -delta_close / (std_close + 1e-8)
        raw = ts_mean_std + delta_close_std
        raw_ffill = raw.ffill().fillna(0)
        normalized = np.sign(raw_ffill)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_132_r(df, window=60, tail_percentile=0.1):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        ts_mean_std = mean_ret / (std_ret + 1e-8)
        delta_close = df['close'].diff(1).fillna(0)
        std_close = df['close'].rolling(window).std().replace(0, np.nan)
        delta_close_std = -delta_close / (std_close + 1e-8)
        raw = ts_mean_std + delta_close_std
        raw_ffill = raw.ffill().fillna(0)
        low = raw_ffill.rolling(window).quantile(tail_percentile)
        high = raw_ffill.rolling(window).quantile(1 - tail_percentile)
        winsorized = raw_ffill.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_133_rank(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(5).std()
        mean_volume = volume.rolling(5).mean()
        delta_std_close = (std_close - std_close.shift(1)) / (std_close + 1e-8)
        delta_mean_volume = (mean_volume - mean_volume.shift(1)) / (mean_volume + 1e-8)
        delta_std_close = delta_std_close.replace([np.inf, -np.inf], np.nan)
        delta_mean_volume = delta_mean_volume.replace([np.inf, -np.inf], np.nan)
        corr = delta_std_close.rolling(window).corr(delta_mean_volume)
        raw = corr.fillna(0)
        # Phương pháp A: Rolling Rank
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_133_tanh(df, window=65):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(5).std()
        mean_volume = volume.rolling(5).mean()
        delta_std_close = (std_close - std_close.shift(1)) / (std_close + 1e-8)
        delta_mean_volume = (mean_volume - mean_volume.shift(1)) / (mean_volume + 1e-8)
        delta_std_close = delta_std_close.replace([np.inf, -np.inf], np.nan)
        delta_mean_volume = delta_mean_volume.replace([np.inf, -np.inf], np.nan)
        corr = delta_std_close.rolling(window).corr(delta_mean_volume)
        raw = corr.fillna(0)
        # Phương pháp B: Dynamic Tanh
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_133_zscore(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(5).std()
        mean_volume = volume.rolling(5).mean()
        delta_std_close = (std_close - std_close.shift(1)) / (std_close + 1e-8)
        delta_mean_volume = (mean_volume - mean_volume.shift(1)) / (mean_volume + 1e-8)
        delta_std_close = delta_std_close.replace([np.inf, -np.inf], np.nan)
        delta_mean_volume = delta_mean_volume.replace([np.inf, -np.inf], np.nan)
        corr = delta_std_close.rolling(window).corr(delta_mean_volume)
        raw = corr.fillna(0)
        # Phương pháp C: Rolling Z-Score/Clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_133_sign(df, window=90):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(5).std()
        mean_volume = volume.rolling(5).mean()
        delta_std_close = (std_close - std_close.shift(1)) / (std_close + 1e-8)
        delta_mean_volume = (mean_volume - mean_volume.shift(1)) / (mean_volume + 1e-8)
        delta_std_close = delta_std_close.replace([np.inf, -np.inf], np.nan)
        delta_mean_volume = delta_mean_volume.replace([np.inf, -np.inf], np.nan)
        corr = delta_std_close.rolling(window).corr(delta_mean_volume)
        raw = corr.fillna(0)
        # Phương pháp D: Sign/Binary Soft
        signal = np.sign(raw)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_133_wf(df, window=80, p1=0.7):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(5).std()
        mean_volume = volume.rolling(5).mean()
        delta_std_close = (std_close - std_close.shift(1)) / (std_close + 1e-8)
        delta_mean_volume = (mean_volume - mean_volume.shift(1)) / (mean_volume + 1e-8)
        delta_std_close = delta_std_close.replace([np.inf, -np.inf], np.nan)
        delta_mean_volume = delta_mean_volume.replace([np.inf, -np.inf], np.nan)
        corr = delta_std_close.rolling(window).corr(delta_mean_volume)
        raw = corr.fillna(0)
        # Phương pháp E: Winsorized Fisher
        p2 = 2 * window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_134_rank(df, window=100):
        close = df['close']
        std_close = close.rolling(window).std()
        raw = std_close / (std_close + 1e-9)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_134_tanh(df, window=5):
        close = df['close']
        std_close = close.rolling(window).std()
        raw = std_close / (std_close + 1e-9)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        result = normalized.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_134_zscore(df, window=35):
        close = df['close']
        std_close = close.rolling(window).std()
        raw = std_close / (std_close + 1e-9)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        result = normalized.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_134_sign(df, window=30):
        close = df['close']
        std_close = close.rolling(window).std()
        raw = std_close / (std_close + 1e-9)
        normalized = np.sign(raw)
        result = normalized.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_134_wf(df, window=80, p1=0.9, p2=40):
        close = df['close']
        std_close = close.rolling(window).std()
        raw = std_close / (std_close + 1e-9)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_135_5(df, window=65):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        # TS_CORR of a series with itself is always 1, so RANK(1) is constant 1
        # To make meaningful, compute rolling correlation of vol_ratio with shifted itself?
        # Original: TS_CORR(delta_vol/volume, delta_vol/volume, 5) => correlation of series with itself = 1
        # RANK(1) = constant ~ 0.999 => normalize to [-1,1] by (rank(pct=True)*2-1)
        ts_corr = vol_ratio.rolling(window=window).corr(vol_ratio).fillna(0)
        raw = ts_corr.rolling(window=window).rank(pct=True) * 2 - 1
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_135_rank(df, window=65):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        corr_series = vol_ratio.rolling(window).corr(vol_ratio)
        rank_series = (corr_series.rolling(window).rank(pct=True) * 2) - 1
        return -rank_series.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_135_tanh(df, window=15):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        corr_series = vol_ratio.rolling(window).corr(vol_ratio)
        normalized = np.tanh(corr_series / (corr_series.rolling(window).std() + 1e-8))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_135_zscore(df, window=60):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        corr_series = vol_ratio.rolling(window).corr(vol_ratio)
        mean = corr_series.rolling(window).mean()
        std = corr_series.rolling(window).std().replace(0, np.nan)
        normalized = ((corr_series - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_135_sign(df, window=85):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        corr_series = vol_ratio.rolling(window).corr(vol_ratio)
        normalized = np.sign(corr_series)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_135_wf(df, window=5, p1=0.1):
        vol = df['matchingVolume']
        delta_vol = vol.diff(1)
        vol_ratio = delta_vol / (vol + 1e-8)
        corr_series = vol_ratio.rolling(window).corr(vol_ratio)
        low = corr_series.rolling(p1).quantile(p1)
        high = corr_series.rolling(p1).quantile(1 - p1)
        winsorized = corr_series.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9)
        numerator = numerator.clip(0, 1)
        normalized = np.arctanh(numerator * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_136_rank(df, window=5):
        x = pd.Series(np.arange(len(df)), index=df.index)
        y = df['close']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_136_tanh(df, window=5):
        x = pd.Series(np.arange(len(df)), index=df.index)
        y = df['close']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_136_zscore(df, window=5):
        x = pd.Series(np.arange(len(df)), index=df.index)
        y = df['close']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_136_sign(df, window=5):
        x = pd.Series(np.arange(len(df)), index=df.index)
        y = df['close']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_136_wf(df, window=40, p1=0.1):
        x = pd.Series(np.arange(len(df)), index=df.index)
        y = df['close']
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_137_k(df, window=25):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(5).corr(volume)
        raw = -np.sign(corr) * corr.rolling(window).std()
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_137_h(df, window=20):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(5).corr(volume)
        raw = -np.sign(corr) * corr.rolling(window).std()
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_137_e(df, window=35):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(5).corr(volume)
        raw = -np.sign(corr) * corr.rolling(window).std()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_137_y(df, window=30):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(5).corr(volume)
        raw = -np.sign(corr) * corr.rolling(window).std()
        result = np.sign(raw)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_137_r(df, window=50, p1=0.7):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(5).corr(volume)
        raw = -np.sign(corr) * corr.rolling(window).std()
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_138_k(df, window=45):
        raw = np.log(df['high'] - df['low'] + 1e-8)
        mean = raw.rolling(window).mean() + 1e-8
        ratio = raw / mean
        signal = (ratio.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_138_h(df, window=5):
        raw = np.log(df['high'] - df['low'] + 1e-8)
        mean = raw.rolling(window).mean() + 1e-8
        ratio = raw / mean
        signal = np.tanh(ratio / ratio.rolling(window).std().replace(0, np.nan).ffill())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_138_p(df, window=70):
        raw = np.log(df['high'] - df['low'] + 1e-8)
        mean = raw.rolling(window).mean() + 1e-8
        ratio = raw / mean
        rolling_mean = ratio.rolling(window).mean()
        rolling_std = ratio.rolling(window).std().replace(0, np.nan).ffill()
        signal = ((ratio - rolling_mean) / rolling_std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_138_y(df, window=5):
        raw = np.log(df['high'] - df['low'] + 1e-8)
        mean = raw.rolling(window).mean() + 1e-8
        ratio = raw / mean
        signal = np.sign(ratio)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_138_r(df, window=40, p1=0.7):
        raw = np.log(df['high'] - df['low'] + 1e-8)
        mean = raw.rolling(window).mean() + 1e-8
        ratio = raw / mean
        p2 = window
        low = ratio.rolling(p2).quantile(p1)
        high = ratio.rolling(p2).quantile(1 - p1)
        winsorized = ratio.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_139_zscore(df, window=20, sub_window=40):
        # Compute days index
        days = pd.Series(np.arange(len(df)), index=df.index)
        # Compute regression slope for volume: y=volume, x=days, rolling window=20
        y = df['matchingVolume'].fillna(method='ffill')
        # y.rolling(20).cov(days) / x.rolling(20).var()
        x = days
        w = 20
        cov = y.rolling(w).cov(x)
        var_x = x.rolling(w).var().replace(0, np.nan)
        slope = cov / var_x
        # Compute TS_VAR of slope over 60 periods
        ts_var = slope.rolling(window).var()
        # Add 1 then inverse (add small epsilon to avoid div by 0)
        raw_inv = 1.0 / (ts_var + 1e-9 + 1)
        # Rolling Z-Score normalization (Case C): ensure clip [-1,1]
        mean = raw_inv.rolling(sub_window).mean()
        std = raw_inv.rolling(sub_window).std().replace(0, np.nan)
        normalized = ((raw_inv - mean) / std).clip(-1, 1)
        # Fill remaining NaNs with 0
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_139_rank(df, window=50):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = volume
        cov = y.rolling(window).cov(x)
        var_x = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var_x
        ts_var = slope.rolling(60).var()
        raw = 1.0 / (ts_var + 1)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_139_tanh(df, window=85):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = volume
        cov = y.rolling(window).cov(x)
        var_x = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var_x
        ts_var = slope.rolling(60).var()
        raw = 1.0 / (ts_var + 1)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_139_sign(df, window=75):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = volume
        cov = y.rolling(window).cov(x)
        var_x = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var_x
        ts_var = slope.rolling(60).var()
        raw = 1.0 / (ts_var + 1)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_139_wf(df, window=80, p1=0.3):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        x = days
        y = volume
        cov = y.rolling(window).cov(x)
        var_x = x.rolling(window).var().replace(0, np.nan)
        slope = cov / var_x
        ts_var = slope.rolling(60).var()
        raw = 1.0 / (ts_var + 1)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], 0).fillna(0)
        return -normalized.clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_140_rank(df, window=5):
        ret = df['close'].pct_change()
        hl_range = (df['high'] - df['low']) / (df['low'] + 1e-8)
        hl_range_lag = hl_range.shift(1)
        corr = ret.rolling(window).corr(hl_range_lag)
        rank_corr = corr.rolling(window).rank(pct=True)
        signal = rank_corr * 2 - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_140_tanh(df, window=45):
        ret = df['close'].pct_change()
        hl_range = (df['high'] - df['low']) / (df['low'] + 1e-8)
        hl_range_lag = hl_range.shift(1)
        corr = ret.rolling(window).corr(hl_range_lag)
        signal = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_140_zscore(df, window=35):
        ret = df['close'].pct_change()
        hl_range = (df['high'] - df['low']) / (df['low'] + 1e-8)
        hl_range_lag = hl_range.shift(1)
        corr = ret.rolling(window).corr(hl_range_lag)
        mean_corr = corr.rolling(window).mean()
        std_corr = corr.rolling(window).std().replace(0, np.nan)
        signal = ((corr - mean_corr) / std_corr).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_140_sign(df, window=95):
        ret = df['close'].pct_change()
        hl_range = (df['high'] - df['low']) / (df['low'] + 1e-8)
        hl_range_lag = hl_range.shift(1)
        corr = ret.rolling(window).corr(hl_range_lag)
        signal = np.sign(corr)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_140_wf(df, p1=0.1, p2=70):
        ret = df['close'].pct_change()
        hl_range = (df['high'] - df['low']) / (df['low'] + 1e-8)
        hl_range_lag = hl_range.shift(1)
        corr = ret.rolling(15).corr(hl_range_lag)
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high)
        norm_val = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        signal = np.arctanh(norm_val.clip(-0.99, 0.99))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_141_rank(df, window=35):
        hi_lo = df['high'] - df['low']
        mean_hl = hi_lo.rolling(window).mean()
        raw = hi_lo / (mean_hl + 1e-8) - 1
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_141_tanh(df, window=60):
        hi_lo = df['high'] - df['low']
        mean_hl = hi_lo.rolling(window).mean()
        raw = hi_lo / (mean_hl + 1e-8) - 1
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_141_zscore(df, window=80):
        hi_lo = df['high'] - df['low']
        mean_hl = hi_lo.rolling(window).mean()
        raw = hi_lo / (mean_hl + 1e-8) - 1
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_141_sign(df, window=95):
        hi_lo = df['high'] - df['low']
        mean_hl = hi_lo.rolling(window).mean()
        raw = hi_lo / (mean_hl + 1e-8) - 1
        result = np.sign(raw)
        result = pd.Series(result, index=df.index).ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_141_wf(df, window_obs=100, p_quantile=0.7):
        hi_lo = df['high'] - df['low']
        mean_hl = hi_lo.rolling(window_obs).mean()
        raw = hi_lo / (mean_hl + 1e-8) - 1
        low = raw.rolling(window_obs).quantile(p_quantile)
        high = raw.rolling(window_obs).quantile(1 - p_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = pd.Series(normalized, index=df.index)
        result = normalized.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_142_rank(df, window=25):
        _return -= df['close'].pct_change()
        ret_mean = _return.rolling(window).mean()
        volume = df.get('matchingVolume', df.get('volume', 1))
        close_vol = df['close'] * volume
        corr = _return.rolling(window).corr(close_vol)
        raw = ret_mean - corr
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = normalized.ffill().fillna(0.0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_142_tanh(df, window=5):
        _return -= df['close'].pct_change()
        ret_mean = _return.rolling(window).mean()
        volume = df.get('matchingVolume', df.get('volume', 1))
        close_vol = df['close'] * volume
        corr = _return.rolling(window).corr(close_vol)
        raw = ret_mean - corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = normalized.ffill().fillna(0.0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_142_zscore(df, window=5):
        _return -= df['close'].pct_change()
        ret_mean = _return.rolling(window).mean()
        volume = df.get('matchingVolume', df.get('volume', 1))
        close_vol = df['close'] * volume
        corr = _return.rolling(window).corr(close_vol)
        raw = ret_mean - corr
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        signal = normalized.ffill().fillna(0.0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_142_sign(df, window=5):
        _return -= df['close'].pct_change()
        ret_mean = _return.rolling(window).mean()
        volume = df.get('matchingVolume', df.get('volume', 1))
        close_vol = df['close'] * volume
        corr = _return.rolling(window).corr(close_vol)
        raw = ret_mean - corr
        normalized = np.sign(raw)
        signal = normalized.ffill().fillna(0.0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_142_wf(df, window=20, p1=0.3):
        p2 = window
        _return -= df['close'].pct_change()
        ret_mean = _return.rolling(window).mean()
        volume = df.get('matchingVolume', df.get('volume', 1))
        close_vol = df['close'] * volume
        corr = _return.rolling(window).corr(close_vol)
        raw = ret_mean - corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0.0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_143_rank(df, window=85):
        # Tính return: đóng cửa hiện tại trừ đóng cửa trước, chia đóng cửa trước
        ret = df['close'].pct_change()
        # Tính range: (high - low)
        hl = df['high'] - df['low']
        # Chuẩn hóa range bằng rolling std (window=5 cố định)
        hl_std = hl.rolling(window).std() + 1e-8
        hl_norm = hl / hl_std
        # Tính tương quan rolling giữa return -và hl_norm
        corr = ret.rolling(window).corr(hl_norm)
        # Rank rolling theo window (để linh hoạt, param dùng sub_window cho rank, hardcode window=5 cho corr)
        sub_window = window  # mặc định dùng chung
        rank = corr.rolling(sub_window).rank(pct=True) * 2 - 1
        return rank.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_143_tanh(df, window=45):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        hl_std = hl.rolling(window).std() + 1e-8
        hl_norm = hl / hl_std
        corr = ret.rolling(window).corr(hl_norm)
        # Dynamic tanh với rolling std
        result = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return -result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_143_zscore(df, window=5):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        hl_std = hl.rolling(window).std() + 1e-8
        hl_norm = hl / hl_std
        corr = ret.rolling(window).corr(hl_norm)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        result = ((corr - mean) / std).clip(-1, 1)
        return result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_143_sign(df, window=15):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        hl_std = hl.rolling(window).std() + 1e-8
        hl_norm = hl / hl_std
        corr = ret.rolling(window).corr(hl_norm)
        # Sign của corr
        result = pd.Series(np.sign(corr), index=df.index)
        return -result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_143_wf(df, window=80, p1=0.9):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        hl_std = hl.rolling(window).std() + 1e-8
        hl_norm = hl / hl_std
        corr = ret.rolling(window).corr(hl_norm)
        # Winsorized Fisher
        p2 = window  # dùng window cho rolling quantile
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_144_zscore(df, window=10):
        delta_close = df['close'].diff(1)
        delta_volume = df['volume'].diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume)
        rank_corr = rolling_corr.rolling(window).rank(pct=True) * 2 - 1
        rolling_std_vol = df['volume'].rolling(window).std() + 1e-8
        raw = rank_corr / rolling_std_vol
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_144_rank(df, window=10):
        delta_close = df['close'].diff(1)
        delta_volume = df['volume'].diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume)
        rank_corr = rolling_corr.rolling(window).rank(pct=True) * 2 - 1
        rolling_std_vol = df['volume'].rolling(window).std() + 1e-8
        raw = rank_corr / rolling_std_vol
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_144_tanh(df, window=10):
        delta_close = df['close'].diff(1)
        delta_volume = df['volume'].diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume)
        rank_corr = rolling_corr.rolling(window).rank(pct=True) * 2 - 1
        rolling_std_vol = df['volume'].rolling(window).std() + 1e-8
        raw = rank_corr / rolling_std_vol
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_144_sign(df, window=10):
        delta_close = df['close'].diff(1)
        delta_volume = df['volume'].diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume)
        rank_corr = rolling_corr.rolling(window).rank(pct=True) * 2 - 1
        rolling_std_vol = df['volume'].rolling(window).std() + 1e-8
        raw = rank_corr / rolling_std_vol
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_144_wf(df, window=10, p1=0.05):
        delta_close = df['close'].diff(1)
        delta_volume = df['volume'].diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume)
        rank_corr = rolling_corr.rolling(window).rank(pct=True) * 2 - 1
        rolling_std_vol = df['volume'].rolling(window).std() + 1e-8
        raw = rank_corr / rolling_std_vol
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_145_rank(df, window=25):
        data = ((df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8))
        rank = data.rolling(window).rank(pct=True) * 2 - 1
        vol_close_corr = df['close'].rolling(window).corr(df['matchingVolume'])
        delta = vol_close_corr.diff()
        delta_normalized = delta.rolling(window).rank(pct=True) * 2 - 1
        signal = rank * delta_normalized
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_145_tanh(df, window=5):
        data = ((df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8))
        norm_data = np.tanh(data / data.rolling(window).std())
        vol_close_corr = df['close'].rolling(window).corr(df['matchingVolume'])
        delta = vol_close_corr.diff()
        norm_delta = np.tanh(delta / delta.rolling(window).std())
        signal = norm_data * norm_delta
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_145_zscore(df, window=20):
        data = ((df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8))
        z_data = ((data - data.rolling(window).mean()) / data.rolling(window).std()).clip(-1, 1)
        vol_close_corr = df['close'].rolling(window).corr(df['matchingVolume'])
        delta = vol_close_corr.diff()
        z_delta = ((delta - delta.rolling(window).mean()) / delta.rolling(window).std()).clip(-1, 1)
        signal = z_data * z_delta
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_145_sign(df, window=5):
        data = ((df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8))
        sign_data = np.sign(data)
        vol_close_corr = df['close'].rolling(window).corr(df['matchingVolume'])
        delta = vol_close_corr.diff()
        sign_delta = np.sign(delta)
        signal = sign_data * sign_delta
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_145_wf(df, window=20):
        data = ((df['high'] - df['low']) / (df['close'].rolling(window).std() + 1e-8))
        p1 = 0.1
        p2 = window
        low = data.rolling(p2).quantile(p1)
        high = data.rolling(p2).quantile(1 - p1)
        winsorized = data.clip(lower=low, upper=high, axis=0)
        norm_data = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        vol_close_corr = df['close'].rolling(window).corr(df['matchingVolume'])
        delta = vol_close_corr.diff()
        low_delta = delta.rolling(p2).quantile(p1)
        high_delta = delta.rolling(p2).quantile(1 - p1)
        winsorized_delta = delta.clip(lower=low_delta, upper=high_delta, axis=0)
        norm_delta = np.arctanh(((winsorized_delta - low_delta) / (high_delta - low_delta + 1e-9)) * 1.98 - 0.99)
        signal = norm_data * norm_delta
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_146_rank(df, window=90):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        corr_vol = ret.rolling(window).corr(volume)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(20).std()
        raw = corr_vol * mean_ret / (std_ret + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_146_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        corr_vol = ret.rolling(window).corr(volume)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(20).std()
        raw = corr_vol * mean_ret / (std_ret + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_146_zscore(df, window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        corr_vol = ret.rolling(window).corr(volume)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(20).std()
        raw = corr_vol * mean_ret / (std_ret + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_146_sign(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        corr_vol = ret.rolling(window).corr(volume)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(20).std()
        raw = corr_vol * mean_ret / (std_ret + 1e-8)
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_146_wf(df, window=70, p1=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        corr_vol = ret.rolling(window).corr(volume)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(20).std()
        raw = corr_vol * mean_ret / (std_ret + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_147_rank(df, window=50):
        close = df['close']
        ret = close.pct_change()
        corr = ret.rolling(20).corr(ret.rolling(20).mean())
        mean_ret = ret.rolling(5).mean()
        std_close = close.rolling(20).std()
        raw = (corr * mean_ret) / (std_close + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_147_tanh(df, window=30):
        close = df['close']
        ret = close.pct_change()
        corr = ret.rolling(20).corr(ret.rolling(20).mean())
        mean_ret = ret.rolling(5).mean()
        std_close = close.rolling(20).std()
        raw = (corr * mean_ret) / (std_close + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_147_zscore(df, window=35):
        close = df['close']
        ret = close.pct_change()
        corr = ret.rolling(20).corr(ret.rolling(20).mean())
        mean_ret = ret.rolling(5).mean()
        std_close = close.rolling(20).std()
        raw = (corr * mean_ret) / (std_close + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_147_sign(df):
        close = df['close']
        ret = close.pct_change()
        corr = ret.rolling(20).corr(ret.rolling(20).mean())
        mean_ret = ret.rolling(5).mean()
        std_close = close.rolling(20).std()
        raw = (corr * mean_ret) / (std_close + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_147_wf(df, window=20, winsor_percentile=0.3):
        close = df['close']
        ret = close.pct_change()
        corr = ret.rolling(20).corr(ret.rolling(20).mean())
        mean_ret = ret.rolling(5).mean()
        std_close = close.rolling(20).std()
        raw = (corr * mean_ret) / (std_close + 1e-8)
        low = raw.rolling(window).quantile(winsor_percentile)
        high = raw.rolling(window).quantile(1 - winsor_percentile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_148_rank(df, window=25):
        delta_close = df['close'].diff(1)
        volume = df['matchingVolume']
        ts_corr = delta_close.rolling(window=window).corr(volume)
        delta_close_scaled = delta_close / (df['close'] + 1e-8)
        ts_std = delta_close_scaled.rolling(window=window).std()
        signal = ts_corr * np.sign(ts_std)
        rank_signal = signal.rolling(window=window).rank(pct=True) * 2 - 1
        return rank_signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_148_tanh(df, window=5):
        delta_close = df['close'].diff(1)
        volume = df['matchingVolume']
        ts_corr = delta_close.rolling(window=window).corr(volume)
        delta_close_scaled = delta_close / (df['close'] + 1e-8)
        ts_std = delta_close_scaled.rolling(window=window).std()
        signal = ts_corr * np.sign(ts_std)
        norm_signal = np.tanh(signal / signal.rolling(window=window).std().replace(0, np.nan))
        return norm_signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_148_zscore(df, window=5):
        delta_close = df['close'].diff(1)
        volume = df['matchingVolume']
        ts_corr = delta_close.rolling(window=window).corr(volume)
        delta_close_scaled = delta_close / (df['close'] + 1e-8)
        ts_std = delta_close_scaled.rolling(window=window).std()
        signal = ts_corr * np.sign(ts_std)
        rolling_mean = signal.rolling(window=window).mean()
        rolling_std = signal.rolling(window=window).std().replace(0, np.nan)
        norm_signal = ((signal - rolling_mean) / rolling_std).clip(-1, 1)
        return norm_signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_148_sign(df, window=5):
        delta_close = df['close'].diff(1)
        volume = df['matchingVolume']
        ts_corr = delta_close.rolling(window=window).corr(volume)
        delta_close_scaled = delta_close / (df['close'] + 1e-8)
        ts_std = delta_close_scaled.rolling(window=window).std()
        signal = ts_corr * np.sign(ts_std)
        norm_signal = np.sign(signal)
        return norm_signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_148_wf(df, window=10, quantile=0.3):
        delta_close = df['close'].diff(1)
        volume = df['matchingVolume']
        ts_corr = delta_close.rolling(window=window).corr(volume)
        delta_close_scaled = delta_close / (df['close'] + 1e-8)
        ts_std = delta_close_scaled.rolling(window=window).std()
        signal = ts_corr * np.sign(ts_std)
        low = signal.rolling(window=window).quantile(quantile)
        high = signal.rolling(window=window).quantile(1 - quantile)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        denominator = (high - low).replace(0, np.nan)
        normalized = np.arctanh(((winsorized - low) / (denominator + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_149_rank(df, window=50):
        raw = (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) * (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) / (df['close'].rolling(window).std() + 1e-8)
        raw = np.sign(raw) * raw
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_149_tanh(df, window=20):
        raw = (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) * (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) / (df['close'].rolling(window).std() + 1e-8)
        raw = np.sign(raw) * raw
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_149_zscore(df, window=90):
        raw = (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) * (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) / (df['close'].rolling(window).std() + 1e-8)
        raw = np.sign(raw) * raw
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_149_sign(df, window=30):
        raw = (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) * (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) / (df['close'].rolling(window).std() + 1e-8)
        raw = np.sign(raw) * raw
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_149_wf(df, window=80, p=0.1):
        raw = (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) * (df['close'] - df['open']).rolling(window).corr(df['matchingVolume']) / (df['close'].rolling(window).std() + 1e-8)
        raw = np.sign(raw) * raw
        low = raw.rolling(window).quantile(p)
        high = raw.rolling(window).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_150_k(df, window=20):
        # Calculate mean crossovers for close
        close_ma_5 = df['close'].rolling(5).mean()
        close_ma_20 = df['close'].rolling(20).mean()
        close_std_20 = df['close'].rolling(20).std()
        close_z = (close_ma_5 - close_ma_20) / (close_std_20 + 1e-8)
        # Calculate mean crossovers for volume
        vol_ma_5 = df['volume'].rolling(5).mean()
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_z = (vol_ma_5 - vol_ma_20) / (vol_std_20 + 1e-8)
        # Difference then normalize
        raw = close_z - vol_z
        # Handle NaNs
        raw = raw.ffill().fillna(0)
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_full_base_150_h(df, window=20):
        close_ma_5 = df['close'].rolling(5).mean()
        close_ma_20 = df['close'].rolling(20).mean()
        close_std_20 = df['close'].rolling(20).std()
        close_z = (close_ma_5 - close_ma_20) / (close_std_20 + 1e-8)
        vol_ma_5 = df['volume'].rolling(5).mean()
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_z = (vol_ma_5 - vol_ma_20) / (vol_std_20 + 1e-8)
        raw = close_z - vol_z
        raw = raw.ffill().fillna(0)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized

    @staticmethod
    def alpha_quanta_full_base_150_p(df, window=20):
        close_ma_5 = df['close'].rolling(5).mean()
        close_ma_20 = df['close'].rolling(20).mean()
        close_std_20 = df['close'].rolling(20).std()
        close_z = (close_ma_5 - close_ma_20) / (close_std_20 + 1e-8)
        vol_ma_5 = df['volume'].rolling(5).mean()
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_z = (vol_ma_5 - vol_ma_20) / (vol_std_20 + 1e-8)
        raw = close_z - vol_z
        raw = raw.ffill().fillna(0)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_150_t(df, window=20):
        close_ma_5 = df['close'].rolling(5).mean()
        close_ma_20 = df['close'].rolling(20).mean()
        close_std_20 = df['close'].rolling(20).std()
        close_z = (close_ma_5 - close_ma_20) / (close_std_20 + 1e-8)
        vol_ma_5 = df['volume'].rolling(5).mean()
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_z = (vol_ma_5 - vol_ma_20) / (vol_std_20 + 1e-8)
        raw = close_z - vol_z
        raw = raw.ffill().fillna(0)
        normalized = pd.Series(np.sign(raw), index=df.index)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_150_r(df, window=20, sub_window=10):
        close_ma_5 = df['close'].rolling(5).mean()
        close_ma_20 = df['close'].rolling(20).mean()
        close_std_20 = df['close'].rolling(20).std()
        close_z = (close_ma_5 - close_ma_20) / (close_std_20 + 1e-8)
        vol_ma_5 = df['volume'].rolling(5).mean()
        vol_ma_20 = df['volume'].rolling(20).mean()
        vol_std_20 = df['volume'].rolling(20).std()
        vol_z = (vol_ma_5 - vol_ma_20) / (vol_std_20 + 1e-8)
        raw = close_z - vol_z
        raw = raw.ffill().fillna(0)
        p1 = 0.05  # fixed quantile for winsorization
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_151_rank(df, window=75):
        try:
            volume = df['matchingVolume']
            close = df['close']
            ts_corr = volume.rolling(window).corr(close)
            ts_mean_vol = volume.rolling(window).mean()
            ts_std_close = close.rolling(window).std().replace(0, np.nan)
            raw = ts_corr * ts_mean_vol / (ts_std_close + 1e-8)
            raw = raw.ffill().fillna(0)
            normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
            return -normalized.ffill().fillna(0)
        except Exception as e:
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_151_tanh(df, window=5):
        try:
            volume = df['matchingVolume']
            close = df['close']
            ts_corr = volume.rolling(window).corr(close)
            ts_mean_vol = volume.rolling(window).mean()
            ts_std_close = close.rolling(window).std().replace(0, np.nan)
            raw = ts_corr * ts_mean_vol / (ts_std_close + 1e-8)
            raw = raw.ffill().fillna(0)
            normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
            return normalized.ffill().fillna(0)
        except Exception as e:
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_151_zscore(df, window=70):
        try:
            volume = df['matchingVolume']
            close = df['close']
            ts_corr = volume.rolling(window).corr(close)
            ts_mean_vol = volume.rolling(window).mean()
            ts_std_close = close.rolling(window).std().replace(0, np.nan)
            raw = ts_corr * ts_mean_vol / (ts_std_close + 1e-8)
            raw = raw.ffill().fillna(0)
            normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
            return -normalized.ffill().fillna(0)
        except Exception as e:
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_151_sign(df, window=5):
        try:
            volume = df['matchingVolume']
            close = df['close']
            ts_corr = volume.rolling(window).corr(close)
            ts_mean_vol = volume.rolling(window).mean()
            ts_std_close = close.rolling(window).std().replace(0, np.nan)
            raw = ts_corr * ts_mean_vol / (ts_std_close + 1e-8)
            raw = raw.ffill().fillna(0)
            normalized = np.sign(raw)
            return normalized.ffill().fillna(0)
        except Exception as e:
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_151_wf(df, window=30, quantile_param=0.1):
        try:
            volume = df['matchingVolume']
            close = df['close']
            ts_corr = volume.rolling(window).corr(close)
            ts_mean_vol = volume.rolling(window).mean()
            ts_std_close = close.rolling(window).std().replace(0, np.nan)
            raw = ts_corr * ts_mean_vol / (ts_std_close + 1e-8)
            raw = raw.ffill().fillna(0)
            low = raw.rolling(window).quantile(quantile_param)
            high = raw.rolling(window).quantile(1 - quantile_param)
            winsorized = raw.clip(lower=low, upper=high, axis=0)
            normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
            return normalized.ffill().fillna(0)
        except Exception as e:
            return pd.Series(np.nan, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_152_rank(df: pd.DataFrame, window: int = 8) -> pd.Series:
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        # return = close.pct_change()
        ret = close.pct_change()
        # delta close sign
        delta_sign = np.sign(close - close.shift(1))
        # volume ratio
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        ts_corr = ret.rolling(window).corr(delta_sign * vol_ratio)
        raw = ts_corr
        # Rolling Rank normalization
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_152_tanh(df: pd.DataFrame, window: int = 8) -> pd.Series:
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        delta_sign = np.sign(close - close.shift(1))
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        ts_corr = ret.rolling(window).corr(delta_sign * vol_ratio)
        raw = ts_corr
        # Dynamic Tanh normalization
        result = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_152_zscore(df: pd.DataFrame, window: int = 8) -> pd.Series:
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        delta_sign = np.sign(close - close.shift(1))
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        ts_corr = ret.rolling(window).corr(delta_sign * vol_ratio)
        raw = ts_corr
        # Rolling Z-Score normalization
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        result = ((raw - mean) / (std + 1e-9)).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_152_sign(df: pd.DataFrame, window: int = 8) -> pd.Series:
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        delta_sign = np.sign(close - close.shift(1))
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        ts_corr = ret.rolling(window).corr(delta_sign * vol_ratio)
        raw = ts_corr
        # Sign/Binary normalization
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_152_wf(df: pd.DataFrame, window: int = 8, sub_window: int = 8) -> pd.Series:
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        delta_sign = np.sign(close - close.shift(1))
        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        ts_corr = ret.rolling(window).corr(delta_sign * vol_ratio)
        raw = ts_corr
        # Winsorized Fisher normalization
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0)
        return result.reindex(df.index, fill_value=0)

    @staticmethod
    def alpha_quanta_full_base_153_rank(df, window=65):
        high_low = df['high'] - df['low']
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1)
        corr = high_low.rolling(window).corr(volume_delta)
        rank_corr = (corr.rolling(window).rank(pct=True) * 2) - 1
        return rank_corr.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_153_tanh(df, window=10):
        high_low = df['high'] - df['low']
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1)
        corr = high_low.rolling(window).corr(volume_delta)
        raw = corr
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_153_zscore(df, window=70):
        high_low = df['high'] - df['low']
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1)
        corr = high_low.rolling(window).corr(volume_delta)
        raw = corr
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_153_sign(df, window=10):
        high_low = df['high'] - df['low']
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1)
        corr = high_low.rolling(window).corr(volume_delta)
        raw = corr
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_153_wf(df, window=50, p1=0.9):
        high_low = df['high'] - df['low']
        volume_delta = df.get('matchingVolume', df.get('volume', 1)).diff(1)
        corr = high_low.rolling(window).corr(volume_delta)
        raw = corr
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_154_k(df, window=20):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        volume = np.log1p(vol)
        raw_corr = volume.rolling(window).corr(ret)
        ret_std = ret.rolling(20).std()
        ret_abs_mean = ret.abs().rolling(window).mean()
        exp_term = np.exp(-ret_std / (ret_abs_mean + 1e-8))
        raw = raw_corr * exp_term
        result = raw.rolling(5).rank(pct=True) * 2 - 1
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_154_h(df, window=5):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        volume = np.log1p(vol)
        raw_corr = volume.rolling(window).corr(ret)
        ret_std = ret.rolling(20).std()
        ret_abs_mean = ret.abs().rolling(window).mean()
        exp_term = np.exp(-ret_std / (ret_abs_mean + 1e-8))
        raw = raw_corr * exp_term
        result = np.tanh(raw / raw.rolling(5).std().replace(0, np.nan).ffill())
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_154_e(df, window=5):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        volume = np.log1p(vol)
        raw_corr = volume.rolling(window).corr(ret)
        ret_std = ret.rolling(20).std()
        ret_abs_mean = ret.abs().rolling(window).mean()
        exp_term = np.exp(-ret_std / (ret_abs_mean + 1e-8))
        raw = raw_corr * exp_term
        result = ((raw - raw.rolling(10).mean()) / raw.rolling(10).std().replace(0, np.nan)).clip(-1, 1)
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_154_t(df, window=5):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        volume = np.log1p(vol)
        raw_corr = volume.rolling(window).corr(ret)
        ret_std = ret.rolling(20).std()
        ret_abs_mean = ret.abs().rolling(window).mean()
        exp_term = np.exp(-ret_std / (ret_abs_mean + 1e-8))
        raw = raw_corr * exp_term
        result = np.sign(raw)
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_154_r(df, window=20, p1=0.3, p2=20):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        volume = np.log1p(vol)
        raw_corr = volume.rolling(window).corr(ret)
        ret_std = ret.rolling(20).std()
        ret_abs_mean = ret.abs().rolling(window).mean()
        exp_term = np.exp(-ret_std / (ret_abs_mean + 1e-8))
        raw = raw_corr * exp_term
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_155_k(df, window=85):
        ret = df['close'].pct_change()
        raw = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
        corr = df['close'].pct_change().rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        sig = np.sign(corr)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = result * sig
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_155_h(df, window=90):
        ret = df['close'].pct_change()
        zscore = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
        corr = df['close'].pct_change().rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        sig = np.sign(corr)
        raw = zscore * sig
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_155_p(df, window=85):
        ret = df['close'].pct_change()
        zscore = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
        corr = df['close'].pct_change().rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        sig = np.sign(corr)
        raw = zscore * sig
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean_) / std_).clip(-1, 1)
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_155_t(df, window=90):
        ret = df['close'].pct_change()
        zscore = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
        corr = df['close'].pct_change().rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        sig = np.sign(corr)
        raw = zscore * sig
        result = np.sign(raw)
        result = result.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_155_r(df, window=90, p1=0.1):
        ret = df['close'].pct_change()
        zscore = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
        corr = df['close'].pct_change().rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        sig = np.sign(corr)
        raw = zscore * sig
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.ffill().fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_156_rank(df, window=10):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        close_ts_mean = df['close'].rolling(window=window).mean()
        close_ts_std = df['close'].rolling(window=window).std()
        close_z = (df['close'] - close_ts_mean) / (close_ts_std + 1e-8)
        corr = high_low.rolling(window=window).corr(close_z)
        vol_mean = df['matchingVolume'].rolling(window=window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        vol_factor = vol_ratio.rolling(window=window).mean()
        raw = corr * vol_factor
        normalized = (raw.rolling(window=window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_full_base_156_tanh(df, window=5):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        close_ts_mean = df['close'].rolling(window=window).mean()
        close_ts_std = df['close'].rolling(window=window).std()
        close_z = (df['close'] - close_ts_mean) / (close_ts_std + 1e-8)
        corr = high_low.rolling(window=window).corr(close_z)
        vol_mean = df['matchingVolume'].rolling(window=window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        vol_factor = vol_ratio.rolling(window=window).mean()
        raw = corr * vol_factor
        normalized = np.tanh(raw / raw.rolling(window=window).std())
        return normalized

    @staticmethod
    def alpha_quanta_full_base_156_zscore(df, window=5):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        close_ts_mean = df['close'].rolling(window=window).mean()
        close_ts_std = df['close'].rolling(window=window).std()
        close_z = (df['close'] - close_ts_mean) / (close_ts_std + 1e-8)
        corr = high_low.rolling(window=window).corr(close_z)
        vol_mean = df['matchingVolume'].rolling(window=window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        vol_factor = vol_ratio.rolling(window=window).mean()
        raw = corr * vol_factor
        normalized = ((raw - raw.rolling(window=window).mean()) / raw.rolling(window=window).std()).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_156_sign(df, window=5):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        close_ts_mean = df['close'].rolling(window=window).mean()
        close_ts_std = df['close'].rolling(window=window).std()
        close_z = (df['close'] - close_ts_mean) / (close_ts_std + 1e-8)
        corr = high_low.rolling(window=window).corr(close_z)
        vol_mean = df['matchingVolume'].rolling(window=window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        vol_factor = vol_ratio.rolling(window=window).mean()
        raw = corr * vol_factor
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_156_wf(df, window=40, p1=0.1):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        close_ts_mean = df['close'].rolling(window=window).mean()
        close_ts_std = df['close'].rolling(window=window).std()
        close_z = (df['close'] - close_ts_mean) / (close_ts_std + 1e-8)
        corr = high_low.rolling(window=window).corr(close_z)
        vol_mean = df['matchingVolume'].rolling(window=window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        vol_factor = vol_ratio.rolling(window=window).mean()
        raw = corr * vol_factor
        low = raw.rolling(window=window).quantile(p1)
        high = raw.rolling(window=window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_157_rank(df, window=70):
        raw = ((df['high'] - df['low']) / (df['low'] + 1e-8)).rolling(window).corr((df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)).shift(0)).fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_157_tanh(df, window=35):
        raw = ((df['high'] - df['low']) / (df['low'] + 1e-8)).rolling(window).corr((df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)).shift(0)).fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_157_zscore(df, window=75):
        raw = ((df['high'] - df['low']) / (df['low'] + 1e-8)).rolling(window).corr((df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)).shift(0)).fillna(0)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_157_sign(df, window=15):
        raw = ((df['high'] - df['low']) / (df['low'] + 1e-8)).rolling(window).corr((df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)).shift(0)).fillna(0)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_157_wf(df, window=30):
        raw = ((df['high'] - df['low']) / (df['low'] + 1e-8)).rolling(window).corr((df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)).shift(0)).fillna(0)
        p1 = 0.05
        p2 = window * 5
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_158_k(df, window=45):
        ret = df['close'].pct_change()
        dollar_vol = df['high'] - df['low']
        dollar_vol_sum = df['high'] + df['low']
        hl_ratio = dollar_vol / (dollar_vol_sum + 1e-8)
        raw = (ret.rolling(window).corr(hl_ratio)).rank(pct=True) * 2 - 1
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_158_h(df, window=45):
        ret = df['close'].pct_change()
        dollar_vol = df['high'] - df['low']
        dollar_vol_sum = df['high'] + df['low']
        hl_ratio = dollar_vol / (dollar_vol_sum + 1e-8)
        raw = ret.rolling(window).corr(hl_ratio)
        std = raw.rolling(window).std().replace(0, np.nan).ffill()
        normalized = np.tanh(raw / std)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_158_e(df, window=5):
        ret = df['close'].pct_change()
        dollar_vol = df['high'] - df['low']
        dollar_vol_sum = df['high'] + df['low']
        hl_ratio = dollar_vol / (dollar_vol_sum + 1e-8)
        raw = ret.rolling(window).corr(hl_ratio)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).ffill()
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_158_y(df, window=75):
        ret = df['close'].pct_change()
        dollar_vol = df['high'] - df['low']
        dollar_vol_sum = df['high'] + df['low']
        hl_ratio = dollar_vol / (dollar_vol_sum + 1e-8)
        raw = ret.rolling(window).corr(hl_ratio)
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_158_r(df, window=80, p1=0.1):
        ret = df['close'].pct_change()
        dollar_vol = df['high'] - df['low']
        dollar_vol_sum = df['high'] + df['low']
        hl_ratio = dollar_vol / (dollar_vol_sum + 1e-8)
        raw = ret.rolling(window).corr(hl_ratio)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = winsorized - low
        denominator = (high - low).replace(0, np.nan).ffill() + 1e-9
        normalized = np.arctanh((numerator / denominator) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_159_rank(df, window=70):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = ret.rolling(window).mean()
        std_close = df['close'].rolling(window).std()
        raw = np.sign(mean_ret) * mean_ret / (std_close + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_159_tanh(df, window=75):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = ret.rolling(window).mean()
        std_close = df['close'].rolling(window).std()
        raw = np.sign(mean_ret) * mean_ret / (std_close + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_159_zscore(df, window=90):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = ret.rolling(window).mean()
        std_close = df['close'].rolling(window).std()
        raw = np.sign(mean_ret) * mean_ret / (std_close + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_159_sign(df, window=10):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = ret.rolling(window).mean()
        std_close = df['close'].rolling(window).std()
        raw = np.sign(mean_ret) * mean_ret / (std_close + 1e-8)
        normalized = np.sign(raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_159_wf(df, window=90, p1=0.7):
        p2 = window
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        mean_ret = ret.rolling(window).mean()
        std_close = df['close'].rolling(window).std()
        raw = np.sign(mean_ret) * mean_ret / (std_close + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_160_rank(df, window=45):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        hl = df['high'] - df['low']
        corr = delta_volume.rolling(window).corr(hl)
        raw = (corr - corr.rolling(window*2).mean()) / corr.rolling(window*2).std()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_160_tanh(df, window=50):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        hl = df['high'] - df['low']
        corr = delta_volume.rolling(window).corr(hl)
        raw = (corr - corr.rolling(window*2).mean()) / corr.rolling(window*2).std()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_160_zscore(df, window=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        hl = df['high'] - df['low']
        corr = delta_volume.rolling(window).corr(hl)
        raw = (corr - corr.rolling(window*2).mean()) / corr.rolling(window*2).std()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_160_sign(df, window=60):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        hl = df['high'] - df['low']
        corr = delta_volume.rolling(window).corr(hl)
        raw = (corr - corr.rolling(window*2).mean()) / corr.rolling(window*2).std()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_160_wf(df, window=60, p1=0.1):
        p2 = window * 2
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_volume = volume.diff(1)
        hl = df['high'] - df['low']
        corr = delta_volume.rolling(window).corr(hl)
        raw = (corr - corr.rolling(p2).mean()) / corr.rolling(p2).std()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_161_rank(df, window=30):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.1))
        rs = vol.rolling(window).std()
        rm = vol.rolling(window).mean()
        z = (rs / (rm + 1e-8))
        raw = (z - z.rolling(window).mean()) / z.rolling(window).std()
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_161_tanh(df, window=20):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.1))
        rs = vol.rolling(window).std()
        rm = vol.rolling(window).mean()
        z = (rs / (rm + 1e-8))
        raw = (z - z.rolling(window).mean()) / z.rolling(window).std()
        result = np.tanh(raw / raw.rolling(window).std())
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_161_zscore(df, window=30):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.1))
        rs = vol.rolling(window).std()
        rm = vol.rolling(window).mean()
        z = (rs / (rm + 1e-8))
        raw = (z - z.rolling(window).mean()) / z.rolling(window).std()
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_161_sign(df, window=20):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.1))
        rs = vol.rolling(window).std()
        rm = vol.rolling(window).mean()
        z = (rs / (rm + 1e-8))
        raw = (z - z.rolling(window).mean()) / z.rolling(window).std()
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_161_wf(df, window=40, p1=0.9):
        p2 = window
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.1))
        rs = vol.rolling(p2).std()
        rm = vol.rolling(p2).mean()
        z = (rs / (rm + 1e-8))
        raw = (z - z.rolling(p2).mean()) / z.rolling(p2).std()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_162_k(df, window=5):
        ret = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        raw = ret.rolling(10).corr(volume_delta)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_162_h(df, window=40):
        ret = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        raw = ret.rolling(10).corr(volume_delta)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_162_p(df, window=5):
        ret = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        raw = ret.rolling(10).corr(volume_delta)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_162_t(df):
        ret = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        raw = ret.rolling(10).corr(volume_delta)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_162_r(df, p1=0.1, p2=90):
        ret = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        raw = ret.rolling(10).corr(volume_delta)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_163_rank(df, window=30):
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) / (ret.rolling(window).std() + 1e-8)
        signal = raw.fillna(0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_163_tanh(df, window=30):
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) / (ret.rolling(window).std() + 1e-8)
        signal = np.tanh(raw)
        return signal

    @staticmethod
    def alpha_quanta_full_base_163_zscore(df, window=30):
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) / (ret.rolling(window).std() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_163_sign(df, window=30):
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) / (ret.rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_full_base_163_wf(df, window=30, quantile=0.1):
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) / (ret.rolling(window).std() + 1e-8)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_164_4(df, window=30, sub_window=10):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        std_ret_rolling = ret.rolling(window).std().replace(0, np.nan)
        mean_std_rolling = std_ret_rolling.rolling(sub_window).mean().replace(0, np.nan)
        part1 = (1 + std_ret_rolling / (mean_std_rolling + 1e-8))
        corr_ret_vol = ret.rolling(5).corr(volume).fillna(0)
        rank_corr = corr_ret_vol.rolling(window).rank(pct=True)
        sign_mean_ret = pd.Series(np.sign(ret.rolling(10).mean().fillna(0)), index=df.index)
        raw = part1 * rank_corr * sign_mean_ret
        # Chuan hoa: Truong hop C - Z-Score Clip
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        zscore = ((raw - mean_raw) / std_raw).fillna(0)
        signal = zscore.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_164_rank(df, window=40):
        returns = df['close'].pct_change().fillna(0)
        std_20 = returns.rolling(window).std()
        mean_std_60 = std_20.rolling(60).mean()
        factor = 1 + std_20 / (mean_std_60 + 1e-8)
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = returns.rolling(5).corr(volume_series).fillna(0)
        rank_corr = corr_5.rolling(window).rank(pct=True).fillna(0.5)
        sign_mean = (returns.rolling(10).mean() > 0).astype(float) * 2 - 1
        sign_mean = sign_mean.fillna(0)
        raw = factor * rank_corr * sign_mean
        result = (raw.rolling(window).rank(pct=True).fillna(0.5) * 2) - 1
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_164_tanh(df, window=10):
        returns = df['close'].pct_change().fillna(0)
        std_20 = returns.rolling(window).std()
        mean_std_60 = std_20.rolling(60).mean()
        factor = 1 + std_20 / (mean_std_60 + 1e-8)
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = returns.rolling(5).corr(volume_series).fillna(0)
        rank_corr = corr_5.rolling(window).rank(pct=True).fillna(0.5)
        sign_mean = (returns.rolling(10).mean() > 0).astype(float) * 2 - 1
        sign_mean = sign_mean.fillna(0)
        raw = factor * rank_corr * sign_mean
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_164_zscore(df, window=40):
        returns = df['close'].pct_change().fillna(0)
        std_20 = returns.rolling(window).std()
        mean_std_60 = std_20.rolling(60).mean()
        factor = 1 + std_20 / (mean_std_60 + 1e-8)
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = returns.rolling(5).corr(volume_series).fillna(0)
        rank_corr = corr_5.rolling(window).rank(pct=True).fillna(0.5)
        sign_mean = (returns.rolling(10).mean() > 0).astype(float) * 2 - 1
        sign_mean = sign_mean.fillna(0)
        raw = factor * rank_corr * sign_mean
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_164_sign(df, window=40):
        returns = df['close'].pct_change().fillna(0)
        std_20 = returns.rolling(window).std()
        mean_std_60 = std_20.rolling(60).mean()
        factor = 1 + std_20 / (mean_std_60 + 1e-8)
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = returns.rolling(5).corr(volume_series).fillna(0)
        rank_corr = corr_5.rolling(window).rank(pct=True).fillna(0.5)
        sign_mean = (returns.rolling(10).mean() > 0).astype(float) * 2 - 1
        sign_mean = sign_mean.fillna(0)
        raw = factor * rank_corr * sign_mean
        result = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_164_wf(df, window=50, p1=0.3):
        p2 = window
        returns = df['close'].pct_change().fillna(0)
        std_20 = returns.rolling(window).std()
        mean_std_60 = std_20.rolling(60).mean()
        factor = 1 + std_20 / (mean_std_60 + 1e-8)
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = returns.rolling(5).corr(volume_series).fillna(0)
        rank_corr = corr_5.rolling(window).rank(pct=True).fillna(0.5)
        sign_mean = (returns.rolling(10).mean() > 0).astype(float) * 2 - 1
        sign_mean = sign_mean.fillna(0)
        raw = factor * rank_corr * sign_mean
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0).clip(-1, 1)
        return result

    @staticmethod
    def alpha_quanta_full_base_165_rank(df, window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        # Tính numerator hiệu chỉnh epsilon
        eps = 1e-8
        numerator = (high - low) / (close + eps)
        # Rolling correlation với volume
        roll_corr = numerator.rolling(window).corr(volume)
        # rank rolling corr về [-1,1] theo phương pháp A
        rank_corr = (roll_corr.rolling(window).rank(pct=True, method='min') * 2) - 1
        # Delta close pct
        delta_close = close.diff(1) / (close + eps)
        # Mean rolling 15 của delta close
        ts_mean_delta = delta_close.rolling(window=15).mean()
        # rank của ts_mean_delta (negative side)
        rank_delta = (ts_mean_delta.rolling(window).rank(pct=True, method='min') * 2) - 1
        # Kết hợp signal: vì alpha gốc là hiệu, ta lấy hiệu của 2 rank
        raw_signal = rank_corr - rank_delta
        # Chuẩn hóa cuối (có thể dùng tanh nhẹ)
        signal = np.tanh(raw_signal)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_165_tanh(df, window=50):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        eps = 1e-8
        numerator = (high - low) / (close + eps)
        roll_corr = numerator.rolling(window).corr(volume)
        # Dynamic tanh normalization B
        std_corr = roll_corr.rolling(window).std()
        # Tránh divide by zero
        std_corr = std_corr.replace(0, np.nan)
        norm_corr = np.tanh(roll_corr / std_corr)
        # Delta close pct
        delta_close = close.diff(1) / (close + eps)
        ts_mean_delta = delta_close.rolling(window=15).mean()
        std_delta = ts_mean_delta.rolling(window).std()
        std_delta = std_delta.replace(0, np.nan)
        norm_delta = np.tanh(ts_mean_delta / std_delta)
        raw_signal = norm_corr - norm_delta
        # Cuối cùng sigmoid-like để giữ trong [-1,1]
        signal = np.tanh(raw_signal)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_165_zscore(df, window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        eps = 1e-8
        numerator = (high - low) / (close + eps)
        # Rolling correlation
        roll_corr = numerator.rolling(window).corr(volume)
        # Z-score normalization C
        mean_corr = roll_corr.rolling(window).mean()
        std_corr = roll_corr.rolling(window).std().replace(0, np.nan)
        z_corr = ((roll_corr - mean_corr) / std_corr).clip(-1, 1)
        # Delta close pct
        delta_close = close.diff(1) / (close + eps)
        ts_mean_delta = delta_close.rolling(window=15).mean()
        mean_delta = ts_mean_delta.rolling(window).mean()
        std_delta = ts_mean_delta.rolling(window).std().replace(0, np.nan)
        z_delta = ((ts_mean_delta - mean_delta) / std_delta).clip(-1, 1)
        raw_signal = z_corr - z_delta
        # Clip lại cho [-1,1]
        signal = raw_signal.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_165_sign(df, window=55):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        eps = 1e-8
        numerator = (high - low) / (close + eps)
        roll_corr = numerator.rolling(window).corr(volume)
        # sign binary normalization D
        corr_sign = np.sign(roll_corr)
        # Delta close pct
        delta_close = close.diff(1) / (close + eps)
        ts_mean_delta = delta_close.rolling(window=15).mean()
        delta_sign = np.sign(ts_mean_delta)
        raw_signal = corr_sign - delta_sign
        # Chuẩn hóa về [-1,1] bằng sign lần nữa hoặc clip
        signal = raw_signal.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_165_wf(df, window=7, p2=40):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        eps = 1e-8
        p1 = 0.1
        numerator = (high - low) / (close + eps)
        roll_corr = numerator.rolling(window).corr(volume)
        # Winsorized Fisher E
        low_q = roll_corr.rolling(p2).quantile(p1)
        high_q = roll_corr.rolling(p2).quantile(1 - p1)
        winsorized_corr = roll_corr.clip(lower=low_q, upper=high_q, axis=0)
        # Fisher transform on corr
        range_corr = (high_q - low_q).replace(0, np.nan)
        scaled_corr = ((winsorized_corr - low_q) / range_corr) * 1.98 - 0.99
        scaled_corr = scaled_corr.clip(-0.99, 0.99)
        fish_corr = np.arctanh(scaled_corr)
        # Delta close pct
        delta_close = close.diff(1) / (close + eps)
        ts_mean_delta = delta_close.rolling(window=15).mean()
        low_q_delta = ts_mean_delta.rolling(p2).quantile(p1)
        high_q_delta = ts_mean_delta.rolling(p2).quantile(1 - p1)
        winsorized_delta = ts_mean_delta.clip(lower=low_q_delta, upper=high_q_delta, axis=0)
        range_delta = (high_q_delta - low_q_delta).replace(0, np.nan)
        scaled_delta = ((winsorized_delta - low_q_delta) / range_delta) * 1.98 - 0.99
        scaled_delta = scaled_delta.clip(-0.99, 0.99)
        fish_delta = np.arctanh(scaled_delta)
        raw_signal = fish_corr - fish_delta
        # Chuẩn hóa cuối về [-1,1] bằng tanh
        signal = np.tanh(raw_signal)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_6(df, window=25):
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std().replace(0, np.nan)
        raw = mean_ret / (std_ret + 1e-8)
        rank_raw = (raw.rolling(window).rank(pct=True) * 2) - 1
        max_close = close.rolling(window).max()
        min_close = close.rolling(window).min()
        sign_width = (max_close - min_close).apply(np.sign)
        result = rank_raw * sign_width
        return result.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_rank(df, window=25):
        ret = df['close'].pct_change(fill_method=None)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = (mean_ret / (std_ret + 1e-8)).rolling(window).rank(pct=True)
        close_max = df['close'].rolling(window).max()
        close_min = df['close'].rolling(window).min()
        sign = np.sign(close_max - close_min)
        result = (raw * 2 - 1) * sign
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_tanh(df, window=15):
        ret = df['close'].pct_change(fill_method=None)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        close_max = df['close'].rolling(window).max()
        close_min = df['close'].rolling(window).min()
        sign = np.sign(close_max - close_min)
        result = normalized * sign
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_zscore(df, window=25):
        ret = df['close'].pct_change(fill_method=None)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        close_max = df['close'].rolling(window).max()
        close_min = df['close'].rolling(window).min()
        sign = np.sign(close_max - close_min)
        result = normalized * sign
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_sign(df, window=30):
        ret = df['close'].pct_change(fill_method=None)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        close_max = df['close'].rolling(window).max()
        close_min = df['close'].rolling(window).min()
        sign = np.sign(close_max - close_min)
        result = np.sign(raw) * sign
        return pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_166_wf(df, window=40, p1=0.3):
        ret = df['close'].pct_change(fill_method=None)
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        close_max = df['close'].rolling(window).max()
        close_min = df['close'].rolling(window).min()
        sign = np.sign(close_max - close_min)
        result = normalized * sign
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_167_rank(df, window=5, sub_window=50):
        close = df['close'].values
        volume = df['matchingVolume'].values
        close_delay = np.roll(close, 1)
        close_delay[0] = close[0]
        return_ = np.diff(close, prepend=close[0]) / close
        close_diff = close - close_delay
        corr = pd.Series(close_diff).rolling(window).corr(pd.Series(volume)).fillna(0).values
        std_return = pd.Series(return_).rolling(sub_window).std().fillna(0).values + 1e-8
        vol_ratio = volume / (pd.Series(volume).rolling(sub_window).mean().fillna(0).values + 1e-8)
        raw = corr / std_return * (1 + vol_ratio)
        raw_series = pd.Series(raw, index=df.index)
        norm = (raw_series.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_167_tanh(df, window=30, sub_window=70):
        close = df['close'].values
        volume = df['matchingVolume'].values
        close_delay = np.roll(close, 1)
        close_delay[0] = close[0]
        return_ = np.diff(close, prepend=close[0]) / close
        close_diff = close - close_delay
        corr = pd.Series(close_diff).rolling(window).corr(pd.Series(volume)).fillna(0).values
        std_return -= pd.Series(return_).rolling(sub_window).std().fillna(0).values + 1e-8
        vol_ratio = volume / (pd.Series(volume).rolling(sub_window).mean().fillna(0).values + 1e-8)
        raw = corr / std_return * (1 + vol_ratio)
        raw_series = pd.Series(raw, index=df.index)
        norm = np.tanh(raw_series / raw_series.rolling(window).std().replace(0, np.nan).fillna(method='ffill'))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_167_zscore(df, window=5, sub_window=90):
        close = df['close'].values
        volume = df['matchingVolume'].values
        close_delay = np.roll(close, 1)
        close_delay[0] = close[0]
        return_ = np.diff(close, prepend=close[0]) / close
        close_diff = close - close_delay
        corr = pd.Series(close_diff).rolling(window).corr(pd.Series(volume)).fillna(0).values
        std_return = pd.Series(return_).rolling(sub_window).std().fillna(0).values + 1e-8
        vol_ratio = volume / (pd.Series(volume).rolling(sub_window).mean().fillna(0).values + 1e-8)
        raw = corr / std_return * (1 + vol_ratio)
        raw_series = pd.Series(raw, index=df.index)
        mean = raw_series.rolling(window).mean()
        std = raw_series.rolling(window).std().replace(0, np.nan).fillna(method='ffill')
        norm = ((raw_series - mean) / std).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_167_sign(df, window=5, sub_window=30):
        close = df['close'].values
        volume = df['matchingVolume'].values
        close_delay = np.roll(close, 1)
        close_delay[0] = close[0]
        return_ = np.diff(close, prepend=close[0]) / close
        close_diff = close - close_delay
        corr = pd.Series(close_diff).rolling(window).corr(pd.Series(volume)).fillna(0).values
        std_return = pd.Series(return_).rolling(sub_window).std().fillna(0).values + 1e-8
        vol_ratio = volume / (pd.Series(volume).rolling(sub_window).mean().fillna(0).values + 1e-8)
        raw = corr / std_return * (1 + vol_ratio)
        norm = np.sign(raw)
        return pd.Series(norm, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_167_wf(df, window=3, sub_window=30, quantile=0.05):
        close = df['close'].values
        volume = df['matchingVolume'].values
        close_delay = np.roll(close, 1)
        close_delay[0] = close[0]
        return_ = np.diff(close, prepend=close[0]) / close
        close_diff = close - close_delay
        corr = pd.Series(close_diff).rolling(window).corr(pd.Series(volume)).fillna(0).values
        std_return = pd.Series(return_).rolling(sub_window).std().fillna(0).values + 1e-8
        vol_ratio = volume / (pd.Series(volume).rolling(sub_window).mean().fillna(0).values + 1e-8)
        raw = corr / std_return * (1 + vol_ratio)
        raw_series = pd.Series(raw, index=df.index)
        p2 = sub_window
        p1 = quantile
        low = raw_series.rolling(p2).quantile(p1).fillna(method='ffill').fillna(0)
        high = raw_series.rolling(p2).quantile(1 - p1).fillna(method='ffill').fillna(0)
        winsorized = raw_series.clip(lower=low, upper=high)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_168_k(df, window=75):
        volume = df['matchingVolume']
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = volume.rolling(window).corr(seq)
        raw = corr.rolling(window).rank(pct=True) * 2 - 1
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_168_h(df, window=40):
        volume = df['matchingVolume']
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = volume.rolling(window).corr(seq)
        std = corr.rolling(window).std().replace(0, np.nan)
        raw = np.tanh(corr / std)
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_168_p(df, window=70):
        volume = df['matchingVolume']
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = volume.rolling(window).corr(seq)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        raw = ((corr - mean) / std).clip(-1, 1)
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_168_y(df, window=10):
        volume = df['matchingVolume']
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = volume.rolling(window).corr(seq)
        raw = np.sign(corr)
        signal = pd.Series(raw, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_168_r(df, window=100, p1=0.7):
        volume = df['matchingVolume']
        seq = pd.Series(np.arange(len(df)), index=df.index)
        corr = volume.rolling(window).corr(seq)
        low = corr.rolling(window).quantile(p1)
        high = corr.rolling(window).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(numerator * 1.98 - 0.99)
        raw = normalized.ffill().fillna(0)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        signal = raw.clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_169_rank(df, window=30):
        close = df['close']
        ret = close.pct_change()
        avg_close = close.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = ret.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = slope.rank(pct=True) * 2 - 1
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_169_tanh(df, window=45):
        close = df['close']
        ret = close.pct_change()
        avg_close = close.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = ret.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.tanh(slope / slope.rolling(window).std())
        return raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_169_zscore(df, window=70):
        close = df['close']
        ret = close.pct_change()
        avg_close = close.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = ret.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = ((slope - slope.rolling(window).mean()) / slope.rolling(window).std()).clip(-1, 1)
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_169_sign(df, window=60):
        close = df['close']
        ret = close.pct_change()
        avg_close = close.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = ret.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        raw = np.sign(slope)
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_169_wf(df, window=40, p1=0.1):
        close = df['close']
        ret = close.pct_change()
        avg_close = close.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = ret.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        low = slope.rolling(window).quantile(p1)
        high = slope.rolling(window).quantile(1 - p1)
        winsorized = slope.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_170_rank(df, window=10):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_170_tanh(df, window=15):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_170_zscore(df, window=10):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_170_sign(df, window=65):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_170_wf(df, window=10, p1=0.1):
        p2 = window
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_171_rank(df, window_rank=35):
        raw = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-8)
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_171_tanh(df, window_tanh=75):
        raw = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window_tanh).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_171_zscore(df, window_zscore=10):
        raw = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-8)
        signal = ((raw - raw.rolling(window_zscore).mean()) / raw.rolling(window_zscore).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_171_sign(df):
        raw = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_171_wf(df, p1=0.1, p2=70):
        raw = (df['close'] - df['close'].rolling(5).mean()) / (df['close'].rolling(5).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = norm.fillna(0).replace([np.inf, -np.inf], 0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_172_rank(df, window=45):
        vol = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(window).mean()
        std = log_vol.rolling(window).std().replace(0, np.nan)
        raw = (log_vol - mean) / (std + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_172_tanh(df, window=25):
        vol = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(window).mean()
        std = log_vol.rolling(window).std().replace(0, np.nan)
        raw = (log_vol - mean) / (std + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_172_zscore(df, window=65):
        vol = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(window).mean()
        std = log_vol.rolling(window).std().replace(0, np.nan)
        raw = (log_vol - mean) / (std + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_172_sign(df, window=30):
        vol = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(window).mean()
        std = log_vol.rolling(window).std().replace(0, np.nan)
        raw = (log_vol - mean) / (std + 1e-8)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_172_wf(df, window=100, p1=0.1):
        vol = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(vol)
        mean = log_vol.rolling(window).mean()
        std = log_vol.rolling(window).std().replace(0, np.nan)
        raw = (log_vol - mean) / (std + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_173_rank(df, window=10):
        raw = df['close'] / df['close'].rolling(window).mean().replace(0, np.nan).ffill()
        rank = raw.rolling(window).rank(pct=True)
        normalized = rank * 2 - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_173_tanh(df, window=60):
        raw = df['close'] / df['close'].rolling(window).mean().replace(0, np.nan).ffill()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_173_zscore(df, window=10):
        raw = df['close'] / df['close'].rolling(window).mean().replace(0, np.nan).ffill()
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan).ffill()
        zscore = (raw - mean_) / std_
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_173_sign(df, window=5):
        raw = df['close'] / df['close'].rolling(window).mean().replace(0, np.nan).ffill()
        normalized = np.sign(raw - 1)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_173_wf(df, window=10, p1=0.1):
        p2 = window
        raw = df['close'] / df['close'].rolling(window).mean().replace(0, np.nan).ffill()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.copy()
        winsorized = winsorized.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_174_rank(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1).astype(float))
        ret = close.pct_change().fillna(0)
        abs_ret = ret.abs()
        x = abs_ret.rolling(window, min_periods=window).rank(pct=True) * 2 - 1
        return x.rolling(window, min_periods=window).corr(pd.Series(1, index=df.index)) * 0

    @staticmethod
    def alpha_quanta_full_base_174_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1).astype(float))
        ret = close.pct_change().fillna(0)
        abs_ret = ret.abs()
        raw = abs_ret.rolling(window, min_periods=window).corr(volume.rolling(window, min_periods=window).apply(lambda x: x, raw=True).replace(0, np.nan)) * 0
        return np.tanh(raw / (raw.rolling(window).std().replace(0, np.nan)))

    @staticmethod
    def alpha_quanta_full_base_174_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1).astype(float))
        ret = close.pct_change().fillna(0)
        abs_ret = ret.abs()
        raw = abs_ret.rolling(window, min_periods=window).corr(volume.rolling(window, min_periods=window).apply(lambda x: x, raw=True).replace(0, np.nan)) * 0
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std().replace(0, np.nan)
        return ((raw - raw_mean) / raw_std).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_174_sign(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1).astype(float))
        ret = close.pct_change().fillna(0)
        abs_ret = ret.abs()
        raw = abs_ret.rolling(window, min_periods=window).corr(volume.rolling(window, min_periods=window).apply(lambda x: x, raw=True).replace(0, np.nan)) * 0
        return np.sign(raw).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_174_wf(df, window=5, sub_window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1).astype(float))
        ret = close.pct_change().fillna(0)
        abs_ret = ret.abs()
        raw = abs_ret.rolling(window, min_periods=window).corr(volume.rolling(window, min_periods=window).apply(lambda x: x, raw=True).replace(0, np.nan)) * 0
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        return np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_175_rank(df, window_rank=5):
        raw = df['high'] - df['low']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = raw.rolling(3).corr(vol)
        # Rolling Rank normalization (A)
        normalized = (corr.rolling(window_rank).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_175_tanh(df, window_std=10):
        raw = df['high'] - df['low']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = raw.rolling(3).corr(vol)
        # Dynamic Tanh normalization (B)
        normalized = np.tanh(corr / corr.rolling(window_std).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_175_zscore(df, window_z=65):
        raw = df['high'] - df['low']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = raw.rolling(3).corr(vol)
        # Rolling Z-Score/Clip normalization (C)
        mean = corr.rolling(window_z).mean()
        std = corr.rolling(window_z).std().replace(0, np.nan)
        normalized = ((corr - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_175_sign(df):
        raw = df['high'] - df['low']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = raw.rolling(3).corr(vol)
        # Sign/Binary Soft normalization (D)
        normalized = np.sign(corr)
        return -pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_175_wf(df, p1=0.1, p2=90):
        raw = df['high'] - df['low']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = raw.rolling(3).corr(vol)
        # Winsorized Fisher normalization (E)
        low_bound = corr.rolling(p2).quantile(p1)
        high_bound = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low_bound, upper=high_bound, axis=0)
        # Fisher transform
        x = (winsorized - low_bound) / (high_bound - low_bound + 1e-9)
        x = x * 1.98 - 0.99
        x = x.clip(-0.99, 0.99)
        normalized = np.arctanh(x)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_176_rank(df, window=65):
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8) * np.sign(df['close'] - df['open'])
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_176_tanh(df, window=45):
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8) * np.sign(df['close'] - df['open'])
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_176_zscore(df, window=30):
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8) * np.sign(df['close'] - df['open'])
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_176_sign(df, window=85):
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8) * np.sign(df['close'] - df['open'])
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_176_wf(df, window=10, quantile_factor=0.1):
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8) * np.sign(df['close'] - df['open'])
        p1 = quantile_factor
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-0.99, 0.99) / 0.99
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_177_rank(df, window=45):
        high_low = df['high'] - df['low']
        close_std = df['close'].rolling(window).std()
        raw = (high_low.rolling(window).std() / (close_std + 1e-8)) * np.sign(df['close'].pct_change().rolling(5).mean())
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_177_tanh(df, window=50):
        high_low = df['high'] - df['low']
        close_std = df['close'].rolling(window).std()
        raw = (high_low.rolling(window).std() / (close_std + 1e-8)) * np.sign(df['close'].pct_change().rolling(5).mean())
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_177_zscore(df, window=40):
        high_low = df['high'] - df['low']
        close_std = df['close'].rolling(window).std()
        raw = (high_low.rolling(window).std() / (close_std + 1e-8)) * np.sign(df['close'].pct_change().rolling(5).mean())
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_177_sign(df, window=85):
        high_low = df['high'] - df['low']
        close_std = df['close'].rolling(window).std()
        raw = (high_low.rolling(window).std() / (close_std + 1e-8)) * np.sign(df['close'].pct_change().rolling(5).mean())
        normalized = np.sign(raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_177_wf(df, window=60, p1=0.7):
        high_low = df['high'] - df['low']
        close_std = df['close'].rolling(window).std()
        raw = (high_low.rolling(window).std() / (close_std + 1e-8)) * np.sign(df['close'].pct_change().rolling(5).mean())
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_178_rank(df, window=90, rank_window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        mean_vol = volume_log.rolling(window).mean()
        std_vol = volume_log.rolling(window).std().replace(0, np.nan)
        zscore = (volume_log - mean_vol) / (std_vol + 1e-8)
        raw = zscore.rolling(rank_window).rank(pct=True) * 2 - 1
        signal = raw.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_178_tanh(df, window=30, factor=3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        mean_vol = volume_log.rolling(window).mean()
        std_vol = volume_log.rolling(window).std().replace(0, np.nan)
        zscore = (volume_log - mean_vol) / (std_vol + 1e-8)
        raw = zscore * factor
        signal = np.tanh(raw).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_178_zscore(df, window=50, zscore_window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        mean_vol = volume_log.rolling(window).mean()
        std_vol = volume_log.rolling(window).std().replace(0, np.nan)
        zscore = (volume_log - mean_vol) / (std_vol + 1e-8)
        rolling_mean = zscore.rolling(zscore_window).mean()
        rolling_std = zscore.rolling(zscore_window).std().replace(0, np.nan)
        signal = ((zscore - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_178_sign(df, window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        mean_vol = volume_log.rolling(window).mean()
        std_vol = volume_log.rolling(window).std().replace(0, np.nan)
        zscore = (volume_log - mean_vol) / (std_vol + 1e-8)
        signal = np.sign(zscore).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_178_wf(df, window=50, p1=0.1, p2=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        mean_vol = volume_log.rolling(window).mean()
        std_vol = volume_log.rolling(window).std().replace(0, np.nan)
        zscore = (volume_log - mean_vol) / (std_vol + 1e-8)
        low = zscore.rolling(p2).quantile(p1)
        high = zscore.rolling(p2).quantile(1 - p1)
        winsorized = zscore.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_179_rank(df, window=50, sub_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        raw1 = (high - low).rolling(window).std() / (close.rolling(window).std() + 1e-8)
        raw2 = ret.rolling(5).mean().pipe(np.sign)
        raw3 = (vol - vol.rolling(sub_window).mean()) / (vol.rolling(sub_window).std() + 1e-8)
        raw = raw1 * raw2 * raw3
        raw = raw.ffill().fillna(0)
        sig = (raw.rolling(window).rank(pct=True) * 2) - 1
        return sig

    @staticmethod
    def alpha_quanta_full_base_179_tanh(df, window=30, sub_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        raw1 = (high - low).rolling(window).std() / (close.rolling(window).std() + 1e-8)
        raw2 = ret.rolling(5).mean().pipe(np.sign)
        raw3 = (vol - vol.rolling(sub_window).mean()) / (vol.rolling(sub_window).std() + 1e-8)
        raw = raw1 * raw2 * raw3
        raw = raw.ffill().fillna(0)
        sig = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1))
        return sig

    @staticmethod
    def alpha_quanta_full_base_179_zscore(df, window=30, sub_window=10):
        high = df['high']
        low = df['low']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        raw1 = (high - low).rolling(window).std() / (close.rolling(window).std() + 1e-8)
        raw2 = ret.rolling(5).mean().pipe(np.sign)
        raw3 = (vol - vol.rolling(sub_window).mean()) / (vol.rolling(sub_window).std() + 1e-8)
        raw = raw1 * raw2 * raw3
        raw = raw.ffill().fillna(0)
        sig = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)).clip(-1, 1)
        return sig

    @staticmethod
    def alpha_quanta_full_base_179_sign(df, window=50, sub_window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        raw1 = (high - low).rolling(window).std() / (close.rolling(window).std() + 1e-8)
        raw2 = ret.rolling(5).mean().pipe(np.sign)
        raw3 = (vol - vol.rolling(sub_window).mean()) / (vol.rolling(sub_window).std() + 1e-8)
        raw = raw1 * raw2 * raw3
        raw = raw.ffill().fillna(0)
        sig = np.sign(raw)
        return sig

    @staticmethod
    def alpha_quanta_full_base_179_wf(df, window=20, sub_window=10, p1=0.05, p2=40):
        high = df['high']
        low = df['low']
        close = df['close']
        vol = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        raw1 = (high - low).rolling(window).std() / (close.rolling(window).std() + 1e-8)
        raw2 = ret.rolling(5).mean().pipe(np.sign)
        raw3 = (vol - vol.rolling(sub_window).mean()) / (vol.rolling(sub_window).std() + 1e-8)
        raw = raw1 * raw2 * raw3
        raw = raw.ffill().fillna(0)
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        sig = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        sig = sig.ffill().fillna(0)
        return sig

    @staticmethod
    def alpha_quanta_full_base_180_rank(df, window=15):
        ret = df['close'].pct_change().replace(0, 1e-8)
        delta_ret = ret.diff(2)
        vol = ret.rolling(3).std().replace(0, np.nan).ffill().fillna(1)
        corr = ret.rolling(3).corr(df.get('matchingVolume', np.log1p(df['close'] * df.get('matchingVolume', 1))).fillna(0)).replace(0, np.nan).ffill().fillna(0)
        raw = delta_ret / vol * (1 - corr) * np.sign(ret.fillna(0))
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_180_tanh(df, window=5):
        ret = df['close'].pct_change().replace(0, 1e-8)
        delta_ret = ret.diff(2)
        vol = ret.rolling(3).std().replace(0, np.nan).ffill().fillna(1)
        corr = ret.rolling(3).corr(df.get('matchingVolume', np.log1p(df['close'] * df.get('matchingVolume', 1))).fillna(0)).replace(0, np.nan).ffill().fillna(0)
        raw = delta_ret / vol * (1 - corr) * np.sign(ret.fillna(0))
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_180_zscore(df, window=5):
        ret = df['close'].pct_change().replace(0, 1e-8)
        delta_ret = ret.diff(2)
        vol = ret.rolling(3).std().replace(0, np.nan).ffill().fillna(1)
        corr = ret.rolling(3).corr(df.get('matchingVolume', np.log1p(df['close'] * df.get('matchingVolume', 1))).fillna(0)).replace(0, np.nan).ffill().fillna(0)
        raw = delta_ret / vol * (1 - corr) * np.sign(ret.fillna(0))
        mean_ = raw.rolling(window).mean().fillna(0)
        std_ = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_180_sign(df):
        ret = df['close'].pct_change().replace(0, 1e-8)
        delta_ret = ret.diff(2)
        vol = ret.rolling(3).std().replace(0, np.nan).ffill().fillna(1)
        corr = ret.rolling(3).corr(df.get('matchingVolume', np.log1p(df['close'] * df.get('matchingVolume', 1))).fillna(0)).replace(0, np.nan).ffill().fillna(0)
        raw = delta_ret / vol * (1 - corr) * np.sign(ret.fillna(0))
        normalized = np.sign(raw.fillna(0))
        return normalized

    @staticmethod
    def alpha_quanta_full_base_180_wf(df, window=60, p1=0.3):
        ret = df['close'].pct_change().replace(0, 1e-8)
        delta_ret = ret.diff(2)
        vol = ret.rolling(3).std().replace(0, np.nan).ffill().fillna(1)
        corr = ret.rolling(3).corr(df.get('matchingVolume', np.log1p(df['close'] * df.get('matchingVolume', 1))).fillna(0)).replace(0, np.nan).ffill().fillna(0)
        raw = delta_ret / vol * (1 - corr) * np.sign(ret.fillna(0))
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_181_rank(df, window=100):
        hl = df['high'] - df['low']
        pct_hl = hl.pct_change(2).replace([np.inf, -np.inf], np.nan)
        vol_ratio = 1 - df['matchingVolume'].rolling(2).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        raw = pct_hl * vol_ratio * (hl - hl.rolling(window).mean()) / hl.rolling(window).std().replace(0, np.nan)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_181_tanh(df, window=95):
        hl = df['high'] - df['low']
        pct_hl = hl.pct_change(2).replace([np.inf, -np.inf], np.nan)
        vol_ratio = 1 - df['matchingVolume'].rolling(2).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        raw = pct_hl * vol_ratio * (hl - hl.rolling(window).mean()) / hl.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_181_zscore(df, window=100):
        hl = df['high'] - df['low']
        pct_hl = hl.pct_change(2).replace([np.inf, -np.inf], np.nan)
        vol_ratio = 1 - df['matchingVolume'].rolling(2).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        raw = pct_hl * vol_ratio * (hl - hl.rolling(window).mean()) / hl.rolling(window).std().replace(0, np.nan)
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        signal = zscore.clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_181_sign(df, window=50):
        hl = df['high'] - df['low']
        pct_hl = hl.pct_change(2).replace([np.inf, -np.inf], np.nan)
        vol_ratio = 1 - df['matchingVolume'].rolling(2).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        raw = pct_hl * vol_ratio * (hl - hl.rolling(window).mean()) / hl.rolling(window).std().replace(0, np.nan)
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_181_wf(df, window=100, p1=0.7):
        hl = df['high'] - df['low']
        pct_hl = hl.pct_change(2).replace([np.inf, -np.inf], np.nan)
        vol_ratio = 1 - df['matchingVolume'].rolling(2).mean() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        raw = pct_hl * vol_ratio * (hl - hl.rolling(window).mean()) / hl.rolling(window).std().replace(0, np.nan)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_182_rank(df, window=60):
        _ret = df['close'].pct_change()
        _abs_ret = _ret.abs()
        _abs_ret_max = _abs_ret.rolling(window).max()
        _abs_ret_sum = _abs_ret.rolling(10).sum() + 1e-8
        _ret_std = _ret.rolling(10).std()
        _raw = (_abs_ret_max / _abs_ret_sum) * _ret_std
        return -(_raw.rolling(10).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_full_base_182_tanh(df, window=5):
        _ret = df['close'].pct_change()
        _abs_ret = _ret.abs()
        _abs_ret_max = _abs_ret.rolling(window).max()
        _abs_ret_sum = _abs_ret.rolling(10).sum() + 1e-8
        _ret_std = _ret.rolling(10).std()
        _raw = (_abs_ret_max / _abs_ret_sum) * _ret_std
        return -np.tanh(_raw / _raw.rolling(10).std())

    @staticmethod
    def alpha_quanta_full_base_182_zscore(df, window=45):
        _ret = df['close'].pct_change()
        _abs_ret = _ret.abs()
        _abs_ret_max = _abs_ret.rolling(window).max()
        _abs_ret_sum = _abs_ret.rolling(10).sum() + 1e-8
        _ret_std = _ret.rolling(10).std()
        _raw = (_abs_ret_max / _abs_ret_sum) * _ret_std
        return -((_raw - _raw.rolling(10).mean()) / _raw.rolling(10).std()).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_182_sign(df, window=35):
        _ret = df['close'].pct_change()
        _abs_ret = _ret.abs()
        _abs_ret_max = _abs_ret.rolling(window).max()
        _abs_ret_sum = _abs_ret.rolling(10).sum() + 1e-8
        _ret_std = _ret.rolling(10).std()
        _raw = (_abs_ret_max / _abs_ret_sum) * _ret_std
        return np.sign(_raw)

    @staticmethod
    def alpha_quanta_full_base_182_wf(df, window=30, p1=0.1):
        p2 = 10
        _ret = df['close'].pct_change()
        _abs_ret = _ret.abs()
        _abs_ret_max = _abs_ret.rolling(window).max()
        _abs_ret_sum = _abs_ret.rolling(p2).sum() + 1e-8
        _ret_std = _ret.rolling(p2).std()
        _raw = (_abs_ret_max / _abs_ret_sum) * _ret_std
        _low = _raw.rolling(p2).quantile(p1)
        _high = _raw.rolling(p2).quantile(1 - p1)
        _winsorized = _raw.clip(lower=_low, upper=_high, axis=0)
        _norm = np.arctanh(((_winsorized - _low) / (_high - _low + 1e-9)) * 1.98 - 0.99)
        return -_norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_183_rank(df, window=15):
        close = df['close']
        sma_20 = close.rolling(window).mean().replace(0, np.nan)
        ratio = close / (sma_20 + 1e-8)
        zscore = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        rank = zscore.rolling(window).rank(pct=True) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_183_tanh(df, window=10):
        close = df['close']
        sma_20 = close.rolling(window).mean().replace(0, np.nan)
        ratio = close / (sma_20 + 1e-8)
        zscore = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(zscore / zscore.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_183_zscore(df, window=20):
        close = df['close']
        sma_20 = close.rolling(window).mean().replace(0, np.nan)
        ratio = close / (sma_20 + 1e-8)
        zscore = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        normalized = ((zscore - zscore.rolling(window).mean()) / zscore.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_183_sign(df, window=15):
        close = df['close']
        sma_20 = close.rolling(window).mean().replace(0, np.nan)
        ratio = close / (sma_20 + 1e-8)
        zscore = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        normalized = np.sign(zscore)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_183_wf(df, window=20, p1=0.3):
        p2 = window
        close = df['close']
        sma_20 = close.rolling(window).mean().replace(0, np.nan)
        ratio = close / (sma_20 + 1e-8)
        zscore = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        low = zscore.rolling(p2).quantile(p1)
        high = zscore.rolling(p2).quantile(1 - p1)
        winsorized = zscore.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_184_rank(df, window=15):
        ret = df['close'].pct_change()
        mean40 = ret.rolling(window).mean()
        std40 = ret.rolling(window).std().replace(0, np.nan)
        zscore40 = mean40 / (std40 + 0.01)
        mean10 = ret.rolling(10).mean()
        std20 = ret.rolling(20).std().replace(0, np.nan)
        delta = mean10 - mean10.shift(5)
        raw = zscore40 + delta / (std20 + 0.01)
        result = (raw.rolling(252).rank(pct=True) * 2) - 1
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_184_tanh(df, window=15):
        ret = df['close'].pct_change()
        mean40 = ret.rolling(window).mean()
        std40 = ret.rolling(window).std().replace(0, np.nan)
        zscore40 = mean40 / (std40 + 0.01)
        mean10 = ret.rolling(10).mean()
        std20 = ret.rolling(20).std().replace(0, np.nan)
        delta = mean10 - mean10.shift(5)
        raw = zscore40 + delta / (std20 + 0.01)
        result = np.tanh(raw / raw.rolling(252).std().replace(0, np.nan))
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_184_zscore(df, window=15):
        ret = df['close'].pct_change()
        mean40 = ret.rolling(window).mean()
        std40 = ret.rolling(window).std().replace(0, np.nan)
        zscore40 = mean40 / (std40 + 0.01)
        mean10 = ret.rolling(10).mean()
        std20 = ret.rolling(20).std().replace(0, np.nan)
        delta = mean10 - mean10.shift(5)
        raw = zscore40 + delta / (std20 + 0.01)
        rolling_mean = raw.rolling(252).mean()
        rolling_std = raw.rolling(252).std().replace(0, np.nan)
        result = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_184_sign(df, window=10):
        ret = df['close'].pct_change()
        mean40 = ret.rolling(window).mean()
        std40 = ret.rolling(window).std().replace(0, np.nan)
        zscore40 = mean40 / (std40 + 0.01)
        mean10 = ret.rolling(10).mean()
        std20 = ret.rolling(20).std().replace(0, np.nan)
        delta = mean10 - mean10.shift(5)
        raw = zscore40 + delta / (std20 + 0.01)
        result = np.sign(raw)
        return pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_184_wf(df, window=20, p1=0.3):
        ret = df['close'].pct_change()
        mean40 = ret.rolling(window).mean()
        std40 = ret.rolling(window).std().replace(0, np.nan)
        zscore40 = mean40 / (std40 + 0.01)
        mean10 = ret.rolling(10).mean()
        std20 = ret.rolling(20).std().replace(0, np.nan)
        delta = mean10 - mean10.shift(5)
        raw = zscore40 + delta / (std20 + 0.01)
        p2 = 252
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_185_rank(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        mean_ret = ret.rolling(40).mean()
        log_vol = np.log1p(volume.rolling(20).sum())
        mean_log_vol = log_vol.rolling(40, min_periods=1).mean().replace(0, np.nan)
        raw = mean_ret * log_vol / (mean_log_vol + 1e-8) - mean_ret.rolling(40, min_periods=1).mean()
        normed = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normed.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_185_tanh(df, window=45):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        mean_ret = ret.rolling(40).mean()
        log_vol = np.log1p(volume.rolling(20).sum())
        mean_log_vol = log_vol.rolling(40, min_periods=1).mean().replace(0, np.nan)
        raw = mean_ret * log_vol / (mean_log_vol + 1e-8) - mean_ret.rolling(40, min_periods=1).mean()
        normed = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normed.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_185_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        mean_ret = ret.rolling(40).mean()
        log_vol = np.log1p(volume.rolling(20).sum())
        mean_log_vol = log_vol.rolling(40, min_periods=1).mean().replace(0, np.nan)
        raw = mean_ret * log_vol / (mean_log_vol + 1e-8) - mean_ret.rolling(40, min_periods=1).mean()
        normed = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normed.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_185_sign(df):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        mean_ret = ret.rolling(40).mean()
        log_vol = np.log1p(volume.rolling(20).sum())
        mean_log_vol = log_vol.rolling(40, min_periods=1).mean().replace(0, np.nan)
        raw = mean_ret * log_vol / (mean_log_vol + 1e-8) - mean_ret.rolling(40, min_periods=1).mean()
        normed = np.sign(raw)
        return pd.Series(normed, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_185_wf(df, p1=0.9, p2=10):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        mean_ret = ret.rolling(40).mean()
        log_vol = np.log1p(volume.rolling(20).sum())
        mean_log_vol = log_vol.rolling(40, min_periods=1).mean().replace(0, np.nan)
        raw = mean_ret * log_vol / (mean_log_vol + 1e-8) - mean_ret.rolling(40, min_periods=1).mean()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normed = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normed.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_186_rank(df, window=75):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        corr = high_low.rolling(window).corr(volume_ratio)
        ma5 = high_low.rolling(5).mean()
        delta_ma5 = ma5 - ma5.shift(1)
        raw = corr * delta_ma5
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_186_tanh(df, window=35):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        corr = high_low.rolling(window).corr(volume_ratio)
        ma5 = high_low.rolling(5).mean()
        delta_ma5 = ma5 - ma5.shift(1)
        raw = corr * delta_ma5
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_186_zscore(df, window=25):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        corr = high_low.rolling(window).corr(volume_ratio)
        ma5 = high_low.rolling(5).mean()
        delta_ma5 = ma5 - ma5.shift(1)
        raw = corr * delta_ma5
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_186_sign(df, window=95):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        corr = high_low.rolling(window).corr(volume_ratio)
        ma5 = high_low.rolling(5).mean()
        delta_ma5 = ma5 - ma5.shift(1)
        raw = corr * delta_ma5
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_186_wf(df, window=90, p1=0.1):
        high_low = (df['high'] - df['low']) / (df['close'] + 1e-8)
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean() + 1e-8)
        corr = high_low.rolling(window).corr(volume_ratio)
        ma5 = high_low.rolling(5).mean()
        delta_ma5 = ma5 - ma5.shift(1)
        raw = corr * delta_ma5
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-8)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_187_rank(df, window=95):
        raw = df['high'].sub(df['low']).div(df['close'].add(1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = df['close'].rolling(window).std().add(1e-8)
        ratio = rolling_mean.div(rolling_std)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std().add(1e-8)
        zscore = (ratio - mean_ratio).div(std_ratio)
        signal = zscore.rolling(window).rank(pct=True).mul(2).sub(1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_187_tanh(df, window=30):
        raw = df['high'].sub(df['low']).div(df['close'].add(1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = df['close'].rolling(window).std().add(1e-8)
        ratio = rolling_mean.div(rolling_std)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std().add(1e-8)
        zscore = (ratio - mean_ratio).div(std_ratio)
        signal = np.tanh(zscore.div(zscore.rolling(window).std().add(1e-8)))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_187_zscore(df, window=60):
        raw = df['high'].sub(df['low']).div(df['close'].add(1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = df['close'].rolling(window).std().add(1e-8)
        ratio = rolling_mean.div(rolling_std)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std().add(1e-8)
        zscore = (ratio - mean_ratio).div(std_ratio)
        z_mean = zscore.rolling(window).mean()
        z_std = zscore.rolling(window).std().add(1e-8)
        signal = ((zscore - z_mean).div(z_std)).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_187_sign(df, window=35):
        raw = df['high'].sub(df['low']).div(df['close'].add(1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = df['close'].rolling(window).std().add(1e-8)
        ratio = rolling_mean.div(rolling_std)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std().add(1e-8)
        zscore = (ratio - mean_ratio).div(std_ratio)
        signal = np.sign(zscore)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_187_wf(df, window=90, factor=7):
        raw = df['high'].sub(df['low']).div(df['close'].add(1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = df['close'].rolling(window).std().add(1e-8)
        ratio = rolling_mean.div(rolling_std)
        mean_ratio = ratio.rolling(window).mean()
        std_ratio = ratio.rolling(window).std().add(1e-8)
        zscore = (ratio - mean_ratio).div(std_ratio)
        p = 0.05
        p2 = factor
        low = zscore.rolling(p2).quantile(p)
        high = zscore.rolling(p2).quantile(1 - p)
        winsorized = zscore.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_188_rank(df, window=5):
        volume_inv = 1.0 / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        returns = df['close'].pct_change()
        corr = returns.rolling(10).corr(volume_inv)
        days = pd.Series(np.arange(len(df)), index=df.index)
        days_corr = days.rolling(window).corr(corr)
        corr_var = corr.rolling(window).var().replace(0, np.nan)
        regbeta = days_corr / corr_var
        sign = np.sign(regbeta)
        raw = sign * corr.abs()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(method='ffill').fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_188_tanh(df, window=5):
        volume_inv = 1.0 / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        returns = df['close'].pct_change()
        corr = returns.rolling(10).corr(volume_inv)
        days = pd.Series(np.arange(len(df)), index=df.index)
        days_corr = days.rolling(window).corr(corr)
        corr_var = corr.rolling(window).var().replace(0, np.nan)
        regbeta = days_corr / corr_var
        sign = np.sign(regbeta)
        raw = sign * corr.abs()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(method='ffill').fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_188_zscore(df, window=5):
        volume_inv = 1.0 / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        returns = df['close'].pct_change()
        corr = returns.rolling(10).corr(volume_inv)
        days = pd.Series(np.arange(len(df)), index=df.index)
        days_corr = days.rolling(window).corr(corr)
        corr_var = corr.rolling(window).var().replace(0, np.nan)
        regbeta = days_corr / corr_var
        sign = np.sign(regbeta)
        raw = sign * corr.abs()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.fillna(method='ffill').fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_188_sign(df, window=5):
        volume_inv = 1.0 / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        returns = df['close'].pct_change()
        corr = returns.rolling(10).corr(volume_inv)
        days = pd.Series(np.arange(len(df)), index=df.index)
        days_corr = days.rolling(window).corr(corr)
        corr_var = corr.rolling(window).var().replace(0, np.nan)
        regbeta = days_corr / corr_var
        sign = np.sign(regbeta)
        raw = sign * corr.abs()
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_188_wf(df, window=20, quantile=0.9):
        volume_inv = 1.0 / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        returns = df['close'].pct_change()
        corr = returns.rolling(10).corr(volume_inv)
        days = pd.Series(np.arange(len(df)), index=df.index)
        days_corr = days.rolling(window).corr(corr)
        corr_var = corr.rolling(window).var().replace(0, np.nan)
        regbeta = days_corr / corr_var
        sign = np.sign(regbeta)
        raw = sign * corr.abs()
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        epsilon = 1e-9
        normalized = np.arctanh(((winsorized - low) / (high - low + epsilon)) * 1.98 - 0.99)
        signal = normalized
        signal = signal.fillna(method='ffill').fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_189_rank(df, window=25):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        vol_log = np.log1p(df['matchingVolume'])
        volume_change = vol_log.diff()
        spread = df['high'] - df['low']
        spread = spread.replace(0, np.nan)
        volume_change_spread = volume_change / spread
        volume_change_spread = volume_change_spread.replace([np.inf, -np.inf], np.nan)
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std().replace(0, np.nan)
        sharpe = ret_mean / (ret_std + 1e-8)
        vol_mean = volume_change_spread.rolling(window).mean()
        vol_std = volume_change_spread.rolling(window).std().replace(0, np.nan)
        vol_sharpe = vol_mean / (vol_std + 1e-8)
        raw = sharpe - vol_sharpe
        normalizer = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalizer.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_189_tanh(df, window=5):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        vol_log = np.log1p(df['matchingVolume'])
        volume_change = vol_log.diff()
        spread = df['high'] - df['low']
        spread = spread.replace(0, np.nan)
        volume_change_spread = volume_change / spread
        volume_change_spread = volume_change_spread.replace([np.inf, -np.inf], np.nan)
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std().replace(0, np.nan)
        sharpe = ret_mean / (ret_std + 1e-8)
        vol_mean = volume_change_spread.rolling(window).mean()
        vol_std = volume_change_spread.rolling(window).std().replace(0, np.nan)
        vol_sharpe = vol_mean / (vol_std + 1e-8)
        raw = sharpe - vol_sharpe
        norm = np.tanh(raw / (raw.rolling(window).std().replace(0, np.nan) + 1e-8))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_189_zscore(df, window=5):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        vol_log = np.log1p(df['matchingVolume'])
        volume_change = vol_log.diff()
        spread = df['high'] - df['low']
        spread = spread.replace(0, np.nan)
        volume_change_spread = volume_change / spread
        volume_change_spread = volume_change_spread.replace([np.inf, -np.inf], np.nan)
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std().replace(0, np.nan)
        sharpe = ret_mean / (ret_std + 1e-8)
        vol_mean = volume_change_spread.rolling(window).mean()
        vol_std = volume_change_spread.rolling(window).std().replace(0, np.nan)
        vol_sharpe = vol_mean / (vol_std + 1e-8)
        raw = sharpe - vol_sharpe
        zscore = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_189_sign(df, window=5):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        vol_log = np.log1p(df['matchingVolume'])
        volume_change = vol_log.diff()
        spread = df['high'] - df['low']
        spread = spread.replace(0, np.nan)
        volume_change_spread = volume_change / spread
        volume_change_spread = volume_change_spread.replace([np.inf, -np.inf], np.nan)
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std().replace(0, np.nan)
        sharpe = ret_mean / (ret_std + 1e-8)
        vol_mean = volume_change_spread.rolling(window).mean()
        vol_std = volume_change_spread.rolling(window).std().replace(0, np.nan)
        vol_sharpe = vol_mean / (vol_std + 1e-8)
        raw = sharpe - vol_sharpe
        sign = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return sign

    @staticmethod
    def alpha_quanta_full_base_189_wf(df, window=30, p1=0.3):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        vol_log = np.log1p(df['matchingVolume'])
        volume_change = vol_log.diff()
        spread = df['high'] - df['low']
        spread = spread.replace(0, np.nan)
        volume_change_spread = volume_change / spread
        volume_change_spread = volume_change_spread.replace([np.inf, -np.inf], np.nan)
        ret_mean = ret.rolling(window).mean()
        ret_std = ret.rolling(window).std().replace(0, np.nan)
        sharpe = ret_mean / (ret_std + 1e-8)
        vol_mean = volume_change_spread.rolling(window).mean()
        vol_std = volume_change_spread.rolling(window).std().replace(0, np.nan)
        vol_sharpe = vol_mean / (vol_std + 1e-8)
        raw = sharpe - vol_sharpe
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_190_rank(df, window=75):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff(1)
        corr = ret.rolling(window).corr(delta_vol)
        mean_ret = ret.rolling(window).mean()
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_190_tanh(df, window=35):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff(1)
        corr = ret.rolling(window).corr(delta_vol)
        mean_ret = ret.rolling(window).mean()
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_190_zscore(df, window=55):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff(1)
        corr = ret.rolling(window).corr(delta_vol)
        mean_ret = ret.rolling(window).mean()
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_190_sign(df, window=40):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff(1)
        corr = ret.rolling(window).corr(delta_vol)
        mean_ret = ret.rolling(window).mean()
        sign = np.sign(mean_ret)
        raw = corr * sign
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_190_wf(df, window=35):
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = vol.diff(1)
        corr = ret.rolling(window).corr(delta_vol)
        mean_ret = ret.rolling(window).mean()
        sign = np.sign(mean_ret)
        raw = corr * sign
        p1 = 0.05
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_191_rank(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change(fill_method=None).fillna(0)
        pos_ret = ret.clip(0, None)
        sum_pos_ret = pos_ret.rolling(window=window).sum()
        vol_ratio = volume / (volume.shift(1) + 1e-8)
        delta_vol = vol_ratio - 1
        std_vol = delta_vol.rolling(window=window).std()
        raw = sum_pos_ret / (std_vol + 1e-8)
        raw_rank = raw.rolling(window=window).rank(pct=True) * 2 - 1
        signal = raw_rank.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_191_tanh(df, window=95):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change(fill_method=None).fillna(0)
        pos_ret = ret.clip(0, None)
        sum_pos_ret = pos_ret.rolling(window=window).sum()
        vol_ratio = volume / (volume.shift(1) + 1e-8)
        delta_vol = vol_ratio - 1
        std_vol = delta_vol.rolling(window=window).std()
        raw = sum_pos_ret / (std_vol + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_191_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change(fill_method=None).fillna(0)
        pos_ret = ret.clip(0, None)
        sum_pos_ret = pos_ret.rolling(window=window).sum()
        vol_ratio = volume / (volume.shift(1) + 1e-8)
        delta_vol = vol_ratio - 1
        std_vol = delta_vol.rolling(window=window).std()
        raw = sum_pos_ret / (std_vol + 1e-8)
        raw_mean = raw.rolling(window=window).mean()
        raw_std = raw.rolling(window=window).std().replace(0, np.nan)
        signal = ((raw - raw_mean) / raw_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_191_sign(df, window=35):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change(fill_method=None).fillna(0)
        pos_ret = ret.clip(0, None)
        sum_pos_ret = pos_ret.rolling(window=window).sum()
        vol_ratio = volume / (volume.shift(1) + 1e-8)
        delta_vol = vol_ratio - 1
        std_vol = delta_vol.rolling(window=window).std()
        raw = sum_pos_ret / (std_vol + 1e-8)
        signal = np.sign(raw) * 1.0
        return signal

    @staticmethod
    def alpha_quanta_full_base_191_wf(df, window=20, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change(fill_method=None).fillna(0)
        pos_ret = ret.clip(0, None)
        sum_pos_ret = pos_ret.rolling(window=window).sum()
        vol_ratio = volume / (volume.shift(1) + 1e-8)
        delta_vol = vol_ratio - 1
        std_vol = delta_vol.rolling(window=window).std()
        raw = sum_pos_ret / (std_vol + 1e-8)
        low = raw.rolling(window=window).quantile(p1)
        high = raw.rolling(window=window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_192_2(df, window=15):
        # Tính return (phần trăm thay đổi của close)
        close = df['close']
        ret = close.pct_change()
        # Lấy dấu của return: 1 nếu dương, -1 nếu âm, 0 nếu bằng 0
        sign_ret = np.sign(ret)
        # Tính delta volume (sự thay đổi khối lượng so với phiên trước)
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff()
        # Rolling correlation giữa sign_ret và delta_vol trong window phiên
        # Không dùng .corr pandas có thể chậm, dùng rolling corr vectorized
        # Sử dụng fillna và clip để đảm bảo ổn định
        cov = sign_ret.rolling(window).cov(delta_vol)
        var_s = sign_ret.rolling(window).var().replace(0, np.nan)
        var_d = delta_vol.rolling(window).var().replace(0, np.nan)
        corr = cov / (var_s * var_d).pow(0.5)
        # Xử lý giá trị thiếu và nhiễu
        corr = corr.fillna(0)

        # Chuẩn hóa về [-1,1] dùng Rolling Rank
        raw = corr
        param = window
        # Trường hợp A: Rolling Rank
        rank_val = raw.rolling(param).rank(pct=True) * 2 - 1
        rank_val = rank_val.fillna(0)

        # Trường hợp B: Dynamic Tanh
        std_val = raw.rolling(param).std().replace(0, np.nan)
        tanh_val = np.tanh(raw / std_val)
        tanh_val = tanh_val.fillna(0)

        # Trường hợp C: Rolling Z-Score/Clip
        mean_val = raw.rolling(param).mean()
        std_val2 = raw.rolling(param).std().replace(0, np.nan)
        zscore_val = ((raw - mean_val) / std_val2).clip(-1, 1)
        zscore_val = zscore_val.fillna(0)

        # Trường hợp D: Sign/Binary Soft
        sign_val = np.sign(raw).fillna(0)

        # Trường hợp E: Winsorized Fisher với p1=0.05, p2=param, p3=param tùy chỉnh
        p1 = 0.05
        p2 = param
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        eps = 1e-9
        normalized = np.arctanh(((winsorized - low) / (high - low + eps)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)

        # Trả về tín hiệu cuối cùng: chọn phương pháp Rolling Rank làm đại diện
        return rank_val

    @staticmethod
    def alpha_quanta_full_base_192_rank(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        ret_sign = np.sign(ret)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        raw = ret_sign.rolling(window).corr(vol_delta)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_192_tanh(df, window=80):
        ret = df['close'].pct_change().fillna(0)
        ret_sign = np.sign(ret)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        raw = ret_sign.rolling(window).corr(vol_delta)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_192_zscore(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        ret_sign = np.sign(ret)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        raw = ret_sign.rolling(window).corr(vol_delta)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_192_sign(df, window=10):
        ret = df['close'].pct_change().fillna(0)
        ret_sign = np.sign(ret)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        raw = ret_sign.rolling(window).corr(vol_delta)
        normalized = np.sign(raw)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_192_wf(df, p1=0.1, window=50):
        ret = df['close'].pct_change().fillna(0)
        ret_sign = np.sign(ret)
        vol_delta = df['matchingVolume'].diff().fillna(0)
        raw = ret_sign.rolling(window).corr(vol_delta)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_193_rank(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).skew() * (ret.rolling(5).std() - ret.rolling(5).std().rolling(10).mean())
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_193_tanh(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).skew() * (ret.rolling(5).std() - ret.rolling(5).std().rolling(10).mean())
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_193_zscore(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).skew() * (ret.rolling(5).std() - ret.rolling(5).std().rolling(10).mean())
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_193_sign(df, window=10):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).skew() * (ret.rolling(5).std() - ret.rolling(5).std().rolling(10).mean())
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_193_wf(df, window=10):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).skew() * (ret.rolling(5).std() - ret.rolling(5).std().rolling(10).mean())
        low = raw.rolling(window).quantile(0.05)
        high = raw.rolling(window).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized_raw = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(normalized_raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_194_rank(df, window=100, sub_window=7):
        vol = df['matchingVolume']
        hl = df['high'] - df['low']
        corr = vol.rolling(window=window).corr(hl)
        delta = corr.diff(sub_window)
        raw = delta
        raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return -raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_194_tanh(df, window=100, sub_window=5):
        vol = df['matchingVolume']
        hl = df['high'] - df['low']
        corr = vol.rolling(window=window).corr(hl)
        delta = corr.diff(sub_window)
        raw = delta
        raw = np.tanh(raw / raw.abs().rolling(window).mean().replace(0, np.nan))
        return -raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_194_zscore(df, window=80, sub_window=7):
        vol = df['matchingVolume']
        hl = df['high'] - df['low']
        corr = vol.rolling(window=window).corr(hl)
        delta = corr.diff(sub_window)
        raw = delta
        mean = raw.rolling(window=window).mean()
        std = raw.rolling(window=window).std().replace(0, np.nan)
        raw = ((raw - mean) / std).clip(-1, 1)
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_194_sign(df, window=80, sub_window=4):
        vol = df['matchingVolume']
        hl = df['high'] - df['low']
        corr = vol.rolling(window=window).corr(hl)
        delta = corr.diff(sub_window)
        raw = delta
        raw = np.sign(raw)
        return -raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_194_wf(df, window=80, sub_window=7, p1=0.05, p2=30):
        vol = df['matchingVolume']
        hl = df['high'] - df['low']
        corr = vol.rolling(window=window).corr(hl)
        delta = corr.diff(sub_window)
        raw = delta
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0).clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_195_rank(df, window_std=7, window_mean=10):
        close = df['close']
        ret = close.pct_change()
        raw = np.log1p(close.rolling(window_std).std()) / (np.log1p(close.rolling(15).std()) + 1e-8) * ret.rolling(window_mean).mean()
        normalized = (raw.rolling(window_mean).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_195_tanh(df, window_std=3, window_mean=20):
        close = df['close']
        ret = close.pct_change()
        raw = np.log1p(close.rolling(window_std).std()) / (np.log1p(close.rolling(15).std()) + 1e-8) * ret.rolling(window_mean).mean()
        normalized = np.tanh(raw / raw.rolling(window_mean).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_195_zscore(df, window_std=7, window_mean=10):
        close = df['close']
        ret = close.pct_change()
        raw = np.log1p(close.rolling(window_std).std()) / (np.log1p(close.rolling(15).std()) + 1e-8) * ret.rolling(window_mean).mean()
        normalized = ((raw - raw.rolling(window_mean).mean()) / raw.rolling(window_mean).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_195_sign(df, window_std=30, window_mean=30):
        close = df['close']
        ret = close.pct_change()
        raw = np.log1p(close.rolling(window_std).std()) / (np.log1p(close.rolling(15).std()) + 1e-8) * ret.rolling(window_mean).mean()
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_195_wf(df, window_std=7, window_mean=10, p1=0.05):
        close = df['close']
        ret = close.pct_change()
        raw = np.log1p(close.rolling(window_std).std()) / (np.log1p(close.rolling(15).std()) + 1e-8) * ret.rolling(window_mean).mean()
        p2 = window_std * 3
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_196_rank(df, window=70):
        high_low_1 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(10).mean() + 1e-8)
        volume_delta = df['matchingVolume'].diff() / (df['matchingVolume'] + 1e-8)
        corr_1 = high_low_1.rolling(window).corr(volume_delta)
        high_low_2 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-8)
        corr_2 = high_low_2.rolling(20).corr(volume_delta)
        raw = corr_1 - corr_2
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_196_tanh(df, window=20):
        high_low_1 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(10).mean() + 1e-8)
        volume_delta = df['matchingVolume'].diff() / (df['matchingVolume'] + 1e-8)
        corr_1 = high_low_1.rolling(window).corr(volume_delta)
        high_low_2 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-8)
        corr_2 = high_low_2.rolling(20).corr(volume_delta)
        raw = corr_1 - corr_2
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_196_zscore(df, window=60):
        high_low_1 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(10).mean() + 1e-8)
        volume_delta = df['matchingVolume'].diff() / (df['matchingVolume'] + 1e-8)
        corr_1 = high_low_1.rolling(window).corr(volume_delta)
        high_low_2 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-8)
        corr_2 = high_low_2.rolling(20).corr(volume_delta)
        raw = corr_1 - corr_2
        normalized = ((raw - raw.rolling(window).mean()) / (raw.rolling(window).std() + 1e-9)).clip(-1, 1)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_196_sign(df, window=30):
        high_low_1 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(10).mean() + 1e-8)
        volume_delta = df['matchingVolume'].diff() / (df['matchingVolume'] + 1e-8)
        corr_1 = high_low_1.rolling(window).corr(volume_delta)
        high_low_2 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-8)
        corr_2 = high_low_2.rolling(20).corr(volume_delta)
        raw = corr_1 - corr_2
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_196_wf(df, window=10, p1=0.7):
        high_low_1 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(10).mean() + 1e-8)
        volume_delta = df['matchingVolume'].diff() / (df['matchingVolume'] + 1e-8)
        corr_1 = high_low_1.rolling(window).corr(volume_delta)
        high_low_2 = (df['high'] - df['low']) / ((df['high'] - df['low']).rolling(20).mean() + 1e-8)
        corr_2 = high_low_2.rolling(20).corr(volume_delta)
        raw = corr_1 - corr_2
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_197_7(df, window=15):
        # Tính return
        ret = df['close'].pct_change()
        # Tính correlation giữa ret và ret.shift(1) trong window=5 (không lookahead)
        corr5 = ret.rolling(5).corr(ret.shift(1))
        # Tính correlation giữa ret và ret.shift(3) trong window=10
        corr10 = ret.rolling(10).corr(ret.shift(3))
        # Tín hiệu sign * mean return
        diff = corr5 - corr10
        signal_raw = np.sign(diff) * ret.rolling(window).mean()
        # Chuẩn hóa Rolling Z-score về [-1, 1]
        mean = signal_raw.rolling(window).mean()
        std = signal_raw.rolling(window).std().replace(0, np.nan)
        signal = ((signal_raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_197_rank(df, window1=3, window2=50):
        returns = df['close'].pct_change().fillna(0)
        raw = np.sign(returns.rolling(window1).corr(returns.shift(1)) - returns.rolling(window2).corr(returns.shift(3))) * returns.rolling(window1).mean()
        normalized = (raw.rolling(window1).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_197_tanh(df, window1=5, window2=40):
        returns = df['close'].pct_change().fillna(0)
        raw = np.sign(returns.rolling(window1).corr(returns.shift(1)) - returns.rolling(window2).corr(returns.shift(3))) * returns.rolling(window1).mean()
        std_raw = raw.rolling(window1).std()
        normalized = np.tanh(raw / std_raw.replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_197_zscore(df, window1=3, window2=60):
        returns = df['close'].pct_change().fillna(0)
        raw = np.sign(returns.rolling(window1).corr(returns.shift(1)) - returns.rolling(window2).corr(returns.shift(3))) * returns.rolling(window1).mean()
        mean_raw = raw.rolling(window1).mean()
        std_raw = raw.rolling(window1).std()
        normalized = ((raw - mean_raw) / std_raw.replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_197_sign(df, window1=5, window2=50):
        returns = df['close'].pct_change().fillna(0)
        raw = np.sign(returns.rolling(window1).corr(returns.shift(1)) - returns.rolling(window2).corr(returns.shift(3))) * returns.rolling(window1).mean()
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_197_wf(df, window1=5, window2=100, norm_window=20):
        returns = df['close'].pct_change().fillna(0)
        raw = np.sign(returns.rolling(window1).corr(returns.shift(1)) - returns.rolling(window2).corr(returns.shift(3))) * returns.rolling(window1).mean()
        p1 = 0.1
        p2 = norm_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm_series = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        norm_series = norm_series.clip(-0.99 + 1e-9, 0.99 - 1e-9)
        normalized = np.arctanh(norm_series)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_198_rank(df, window=10, sub_window=90):
        ret = df['close'].pct_change()
        mean_20 = ret.rolling(window).mean()
        rank_20 = mean_20.rolling(window).rank(pct=True) * 2 - 1
        mean_60 = ret.rolling(sub_window).mean()
        std_60 = ret.rolling(sub_window).std()
        sign = np.sign(mean_60 - std_60)
        raw = rank_20 * sign
        return -raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_198_tanh(df, window=3, sub_window=40):
        ret = df['close'].pct_change()
        mean_20 = ret.rolling(window).mean()
        mean_60 = ret.rolling(sub_window).mean()
        std_60 = ret.rolling(sub_window).std()
        sign = np.sign(mean_60 - std_60)
        raw = mean_20 * sign
        norm = np.tanh(raw / raw.rolling(window).std())
        return -norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_198_zscore(df, window=30, sub_window=90):
        ret = df['close'].pct_change()
        mean_20 = ret.rolling(window).mean()
        mean_60 = ret.rolling(sub_window).mean()
        std_60 = ret.rolling(sub_window).std()
        sign = np.sign(mean_60 - std_60)
        raw = mean_20 * sign
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_198_sign(df, window=3, sub_window=50):
        ret = df['close'].pct_change()
        mean_20 = ret.rolling(window).mean()
        mean_60 = ret.rolling(sub_window).mean()
        std_60 = ret.rolling(sub_window).std()
        sign = np.sign(mean_60 - std_60)
        raw = mean_20 * sign
        norm = np.sign(raw)
        return -norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_198_wf(df, window=1, sub_window=80, p1=0.05, p2=120):
        ret = df['close'].pct_change()
        mean_20 = ret.rolling(window).mean()
        mean_60 = ret.rolling(sub_window).mean()
        std_60 = ret.rolling(sub_window).std()
        sign = np.sign(mean_60 - std_60)
        raw = mean_20 * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_199_rank(df, window=100):
        # Tinh high-low range normalized by close
        raw_hl = (df['high'] - df['low']) / (df['close'] + 1e-8)
        # Tinh volume delta normalized by volume
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        # Rolling correlation
        corr = raw_hl.rolling(window).corr(vol_delta)
        # Tinh return
        ret = df['close'].pct_change()
        # Rolling std and mean of return
        std_ret = ret.rolling(window).std()
        mean_ret = ret.rolling(window).mean()
        # Component 2
        component2 = 1 - std_ret / (mean_ret.abs() + 1e-8)
        raw = corr * component2
        # Rolling rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_199_tanh(df, window=55):
        raw_hl = (df['high'] - df['low']) / (df['close'] + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw_hl.rolling(window).corr(vol_delta)
        ret = df['close'].pct_change()
        std_ret = ret.rolling(window).std()
        mean_ret = ret.rolling(window).mean()
        component2 = 1 - std_ret / (mean_ret.abs() + 1e-8)
        raw = corr * component2
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_199_zscore(df, window=55):
        raw_hl = (df['high'] - df['low']) / (df['close'] + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw_hl.rolling(window).corr(vol_delta)
        ret = df['close'].pct_change()
        std_ret = ret.rolling(window).std()
        mean_ret = ret.rolling(window).mean()
        component2 = 1 - std_ret / (mean_ret.abs() + 1e-8)
        raw = corr * component2
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_199_sign(df, window=5):
        raw_hl = (df['high'] - df['low']) / (df['close'] + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw_hl.rolling(window).corr(vol_delta)
        ret = df['close'].pct_change()
        std_ret = ret.rolling(window).std()
        mean_ret = ret.rolling(window).mean()
        component2 = 1 - std_ret / (mean_ret.abs() + 1e-8)
        raw = corr * component2
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_199_wf(df, window=100, p1=0.1):
        raw_hl = (df['high'] - df['low']) / (df['close'] + 1e-8)
        vol_delta = df['matchingVolume'].diff(1) / (df['matchingVolume'] + 1e-8)
        corr = raw_hl.rolling(window).corr(vol_delta)
        ret = df['close'].pct_change()
        std_ret = ret.rolling(window).std()
        mean_ret = ret.rolling(window).mean()
        component2 = 1 - std_ret / (mean_ret.abs() + 1e-8)
        raw = corr * component2
        # Winsorized Fisher normalization
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_200_rank(df, window=1, sub_window=40):
        # Calculate returns
        ret = df['close'].pct_change().fillna(0)
        # Rolling count of positive returns over window
        pos_count = ret.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        ratio = pos_count / window
        # TS_MEAN and TS_MEDIAN of returns over sub_window
        mean_ret = ret.rolling(sub_window).mean()
        median_ret = ret.rolling(sub_window).median()
        raw = ratio * mean_ret / (median_ret + 1e-8)
        # Rolling Rank normalization (Case A)
        normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_200_tanh(df, window=1, sub_window=20):
        ret = df['close'].pct_change().fillna(0)
        pos_count = ret.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        ratio = pos_count / window
        mean_ret = ret.rolling(sub_window).mean()
        median_ret = ret.rolling(sub_window).median()
        raw = ratio * mean_ret / (median_ret + 1e-8)
        # Dynamic Tanh normalization (Case B)
        normalized = np.tanh(raw / raw.rolling(sub_window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_200_zscore(df, window=1, sub_window=30):
        ret = df['close'].pct_change().fillna(0)
        pos_count = ret.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        ratio = pos_count / window
        mean_ret = ret.rolling(sub_window).mean()
        median_ret = ret.rolling(sub_window).median()
        raw = ratio * mean_ret / (median_ret + 1e-8)
        # Rolling Z-score normalization (Case C)
        mean_raw = raw.rolling(sub_window).mean()
        std_raw = raw.rolling(sub_window).std().replace(0, np.nan)
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_200_sign(df, window=1, sub_window=20):
        ret = df['close'].pct_change().fillna(0)
        pos_count = ret.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        ratio = pos_count / window
        mean_ret = ret.rolling(sub_window).mean()
        median_ret = ret.rolling(sub_window).median()
        raw = ratio * mean_ret / (median_ret + 1e-8)
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_200_wf(df, window=3, sub_window=100, quantile=0.05):
        ret = df['close'].pct_change().fillna(0)
        pos_count = ret.rolling(window).apply(lambda x: (x > 0).sum(), raw=True)
        ratio = pos_count / window
        mean_ret = ret.rolling(sub_window).mean()
        median_ret = ret.rolling(sub_window).median()
        raw = ratio * mean_ret / (median_ret + 1e-8)
        # Winsorized Fisher normalization (Case E)
        low = raw.rolling(sub_window).quantile(quantile)
        high = raw.rolling(sub_window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_201_rank(df, window_rank=40, window_std=30):
        raw = df['high'] - df['low']
        ts_mean = raw.rolling(window_rank).mean()
        ts_std = df['close'].rolling(window_std).std()
        ratio = ts_mean / (ts_std + 1e-8)
        signal = (ratio.rolling(window_rank).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_201_tanh(df, window_rank=30, window_std=70):
        raw = df['high'] - df['low']
        ts_mean = raw.rolling(window_rank).mean()
        ts_std = df['close'].rolling(window_std).std()
        ratio = ts_mean / (ts_std + 1e-8)
        signal = np.tanh(ratio / ratio.rolling(window_rank).std())
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_201_zscore(df, window_rank=20, window_std=60):
        raw = df['high'] - df['low']
        ts_mean = raw.rolling(window_rank).mean()
        ts_std = df['close'].rolling(window_std).std()
        ratio = ts_mean / (ts_std + 1e-8)
        signal = ((ratio - ratio.rolling(window_rank).mean()) / ratio.rolling(window_rank).std()).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_201_sign(df, window_rank=30, window_std=90):
        raw = df['high'] - df['low']
        ts_mean = raw.rolling(window_rank).mean()
        ts_std = df['close'].rolling(window_std).std()
        ratio = ts_mean / (ts_std + 1e-8)
        signal = np.sign(ratio)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_201_wf(df, window_rank=40, window_std=30):
        raw = df['high'] - df['low']
        ts_mean = raw.rolling(window_rank).mean()
        ts_std = df['close'].rolling(window_std).std()
        ratio = ts_mean / (ts_std + 1e-8)
        p1 = 0.05
        p2 = window_rank
        low = ratio.rolling(p2).quantile(p1)
        high = ratio.rolling(p2).quantile(1 - p1)
        winsorized = ratio.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_202_k(df, window=25):
        returns = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = returns.rolling(window).corr(volume)
        raw = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        normed = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normed.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_202_h(df, window=5):
        returns = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = returns.rolling(window).corr(volume)
        raw = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        normed = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normed.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_202_p(df, window=30):
        returns = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = returns.rolling(window).corr(volume)
        raw = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        normed = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normed.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_202_n(df, window=100):
        returns = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = returns.rolling(window).corr(volume)
        raw = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        normed = np.sign(raw)
        return -normed.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_202_r(df, window=30):
        p1 = 0.1
        p2 = 20
        returns = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr = returns.rolling(window).corr(volume)
        raw = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normed = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normed.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_203_rank(df, window=30):
        close = df['close']
        high = df['high']
        low = df['low']
        sign_val = np.sign(close.rolling(5).mean())
        raw = sign_val * (high - low).rolling(10).mean() / (close.rolling(10).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_203_tanh(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        sign_val = np.sign(close.rolling(5).mean())
        raw = sign_val * (high - low).rolling(10).mean() / (close.rolling(10).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_203_zscore(df, window=30):
        close = df['close']
        high = df['high']
        low = df['low']
        sign_val = np.sign(close.rolling(5).mean())
        raw = sign_val * (high - low).rolling(10).mean() / (close.rolling(10).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_203_sign(df):
        close = df['close']
        high = df['high']
        low = df['low']
        sign_val = np.sign(close.rolling(5).mean())
        raw = sign_val * (high - low).rolling(10).mean() / (close.rolling(10).std() + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_203_wf(df, p1=0.1, p2=30):
        close = df['close']
        high = df['high']
        low = df['low']
        sign_val = np.sign(close.rolling(5).mean())
        raw = sign_val * (high - low).rolling(10).mean() / (close.rolling(10).std() + 1e-8)
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_204_rank(df, window=30, sub_window=10):
        close = df['close']
        ts_std_5 = close.rolling(window=window).std()
        ts_std_20 = close.rolling(window=sub_window).std()
        ts_zcore = (close - close.rolling(window=sub_window).mean()) / ts_std_20.replace(0, np.nan)
        raw = np.sign(ts_zcore) * np.log1p(ts_std_20.clip(lower=0))
        signal = (raw.rolling(window=sub_window).rank(pct=True) * 2) - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_204_tanh(df, window=5, sub_window=20):
        close = df['close']
        ts_std_5 = close.rolling(window=window).std()
        ts_std_20 = close.rolling(window=sub_window).std()
        ts_zcore = (close - close.rolling(window=sub_window).mean()) / ts_std_20.replace(0, np.nan)
        raw = np.sign(ts_zcore) * np.log1p(ts_std_20.clip(lower=0))
        signal = np.tanh(raw / raw.rolling(window=sub_window).std().replace(0, np.nan))
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_204_zscore(df, window=10, sub_window=20):
        close = df['close']
        ts_std_5 = close.rolling(window=window).std()
        ts_std_20 = close.rolling(window=sub_window).std()
        ts_zcore = (close - close.rolling(window=sub_window).mean()) / ts_std_20.replace(0, np.nan)
        raw = np.sign(ts_zcore) * np.log1p(ts_std_20.clip(lower=0))
        signal = ((raw - raw.rolling(window=sub_window).mean()) / raw.rolling(window=sub_window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_204_sign(df, window=1, sub_window=10):
        close = df['close']
        ts_std_5 = close.rolling(window=window).std()
        ts_std_20 = close.rolling(window=sub_window).std()
        ts_zcore = (close - close.rolling(window=sub_window).mean()) / ts_std_20.replace(0, np.nan)
        raw = np.sign(ts_zcore) * np.log1p(ts_std_20.clip(lower=0))
        signal = np.sign(raw)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_204_wf(df, window=40, sub_window=30):
        close = df['close']
        ts_std_5 = close.rolling(window=window).std()
        ts_std_20 = close.rolling(window=sub_window).std()
        ts_zcore = (close - close.rolling(window=sub_window).mean()) / ts_std_20.replace(0, np.nan)
        raw = np.sign(ts_zcore) * np.log1p(ts_std_20.clip(lower=0))
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(window=p2).quantile(p1)
        high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_205_k(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw = rolling_corr * (close / (volume + 1e-8)).rolling(10).mean()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_full_base_205_h(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw = rolling_corr * (close / (volume + 1e-8)).rolling(10).mean()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8))
        return normalized

    @staticmethod
    def alpha_quanta_full_base_205_e(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw = rolling_corr * (close / (volume + 1e-8)).rolling(10).mean()
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_205_y(df, window=15):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw = rolling_corr * (close / (volume + 1e-8)).rolling(10).mean()
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_205_r(df, window=10, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        rolling_corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw = rolling_corr * (close / (volume + 1e-8)).rolling(10).mean()
        p2 = int(window / 2) if window >= 2 else 2
        low = raw.rolling(p2).quantile(p1).ffill().fillna(raw.min())
        high = raw.rolling(p2).quantile(1 - p1).ffill().fillna(raw.max())
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_206_rank(df, window=25):
        close = df['close']
        mean_5 = close.rolling(window).mean()
        std_30 = close.rolling(30).std()
        raw = mean_5 / (std_30 + 1)
        rank_raw = raw.rolling(5).rank(pct=True) * 2 - 1
        mean_30 = close.rolling(30).mean()
        zscore = (mean_30 - mean_30.rolling(10).mean()) / mean_30.rolling(10).std()
        zscore = zscore.clip(-1, 1)
        signal = rank_raw * zscore
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_206_tanh(df, window=10):
        close = df['close']
        mean_5 = close.rolling(window).mean()
        std_30 = close.rolling(30).std()
        raw = mean_5 / (std_30 + 1)
        tanh_raw = np.tanh(raw / raw.rolling(30).std())
        mean_30 = close.rolling(30).mean()
        zscore = (mean_30 - mean_30.rolling(10).mean()) / mean_30.rolling(10).std()
        zscore = np.tanh(zscore)
        signal = tanh_raw * zscore
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_206_zscore(df, window=60):
        close = df['close']
        mean_5 = close.rolling(window).mean()
        std_30 = close.rolling(30).std()
        raw = mean_5 / (std_30 + 1)
        zscore_raw = (raw - raw.rolling(30).mean()) / raw.rolling(30).std()
        zscore_raw = zscore_raw.clip(-1, 1)
        mean_30 = close.rolling(30).mean()
        zscore = (mean_30 - mean_30.rolling(10).mean()) / mean_30.rolling(10).std()
        zscore = zscore.clip(-1, 1)
        signal = zscore_raw * zscore
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_206_sign(df, window=70):
        close = df['close']
        mean_5 = close.rolling(window).mean()
        std_30 = close.rolling(30).std()
        raw = mean_5 / (std_30 + 1)
        sign_raw = np.sign(raw)
        mean_30 = close.rolling(30).mean()
        zscore = (mean_30 - mean_30.rolling(10).mean()) / mean_30.rolling(10).std()
        zscore = np.sign(zscore)
        signal = sign_raw * zscore
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_206_wf(df, window=40, p1=0.9):
        close = df['close']
        mean_5 = close.rolling(window).mean()
        std_30 = close.rolling(30).std()
        raw = mean_5 / (std_30 + 1)
        p2 = 30
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        mean_30 = close.rolling(30).mean()
        zscore = (mean_30 - mean_30.rolling(10).mean()) / mean_30.rolling(10).std()
        zscore_low = zscore.rolling(p2).quantile(p1)
        zscore_high = zscore.rolling(p2).quantile(1 - p1)
        zscore_wins = zscore.clip(lower=zscore_low, upper=zscore_high, axis=0)
        zscore_norm = np.arctanh(((zscore_wins - zscore_low) / (zscore_high - zscore_low + 1e-9)) * 1.98 - 0.99)
        signal = normalized * zscore_norm
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_207_rank(df, window=65):
        delta_window = 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        mean_ret_delta = mean_ret.diff(delta_window)
        signal = raw * np.sign(mean_ret_delta)
        normalized = (signal.rolling(window * 2).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_207_tanh(df, window=90):
        delta_window = 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        mean_ret_delta = mean_ret.diff(delta_window)
        signal = raw * np.sign(mean_ret_delta)
        normalized = np.tanh(signal / signal.rolling(window * 2).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_207_zscore(df, window=90):
        delta_window = 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        mean_ret_delta = mean_ret.diff(delta_window)
        signal = raw * np.sign(mean_ret_delta)
        normalized = ((signal - signal.rolling(window * 2).mean()) / signal.rolling(window * 2).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_207_sign(df, window=90):
        delta_window = 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        mean_ret_delta = mean_ret.diff(delta_window)
        signal = raw * np.sign(mean_ret_delta)
        normalized = np.sign(signal)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_207_wf(df, window=20, p1=0.1):
        p2 = window * 2
        delta_window = 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_ret = ret.rolling(window).std()
        raw = mean_ret / (std_ret + 1e-8)
        mean_ret_delta = mean_ret.diff(delta_window)
        signal = raw * np.sign(mean_ret_delta)
        low = signal.rolling(p2).quantile(p1)
        high = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_208_rank(df, window=85):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        mean_hl = hl.rolling(window).mean()
        ratio = hl / (mean_hl + 1e-8)
        corr = ret.rolling(window).corr(ratio)
        rank = corr.rolling(window).rank(pct=True)
        signal = (rank * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_208_tanh(df, window=45):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        mean_hl = hl.rolling(window).mean()
        ratio = hl / (mean_hl + 1e-8)
        corr = ret.rolling(window).corr(ratio)
        raw = corr
        std = raw.rolling(window).std().replace(0, 1e-8)
        signal = np.tanh(raw / std)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_208_zscore(df, window=5):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        mean_hl = hl.rolling(window).mean()
        ratio = hl / (mean_hl + 1e-8)
        corr = ret.rolling(window).corr(ratio)
        raw = corr
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, 1e-8)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_208_sign(df, window=5):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        mean_hl = hl.rolling(window).mean()
        ratio = hl / (mean_hl + 1e-8)
        corr = ret.rolling(window).corr(ratio)
        signal = np.sign(corr)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_208_wf(df, window=60, p1=0.1, p2=30):
        ret = df['close'].pct_change()
        hl = df['high'] - df['low']
        mean_hl = hl.rolling(window).mean()
        ratio = hl / (mean_hl + 1e-8)
        corr = ret.rolling(window).corr(ratio)
        raw = corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        low = low.fillna(raw.quantile(p1))
        high = high.fillna(raw.quantile(1 - p1))
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_209_9(df, window=60, sub_window=40):
        returns = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        volatility = (df['high'] - df['low']) / (df['low'] + 1e-8).replace([np.inf, -np.inf], np.nan)
        corr_window = window
        # Rolling correlation using covariance and variance
        mean_ret = returns.rolling(corr_window).mean()
        mean_vol = volatility.rolling(corr_window).mean()
        diff_ret = returns - mean_ret
        diff_vol = volatility - mean_vol
        cov = (diff_ret * diff_vol).rolling(corr_window).sum() / corr_window
        var_ret = (diff_ret ** 2).rolling(corr_window).sum() / corr_window
        var_vol = (diff_vol ** 2).rolling(corr_window).sum() / corr_window
        std_ret = np.sqrt(var_ret.replace(0, np.nan))
        std_vol = np.sqrt(var_vol.replace(0, np.nan))
        ts_corr = cov / (std_ret * std_vol + 1e-9)
        sign = pd.Series(np.sign(ts_corr), index=df.index)
        std_returns = returns.rolling(sub_window).std().replace(0, np.nan)
        raw = sign * std_returns
        # Chuẩn hóa loại B: Dynamic Tanh
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_209_rank(df, window=10, sub_window=30):
        # Normalization Method A: Rolling Rank
        ret = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        ret = ret.fillna(0)
        hl_ratio = (df['high'] - df['low']) / (df['low'] + 1e-8)
        corr_rolling = ret.rolling(window).corr(hl_ratio)
        raw = np.sign(corr_rolling) * ret.rolling(sub_window).std()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_209_tanh(df, window=70, sub_window=40):
        # Normalization Method B: Dynamic Tanh
        ret = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        ret = ret.fillna(0)
        hl_ratio = (df['high'] - df['low']) / (df['low'] + 1e-8)
        corr_rolling = ret.rolling(window).corr(hl_ratio)
        raw = np.sign(corr_rolling) * ret.rolling(sub_window).std()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_209_zscore(df, window=10, sub_window=30):
        # Normalization Method C: Rolling Z-Score Clip
        ret = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        ret = ret.fillna(0)
        hl_ratio = (df['high'] - df['low']) / (df['low'] + 1e-8)
        corr_rolling = ret.rolling(window).corr(hl_ratio)
        raw = np.sign(corr_rolling) * ret.rolling(sub_window).std()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_209_sign(df, window=60, sub_window=40):
        # Normalization Method D: Sign Binary Soft
        ret = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        ret = ret.fillna(0)
        hl_ratio = (df['high'] - df['low']) / (df['low'] + 1e-8)
        corr_rolling = ret.rolling(window).corr(hl_ratio)
        raw = np.sign(corr_rolling) * ret.rolling(sub_window).std()
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_209_wf(df, window=30, sub_window=7, winsor_quantile=0.05):
        # Normalization Method E: Winsorized Fisher
        ret = (df['close'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        ret = ret.fillna(0)
        hl_ratio = (df['high'] - df['low']) / (df['low'] + 1e-8)
        corr_rolling = ret.rolling(window).corr(hl_ratio)
        raw = np.sign(corr_rolling) * ret.rolling(sub_window).std()
        low = raw.rolling(window).quantile(winsor_quantile)
        high = raw.rolling(window).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_210_0(df, window_corr=30, window_rank=1):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close - close.shift(1)
        vol_log = np.log1p(volume)
        corr = delta_close.rolling(window_corr).corr(vol_log)
        vol_rank = volume.rolling(window_rank).rank(pct=True)
        raw = corr * vol_rank
        normalized = ((raw - raw.rolling(window_corr).mean()) / raw.rolling(window_corr).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_210_rank(df, window_corr=10, window_rank=30):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        volume_log = np.log1p(volume)
        corr_series = delta_close.rolling(window=window_corr).corr(volume_log)
        rank_volume = volume.rolling(window=window_rank).rank(pct=True) * 2 - 1
        raw = corr_series * rank_volume
        result = (raw.rolling(window=window_rank).rank(pct=True) * 2) - 1
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_210_tanh(df, window_corr=10, window_rank=40):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        volume_log = np.log1p(volume)
        corr_series = delta_close.rolling(window=window_corr).corr(volume_log)
        rank_volume = volume.rolling(window=window_rank).rank(pct=True) * 2 - 1
        raw = corr_series * rank_volume
        std_raw = raw.rolling(window=window_rank).std().replace(0, np.nan)
        result = np.tanh(raw / std_raw)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_210_zscore(df, window_corr=10, window_rank=10):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        volume_log = np.log1p(volume)
        corr_series = delta_close.rolling(window=window_corr).corr(volume_log)
        rank_volume = volume.rolling(window=window_rank).rank(pct=True) * 2 - 1
        raw = corr_series * rank_volume
        mean_raw = raw.rolling(window=window_rank).mean()
        std_raw = raw.rolling(window=window_rank).std().replace(0, np.nan)
        result = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_210_sign(df, window_corr=20, window_rank=1):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        volume_log = np.log1p(volume)
        corr_series = delta_close.rolling(window=window_corr).corr(volume_log)
        rank_volume = volume.rolling(window=window_rank).rank(pct=True) * 2 - 1
        raw = corr_series * rank_volume
        result = np.sign(raw)
        return result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_210_wf(df, window_corr=10, window_rank=20):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(1)
        volume_log = np.log1p(volume)
        corr_series = delta_close.rolling(window=window_corr).corr(volume_log)
        rank_volume = volume.rolling(window=window_rank).rank(pct=True) * 2 - 1
        raw = corr_series * rank_volume
        quantile_low = 0.05
        low = raw.rolling(window=window_rank).quantile(quantile_low)
        high = raw.rolling(window=window_rank).quantile(1 - quantile_low)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0).clip(-1, 1)
        return result

    @staticmethod
    def alpha_quanta_full_base_211_k(df, window=80, sub_window=20):
        range_mean = (df['high'] - df['low']).rolling(window).mean()
        vol_mean = df['matchingVolume'].rolling(window).mean().replace(0, 1e-8)
        raw = range_mean / vol_mean
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        volume_z = (df['matchingVolume'] - df['matchingVolume'].rolling(sub_window).mean()) / df['matchingVolume'].rolling(sub_window).std()
        volume_z = volume_z.clip(-1, 1).fillna(0)
        result = signal * volume_z
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_211_h(df, window=50, sub_window=30):
        range_mean = (df['high'] - df['low']).rolling(window).mean()
        vol_mean = df['matchingVolume'].rolling(window).mean().replace(0, 1e-8)
        raw = range_mean / vol_mean
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        volume_z = (df['matchingVolume'] - df['matchingVolume'].rolling(sub_window).mean()) / df['matchingVolume'].rolling(sub_window).std()
        volume_z = np.tanh(volume_z).fillna(0)
        result = signal * volume_z
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_211_e(df, window=50, sub_window=40):
        range_mean = (df['high'] - df['low']).rolling(window).mean()
        vol_mean = df['matchingVolume'].rolling(window).mean().replace(0, 1e-8)
        raw = range_mean / vol_mean
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        volume_z = ((df['matchingVolume'] - df['matchingVolume'].rolling(sub_window).mean()) / df['matchingVolume'].rolling(sub_window).std().replace(0, np.nan)).clip(-1, 1)
        result = signal * volume_z
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_211_y(df, window=10, sub_window=30):
        range_mean = (df['high'] - df['low']).rolling(window).mean()
        vol_mean = df['matchingVolume'].rolling(window).mean().replace(0, 1e-8)
        raw = range_mean / vol_mean
        signal = np.sign(raw)
        volume_sign = np.sign(df['matchingVolume'] - df['matchingVolume'].rolling(sub_window).mean())
        result = signal * volume_sign
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_211_r(df, window=70, sub_window=10):
        range_mean = (df['high'] - df['low']).rolling(window).mean()
        vol_mean = df['matchingVolume'].rolling(window).mean().replace(0, 1e-8)
        raw = range_mean / vol_mean
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        volume_z = (df['matchingVolume'] - df['matchingVolume'].rolling(sub_window).mean()) / df['matchingVolume'].rolling(sub_window).std().replace(0, np.nan)
        vol_effect = np.tanh(volume_z).fillna(0)
        result = signal * vol_effect
        return -result.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_212_rank(df, window=5):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(5).mean()
        hl = df['high'] - df['low']
        std_hl = hl.rolling(window).std()
        raw = mean_ret / (std_hl + 1e-8)
        corr = df['close'].rolling(window).corr(hl)
        raw2 = raw - corr
        norm = (raw2.rolling(window).rank(pct=True) * 2) - 1
        return -norm

    @staticmethod
    def alpha_quanta_full_base_212_tanh(df, window=5):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(5).mean()
        hl = df['high'] - df['low']
        std_hl = hl.rolling(window).std()
        raw = mean_ret / (std_hl + 1e-8)
        corr = df['close'].rolling(window).corr(hl)
        raw2 = raw - corr
        norm = np.tanh(raw2 / raw2.rolling(window).std())
        return -norm

    @staticmethod
    def alpha_quanta_full_base_212_zscore(df, window=5):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(5).mean()
        hl = df['high'] - df['low']
        std_hl = hl.rolling(window).std()
        raw = mean_ret / (std_hl + 1e-8)
        corr = df['close'].rolling(window).corr(hl)
        raw2 = raw - corr
        mean_raw = raw2.rolling(window).mean()
        std_raw = raw2.rolling(window).std()
        norm = ((raw2 - mean_raw) / std_raw).clip(-1, 1)
        return -norm

    @staticmethod
    def alpha_quanta_full_base_212_sign(df, window=5):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(5).mean()
        hl = df['high'] - df['low']
        std_hl = hl.rolling(window).std()
        raw = mean_ret / (std_hl + 1e-8)
        corr = df['close'].rolling(window).corr(hl)
        raw2 = raw - corr
        norm = np.sign(raw2)
        return -norm

    @staticmethod
    def alpha_quanta_full_base_212_wf(df, window=15):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(5).mean()
        hl = df['high'] - df['low']
        std_hl = hl.rolling(window).std()
        raw = mean_ret / (std_hl + 1e-8)
        corr = df['close'].rolling(window).corr(hl)
        raw2 = raw - corr
        p1 = 0.05
        low = raw2.rolling(window).quantile(p1)
        high = raw2.rolling(window).quantile(1 - p1)
        winsorized = raw2.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm

    @staticmethod
    def alpha_quanta_full_base_213_rank(df, window=55):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff() / close
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        ts_mean = corr.rolling(4).mean()
        rank = ts_mean.rolling(window).rank(pct=True) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_213_tanh(df, window=50):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff() / close
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        ts_mean = corr.rolling(4).mean()
        normalized = np.tanh(ts_mean / ts_mean.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_213_zscore(df, window=60):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff() / close
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        ts_mean = corr.rolling(4).mean()
        zscore = ((ts_mean - ts_mean.rolling(window).mean()) / ts_mean.rolling(window).std()).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_213_sign(df, window=55):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff() / close
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        ts_mean = corr.rolling(4).mean()
        sign = np.sign(ts_mean)
        return pd.Series(sign, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_213_wf(df, window=10, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff() / close
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        ts_mean = corr.rolling(4).mean()
        low = ts_mean.rolling(window).quantile(p1)
        high = ts_mean.rolling(window).quantile(1 - p1)
        winsorized = ts_mean.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_214_rank(df, window=60):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = df['close'] / (volume + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        inv_std = 1.0 / (std + 1e-8)
        formula = mean * inv_std
        signal = (formula.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_214_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = df['close'] / (volume + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        inv_std = 1.0 / (std + 1e-8)
        formula = mean * inv_std
        signal = np.tanh(formula / formula.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_214_zscore(df, window=75):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = df['close'] / (volume + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        inv_std = 1.0 / (std + 1e-8)
        formula = mean * inv_std
        rolling_mean = formula.rolling(window).mean()
        rolling_std = formula.rolling(window).std()
        signal = ((formula - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_214_sign(df, window=75):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = df['close'] / (volume + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        inv_std = 1.0 / (std + 1e-8)
        formula = mean * inv_std
        signal = pd.Series(np.sign(formula), index=df.index)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_214_wf(df, window=60, quantile=0.3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = df['close'] / (volume + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        inv_std = 1.0 / (std + 1e-8)
        formula = mean * inv_std
        p = quantile
        low = formula.rolling(window * 2).quantile(p)
        high = formula.rolling(window * 2).quantile(1 - p)
        winsorized = formula.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_215_rank(df, window=80):
        high_low_diff = df['high'] - df['low']
        vol_log = np.log1p(df['matchingVolume'])
        corr = high_low_diff.rolling(window).corr(vol_log)
        ts_std = df['close'].rolling(window).std()
        ts_mean = df['close'].rolling(window).mean()
        raw = corr * (1 - ts_std / (ts_mean + 1e-9))
        rank_pct = raw.rolling(window).rank(pct=True)
        signal = rank_pct * 2 - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_215_tanh(df, window=90):
        high_low_diff = df['high'] - df['low']
        vol_log = np.log1p(df['matchingVolume'])
        corr = high_low_diff.rolling(window).corr(vol_log)
        ts_std = df['close'].rolling(window).std()
        ts_mean = df['close'].rolling(window).mean()
        raw = corr * (1 - ts_std / (ts_mean + 1e-9))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_215_zscore(df, window=50):
        high_low_diff = df['high'] - df['low']
        vol_log = np.log1p(df['matchingVolume'])
        corr = high_low_diff.rolling(window).corr(vol_log)
        ts_std = df['close'].rolling(window).std()
        ts_mean = df['close'].rolling(window).mean()
        raw = corr * (1 - ts_std / (ts_mean + 1e-9))
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / std.replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_215_sign(df, window=5):
        high_low_diff = df['high'] - df['low']
        vol_log = np.log1p(df['matchingVolume'])
        corr = high_low_diff.rolling(window).corr(vol_log)
        ts_std = df['close'].rolling(window).std()
        ts_mean = df['close'].rolling(window).mean()
        raw = corr * (1 - ts_std / (ts_mean + 1e-9))
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_215_wf(df, window=80, factor=0.7):
        high_low_diff = df['high'] - df['low']
        vol_log = np.log1p(df['matchingVolume'])
        corr = high_low_diff.rolling(window).corr(vol_log)
        ts_std = df['close'].rolling(window).std()
        ts_mean = df['close'].rolling(window).mean()
        raw = corr * (1 - ts_std / (ts_mean + 1e-9))
        p = factor
        low = raw.rolling(window).quantile(p)
        high = raw.rolling(window).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        temp = (winsorized - low) / (high - low + 1e-9)
        signal = np.arctanh(temp * 1.98 - 0.99)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_216_rank(df, window=85):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0))
        if volume.sum() == 0:
            volume = df['close'] * 1e6
        ret = close - open_
        corr = ret.rolling(window).corr(volume)
        mean_vol = volume.rolling(window).mean()
        raw = np.sign(ret) * corr * mean_vol
        ranked = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -ranked.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_216_tanh(df, window=70):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0))
        if volume.sum() == 0:
            volume = df['close'] * 1e6
        ret = close - open_
        corr = ret.rolling(window).corr(volume)
        mean_vol = volume.rolling(window).mean()
        raw = np.sign(ret) * corr * mean_vol
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_216_zscore(df, window=70):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0))
        if volume.sum() == 0:
            volume = df['close'] * 1e6
        ret = close - open_
        corr = ret.rolling(window).corr(volume)
        mean_vol = volume.rolling(window).mean()
        raw = np.sign(ret) * corr * mean_vol
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_216_sign(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0))
        if volume.sum() == 0:
            volume = df['close'] * 1e6
        ret = close - open_
        corr = ret.rolling(window).corr(volume)
        mean_vol = volume.rolling(window).mean()
        raw = np.sign(ret) * corr * mean_vol
        normalized = np.sign(raw)
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_full_base_216_wf(df, window=90, quantile=0.7):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0))
        if volume.sum() == 0:
            volume = df['close'] * 1e6
        ret = close - open_
        corr = ret.rolling(window).corr(volume)
        mean_vol = volume.rolling(window).mean()
        raw = np.sign(ret) * corr * mean_vol
        p1 = raw.rolling(window).quantile(quantile)
        p99 = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=p1, upper=p99, axis=0)
        norm_val = ((winsorized - p1) / (p99 - p1 + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(norm_val.clip(-0.99, 0.99))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_217_k(df, window=20, delta=2):
        close = df['close'].astype(float)
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        delta_ts = ts_mean.diff(delta)
        ts_std = close.rolling(window).std()
        raw = delta_ts / (ts_std + 1e-8)
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_217_h(df, window=40, delta=2):
        close = df['close'].astype(float)
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        delta_ts = ts_mean.diff(delta)
        ts_std = close.rolling(window).std()
        raw = delta_ts / (ts_std + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_217_e(df, window=40, delta=2):
        close = df['close'].astype(float)
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        delta_ts = ts_mean.diff(delta)
        ts_std = close.rolling(window).std()
        raw = delta_ts / (ts_std + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_217_n(df, window=10, delta=2):
        close = df['close'].astype(float)
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        delta_ts = ts_mean.diff(delta)
        ts_std = close.rolling(window).std()
        raw = delta_ts / (ts_std + 1e-8)
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_217_r(df, window=40, delta=1, percentile=0.05):
        close = df['close'].astype(float)
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        delta_ts = ts_mean.diff(delta)
        ts_std = close.rolling(window).std()
        raw = delta_ts / (ts_std + 1e-8)
        low = raw.rolling(window).quantile(percentile)
        high = raw.rolling(window).quantile(1 - percentile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_full_base_218_rank(df, window=65):
        hl = df['high'] - df['low']
        vol_delta = df['matchingVolume'].diff()
        valid = hl.notna() & vol_delta.notna()
        corr = hl.rolling(window).corr(vol_delta).fillna(0)
        rank = corr.rolling(window).rank(pct=True) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_218_tanh(df, window=10):
        hl = df['high'] - df['low']
        vol_delta = df['matchingVolume'].diff()
        valid = hl.notna() & vol_delta.notna()
        corr = hl.rolling(window).corr(vol_delta).fillna(0)
        std = corr.rolling(window).std().replace(0, np.nan)
        return -np.tanh(corr / std).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_218_zscore(df, window=70):
        hl = df['high'] - df['low']
        vol_delta = df['matchingVolume'].diff()
        corr = hl.rolling(window).corr(vol_delta).fillna(0)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        return ((corr - mean) / std).clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_218_sign(df, window=5):
        hl = df['high'] - df['low']
        vol_delta = df['matchingVolume'].diff()
        corr = hl.rolling(window).corr(vol_delta).fillna(0)
        return -np.sign(corr).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_218_wf(df, window=30, p2=100, p1=0.05):
        hl = df['high'] - df['low']
        vol_delta = df['matchingVolume'].diff()
        corr = hl.rolling(window).corr(vol_delta).fillna(0)
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_219_k(df, window_rank=40):
        close = df['close']
        ret = close.pct_change()
        ratio = ret.rolling(5).mean() / (ret.rolling(10).std() + 1e-8)
        sign = (ret.rolling(10).mean() > 0).astype(int) * 2 - 1
        raw = ratio * sign
        signal = raw.rolling(window_rank).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_219_h(df, window=100):
        close = df['close']
        ret = close.pct_change()
        ratio = ret.rolling(5).mean() / (ret.rolling(10).std() + 1e-8)
        sign = (ret.rolling(10).mean() > 0).astype(int) * 2 - 1
        raw = ratio * sign
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_219_e(df, window_z=60):
        close = df['close']
        ret = close.pct_change()
        ratio = ret.rolling(5).mean() / (ret.rolling(10).std() + 1e-8)
        sign = (ret.rolling(10).mean() > 0).astype(int) * 2 - 1
        raw = ratio * sign
        mean_ = raw.rolling(window_z).mean()
        std_ = raw.rolling(window_z).std().replace(0, np.nan)
        signal = ((raw - mean_) / std_).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_219_y(df, window=90):
        close = df['close']
        ret = close.pct_change()
        ratio = ret.rolling(5).mean() / (ret.rolling(10).std() + 1e-8)
        sign = (ret.rolling(10).mean() > 0).astype(int) * 2 - 1
        raw = ratio * sign
        signal = np.sign(raw).rolling(window).mean().fillna(0)
        return pd.Series(np.sign(signal), index=df.index) * 1.0

    @staticmethod
    def alpha_quanta_full_base_219_r(df, window_rank=75):
        close = df['close']
        ret = close.pct_change()
        ratio = ret.rolling(5).mean() / (ret.rolling(10).std() + 1e-8)
        sign = (ret.rolling(10).mean() > 0).astype(int) * 2 - 1
        raw = ratio * sign
        p1 = 0.05
        p2 = window_rank
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9) * 1.98 - 0.99
        numerator = numerator.clip(-0.99, 0.99)
        signal = np.arctanh(numerator)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_220_rank(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct_close = close.diff() / close
        vol_mean = volume.rolling(window).mean()
        pct_vol = volume.diff() / (vol_mean + 1e-8)
        corr = pct_close.rolling(window).corr(pct_vol)
        rank = corr.rolling(window).rank(pct=True)
        signal = (rank * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_220_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct_close = close.diff() / close
        vol_mean = volume.rolling(window).mean()
        pct_vol = volume.diff() / (vol_mean + 1e-8)
        corr = pct_close.rolling(window).corr(pct_vol)
        signal = np.tanh(corr / corr.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_220_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        pct_close = close.diff() / close
        vol_mean = volume.rolling(window).mean()
        pct_vol = volume.diff() / (vol_mean + 1e-8)
        corr = pct_close.rolling(window).corr(pct_vol)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std()
        zscore = (corr - mean) / (std + 1e-9)
        signal = zscore.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_220_sign(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        pct_close = close.diff() / close
        vol_mean = volume.rolling(window).mean()
        pct_vol = volume.diff() / (vol_mean + 1e-8)
        corr = pct_close.rolling(window).corr(pct_vol)
        signal = np.sign(corr)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_220_wf(df, window_rank=10, winsor_quantile=0.1):
        close = df['close']
        volume = df['matchingVolume']
        pct_close = close.diff() / close
        vol_mean = volume.rolling(window_rank).mean()
        pct_vol = volume.diff() / (vol_mean + 1e-8)
        corr = pct_close.rolling(window_rank).corr(pct_vol)
        low = corr.rolling(window_rank).quantile(winsor_quantile)
        high = corr.rolling(window_rank).quantile(1 - winsor_quantile)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_221_rank(df: pd.DataFrame, window: int = 15) -> pd.Series:
        ret = df['close'].pct_change()
        ret_ratio = ret.rolling(5).mean() / (ret.rolling(15).std() + 1e-8)
        hl_range = (df['high'] - df['low']) / df['close']
        norm_hl = hl_range / (hl_range.rolling(15).mean() + 1e-8)
        raw = ret_ratio - norm_hl
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_221_tanh(df: pd.DataFrame, window: int = 15) -> pd.Series:
        ret = df['close'].pct_change()
        ret_ratio = ret.rolling(5).mean() / (ret.rolling(15).std() + 1e-8)
        hl_range = (df['high'] - df['low']) / df['close']
        norm_hl = hl_range / (hl_range.rolling(15).mean() + 1e-8)
        raw = ret_ratio - norm_hl
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_221_zscore(df: pd.DataFrame, window: int = 15) -> pd.Series:
        ret = df['close'].pct_change()
        ret_ratio = ret.rolling(5).mean() / (ret.rolling(15).std() + 1e-8)
        hl_range = (df['high'] - df['low']) / df['close']
        norm_hl = hl_range / (hl_range.rolling(15).mean() + 1e-8)
        raw = ret_ratio - norm_hl
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_221_sign(df: pd.DataFrame, window: int = 15) -> pd.Series:
        ret = df['close'].pct_change()
        ret_ratio = ret.rolling(5).mean() / (ret.rolling(15).std() + 1e-8)
        hl_range = (df['high'] - df['low']) / df['close']
        norm_hl = hl_range / (hl_range.rolling(15).mean() + 1e-8)
        raw = ret_ratio - norm_hl
        signal = np.sign(raw).astype(float)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_221_wf(df: pd.DataFrame, p1: float = 0.05, p2: int = 20) -> pd.Series:
        ret = df['close'].pct_change()
        ret_ratio = ret.rolling(5).mean() / (ret.rolling(15).std() + 1e-8)
        hl_range = (df['high'] - df['low']) / df['close']
        norm_hl = hl_range / (hl_range.rolling(15).mean() + 1e-8)
        raw = ret_ratio - norm_hl
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_222_rank(df, window=100):
        returns = df['close'].pct_change()
        vol = df['matchingVolume']
        vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
        corr = returns.rolling(window).corr(vol_ratio)
        mean_ret = returns.rolling(window).mean()
        raw = corr * mean_ret
        norm = (raw.rolling(2*window).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_222_tanh(df, window=100):
        returns = df['close'].pct_change()
        vol = df['matchingVolume']
        vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
        corr = returns.rolling(window).corr(vol_ratio)
        mean_ret = returns.rolling(window).mean()
        raw = corr * mean_ret
        norm = np.tanh(raw / (raw.rolling(2*window).std() + 1e-9))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_222_zscore(df, window=90):
        returns = df['close'].pct_change()
        vol = df['matchingVolume']
        vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
        corr = returns.rolling(window).corr(vol_ratio)
        mean_ret = returns.rolling(window).mean()
        raw = corr * mean_ret
        norm = ((raw - raw.rolling(2*window).mean()) / (raw.rolling(2*window).std() + 1e-9)).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_222_sign(df, window=20):
        returns = df['close'].pct_change()
        vol = df['matchingVolume']
        vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
        corr = returns.rolling(window).corr(vol_ratio)
        mean_ret = returns.rolling(window).mean()
        raw = corr * mean_ret
        norm = np.sign(raw)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_222_wf(df, window=100, p1=0.3):
        returns = df['close'].pct_change()
        vol = df['matchingVolume']
        vol_ratio = vol / (vol.rolling(20).mean() + 1e-8)
        corr = returns.rolling(window).corr(vol_ratio)
        mean_ret = returns.rolling(window).mean()
        raw = corr * mean_ret
        p2 = int(window * 4)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        ratio = (raw.clip(lower=low, upper=high, axis=0) - low) / ((high - low) + 1e-9)
        norm = np.arctanh(ratio * 1.98 - 0.99)
        return -pd.Series(np.where((high - low).abs() > 1e-9, norm, 0), index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_223_zscore(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        vwap_approx = (close * volume).rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        raw = (close - vwap_approx) / (high.rolling(1).max() - low.rolling(1).min()).rolling(window).mean() + 1e-8
        delay_close = close.shift(1)
        true_range = pd.concat([high - low, (high - delay_close).abs(), (low - delay_close).abs()], axis=1).max(axis=1)
        denom = true_range.rolling(window).mean() + 1e-8
        raw = (close * volume).rolling(window).mean() / (volume.rolling(window).mean() + 1e-8)
        raw = (close - raw) / denom
        std = raw.rolling(window).std()
        mean = raw.rolling(window).mean()
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_223_rank(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        mean_close_vol = (close * volume).rolling(window).mean()
        mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = close - mean_close_vol / (mean_vol + 1e-8)
        denom = np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))
        denom_mean = denom.rolling(window).mean().replace(0, np.nan)
        raw = raw / (denom_mean + 1e-8)
        raw = raw.fillna(0)
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_223_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        mean_close_vol = (close * volume).rolling(window).mean()
        mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = close - mean_close_vol / (mean_vol + 1e-8)
        denom = np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))
        denom_mean = denom.rolling(window).mean().replace(0, np.nan)
        raw = raw / (denom_mean + 1e-8)
        raw = raw.fillna(0)
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_223_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        mean_close_vol = (close * volume).rolling(window).mean()
        mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = close - mean_close_vol / (mean_vol + 1e-8)
        denom = np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))
        denom_mean = denom.rolling(window).mean().replace(0, np.nan)
        raw = raw / (denom_mean + 1e-8)
        raw = raw.fillna(0)
        norm = np.sign(raw)
        return norm

    @staticmethod
    def alpha_quanta_full_base_223_wf(df, window=10, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        high = df['high']
        low = df['low']
        mean_close_vol = (close * volume).rolling(window).mean()
        mean_vol = volume.rolling(window).mean().replace(0, np.nan)
        raw = close - mean_close_vol / (mean_vol + 1e-8)
        denom = np.maximum(np.maximum(high - low, np.abs(high - close.shift(1))), np.abs(low - close.shift(1)))
        denom_mean = denom.rolling(window).mean().replace(0, np.nan)
        raw = raw / (denom_mean + 1e-8)
        raw = raw.fillna(0)
        low_bound = raw.rolling(window).quantile(p1)
        high_bound = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        norm = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_224_rank(df, window=90):
        raw = df['close'] * df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_224_tanh(df, window=5):
        raw = df['close'] * df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_224_zscore(df, window=95):
        raw = df['close'] * df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_224_sign(df, window=95):
        raw = df['close'] * df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        signal = np.sign(raw)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_224_wf(df, window=60, quantile=0.7):
        raw = df['close'] * df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_225_k(df, window=3, rank_window=30):
        close = df['close']
        volume = df['matchingVolume']
        vwap_est = (close * volume).rolling(window=window).mean() / (volume.rolling(window=window).mean() + 1e-8)
        raw_a = (close - vwap_est) / (vwap_est + 1e-8)
        ret = close.pct_change()
        ts_std_ret = ret.rolling(window=window).std()
        ts_mean_std_ret = ts_std_ret.rolling(window=rank_window).mean()
        raw_b = ts_std_ret / (ts_mean_std_ret + 1e-8)
        raw = raw_a * raw_b
        signal = raw.rolling(rank_window).rank(pct=True) * 2 - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_225_h(df, window=7, norm_window=10):
        close = df['close']
        volume = df['matchingVolume']
        vwap_est = (close * volume).rolling(window=window).mean() / (volume.rolling(window=window).mean() + 1e-8)
        raw_a = (close - vwap_est) / (vwap_est + 1e-8)
        ret = close.pct_change()
        ts_std_ret = ret.rolling(window=window).std()
        ts_mean_std_ret = ts_std_ret.rolling(window=norm_window).mean()
        raw_b = ts_std_ret / (ts_mean_std_ret + 1e-8)
        raw = raw_a * raw_b
        signal = np.tanh(raw / raw.rolling(norm_window).std())
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_225_p(df, window=3, zscore_window=10):
        close = df['close']
        volume = df['matchingVolume']
        vwap_est = (close * volume).rolling(window=window).mean() / (volume.rolling(window=window).mean() + 1e-8)
        raw_a = (close - vwap_est) / (vwap_est + 1e-8)
        ret = close.pct_change()
        ts_std_ret = ret.rolling(window=window).std()
        ts_mean_std_ret = ts_std_ret.rolling(window=zscore_window).mean()
        raw_b = ts_std_ret / (ts_mean_std_ret + 1e-8)
        raw = raw_a * raw_b
        rolling_mean = raw.rolling(zscore_window).mean()
        rolling_std = raw.rolling(zscore_window).std()
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_225_t(df, window=3, dummy_window=60):
        close = df['close']
        volume = df['matchingVolume']
        vwap_est = (close * volume).rolling(window=window).mean() / (volume.rolling(window=window).mean() + 1e-8)
        raw_a = (close - vwap_est) / (vwap_est + 1e-8)
        ret = close.pct_change()
        ts_std_ret = ret.rolling(window=window).std()
        ts_mean_std_ret = ts_std_ret.rolling(window=dummy_window).mean()
        raw_b = ts_std_ret / (ts_mean_std_ret + 1e-8)
        raw = raw_a * raw_b
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_225_r(df, window=20, quantile=0.3):
        if quantile <= 0 or quantile >= 0.5:
            quantile = 0.1
        p2_rolling = 20
        close = df['close']
        volume = df['matchingVolume']
        vwap_est = (close * volume).rolling(window=window).mean() / (volume.rolling(window=window).mean() + 1e-8)
        raw_a = (close - vwap_est) / (vwap_est + 1e-8)
        ret = close.pct_change()
        ts_std_ret = ret.rolling(window=window).std()
        ts_mean_std_ret = ts_std_ret.rolling(window=p2_rolling).mean()
        raw_b = ts_std_ret / (ts_mean_std_ret + 1e-8)
        raw = raw_a * raw_b
        low = raw.rolling(p2_rolling).quantile(quantile)
        high = raw.rolling(p2_rolling).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_226_k(df, window_rank_1=55):
        h_l = df['high'] - df['low']
        ts_mean = h_l.rolling(window=window_rank_1).mean()
        ts_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window=window_rank_1).mean() + 1e-8
        raw = ts_mean / ts_volume
        sign = pd.Series(np.sign(h_l.rolling(window=window_rank_1).std()), index=df.index)
        raw = raw * sign
        signal = (raw.rolling(window=window_rank_1).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_226_h(df, window=5):
        h_l = df['high'] - df['low']
        ts_mean = h_l.rolling(window=window).mean()
        ts_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8
        raw = ts_mean / ts_volume
        sign = pd.Series(np.sign(h_l.rolling(window=window).std()), index=df.index)
        raw = raw * sign
        roll_std = raw.rolling(window=window).std().replace(0, np.nan)
        signal = np.tanh(raw / roll_std)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_226_e(df, window=70):
        h_l = df['high'] - df['low']
        ts_mean = h_l.rolling(window=window).mean()
        ts_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8
        raw = ts_mean / ts_volume
        sign = pd.Series(np.sign(h_l.rolling(window=window).std()), index=df.index)
        raw = raw * sign
        roll_mean = raw.rolling(window=window).mean()
        roll_std = raw.rolling(window=window).std().replace(0, np.nan)
        signal = ((raw - roll_mean) / roll_std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_226_y(df, window=80):
        h_l = df['high'] - df['low']
        ts_mean = h_l.rolling(window=window).mean()
        ts_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8
        raw = ts_mean / ts_volume
        sign = pd.Series(np.sign(h_l.rolling(window=window).std()), index=df.index)
        raw = raw * sign
        signal = pd.Series(np.sign(raw), index=df.index)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_226_r(df, window=70, p_quantile=0.1):
        h_l = df['high'] - df['low']
        ts_mean = h_l.rolling(window=window).mean()
        ts_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(window=window).mean() + 1e-8
        raw = ts_mean / ts_volume
        sign = pd.Series(np.sign(h_l.rolling(window=window).std()), index=df.index)
        raw = raw * sign
        low = raw.rolling(window=window).quantile(p_quantile)
        high = raw.rolling(window=window).quantile(1 - p_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_227_k(df, window=85):
        hl_range = df['high'] - df['low']
        mean_range = hl_range.rolling(window).mean()
        avg_close = df['close'].rolling(window).mean() + 1e-8
        raw = mean_range / avg_close
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        delta_close = df['close'].diff()
        sign = pd.Series(np.sign(delta_close), index=df.index)
        signal = z * sign
        normalized = (signal.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_227_h(df, window=60):
        hl_range = df['high'] - df['low']
        mean_range = hl_range.rolling(window).mean()
        avg_close = df['close'].rolling(window).mean() + 1e-8
        raw = mean_range / avg_close
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        delta_close = df['close'].diff()
        sign = pd.Series(np.sign(delta_close), index=df.index)
        signal = z * sign
        normalized = np.tanh(signal / signal.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_227_p(df, window=5):
        hl_range = df['high'] - df['low']
        mean_range = hl_range.rolling(window).mean()
        avg_close = df['close'].rolling(window).mean() + 1e-8
        raw = mean_range / avg_close
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        delta_close = df['close'].diff()
        sign = pd.Series(np.sign(delta_close), index=df.index)
        signal = z * sign
        normalized = ((signal - signal.rolling(window).mean()) / signal.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_227_t(df, window=5):
        hl_range = df['high'] - df['low']
        mean_range = hl_range.rolling(window).mean()
        avg_close = df['close'].rolling(window).mean() + 1e-8
        raw = mean_range / avg_close
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        delta_close = df['close'].diff()
        sign = pd.Series(np.sign(delta_close), index=df.index)
        signal = z * sign
        normalized = pd.Series(np.sign(signal), index=df.index)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_227_r(df, window=60, p1=0.1):
        hl_range = df['high'] - df['low']
        mean_range = hl_range.rolling(window).mean()
        avg_close = df['close'].rolling(window).mean() + 1e-8
        raw = mean_range / avg_close
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        delta_close = df['close'].diff()
        sign = pd.Series(np.sign(delta_close), index=df.index)
        signal = z * sign
        p2 = window
        low = signal.rolling(p2).quantile(p1)
        high = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_228_rank(df, window=70):
        ret = df['close'].pct_change()
        ts_sum = ret.rolling(window).sum()
        rank_ts_sum = ts_sum.rolling(window).rank(pct=True) * 2 - 1
        ts_std = ret.rolling(window).std()
        max_std = ts_std.rolling(window).max()
        factor = 1 - ts_std / (max_std + 1e-8)
        raw = rank_ts_sum * factor
        return raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_228_tanh(df, window=30):
        ret = df['close'].pct_change()
        ts_sum = ret.rolling(window).sum()
        ts_std = ret.rolling(window).std()
        max_std = ts_std.rolling(window).max()
        factor = 1 - ts_std / (max_std + 1e-8)
        raw = ts_sum * factor
        return np.tanh(raw / raw.rolling(window).std()).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_228_zscore(df, window=30):
        ret = df['close'].pct_change()
        ts_sum = ret.rolling(window).sum()
        ts_std = ret.rolling(window).std()
        max_std = ts_std.rolling(window).max()
        factor = 1 - ts_std / (max_std + 1e-8)
        raw = ts_sum * factor
        return ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_228_sign(df, window=30):
        ret = df['close'].pct_change()
        ts_sum = ret.rolling(window).sum()
        ts_std = ret.rolling(window).std()
        max_std = ts_std.rolling(window).max()
        factor = 1 - ts_std / (max_std + 1e-8)
        raw = ts_sum * factor
        return np.sign(raw).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_228_wf(df, window=30, p1=0.3):
        p2 = window
        ret = df['close'].pct_change()
        ts_sum = ret.rolling(window).sum()
        ts_std = ret.rolling(window).std()
        max_std = ts_std.rolling(window).max()
        factor = 1 - ts_std / (max_std + 1e-8)
        raw = ts_sum * factor
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_229_rank(df: pd.DataFrame, window: int = 5, volume_window: int = 20, std_window: int = 5) -> pd.Series:
        # Calculate percentage change of close over window periods
        close_pct = df['close'].pct_change(periods=window)
        # Calculate volume mean and std
        vol_mean = df['matchingVolume'].rolling(volume_window).mean()
        vol_std = df['matchingVolume'].rolling(std_window).std()
        # Compute raw signal: pctchange / (vol_mean / (vol_std + 1e-8))
        raw = close_pct / (vol_mean / (vol_std + 1e-8))
        # Rolling rank normalization to [-1, 1]
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_229_tanh(df: pd.DataFrame, window: int = 5, volume_window: int = 20, std_window: int = 5) -> pd.Series:
        close_pct = df['close'].pct_change(periods=window)
        vol_mean = df['matchingVolume'].rolling(volume_window).mean()
        vol_std = df['matchingVolume'].rolling(std_window).std()
        raw = close_pct / (vol_mean / (vol_std + 1e-8))
        # Dynamic Tanh normalization: keep magnitude
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_229_zscore(df: pd.DataFrame, window: int = 5, volume_window: int = 20, std_window: int = 5) -> pd.Series:
        close_pct = df['close'].pct_change(periods=window)
        vol_mean = df['matchingVolume'].rolling(volume_window).mean()
        vol_std = df['matchingVolume'].rolling(std_window).std()
        raw = close_pct / (vol_mean / (vol_std + 1e-8))
        # Rolling Z-Score/Clip normalization
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_229_sign(df: pd.DataFrame, window: int = 5, volume_window: int = 20, std_window: int = 5) -> pd.Series:
        close_pct = df['close'].pct_change(periods=window)
        vol_mean = df['matchingVolume'].rolling(volume_window).mean()
        vol_std = df['matchingVolume'].rolling(std_window).std()
        raw = close_pct / (vol_mean / (vol_std + 1e-8))
        # Sign/Binary Soft normalization
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_229_wf(df: pd.DataFrame, window: int = 5, volume_window: int = 20, std_window: int = 5, quantile_low: float = 0.05) -> pd.Series:
        close_pct = df['close'].pct_change(periods=window)
        vol_mean = df['matchingVolume'].rolling(volume_window).mean()
        vol_std = df['matchingVolume'].rolling(std_window).std()
        raw = close_pct / (vol_mean / (vol_std + 1e-8))
        # Winsorized Fisher normalization
        low = raw.rolling(window).quantile(quantile_low)
        high = raw.rolling(window).quantile(1 - quantile_low)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = signal.ffill().fillna(0)
        signal = signal.clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_230_k(df, window=35):
        pct_chg_10 = df['close'].pct_change(10)
        vol_ratio = df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).fillna(0) / (df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).rolling(window).mean() + 1e-8)
        raw = np.sign(pct_chg_10) * pct_chg_10 / vol_ratio
        return (raw.rolling(window).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_full_base_230_h(df, window=75):
        pct_chg_10 = df['close'].pct_change(10)
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).fillna(0)
        vol_ratio = vol / (vol.rolling(window).mean() + 1e-8)
        raw = np.sign(pct_chg_10) * pct_chg_10 / vol_ratio
        return np.tanh(raw / raw.rolling(window).std())

    @staticmethod
    def alpha_quanta_full_base_230_e(df, window=30):
        pct_chg_10 = df['close'].pct_change(10)
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).fillna(0)
        vol_ratio = vol / (vol.rolling(window).mean() + 1e-8)
        raw = np.sign(pct_chg_10) * pct_chg_10 / vol_ratio
        return ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_230_y(df, window=95):
        pct_chg_10 = df['close'].pct_change(10)
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).fillna(0)
        vol_ratio = vol / (vol.rolling(window).mean() + 1e-8)
        raw = np.sign(pct_chg_10) * pct_chg_10 / vol_ratio
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_full_base_230_r(df, window=100, p1=0.1):
        pct_chg_10 = df['close'].pct_change(10)
        vol = df.get('matchingVolume', df.get('volume', df['close'] * 0.01)).fillna(0)
        vol_ratio = vol / (vol.rolling(window).mean() + 1e-8)
        raw = np.sign(pct_chg_10) * pct_chg_10 / vol_ratio
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_full_base_231_rank(df, window_rank=30, window_corr=60):
        close_pct = df['close'].pct_change(3).fillna(0)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window_corr).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window_corr).std()
        volume_z = volume_z.fillna(0).replace([np.inf, -np.inf], 0)
        corr = close_pct.rolling(5).corr(volume_z).fillna(0)
        rank = corr.rolling(window_rank).rank(pct=True) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_231_tanh(df, window_std=50, window_corr=5):
        close_pct = df['close'].pct_change(3).fillna(0)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window_std).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window_std).std()
        volume_z = volume_z.fillna(0).replace([np.inf, -np.inf], 0)
        corr = close_pct.rolling(window_corr).corr(volume_z).fillna(0)
        normalized = np.tanh(corr / (corr.rolling(window_std).std().replace(0, np.nan).ffill().fillna(1)))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_231_zscore(df, window_corr=5, window_z=60):
        close_pct = df['close'].pct_change(3).fillna(0)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window_z).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window_z).std()
        volume_z = volume_z.fillna(0).replace([np.inf, -np.inf], 0)
        corr = close_pct.rolling(window_corr).corr(volume_z).fillna(0)
        normalized = ((corr - corr.rolling(window_corr).mean()) / corr.rolling(window_corr).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_231_sign(df, window_corr=5, window_z=70):
        close_pct = df['close'].pct_change(3).fillna(0)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window_z).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window_z).std()
        volume_z = volume_z.fillna(0).replace([np.inf, -np.inf], 0)
        corr = close_pct.rolling(window_corr).corr(volume_z).fillna(0)
        normalized = np.sign(corr)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_231_wf(df, window_corr=100, p_quantile=0.1):
        close_pct = df['close'].pct_change(3).fillna(0)
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_roll = volume.rolling(window_corr * 2)
        low_v = volume_roll.quantile(p_quantile).ffill().fillna(volume.min())
        high_v = volume_roll.quantile(1 - p_quantile).ffill().fillna(volume.max())
        volume_z = ((volume - low_v) / (high_v - low_v + 1e-9) * 1.98 - 0.99).clip(-0.99, 0.99)
        volume_z = volume_z.fillna(0).replace([np.inf, -np.inf], 0)
        corr = close_pct.rolling(window_corr).corr(np.arctanh(volume_z)).fillna(0)
        low_c = corr.rolling(window_corr).quantile(p_quantile).ffill().fillna(corr.min())
        high_c = corr.rolling(window_corr).quantile(1 - p_quantile).ffill().fillna(corr.max())
        winsorized = corr.clip(lower=low_c, upper=high_c, axis=0)
        normalized = np.arctanh(((winsorized - low_c) / (high_c - low_c + 1e-9)) * 1.98 - 0.99).fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_232_rank(df, window=5):
        close = df['close']
        ret = close.pct_change().fillna(0)
        raw = -np.sign(ret.shift(1)) * ret.shift(1) / (ret.rolling(window).std() + 1e-8)
        # Rolling Rank
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_232_tanh(df, window=15):
        close = df['close']
        ret = close.pct_change().fillna(0)
        raw = -np.sign(ret.shift(1)) * ret.shift(1) / (ret.rolling(window).std() + 1e-8)
        # Dynamic Tanh
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_232_zscore(df, window=10):
        close = df['close']
        ret = close.pct_change().fillna(0)
        raw = -np.sign(ret.shift(1)) * ret.shift(1) / (ret.rolling(window).std() + 1e-8)
        # Rolling Z-Score Clip
        rolled_mean = raw.rolling(window).mean()
        rolled_std = raw.rolling(window).std().replace(0, np.nan).ffill()
        signal = ((raw - rolled_mean) / rolled_std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_232_sign(df, window=65):
        close = df['close']
        ret = close.pct_change().fillna(0)
        raw = -np.sign(ret.shift(1)) * ret.shift(1) / (ret.rolling(window).std() + 1e-8)
        # Sign/Binary Soft
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_232_wf(df, window=10, winsorize_quantile=0.3):
        close = df['close']
        ret = close.pct_change().fillna(0)
        raw = -np.sign(ret.shift(1)) * ret.shift(1) / (ret.rolling(window).std() + 1e-8)
        # Winsorized Fisher
        low = raw.rolling(window).quantile(winsorize_quantile)
        high = raw.rolling(window).quantile(1 - winsorize_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_233_rank(df, window=75):
        delta_close = df['close'].diff(5)
        delta_close_mean = delta_close.rolling(window).mean()
        delta_close_std = delta_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (delta_close - delta_close_mean) / delta_close_std
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = ret.rolling(10).corr(vol)
        raw = ts_zscore * (-ts_corr)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_233_tanh(df, window=90):
        delta_close = df['close'].diff(5)
        delta_close_mean = delta_close.rolling(window).mean()
        delta_close_std = delta_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (delta_close - delta_close_mean) / delta_close_std
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = ret.rolling(10).corr(vol)
        raw = ts_zscore * (-ts_corr)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_233_zscore(df, window=75):
        delta_close = df['close'].diff(5)
        delta_close_mean = delta_close.rolling(window).mean()
        delta_close_std = delta_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (delta_close - delta_close_mean) / delta_close_std
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = ret.rolling(10).corr(vol)
        raw = ts_zscore * (-ts_corr)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_233_sign(df, window=25):
        delta_close = df['close'].diff(5)
        delta_close_mean = delta_close.rolling(window).mean()
        delta_close_std = delta_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (delta_close - delta_close_mean) / delta_close_std
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = ret.rolling(10).corr(vol)
        raw = ts_zscore * (-ts_corr)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_233_wf(df, window=30, p1=0.1):
        p2 = window
        delta_close = df['close'].diff(5)
        delta_close_mean = delta_close.rolling(window).mean()
        delta_close_std = delta_close.rolling(window).std().replace(0, np.nan)
        ts_zscore = (delta_close - delta_close_mean) / delta_close_std
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = ret.rolling(10).corr(vol)
        raw = ts_zscore * (-ts_corr)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_234_k(df, window=15):
        close = df['close']
        high = df['high']
        low = df['low']
        sma_close = close.rolling(window, min_periods=1).mean()
        raw = -((close - sma_close) / ((high - low).rolling(window, min_periods=1).mean() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_234_h(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        sma_close = close.rolling(window, min_periods=1).mean()
        raw = -((close - sma_close) / ((high - low).rolling(window, min_periods=1).mean() + 1e-8))
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_234_e(df, window=10):
        close = df['close']
        high = df['high']
        low = df['low']
        sma_close = close.rolling(window, min_periods=1).mean()
        raw = -((close - sma_close) / ((high - low).rolling(window, min_periods=1).mean() + 1e-8))
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std()
        signal = ((raw - mean_) / (std_ + 1e-8)).clip(-1, 1)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_234_y(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        sma_close = close.rolling(window, min_periods=1).mean()
        raw = -((close - sma_close) / ((high - low).rolling(window, min_periods=1).mean() + 1e-8))
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_234_r(df, window=10, p1=0.7):
        close = df['close']
        high = df['high']
        low = df['low']
        sma_close = close.rolling(window, min_periods=1).mean()
        raw = -((close - sma_close) / ((high - low).rolling(window, min_periods=1).mean() + 1e-8))
        p2 = window
        low_quantile = raw.rolling(p2).quantile(p1)
        high_quantile = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_quantile, upper=high_quantile, axis=0)
        normalized = np.arctanh(((winsorized - low_quantile) / (high_quantile - low_quantile + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_235_5(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 0))
        vol_log = np.log1p(volume)
        delta_close = close.diff(1)
        delta_vol = vol_log.diff(1)
        ts_std_close = delta_close.rolling(window).std()
        ts_std_vol = delta_vol.rolling(window).std().replace(0, np.nan)
        ratio = ts_std_close / (ts_std_vol + 1e-8)
        ts_corr_cv = delta_close.rolling(window).corr(delta_vol).fillna(0)
        raw = ratio * ts_corr_cv
        normalized = np.tanh(raw / (raw.rolling(window).std().replace(0, np.nan) + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_235_rank(df, window=5):
        delta_close = df['close'].diff()
        delta_volume = df.get('matchingVolume', df.get('volume', 1)).diff()
        std_close = delta_close.rolling(window).std()
        std_volume = delta_volume.rolling(window).std()
        raw = (std_close / (std_volume + 1e-8)) * delta_close.rolling(window).corr(delta_volume)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_235_tanh(df, window=15):
        delta_close = df['close'].diff()
        delta_volume = df.get('matchingVolume', df.get('volume', 1)).diff()
        std_close = delta_close.rolling(window).std()
        std_volume = delta_volume.rolling(window).std()
        raw = (std_close / (std_volume + 1e-8)) * delta_close.rolling(window).corr(delta_volume)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_235_zscore(df, window=10):
        delta_close = df['close'].diff()
        delta_volume = df.get('matchingVolume', df.get('volume', 1)).diff()
        std_close = delta_close.rolling(window).std()
        std_volume = delta_volume.rolling(window).std()
        raw = (std_close / (std_volume + 1e-8)) * delta_close.rolling(window).corr(delta_volume)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_235_sign(df, window=10):
        delta_close = df['close'].diff()
        delta_volume = df.get('matchingVolume', df.get('volume', 1)).diff()
        std_close = delta_close.rolling(window).std()
        std_volume = delta_volume.rolling(window).std()
        raw = (std_close / (std_volume + 1e-8)) * delta_close.rolling(window).corr(delta_volume)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_235_wf(df, window=10, winsor_quantile=0.3):
        delta_close = df['close'].diff()
        delta_volume = df.get('matchingVolume', df.get('volume', 1)).diff()
        std_close = delta_close.rolling(window).std()
        std_volume = delta_volume.rolling(window).std()
        raw = (std_close / (std_volume + 1e-8)) * delta_close.rolling(window).corr(delta_volume)
        p2 = window
        low = raw.rolling(p2).quantile(winsor_quantile)
        high = raw.rolling(p2).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_236_rank(df, window=15):
        delta = df['close'].diff()
        mean_short = delta.rolling(window).mean()
        mean_long = delta.rolling(15).mean()
        raw = (mean_short / (mean_long + 1e-8)) * np.sign(mean_short)
        signal = ((raw.rolling(window).rank(pct=True) * 2) - 1).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_236_tanh(df, window=25):
        delta = df['close'].diff()
        mean_short = delta.rolling(5).mean()
        mean_long = delta.rolling(window).mean()
        raw = (mean_short / (mean_long + 1e-8)) * np.sign(mean_short)
        signal = np.tanh(raw / (raw.abs().rolling(window).std().replace(0, np.nan))).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_236_zscore(df, window=5):
        delta = df['close'].diff()
        mean_short = delta.rolling(5).mean()
        mean_long = delta.rolling(window).mean()
        raw = (mean_short / (mean_long + 1e-8)) * np.sign(mean_short)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_236_sign(df, window=20):
        delta = df['close'].diff()
        mean_short = delta.rolling(5).mean()
        mean_long = delta.rolling(window).mean()
        raw = (mean_short / (mean_long + 1e-8)) * np.sign(mean_short)
        signal = np.sign(raw).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_236_wf(df, window=5):
        delta = df['close'].diff()
        mean_short = delta.rolling(5).mean()
        mean_long = delta.rolling(window).mean()
        raw = (mean_short / (mean_long + 1e-8)) * np.sign(mean_short)
        low = raw.rolling(window).quantile(0.05)
        high = raw.rolling(window).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_237_rank(df, window=55):
        v = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = v.diff(1).abs()
        close_delta = df['close'].diff(1).abs()
        raw = ((vol_delta - vol_delta.rolling(window).mean()) / vol_delta.rolling(window).std()) - ((close_delta - close_delta.rolling(window).mean()) / close_delta.rolling(window).std())
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_237_tanh(df, window=90):
        v = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = v.diff(1).abs()
        close_delta = df['close'].diff(1).abs()
        raw = ((vol_delta - vol_delta.rolling(window).mean()) / vol_delta.rolling(window).std()) - ((close_delta - close_delta.rolling(window).mean()) / close_delta.rolling(window).std())
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return norm.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_237_zscore(df, window=35):
        v = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = v.diff(1).abs()
        close_delta = df['close'].diff(1).abs()
        raw = ((vol_delta - vol_delta.rolling(window).mean()) / vol_delta.rolling(window).std()) - ((close_delta - close_delta.rolling(window).mean()) / close_delta.rolling(window).std())
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return norm.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_237_sign(df, window=25):
        v = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = v.diff(1).abs()
        close_delta = df['close'].diff(1).abs()
        raw = ((vol_delta - vol_delta.rolling(window).mean()) / vol_delta.rolling(window).std()) - ((close_delta - close_delta.rolling(window).mean()) / close_delta.rolling(window).std())
        norm = np.sign(raw)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_237_wf(df, window=30, quantile_factor=0.3):
        v = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = v.diff(1).abs()
        close_delta = df['close'].diff(1).abs()
        raw = ((vol_delta - vol_delta.rolling(window).mean()) / vol_delta.rolling(window).std()) - ((close_delta - close_delta.rolling(window).mean()) / close_delta.rolling(window).std())
        p1 = quantile_factor
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_238_rank(df, window=10, rank_window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        ret = close.pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        mean_range = (close / (high - low + 1e-8)).rolling(window).mean() + 1e-8
        raw = std_ret / mean_range
        normalized = (raw.rolling(rank_window).rank(pct=True) * 2) - 1
        sign = np.sign(ret.rolling(20).mean().fillna(0))
        signal = normalized * sign
        return signal

    @staticmethod
    def alpha_quanta_full_base_238_tanh(df, window=80, reg_window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        ret = close.pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        mean_range = (close / (high - low + 1e-8)).rolling(window).mean() + 1e-8
        raw = std_ret / mean_range
        normalized = np.tanh(raw / raw.rolling(reg_window).std().replace(0, np.nan).ffill())
        sign = np.sign(ret.rolling(20).mean().fillna(0))
        signal = normalized * sign
        return signal

    @staticmethod
    def alpha_quanta_full_base_238_zscore(df, window=20, z_window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        ret = close.pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        mean_range = (close / (high - low + 1e-8)).rolling(window).mean() + 1e-8
        raw = std_ret / mean_range
        mean_raw = raw.rolling(z_window).mean()
        std_raw = raw.rolling(z_window).std().replace(0, np.nan)
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        sign = np.sign(ret.rolling(20).mean().fillna(0))
        signal = normalized * sign
        return -signal

    @staticmethod
    def alpha_quanta_full_base_238_sign(df, window=40, rank_window=1):
        close = df['close']
        high = df['high']
        low = df['low']
        ret = close.pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        mean_range = (close / (high - low + 1e-8)).rolling(window).mean() + 1e-8
        raw = std_ret / mean_range
        normalized = np.sign(raw.rolling(rank_window).rank(pct=True) - 0.5)
        sign = np.sign(ret.rolling(20).mean().fillna(0))
        signal = normalized * sign
        return signal

    @staticmethod
    def alpha_quanta_full_base_238_wf(df, window=80, p1=0.5, p2=60):
        close = df['close']
        high = df['high']
        low = df['low']
        ret = close.pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        mean_range = (close / (high - low + 1e-8)).rolling(window).mean() + 1e-8
        raw = std_ret / mean_range
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        sign = np.sign(ret.rolling(20).mean().fillna(0))
        signal = normalized * sign
        return -signal

    @staticmethod
    def alpha_quanta_full_base_239_rank(df: pd.DataFrame, window: int = 40) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        midpoint = (high + low) * 0.5
        price_range_norm = (high - low) / (high + low + 1e-9)
        volume_delta = volume.diff().fillna(0)
        volume_delta_norm = volume_delta / (volume + 1e-9)
        cov = price_range_norm.rolling(window).cov(volume_delta_norm)
        var_x = price_range_norm.rolling(window).var().replace(0, np.nan)
        var_y = volume_delta_norm.rolling(window).var().replace(0, np.nan)
        std_product = np.sqrt(var_x * var_y).replace(0, np.nan)
        corr = (cov / std_product).fillna(0).clip(-1, 1)
        raw = -corr.abs()
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_239_tanh(df: pd.DataFrame, window: int = 40) -> pd.Series:
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range_norm = (high - low) / (high + low + 1e-9)
        volume_delta = volume.diff().fillna(0)
        volume_delta_norm = volume_delta / (volume + 1e-9)
        cov = price_range_norm.rolling(window).cov(volume_delta_norm)
        var_x = price_range_norm.rolling(window).var().replace(0, np.nan)
        var_y = volume_delta_norm.rolling(window).var().replace(0, np.nan)
        std_product = np.sqrt(var_x * var_y).replace(0, np.nan)
        corr = (cov / std_product).fillna(0).clip(-1, 1)
        raw = -corr.abs()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_239_zscore(df: pd.DataFrame, window: int = 40) -> pd.Series:
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range_norm = (high - low) / (high + low + 1e-9)
        volume_delta = volume.diff().fillna(0)
        volume_delta_norm = volume_delta / (volume + 1e-9)
        cov = price_range_norm.rolling(window).cov(volume_delta_norm)
        var_x = price_range_norm.rolling(window).var().replace(0, np.nan)
        var_y = volume_delta_norm.rolling(window).var().replace(0, np.nan)
        std_product = np.sqrt(var_x * var_y).replace(0, np.nan)
        corr = (cov / std_product).fillna(0).clip(-1, 1)
        raw = -corr.abs()
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_239_sign(df: pd.DataFrame, window: int = 40) -> pd.Series:
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range_norm = (high - low) / (high + low + 1e-9)
        volume_delta = volume.diff().fillna(0)
        volume_delta_norm = volume_delta / (volume + 1e-9)
        cov = price_range_norm.rolling(window).cov(volume_delta_norm)
        var_x = price_range_norm.rolling(window).var().replace(0, np.nan)
        var_y = volume_delta_norm.rolling(window).var().replace(0, np.nan)
        std_product = np.sqrt(var_x * var_y).replace(0, np.nan)
        corr = (cov / std_product).fillna(0).clip(-1, 1)
        raw = -corr.abs()
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_239_wf(df: pd.DataFrame, window: int = 40, quantile: float = 0.05) -> pd.Series:
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        p1 = quantile
        p2 = window
        price_range_norm = (high - low) / (high + low + 1e-9)
        volume_delta = volume.diff().fillna(0)
        volume_delta_norm = volume_delta / (volume + 1e-9)
        cov = price_range_norm.rolling(window).cov(volume_delta_norm)
        var_x = price_range_norm.rolling(window).var().replace(0, np.nan)
        var_y = volume_delta_norm.rolling(window).var().replace(0, np.nan)
        std_product = np.sqrt(var_x * var_y).replace(0, np.nan)
        corr = (cov / std_product).fillna(0).clip(-1, 1)
        raw = -corr.abs()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_240_k(df, window=30, sub_window=7):
        high = df['high']
        low = df['low']
        close = df['close']
        open = df['open']
        raw = (high - close).rolling(window).mean() - (close - low).rolling(window).mean()
        denom = (high - low).rolling(window).std() + 1e-8
        inner = raw / denom
        inner_rank = inner.rolling(window).rank(pct=True) * 2 - 1
        sign = np.sign((close - open).rolling(sub_window).mean())
        result = inner_rank * sign
        result = result.ffill().fillna(0)
        return result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_240_h(df, window=30, sub_window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        open = df['open']
        raw = (high - close).rolling(window).mean() - (close - low).rolling(window).mean()
        denom = (high - low).rolling(window).std() + 1e-8
        inner = raw / denom
        inner_tanh = np.tanh(inner / inner.rolling(window).std())
        sign = np.sign((close - open).rolling(sub_window).mean())
        result = inner_tanh * sign
        result = result.ffill().fillna(0)
        return result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_240_p(df, window=60, sub_window=1):
        high = df['high']
        low = df['low']
        close = df['close']
        open = df['open']
        raw = (high - close).rolling(window).mean() - (close - low).rolling(window).mean()
        denom = (high - low).rolling(window).std() + 1e-8
        inner = raw / denom
        inner_zscore = ((inner - inner.rolling(window).mean()) / inner.rolling(window).std()).clip(-1, 1)
        sign = np.sign((close - open).rolling(sub_window).mean())
        result = inner_zscore * sign
        result = result.ffill().fillna(0)
        return result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_240_y(df, window=100, sub_window=3):
        high = df['high']
        low = df['low']
        close = df['close']
        open = df['open']
        raw = (high - close).rolling(window).mean() - (close - low).rolling(window).mean()
        denom = (high - low).rolling(window).std() + 1e-8
        inner = raw / denom
        inner_sign = np.sign(inner)
        sign = np.sign((close - open).rolling(sub_window).mean())
        result = inner_sign * sign
        result = result.ffill().fillna(0)
        return -result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_240_r(df, window=90, sub_window=5, p1=0.05, p2=30):
        high = df['high']
        low = df['low']
        close = df['close']
        open = df['open']
        raw = (high - close).rolling(window).mean() - (close - low).rolling(window).mean()
        denom = (high - low).rolling(window).std() + 1e-8
        inner = raw / denom
        low_q = inner.rolling(p2).quantile(p1)
        high_q = inner.rolling(p2).quantile(1 - p1)
        winsorized = inner.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        sign = np.sign((close - open).rolling(sub_window).mean())
        result = normalized * sign
        result = result.ffill().fillna(0)
        return result.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_241_rank(df, window=80):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        spread = high - low
        mean_spread = spread.rolling(window, min_periods=1).mean() + 1e-8
        ratio_spread = spread / mean_spread
        mean_volume = volume.rolling(window, min_periods=1).mean() + 1e-8
        ratio_volume = volume / mean_volume
        corr = ratio_spread.rolling(window, min_periods=1).corr(ratio_volume)
        raw = corr.rolling(window, min_periods=1).rank(pct=True) * 2 - 1
        raw = raw.fillna(0)
        return -raw * 1.0

    @staticmethod
    def alpha_quanta_full_base_241_tanh(df, window=5):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        spread = high - low
        mean_spread = spread.rolling(window, min_periods=1).mean() + 1e-8
        ratio_spread = spread / mean_spread
        mean_volume = volume.rolling(window, min_periods=1).mean() + 1e-8
        ratio_volume = volume / mean_volume
        corr = ratio_spread.rolling(window, min_periods=1).corr(ratio_volume)
        raw = corr
        std = raw.rolling(window, min_periods=1).std().replace(0, 1e-8)
        normalized = np.tanh(raw / std)
        normalized = normalized.fillna(0)
        return -normalized * 1.0

    @staticmethod
    def alpha_quanta_full_base_241_zscore(df, window=70):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        spread = high - low
        mean_spread = spread.rolling(window, min_periods=1).mean() + 1e-8
        ratio_spread = spread / mean_spread
        mean_volume = volume.rolling(window, min_periods=1).mean() + 1e-8
        ratio_volume = volume / mean_volume
        corr = ratio_spread.rolling(window, min_periods=1).corr(ratio_volume)
        raw = corr
        mean = raw.rolling(window, min_periods=1).mean()
        std = raw.rolling(window, min_periods=1).std().replace(0, 1e-8)
        normalized = ((raw - mean) / std).clip(-1, 1)
        normalized = normalized.fillna(0)
        return normalized * 1.0

    @staticmethod
    def alpha_quanta_full_base_241_sign(df, window=20):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        spread = high - low
        mean_spread = spread.rolling(window, min_periods=1).mean() + 1e-8
        ratio_spread = spread / mean_spread
        mean_volume = volume.rolling(window, min_periods=1).mean() + 1e-8
        ratio_volume = volume / mean_volume
        corr = ratio_spread.rolling(window, min_periods=1).corr(ratio_volume)
        raw = corr
        normalized = np.sign(raw)
        normalized = normalized.fillna(0)
        return normalized * 1.0

    @staticmethod
    def alpha_quanta_full_base_241_wf(df, window=70, p1=0.3):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        spread = high - low
        mean_spread = spread.rolling(window, min_periods=1).mean() + 1e-8
        ratio_spread = spread / mean_spread
        mean_volume = volume.rolling(window, min_periods=1).mean() + 1e-8
        ratio_volume = volume / mean_volume
        corr = ratio_spread.rolling(window, min_periods=1).corr(ratio_volume)
        raw = corr
        p2 = window
        low_q = raw.rolling(p2, min_periods=1).quantile(p1)
        high_q = raw.rolling(p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        delta = high_q - low_q + 1e-9
        normalized = np.arctanh(((winsorized - low_q) / delta) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized * 1.0

    @staticmethod
    def alpha_quanta_full_base_242_rank(df, window=10):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df['matchingVolume']
        mean_vol = vol.rolling(window).mean()
        std_vol = vol.rolling(window).std()
        raw = np.sign(mean_ret) * mean_vol / (std_vol + 1)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_242_tanh(df, window=25):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df['matchingVolume']
        mean_vol = vol.rolling(window).mean()
        std_vol = vol.rolling(window).std()
        raw = np.sign(mean_ret) * mean_vol / (std_vol + 1)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_242_zscore(df, window=10):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df['matchingVolume']
        mean_vol = vol.rolling(window).mean()
        std_vol = vol.rolling(window).std()
        raw = np.sign(mean_ret) * mean_vol / (std_vol + 1)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_242_sign(df, window=30):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df['matchingVolume']
        mean_vol = vol.rolling(window).mean()
        std_vol = vol.rolling(window).std()
        raw = np.sign(mean_ret) * mean_vol / (std_vol + 1)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_242_wf(df, window=20, p1=0.1, p2=30):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        vol = df['matchingVolume']
        mean_vol = vol.rolling(window).mean()
        std_vol = vol.rolling(window).std()
        raw = np.sign(mean_ret) * mean_vol / (std_vol + 1)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_243_rank(df, window=95):
        close = df['close']
        volume = df['matchingVolume']
        ts_std_5 = close.rolling(5).std()
        ts_pctchange_volume_1 = volume.pct_change(1).fillna(0)
        ts_corr = ts_std_5.rolling(window).corr(ts_pctchange_volume_1).fillna(0)
        result = (ts_corr.rolling(window).rank(pct=True) * 2) - 1
        return result

    @staticmethod
    def alpha_quanta_full_base_243_tanh(df, window=45):
        close = df['close']
        volume = df['matchingVolume']
        ts_std_5 = close.rolling(5).std()
        ts_pctchange_volume_1 = volume.pct_change(1).fillna(0)
        ts_corr = ts_std_5.rolling(window).corr(ts_pctchange_volume_1).fillna(0)
        result = np.tanh(ts_corr / ts_corr.rolling(window).std().replace(0, np.nan)).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_243_zscore(df, window=75):
        close = df['close']
        volume = df['matchingVolume']
        ts_std_5 = close.rolling(5).std()
        ts_pctchange_volume_1 = volume.pct_change(1).fillna(0)
        ts_corr = ts_std_5.rolling(window).corr(ts_pctchange_volume_1).fillna(0)
        mean = ts_corr.rolling(window).mean()
        std = ts_corr.rolling(window).std().replace(0, np.nan)
        result = ((ts_corr - mean) / std).clip(-1, 1).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_243_sign(df, window=45):
        close = df['close']
        volume = df['matchingVolume']
        ts_std_5 = close.rolling(5).std()
        ts_pctchange_volume_1 = volume.pct_change(1).fillna(0)
        ts_corr = ts_std_5.rolling(window).corr(ts_pctchange_volume_1).fillna(0)
        result = np.sign(ts_corr).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_243_wf(df, window=40, quantile=0.1):
        close = df['close']
        volume = df['matchingVolume']
        ts_std_5 = close.rolling(5).std()
        ts_pctchange_volume_1 = volume.pct_change(1).fillna(0)
        ts_corr = ts_std_5.rolling(window).corr(ts_pctchange_volume_1).fillna(0)
        low = ts_corr.rolling(window).quantile(quantile)
        high = ts_corr.rolling(window).quantile(1 - quantile)
        winsorized = ts_corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_full_base_244_rank(df, window=40, sub_window=80):
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0)
        ts_mean_ret = returns.rolling(window).mean()
        delay_close = close.shift(1)
        range_ratio = (high - low) / (delay_close + 1e-8)
        ts_rank_range = range_ratio.rolling(sub_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        raw = ts_mean_ret * ts_rank_range
        raw = raw.rolling(window).rank(pct=True)
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_244_tanh(df, window=3, sub_window=50):
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0)
        ts_mean_ret = returns.rolling(window).mean()
        delay_close = close.shift(1)
        range_ratio = (high - low) / (delay_close + 1e-8)
        ts_rank_range = range_ratio.rolling(sub_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        raw = ts_mean_ret * ts_rank_range
        signal = np.tanh(raw / raw.rolling(window).std())
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_244_zscore(df, window=3, sub_window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0)
        ts_mean_ret = returns.rolling(window).mean()
        delay_close = close.shift(1)
        range_ratio = (high - low) / (delay_close + 1e-8)
        ts_rank_range = range_ratio.rolling(sub_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        raw = ts_mean_ret * ts_rank_range
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_244_sign(df, window=3, sub_window=90):
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0)
        ts_mean_ret = returns.rolling(window).mean()
        delay_close = close.shift(1)
        range_ratio = (high - low) / (delay_close + 1e-8)
        ts_rank_range = range_ratio.rolling(sub_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        raw = ts_mean_ret * ts_rank_range
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_244_wf(df, window=1, sub_window=60, p1=0.05, p2=20):
        close = df['close']
        high = df['high']
        low = df['low']
        returns = close.pct_change().fillna(0)
        ts_mean_ret = returns.rolling(window).mean()
        delay_close = close.shift(1)
        range_ratio = (high - low) / (delay_close + 1e-8)
        ts_rank_range = range_ratio.rolling(sub_window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        raw = ts_mean_ret * ts_rank_range
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_245_rank(df, window=5):
        # Volume and close handling
        volume = df['matchingVolume']
        close = df['close']
        # Delta calculations
        delta_close = close.diff()
        delta_volume = volume.diff()
        # Rolling correlation
        corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan)
        # Rolling z-score of volume
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().replace(0, np.nan)
        zscore_vol = (volume - vol_mean) / vol_std
        # Raw signal: product then rank
        raw = zscore_vol * corr
        # Rolling rank normalization (case A)
        ranked = raw.rolling(window).rank(pct=True) * 2 - 1
        # Handle remaining NaNs
        ranked = ranked.ffill().fillna(0)
        return ranked

    @staticmethod
    def alpha_quanta_full_base_245_tanh(df, window=5):
        volume = df['matchingVolume']
        close = df['close']
        delta_close = close.diff()
        delta_volume = volume.diff()
        corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().replace(0, np.nan)
        zscore_vol = (volume - vol_mean) / vol_std
        raw = zscore_vol * corr
        # Dynamic tanh normalization (case B)
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std_raw)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_245_zscore(df, window=5):
        volume = df['matchingVolume']
        close = df['close']
        delta_close = close.diff()
        delta_volume = volume.diff()
        corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().replace(0, np.nan)
        zscore_vol = (volume - vol_mean) / vol_std
        raw = zscore_vol * corr
        # Rolling z-score clip normalization (case C)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - raw_mean) / raw_std).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_245_sign(df, window=5):
        volume = df['matchingVolume']
        close = df['close']
        delta_close = close.diff()
        delta_volume = volume.diff()
        corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().replace(0, np.nan)
        zscore_vol = (volume - vol_mean) / vol_std
        raw = zscore_vol * corr
        # Sign soft normalization (case D)
        normalized = np.sign(raw)
        normalized = pd.Series(normalized, index=df.index).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_245_wf(df, window=20, p1=0.1):
        p2 = window
        volume = df['matchingVolume']
        close = df['close']
        delta_close = close.diff()
        delta_volume = volume.diff()
        corr = delta_close.rolling(window).corr(delta_volume).replace([np.inf, -np.inf], np.nan)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std().replace(0, np.nan)
        zscore_vol = (volume - vol_mean) / vol_std
        raw = zscore_vol * corr
        # Winsorized Fisher normalization (case E)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_246_rank(df, window=45):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        inv_close = 1.0 / close
        ts_mean_inv = inv_close.rolling(window).mean()
        sign_raw = np.sign(inv_close - ts_mean_inv)
        seq = pd.Series(np.arange(window), index=range(window))
        x = pd.Series(np.full(len(df), np.nan), index=df.index)
        for i in range(window, len(df)):
            x.iloc[i] = np.corrcoef(volume.iloc[i-window:i], seq)[0, 1]
        x = x.ffill()
        raw = sign_raw * (1 - x)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_246_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        inv_close = 1.0 / close
        ts_mean_inv = inv_close.rolling(window).mean()
        sign_raw = np.sign(inv_close - ts_mean_inv)
        seq = pd.Series(np.arange(window), index=range(window))
        x = pd.Series(np.full(len(df), np.nan), index=df.index)
        for i in range(window, len(df)):
            x.iloc[i] = np.corrcoef(volume.iloc[i-window:i], seq)[0, 1]
        x = x.ffill()
        raw = sign_raw * (1 - x)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_246_zscore(df, window=45):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        inv_close = 1.0 / close
        ts_mean_inv = inv_close.rolling(window).mean()
        sign_raw = np.sign(inv_close - ts_mean_inv)
        seq = pd.Series(np.arange(window), index=range(window))
        x = pd.Series(np.full(len(df), np.nan), index=df.index)
        for i in range(window, len(df)):
            x.iloc[i] = np.corrcoef(volume.iloc[i-window:i], seq)[0, 1]
        x = x.ffill()
        raw = sign_raw * (1 - x)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean_) / std_).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_246_sign(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        inv_close = 1.0 / close
        ts_mean_inv = inv_close.rolling(window).mean()
        sign_raw = np.sign(inv_close - ts_mean_inv)
        seq = pd.Series(np.arange(window), index=range(window))
        x = pd.Series(np.full(len(df), np.nan), index=df.index)
        for i in range(window, len(df)):
            x.iloc[i] = np.corrcoef(volume.iloc[i-window:i], seq)[0, 1]
        x = x.ffill()
        raw = sign_raw * (1 - x)
        result = np.sign(raw)
        return -pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_246_wf(df, window=50, winsor_quantile=0.9):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        inv_close = 1.0 / close
        ts_mean_inv = inv_close.rolling(window).mean()
        sign_raw = np.sign(inv_close - ts_mean_inv)
        seq = pd.Series(np.arange(window), index=range(window))
        x = pd.Series(np.full(len(df), np.nan), index=df.index)
        for i in range(window, len(df)):
            x.iloc[i] = np.corrcoef(volume.iloc[i-window:i], seq)[0, 1]
        x = x.ffill()
        raw = sign_raw * (1 - x)
        low = raw.rolling(window).quantile(winsor_quantile)
        high = raw.rolling(window).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        scale = (high - low).replace(0, np.nan)
        normalized = np.arctanh(((winsorized - low) / scale) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_247_k(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0.01))
        max_close = close.rolling(window).max()
        std_close = close.rolling(window).std().replace(0, np.nan)
        mean_volume = volume.rolling(window).mean().replace(0, np.nan)
        raw = ((close - max_close) / (std_close + 1e-8)) * (std_close / (mean_volume + 1e-8))
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_247_h(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0.01))
        max_close = close.rolling(window).max()
        std_close = close.rolling(window).std().replace(0, np.nan)
        mean_volume = volume.rolling(window).mean().replace(0, np.nan)
        raw = ((close - max_close) / (std_close + 1e-8)) * (std_close / (mean_volume + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_247_e(df, window=10):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0.01))
        max_close = close.rolling(window).max()
        std_close = close.rolling(window).std().replace(0, np.nan)
        mean_volume = volume.rolling(window).mean().replace(0, np.nan)
        raw = ((close - max_close) / (std_close + 1e-8)) * (std_close / (mean_volume + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_247_y(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0.01))
        max_close = close.rolling(window).max()
        std_close = close.rolling(window).std().replace(0, np.nan)
        mean_volume = volume.rolling(window).mean().replace(0, np.nan)
        raw = ((close - max_close) / (std_close + 1e-8)) * (std_close / (mean_volume + 1e-8))
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_247_r(df, window=10, p1=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 0.01))
        max_close = close.rolling(window).max()
        std_close = close.rolling(window).std().replace(0, np.nan)
        mean_volume = volume.rolling(window).mean().replace(0, np.nan)
        raw = ((close - max_close) / (std_close + 1e-8)) * (std_close / (mean_volume + 1e-8))
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_248_rank(df, window=10):
        raw = df['close']
        inv_close = 1.0 / raw.replace(0, np.nan)
        zscore_inv = (inv_close - inv_close.rolling(window).mean()) / inv_close.rolling(window).std().replace(0, np.nan)
        rank_zscore = (zscore_inv.rolling(window).rank(pct=True) * 2) - 1
        ret = df['close'].pct_change()
        mean_ret_5 = ret.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        corr_5 = ret.rolling(5).corr(days).fillna(0)
        sign_mean_ret = np.sign(mean_ret_5).fillna(0)
        combined = rank_zscore * (1 - sign_mean_ret * corr_5.abs())
        result = combined.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_248_tanh(df, window=5):
        raw = df['close']
        inv_close = 1.0 / raw.replace(0, np.nan)
        zscore_inv = (inv_close - inv_close.rolling(window).mean()) / inv_close.rolling(window).std().replace(0, np.nan)
        result = np.tanh(zscore_inv / zscore_inv.rolling(window).std().replace(0, np.nan))
        result = result.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_248_zscore(df, window=30):
        raw = df['close']
        inv_close = 1.0 / raw.replace(0, np.nan)
        zscore_inv = (inv_close - inv_close.rolling(window).mean()) / inv_close.rolling(window).std().replace(0, np.nan)
        ret = df['close'].pct_change()
        mean_ret_5 = ret.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        corr_5 = ret.rolling(5).corr(days).fillna(0)
        sign_mean_ret = np.sign(mean_ret_5).fillna(0)
        combined = zscore_inv * (1 - sign_mean_ret * corr_5.abs())
        result = ((combined - combined.rolling(window).mean()) / combined.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        result = result.fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_248_sign(df, window=5):
        raw = df['close']
        inv_close = 1.0 / raw.replace(0, np.nan)
        zscore_inv = (inv_close - inv_close.rolling(window).mean()) / inv_close.rolling(window).std().replace(0, np.nan)
        ret = df['close'].pct_change()
        mean_ret_5 = ret.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        corr_5 = ret.rolling(5).corr(days).fillna(0)
        sign_mean_ret = np.sign(mean_ret_5).fillna(0)
        raw_signal = zscore_inv * (1 - sign_mean_ret * corr_5.abs())
        result = np.sign(raw_signal).fillna(0)
        return -result

    @staticmethod
    def alpha_quanta_full_base_248_wf(df, window=20, p1=0.3):
        raw = df['close']
        inv_close = 1.0 / raw.replace(0, np.nan)
        zscore_inv = (inv_close - inv_close.rolling(window).mean()) / inv_close.rolling(window).std().replace(0, np.nan)
        ret = df['close'].pct_change()
        mean_ret_5 = ret.rolling(5).mean()
        days = pd.Series(np.arange(len(df)), index=df.index)
        corr_5 = ret.rolling(5).corr(days).fillna(0)
        sign_mean_ret = np.sign(mean_ret_5).fillna(0)
        raw_signal = zscore_inv * (1 - sign_mean_ret * corr_5.abs())
        low = raw_signal.rolling(window).quantile(p1)
        high = raw_signal.rolling(window).quantile(1 - p1)
        winsorized = raw_signal.clip(lower=low, upper=high, axis=0)
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        scaled = scaled.clip(-0.99, 0.99)
        result = np.arctanh(scaled).fillna(0)
        result = result / 3  # scale down to [-1,1] approximately
        result = result.clip(-1, 1)
        return -result

    @staticmethod
    def alpha_quanta_full_base_249_rank(df, window=100):
        raw = df['close'].rolling(5).std() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_249_tanh(df, window=5):
        raw = df['close'].rolling(5).std() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_249_zscore(df, window=25):
        raw = df['close'].rolling(5).std() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_249_sign(df):
        raw = df['close'].rolling(5).std() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_249_wf(df, window_p1=0.1, window_p2=30):
        raw = df['close'].rolling(5).std() / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        low = raw.rolling(int(window_p2)).quantile(window_p1)
        high = raw.rolling(int(window_p2)).quantile(1 - window_p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_250_rank(df, window=100):
        close = df['close']
        volume = df['matchingVolume']
        pct_chg = close.pct_change(window)
        sign_pct = np.sign(pct_chg)
        corr = close.rolling(window).corr(volume).replace(0, np.nan)
        raw = sign_pct * corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_250_tanh(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        pct_chg = close.pct_change(window)
        sign_pct = np.sign(pct_chg)
        corr = close.rolling(window).corr(volume).replace(0, np.nan)
        raw = sign_pct * corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_250_zscore(df, window=85):
        close = df['close']
        volume = df['matchingVolume']
        pct_chg = close.pct_change(window)
        sign_pct = np.sign(pct_chg)
        corr = close.rolling(window).corr(volume).replace(0, np.nan)
        raw = sign_pct * corr
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_250_sign(df, window=70):
        close = df['close']
        volume = df['matchingVolume']
        pct_chg = close.pct_change(window)
        sign_pct = np.sign(pct_chg)
        corr = close.rolling(window).corr(volume).replace(0, np.nan)
        raw = sign_pct * corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_250_wf(df, window=40, p1=0.1):
        close = df['close']
        volume = df['matchingVolume']
        pct_chg = close.pct_change(window)
        sign_pct = np.sign(pct_chg)
        corr = close.rolling(window).corr(volume).replace(0, np.nan)
        raw = sign_pct * corr
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_251_rank(df, window=75):
        mean_close = df['close'].rolling(window).mean()
        range_hl = df['high'] - df['low']
        std_range = range_hl.rolling(window).std()
        raw = mean_close / (std_range + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_251_tanh(df, window=20):
        mean_close = df['close'].rolling(window).mean()
        range_hl = df['high'] - df['low']
        std_range = range_hl.rolling(window).std()
        raw = mean_close / (std_range + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_251_zscore(df, window=30):
        mean_close = df['close'].rolling(window).mean()
        range_hl = df['high'] - df['low']
        std_range = range_hl.rolling(window).std()
        raw = mean_close / (std_range + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_251_sign(df, window=10):
        mean_close = df['close'].rolling(window).mean()
        range_hl = df['high'] - df['low']
        std_range = range_hl.rolling(window).std()
        raw = mean_close / (std_range + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_251_wf(df, window=20, p1=0.1):
        mean_close = df['close'].rolling(window).mean()
        range_hl = df['high'] - df['low']
        std_range = range_hl.rolling(window).std()
        raw = mean_close / (std_range + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_252_rank(df, window=60):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        raw_num = (volume * (high - low)).rolling(window).sum()
        raw_den = (volume.rolling(window).std() + 1e-8)
        raw_sign = np.sign(raw_num / raw_den)
        raw_abs = (abs(high - low) / (volume + 1e-8)).rolling(window).sum()
        raw = raw_sign * raw_abs
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_252_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        raw_num = (volume * (high - low)).rolling(window).sum()
        raw_den = (volume.rolling(window).std() + 1e-8)
        raw_sign = np.sign(raw_num / raw_den)
        raw_abs = (abs(high - low) / (volume + 1e-8)).rolling(window).sum()
        raw = raw_sign * raw_abs
        norm = np.tanh(raw / (raw.abs().rolling(window).std() + 1e-8))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_252_zscore(df, window=70):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        raw_num = (volume * (high - low)).rolling(window).sum()
        raw_den = (volume.rolling(window).std() + 1e-8)
        raw_sign = np.sign(raw_num / raw_den)
        raw_abs = (abs(high - low) / (volume + 1e-8)).rolling(window).sum()
        raw = raw_sign * raw_abs
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std()
        norm = ((raw - mean_) / std_).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_252_sign(df, window=40):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        raw_num = (volume * (high - low)).rolling(window).sum()
        raw_den = (volume.rolling(window).std() + 1e-8)
        raw_sign = np.sign(raw_num / raw_den)
        raw_abs = (abs(high - low) / (volume + 1e-8)).rolling(window).sum()
        raw = raw_sign * raw_abs
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_252_wf(df, window=60, quantile=0.7):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        raw_num = (volume * (high - low)).rolling(window).sum()
        raw_den = (volume.rolling(window).std() + 1e-8)
        raw_sign = np.sign(raw_num / raw_den)
        raw_abs = (abs(high - low) / (volume + 1e-8)).rolling(window).sum()
        raw = raw_sign * raw_abs
        p2 = max(window * 2, 10)
        low_quant = raw.rolling(p2).quantile(quantile)
        high_quant = raw.rolling(p2).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_quant, upper=high_quant, axis=0)
        norm = np.arctanh(((winsorized - low_quant) / (high_quant - low_quant + 1e-9)) * 1.98 - 0.99)
        return -norm.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_253_rank(df, window_rank=100):
        log_close = np.log(df['close'])
        delta_log_close = log_close.diff(1)
        std_log_return -= delta_log_close.rolling(10).std()
        log_vol = np.log1p(df['matchingVolume'])
        mean_log_vol = log_vol.rolling(10).mean()
        raw = std_log_return / (mean_log_vol + 1e-8)
        normalized = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_253_tanh(df, window_tanh=15):
        log_close = np.log(df['close'])
        delta_log_close = log_close.diff(1)
        std_log_return -= delta_log_close.rolling(10).std()
        log_vol = np.log1p(df['matchingVolume'])
        mean_log_vol = log_vol.rolling(10).mean()
        raw = std_log_return / (mean_log_vol + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window_tanh).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_253_zscore(df, window_zscore=100):
        log_close = np.log(df['close'])
        delta_log_close = log_close.diff(1)
        std_log_return -= delta_log_close.rolling(10).std()
        log_vol = np.log1p(df['matchingVolume'])
        mean_log_vol = log_vol.rolling(10).mean()
        raw = std_log_return / (mean_log_vol + 1e-8)
        normalized = ((raw - raw.rolling(window_zscore).mean()) / raw.rolling(window_zscore).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_253_sign(df):
        log_close = np.log(df['close'])
        delta_log_close = log_close.diff(1)
        std_log_return = delta_log_close.rolling(10).std()
        log_vol = np.log1p(df['matchingVolume'])
        mean_log_vol = log_vol.rolling(10).mean()
        raw = std_log_return / (mean_log_vol + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_253_wf(df, p1_quantile=0.1):
        p2_winsor = 50
        log_close = np.log(df['close'])
        delta_log_close = log_close.diff(1)
        std_log_return -= delta_log_close.rolling(10).std()
        log_vol = np.log1p(df['matchingVolume'])
        mean_log_vol = log_vol.rolling(10).mean()
        raw = std_log_return / (mean_log_vol + 1e-8)
        low = raw.rolling(p2_winsor).quantile(p1_quantile)
        high = raw.rolling(p2_winsor).quantile(1 - p1_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_254_rank(df, window=25):
        delta_close = df['close'].diff()
        volume_log = np.log1p(df['matchingVolume'])
        rolling_window = max(window, 2)
        std_series = delta_close.rolling(rolling_window).std()
        mean_abs = delta_close.abs().rolling(rolling_window).mean()
        denom = mean_abs + 1e-8
        ratio = 1 - (std_series / denom)
        corr_series = delta_close.rolling(rolling_window).corr(volume_log)
        raw = corr_series * ratio
        raw = raw.ffill().fillna(0)
        result = (raw.rolling(rolling_window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_254_tanh(df, window=20):
        delta_close = df['close'].diff()
        volume_log = np.log1p(df['matchingVolume'])
        rolling_window = max(window, 2)
        std_series = delta_close.rolling(rolling_window).std()
        mean_abs = delta_close.abs().rolling(rolling_window).mean()
        denom = mean_abs + 1e-8
        ratio = 1 - (std_series / denom)
        corr_series = delta_close.rolling(rolling_window).corr(volume_log)
        raw = corr_series * ratio
        raw = raw.ffill().fillna(0)
        result = np.tanh(raw / (raw.rolling(rolling_window).std().replace(0, np.nan).ffill() + 1e-8))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_254_zscore(df, window=30):
        delta_close = df['close'].diff()
        volume_log = np.log1p(df['matchingVolume'])
        rolling_window = max(window, 2)
        std_series = delta_close.rolling(rolling_window).std()
        mean_abs = delta_close.abs().rolling(rolling_window).mean()
        denom = mean_abs + 1e-8
        ratio = 1 - (std_series / denom)
        corr_series = delta_close.rolling(rolling_window).corr(volume_log)
        raw = corr_series * ratio
        raw = raw.ffill().fillna(0)
        rolling_mean = raw.rolling(rolling_window).mean()
        rolling_std = raw.rolling(rolling_window).std().replace(0, np.nan).ffill()
        result = ((raw - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_254_sign(df, window=25):
        delta_close = df['close'].diff()
        volume_log = np.log1p(df['matchingVolume'])
        rolling_window = max(window, 2)
        std_series = delta_close.rolling(rolling_window).std()
        mean_abs = delta_close.abs().rolling(rolling_window).mean()
        denom = mean_abs + 1e-8
        ratio = 1 - (std_series / denom)
        corr_series = delta_close.rolling(rolling_window).corr(volume_log)
        raw = corr_series * ratio
        raw = raw.ffill().fillna(0)
        result = np.sign(raw)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_254_wf(df, window=30, winsor_percentile=0.1):
        delta_close = df['close'].diff()
        volume_log = np.log1p(df['matchingVolume'])
        rolling_window = max(window, 2)
        std_series = delta_close.rolling(rolling_window).std()
        mean_abs = delta_close.abs().rolling(rolling_window).mean()
        denom = mean_abs + 1e-8
        ratio = 1 - (std_series / denom)
        corr_series = delta_close.rolling(rolling_window).corr(volume_log)
        raw = corr_series * ratio
        raw = raw.ffill().fillna(0)
        p = winsor_percentile
        p1 = max(min(p, 0.5 - 1e-10), 1e-10)
        low = raw.rolling(rolling_window).quantile(p1)
        high = raw.rolling(rolling_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        denom_w = high - low + 1e-9
        scaled = ((winsorized - low) / denom_w) * 1.98 - 0.99
        scaled = scaled.clip(-0.99 + 1e-9, 0.99 - 1e-9)
        result = np.arctanh(scaled)
        result = result.ffill().fillna(0)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_255_rank(df, window=65):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        std_close = close.rolling(5).std()
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        ts_corr = std_close.rolling(window).corr(volume_ratio)
        ranked = ts_corr.rolling(window).rank(pct=True)
        signal = (ranked * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_255_tanh(df, window=35):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        std_close = close.rolling(5).std()
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        ts_corr = std_close.rolling(window).corr(volume_ratio)
        signal = np.tanh(ts_corr / ts_corr.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_255_zscore(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        std_close = close.rolling(5).std()
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        ts_corr = std_close.rolling(window).corr(volume_ratio)
        mean = ts_corr.rolling(window).mean()
        std = ts_corr.rolling(window).std().replace(0, np.nan)
        signal = ((ts_corr - mean) / std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_255_sign(df, window=65):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        std_close = close.rolling(5).std()
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        ts_corr = std_close.rolling(window).corr(volume_ratio)
        signal = np.sign(ts_corr)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_255_wf(df, p1=0.1, p2=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        std_close = close.rolling(5).std()
        delta_volume = volume.diff(1)
        volume_ratio = delta_volume / (volume + 1e-8)
        ts_corr = std_close.rolling(p2).corr(volume_ratio)
        low = ts_corr.rolling(p2).quantile(p1)
        high = ts_corr.rolling(p2).quantile(1 - p1)
        winsorized = ts_corr.clip(lower=low, upper=high, axis=0)
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        scaled = scaled.clip(-0.99, 0.99)
        signal = np.arctanh(scaled)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_256_rank(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        hl_range = (high - low) / (volume + 1e-8)
        mean_hl = hl_range.rolling(window).mean()
        ret = close.pct_change()
        std_ret = ret.rolling(window).std() + 1e-8
        raw = mean_hl / std_ret
        raw = raw.ffill().fillna(0)
        return (raw.rolling(window).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_full_base_256_tanh(df, window=20):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        hl_range = (high - low) / (volume + 1e-8)
        mean_hl = hl_range.rolling(window).mean()
        ret = close.pct_change()
        std_ret = ret.rolling(window).std() + 1e-8
        raw = mean_hl / std_ret
        raw = raw.ffill().fillna(0)
        return -np.tanh(raw / raw.rolling(window).std())

    @staticmethod
    def alpha_quanta_full_base_256_zscore(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        hl_range = (high - low) / (volume + 1e-8)
        mean_hl = hl_range.rolling(window).mean()
        ret = close.pct_change()
        std_ret = ret.rolling(window).std() + 1e-8
        raw = mean_hl / std_ret
        raw = raw.ffill().fillna(0)
        return ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_256_sign(df, window=75):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        hl_range = (high - low) / (volume + 1e-8)
        mean_hl = hl_range.rolling(window).mean()
        ret = close.pct_change()
        std_ret = ret.rolling(window).std() + 1e-8
        raw = mean_hl / std_ret
        raw = raw.ffill().fillna(0)
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_full_base_256_wf(df, window=40, p2=70, p1=0.1):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        hl_range = (high - low) / (volume + 1e-8)
        mean_hl = hl_range.rolling(window).mean()
        ret = close.pct_change()
        std_ret = ret.rolling(window).std() + 1e-8
        raw = mean_hl / std_ret
        raw = raw.ffill().fillna(0)
        low_val = raw.rolling(p2).quantile(p1)
        high_val = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_val, upper=high_val, axis=0)
        normalized = np.arctanh(((winsorized - low_val) / (high_val - low_val + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_full_base_257_rank(df, window=70):
        close = df['close']
        ts_std = close.rolling(window).std()
        ts_rank = ts_std.rolling(20).rank(pct=True)
        raw = 1 - ts_rank
        signal = ((raw.rolling(window).rank(pct=True) * 2) - 1)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_257_tanh(df, window=25):
        close = df['close']
        ts_std = close.rolling(window).std()
        ts_rank = ts_std.rolling(20).rank(pct=True)
        raw = 1 - ts_rank
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_257_zscore(df, window=70):
        close = df['close']
        ts_std = close.rolling(window).std()
        ts_rank = ts_std.rolling(20).rank(pct=True)
        raw = 1 - ts_rank
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_257_sign(df, window=75):
        close = df['close']
        ts_std = close.rolling(window).std()
        ts_rank = ts_std.rolling(20).rank(pct=True)
        raw = 1 - ts_rank
        signal = np.sign(raw - 0.5)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_257_wf(df, window=70, p1=0.3):
        close = df['close']
        ts_std = close.rolling(window).std()
        ts_rank = ts_std.rolling(20).rank(pct=True)
        raw = 1 - ts_rank
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(numerator * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return -signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_258_rank(df, window=5):
        volume = df['matchingVolume']
        ret = df['close'].pct_change()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        ret_mean = ret.rolling(window).mean()
        raw = volume_z * np.sign(-ret_mean)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_258_tanh(df, window=5):
        volume = df['matchingVolume']
        ret = df['close'].pct_change()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        ret_mean = ret.rolling(window).mean()
        raw = volume_z * np.sign(-ret_mean)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_258_zscore(df, window=5):
        volume = df['matchingVolume']
        ret = df['close'].pct_change()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        ret_mean = ret.rolling(window).mean()
        raw = volume_z * np.sign(-ret_mean)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_258_sign(df, window=50):
        volume = df['matchingVolume']
        ret = df['close'].pct_change()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        ret_mean = ret.rolling(window).mean()
        raw = volume_z * np.sign(-ret_mean)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_258_wf(df, window=60, factor=0.1):
        volume = df['matchingVolume']
        ret = df['close'].pct_change()
        volume_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        ret_mean = ret.rolling(window).mean()
        raw = volume_z * np.sign(-ret_mean)
        p1 = factor
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_259_rank(df, window=25):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(window).std().replace(0, np.nan)
        rank_diff = ts_mean.rank(pct=True) - ts_std.rank(pct=True)
        raw = rank_diff
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_259_tanh(df, window=15):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(window).std().replace(0, np.nan)
        rank_diff = ts_mean.rank(pct=True) - ts_std.rank(pct=True)
        raw = rank_diff
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_259_zscore(df, window=35):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(window).std().replace(0, np.nan)
        rank_diff = ts_mean.rank(pct=True) - ts_std.rank(pct=True)
        raw = rank_diff
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_259_sign(df, window=10):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(window).std().replace(0, np.nan)
        rank_diff = ts_mean.rank(pct=True) - ts_std.rank(pct=True)
        raw = rank_diff
        normalized = np.sign(raw)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_259_wf(df, window=30, p1=0.1):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(window).std().replace(0, np.nan)
        rank_diff = ts_mean.rank(pct=True) - ts_std.rank(pct=True)
        raw = rank_diff
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_full_base_260_rank(df, window=10):
        raw = df['close'].pct_change(window).rank(pct=True) * 2 - 1
        return raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_260_tanh(df, window=15):
        ret = df['close'].pct_change(window)
        raw = ret * ((0 - ret.rolling(window).std()) / ret.rolling(window).std())
        raw = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -raw.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_260_zscore(df, window=25):
        ret = df['close'].pct_change(window)
        std = ret.rolling(window).std().replace(0, np.nan)
        z = (0 - std) / std
        raw = (ret * z - ret.rolling(window).mean()) / ret.rolling(window).std().replace(0, np.nan)
        normalized = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        return -normalized.clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_260_sign(df, window=30):
        ret = df['close'].pct_change(window)
        std = ret.rolling(window).std().replace(0, np.nan)
        z = (0 - std) / std
        raw = ret * z
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_260_wf(df, window=40, p1=0.3):
        ret = df['close'].pct_change(window)
        std = ret.rolling(window).std().replace(0, np.nan)
        z = (0 - std) / std
        raw = ret * z
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_261_rank(df, window=75):
        high = df['high']
        low = df['low']
        close = df['close']
        ret = close.pct_change()
        raw = (high - low) / (close + 1e-8)
        std_raw = raw.rolling(5).std()
        std_ret = ret.rolling(window).std()
        raw_ratio = std_raw / np.log(std_ret + 1.1 + 1e-9)
        rank = raw_ratio.rolling(window).rank(pct=True)
        signal = (rank * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_261_tanh(df, factor=5):
        high = df['high']
        low = df['low']
        close = df['close']
        ret = close.pct_change()
        raw = (high - low) / (close + 1e-8)
        std_raw = raw.rolling(5).std()
        std_ret = ret.rolling(20).std()
        raw_ratio = std_raw / np.log(std_ret + 1.1 + 1e-9)
        signal = np.tanh(raw_ratio / factor)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_261_zscore(df, window_std=100):
        high = df['high']
        low = df['low']
        close = df['close']
        ret = close.pct_change()
        raw = (high - low) / (close + 1e-8)
        std_raw = raw.rolling(5).std()
        std_ret = ret.rolling(window_std).std()
        raw_ratio = std_raw / np.log(std_ret + 1.1 + 1e-9)
        mean = raw_ratio.rolling(window_std).mean()
        std = raw_ratio.rolling(window_std).std()
        signal = ((raw_ratio - mean) / std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_261_sign(df, window=85):
        high = df['high']
        low = df['low']
        close = df['close']
        ret = close.pct_change()
        raw = (high - low) / (close + 1e-8)
        std_raw = raw.rolling(5).std()
        std_ret = ret.rolling(window).std()
        raw_ratio = std_raw / np.log(std_ret + 1.1 + 1e-9)
        signal = np.sign(raw_ratio)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_261_wf(df, p1=0.1, p2=100):
        high = df['high']
        low = df['low']
        close = df['close']
        ret = close.pct_change()
        raw = (high - low) / (close + 1e-8)
        std_raw = raw.rolling(5).std()
        std_ret = ret.rolling(20).std()
        raw_ratio = std_raw / np.log(std_ret + 1.1 + 1e-9)
        low_q = raw_ratio.rolling(p2).quantile(p1)
        high_q = raw_ratio.rolling(p2).quantile(1 - p1)
        winsorized = raw_ratio.clip(lower=low_q, upper=high_q)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_262_rank(df, window=40, corr_window=40):
        # Tính return -(phần trăm thay đổi của close)
        ret = df['close'].pct_change().fillna(0)
        # TS_MEAN của return trong 60 period
        mean_ret = ret.rolling(window).mean()
        # TS_CORR giữa volume và close
        corr_vc = df['matchingVolume'].rolling(corr_window).corr(df['close']).fillna(0)
        # SIGN của correlation
        sign_corr = np.sign(corr_vc)
        # TS_STD của (high - low) / (close + epsilon)
        hl_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        std_hl = hl_range.rolling(window).std().fillna(0)
        # Denominator
        denom = std_hl + 0.1
        # Raw công thức
        raw = mean_ret * sign_corr / denom
        # Rolling Rank normalization (A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_262_tanh(df, window=10, corr_window=30):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        corr_vc = df['matchingVolume'].rolling(corr_window).corr(df['close']).fillna(0)
        sign_corr = np.sign(corr_vc)
        hl_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        std_hl = hl_range.rolling(window).std().fillna(0)
        denom = std_hl + 0.1
        raw = mean_ret * sign_corr / denom
        # Dynamic Tanh normalization (B)
        std_raw = raw.rolling(window).std().fillna(1e-8)
        normalized = np.tanh(raw / std_raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_262_zscore(df, window=40, corr_window=40):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        corr_vc = df['matchingVolume'].rolling(corr_window).corr(df['close']).fillna(0)
        sign_corr = np.sign(corr_vc)
        hl_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        std_hl = hl_range.rolling(window).std().fillna(0)
        denom = std_hl + 0.1
        raw = mean_ret * sign_corr / denom
        # Rolling Z-Score/Clip normalization (C)
        mean_raw = raw.rolling(window).mean().fillna(0)
        std_raw = raw.rolling(window).std().fillna(1e-8)
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_262_sign(df, window=20, corr_window=40):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        corr_vc = df['matchingVolume'].rolling(corr_window).corr(df['close']).fillna(0)
        sign_corr = np.sign(corr_vc)
        hl_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        std_hl = hl_range.rolling(window).std().fillna(0)
        denom = std_hl + 0.1
        raw = mean_ret * sign_corr / denom
        # Sign/Binary Soft normalization (D)
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_262_wf(df, window=40, corr_window=40, quantile=0.05, winsor_window=120):
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window).mean()
        corr_vc = df['matchingVolume'].rolling(corr_window).corr(df['close']).fillna(0)
        sign_corr = np.sign(corr_vc)
        hl_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        std_hl = hl_range.rolling(window).std().fillna(0)
        denom = std_hl + 0.1
        raw = mean_ret * sign_corr / denom
        # Winsorized Fisher normalization (E)
        low = raw.rolling(winsor_window).quantile(quantile)
        high = raw.rolling(winsor_window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        ratio = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(ratio.clip(-0.9999, 0.9999))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_263_rank(df, window=65):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        mean_ret_15 = ret.rolling(window=window, min_periods=window).mean()
        std_ret_10 = ret.rolling(window=10, min_periods=10).std()
        delta_vol_10 = vol.diff(10).fillna(0)
        cov = mean_ret_15.rolling(window=10, min_periods=10).cov(delta_vol_10)
        var_delta = delta_vol_10.rolling(window=10, min_periods=10).var().replace(0, np.nan)
        ts_corr = cov / var_delta
        raw = ts_corr * mean_ret_15 / (std_ret_10 + 0.05)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = (raw.rolling(window=window, min_periods=window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_263_tanh(df, window=65):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        mean_ret_15 = ret.rolling(window=window, min_periods=window).mean()
        std_ret_10 = ret.rolling(window=10, min_periods=10).std()
        delta_vol_10 = vol.diff(10).fillna(0)
        cov = mean_ret_15.rolling(window=10, min_periods=10).cov(delta_vol_10)
        var_delta = delta_vol_10.rolling(window=10, min_periods=10).var().replace(0, np.nan)
        ts_corr = cov / var_delta
        raw = ts_corr * mean_ret_15 / (std_ret_10 + 0.05)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.tanh(raw / (raw.rolling(window=window, min_periods=window).std().replace(0, np.nan) + 1e-9))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_263_zscore(df, window=70):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        mean_ret_15 = ret.rolling(window=window, min_periods=window).mean()
        std_ret_10 = ret.rolling(window=10, min_periods=10).std()
        delta_vol_10 = vol.diff(10).fillna(0)
        cov = mean_ret_15.rolling(window=10, min_periods=10).cov(delta_vol_10)
        var_delta = delta_vol_10.rolling(window=10, min_periods=10).var().replace(0, np.nan)
        ts_corr = cov / var_delta
        raw = ts_corr * mean_ret_15 / (std_ret_10 + 0.05)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        mean_raw = raw.rolling(window=window, min_periods=window).mean()
        std_raw = raw.rolling(window=window, min_periods=window).std().replace(0, np.nan)
        normalized = ((raw - mean_raw) / (std_raw + 1e-9)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_263_sign(df, window=45):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        mean_ret_15 = ret.rolling(window=window, min_periods=window).mean()
        std_ret_10 = ret.rolling(window=10, min_periods=10).std()
        delta_vol_10 = vol.diff(10).fillna(0)
        cov = mean_ret_15.rolling(window=10, min_periods=10).cov(delta_vol_10)
        var_delta = delta_vol_10.rolling(window=10, min_periods=10).var().replace(0, np.nan)
        ts_corr = cov / var_delta * 10  # scale factor
        raw = ts_corr * mean_ret_15 / (std_ret_10 + 0.05)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_263_wf(df, window=90, p1=0.9):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        mean_ret_15 = ret.rolling(window=window, min_periods=window).mean()
        std_ret_10 = ret.rolling(window=10, min_periods=10).std()
        delta_vol_10 = vol.diff(10).fillna(0)
        cov = mean_ret_15.rolling(window=10, min_periods=10).cov(delta_vol_10)
        var_delta = delta_vol_10.rolling(window=10, min_periods=10).var().replace(0, np.nan)
        ts_corr = cov / var_delta
        raw = ts_corr * mean_ret_15 / (std_ret_10 + 0.05)
        raw = raw.fillna(0).replace([np.inf, -np.inf], 0)
        p2 = window
        low = raw.rolling(window=p2, min_periods=p2).quantile(p1)
        high = raw.rolling(window=p2, min_periods=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_264_rank(df, window=30):
        ret = df['close'].pct_change()
        std = ret.rolling(window).std() + 1e-8
        mean_abs = ret.abs().rolling(window).mean() + 1e-8
        raw = np.log(std / mean_abs)
        raw_rolling = raw.rolling(window).rank(pct=True) * 2 - 1
        return -raw_rolling

    @staticmethod
    def alpha_quanta_full_base_264_tanh(df, window=10):
        ret = df['close'].pct_change()
        std = ret.rolling(window).std() + 1e-8
        mean_abs = ret.abs().rolling(window).mean() + 1e-8
        raw = np.log(std / mean_abs)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_264_zscore(df, window=30, sub_window=40):
        ret = df['close'].pct_change()
        std = ret.rolling(window).std() + 1e-8
        mean_abs = ret.abs().rolling(window).mean() + 1e-8
        raw = np.log(std / mean_abs)
        raw_mean = raw.rolling(sub_window).mean()
        raw_std = raw.rolling(sub_window).std()
        normalized = ((raw - raw_mean) / raw_std).clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_264_sign(df, window=10):
        ret = df['close'].pct_change()
        std = ret.rolling(window).std() + 1e-8
        mean_abs = ret.abs().rolling(window).mean() + 1e-8
        raw = np.log(std / mean_abs)
        normalized = np.sign(raw)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_264_wf(df, window=20, sub_window=40):
        ret = df['close'].pct_change()
        std = ret.rolling(window).std() + 1e-8
        mean_abs = ret.abs().rolling(window).mean() + 1e-8
        raw = np.log(std / mean_abs)
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized

    @staticmethod
    def alpha_quanta_full_base_265_k(df, window=100):
        hilo = df['high'] - df['low']
        corr = hilo.rolling(window).corr(df['matchingVolume'])
        # TS_ZSCORE: z-score of corr over window
        zscore = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        # RANK on the resulting zscore series -> rolling percentile
        raw = zscore.rolling(window).rank(pct=True)
        signal = (raw * 2) - 1
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_265_h(df, window=55):
        hilo = df['high'] - df['low']
        corr = hilo.rolling(window).corr(df['matchingVolume'])
        zscore = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        raw = zscore.rolling(window).rank(pct=True)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_265_p(df, window=35):
        hilo = df['high'] - df['low']
        corr = hilo.rolling(window).corr(df['matchingVolume'])
        zscore = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        raw = zscore.rolling(window).rank(pct=True)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_265_y(df, window=45):
        hilo = df['high'] - df['low']
        corr = hilo.rolling(window).corr(df['matchingVolume'])
        zscore = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        raw = zscore.rolling(window).rank(pct=True)
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_265_r(df, window=50, p1=0.7):
        hilo = df['high'] - df['low']
        corr = hilo.rolling(window).corr(df['matchingVolume'])
        zscore = (corr - corr.rolling(window).mean()) / corr.rolling(window).std()
        raw = zscore.rolling(window).rank(pct=True)
        # Winsorize
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher transform
        numerator = (winsorized - low) / (high - low + 1e-9)
        signal = np.arctanh(numerator * 1.98 - 0.99)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_full_base_266_rank(df, window=65):
        delta_close = df['close'].diff(1)
        ts_mean = delta_close.rolling(window).mean()
        sign_ts_mean = np.sign(ts_mean)
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = delta_close.rolling(window).corr(volume)
        raw = sign_ts_mean * ts_corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_266_tanh(df, window=20):
        delta_close = df['close'].diff(1)
        ts_mean = delta_close.rolling(window).mean()
        sign_ts_mean = np.sign(ts_mean)
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = delta_close.rolling(window).corr(volume)
        raw = sign_ts_mean * ts_corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_266_zscore(df, window=90):
        delta_close = df['close'].diff(1)
        ts_mean = delta_close.rolling(window).mean()
        sign_ts_mean = np.sign(ts_mean)
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = delta_close.rolling(window).corr(volume)
        raw = sign_ts_mean * ts_corr
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_266_sign(df, window=20):
        delta_close = df['close'].diff(1)
        ts_mean = delta_close.rolling(window).mean()
        sign_ts_mean = np.sign(ts_mean)
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = delta_close.rolling(window).corr(volume)
        raw = sign_ts_mean * ts_corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_266_wf(df, window=90, p1=0.1, p2=60):
        delta_close = df['close'].diff(1)
        ts_mean = delta_close.rolling(window).mean()
        sign_ts_mean = np.sign(ts_mean)
        volume = df.get('matchingVolume', df.get('volume', 1))
        ts_corr = delta_close.rolling(window).corr(volume)
        raw = sign_ts_mean * ts_corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_267_rank(df, window=25):
        ret = df['close'].pct_change()
        delta_close = df['close'].diff()
        corr = delta_close.rolling(window).corr(df['matchingVolume'])
        hl_range = (df['high'] - df['low']) / df['close']
        hl_std = hl_range.rolling(window).std()
        raw = corr / (hl_std + 1e-8) * ret.rolling(5).mean()
        raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_267_tanh(df, window=85):
        ret = df['close'].pct_change()
        delta_close = df['close'].diff()
        corr = delta_close.rolling(window).corr(df['matchingVolume'])
        hl_range = (df['high'] - df['low']) / df['close']
        hl_std = hl_range.rolling(window).std()
        raw = corr / (hl_std + 1e-8) * ret.rolling(5).mean()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_267_zscore(df, window=85):
        ret = df['close'].pct_change()
        delta_close = df['close'].diff()
        corr = delta_close.rolling(window).corr(df['matchingVolume'])
        hl_range = (df['high'] - df['low']) / df['close']
        hl_std = hl_range.rolling(window).std()
        raw = corr / (hl_std + 1e-8) * ret.rolling(5).mean()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_267_sign(df, window=25):
        ret = df['close'].pct_change()
        delta_close = df['close'].diff()
        corr = delta_close.rolling(window).corr(df['matchingVolume'])
        hl_range = (df['high'] - df['low']) / df['close']
        hl_std = hl_range.rolling(window).std()
        raw = corr / (hl_std + 1e-8) * ret.rolling(5).mean()
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_full_base_267_wf(df, window=30, p1=0.7):
        ret = df['close'].pct_change()
        delta_close = df['close'].diff()
        corr = delta_close.rolling(window).corr(df['matchingVolume'])
        hl_range = (df['high'] - df['low']) / df['close']
        hl_std = hl_range.rolling(window).std()
        raw = corr / (hl_std + 1e-8) * ret.rolling(5).mean()
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_full_base_268_rank(df, window=55):
        raw_data = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff() / df['matchingVolume']
        return_zscore = (raw_data - raw_data.rolling(window).mean()) / raw_data.rolling(window).std()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        temp = raw_data.rolling(10).mean()
        sign_component = np.sign(temp)
        raw = return_zscore - volume_zscore * sign_component
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_268_tanh(df, window=95):
        raw_data = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff() / df['matchingVolume']
        return_zscore = (raw_data - raw_data.rolling(window).mean()) / raw_data.rolling(window).std()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        temp = raw_data.rolling(10).mean()
        sign_component = np.sign(temp)
        raw = return_zscore - volume_zscore * sign_component
        signal = np.tanh(raw / raw.rolling(window).std())
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_268_zscore(df, window=50):
        raw_data = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff() / df['matchingVolume']
        return_zscore = (raw_data - raw_data.rolling(window).mean()) / raw_data.rolling(window).std()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        temp = raw_data.rolling(10).mean()
        sign_component = np.sign(temp)
        raw = return_zscore - volume_zscore * sign_component
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_268_sign(df, window=85):
        raw_data = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff() / df['matchingVolume']
        return_zscore = (raw_data - raw_data.rolling(window).mean()) / raw_data.rolling(window).std()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        temp = raw_data.rolling(10).mean()
        sign_component = np.sign(temp)
        raw = return_zscore - volume_zscore * sign_component
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_268_wf(df, window=80, p1=0.1):
        raw_data = df['close'].pct_change()
        volume_delta = df['matchingVolume'].diff() / df['matchingVolume']
        return_zscore = (raw_data - raw_data.rolling(window).mean()) / raw_data.rolling(window).std()
        volume_zscore = (volume_delta - volume_delta.rolling(window).mean()) / volume_delta.rolling(window).std()
        temp = raw_data.rolling(10).mean()
        sign_component = np.sign(temp)
        raw = return_zscore - volume_zscore * sign_component
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_full_base_269_rank(df, window_rank=100):
        raw = df['close'] - df['low'].rolling(10).min()
        corr = raw.rolling(window_rank).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0)))
        vol_range = df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_rank).max() - df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_rank).min() + 1e-8
        ratio = corr / vol_range
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(15).mean()
        value = ratio * mean_ret
        signal = value.rolling(window_rank).rank(pct=True) * 2 - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_269_tanh(df, window_corr=65):
        raw = df['close'] - df['low'].rolling(10).min()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0)))
        vol_range = df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_corr).max() - df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_corr).min() + 1e-8
        ratio = corr / vol_range
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(15).mean()
        value = ratio * mean_ret
        signal = np.tanh(value / value.rolling(window_corr).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_269_zscore(df, window_z=80):
        raw = df['close'] - df['low'].rolling(10).min()
        corr = raw.rolling(window_z).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0)))
        vol_range = df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_z).max() - df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_z).min() + 1e-8
        ratio = corr / vol_range
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(15).mean()
        value = ratio * mean_ret
        mean_val = value.rolling(window_z).mean()
        std_val = value.rolling(window_z).std().replace(0, np.nan)
        signal = ((value - mean_val) / std_val).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_269_sign(df, window_sign=90):
        raw = df['close'] - df['low'].rolling(10).min()
        corr = raw.rolling(window_sign).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0)))
        vol_range = df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_sign).max() - df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(window_sign).min() + 1e-8
        ratio = corr / vol_range
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(15).mean()
        value = ratio * mean_ret
        signal = np.sign(value)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_269_wf(df, p1=0.9, p2=100):
        raw = df['close'] - df['low'].rolling(10).min()
        corr = raw.rolling(p2).corr(df.get('matchingVolume', df.get('volume', df['close'] * 0)))
        vol_range = df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(p2).max() - df.get('matchingVolume', df.get('volume', df['close'] * 0)).rolling(p2).min() + 1e-8
        ratio = corr / vol_range
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(15).mean()
        value = ratio * mean_ret
        low = value.rolling(p2).quantile(p1)
        high = value.rolling(p2).quantile(1 - p1)
        winsorized = value.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_270_rank(df, window=45):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff()
        corr = high_low.rolling(window).corr(volume_delta)
        sign = np.sign(corr)
        ret = df['close'].pct_change()
        std_short = ret.rolling(window).std()
        std_long = ret.rolling(window * 2).std()
        raw = sign * (std_short / (std_long + 1e-8))
        normalized = (raw.rolling(window * 2).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_270_tanh(df, window=10):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff()
        corr = high_low.rolling(window).corr(volume_delta)
        sign = np.sign(corr)
        ret = df['close'].pct_change()
        std_short = ret.rolling(window).std()
        std_long = ret.rolling(window * 2).std()
        raw = sign * (std_short / (std_long + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window * 2).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_270_zscore(df, window=45):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff()
        corr = high_low.rolling(window).corr(volume_delta)
        sign = np.sign(corr)
        ret = df['close'].pct_change()
        std_short = ret.rolling(window).std()
        std_long = ret.rolling(window * 2).std()
        raw = sign * (std_short / (std_long + 1e-8))
        normalized = ((raw - raw.rolling(window * 2).mean()) / raw.rolling(window * 2).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_270_sign(df, window=5):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff()
        corr = high_low.rolling(window).corr(volume_delta)
        sign = np.sign(corr)
        ret = df['close'].pct_change()
        std_short = ret.rolling(window).std()
        std_long = ret.rolling(window * 2).std()
        raw = sign * (std_short / (std_long + 1e-8))
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_270_wf(df, window=80, p1=0.9):
        high_low = df['high'] - df['low']
        volume_delta = df['matchingVolume'].diff()
        corr = high_low.rolling(window).corr(volume_delta)
        sign = np.sign(corr)
        ret = df['close'].pct_change()
        std_short = ret.rolling(window).std()
        std_long = ret.rolling(window * 2).std()
        raw = sign * (std_short / (std_long + 1e-8))
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_271_rank(df, window=40):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        mean_ret = ret.rolling(5).mean()
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        corr = delta_close.rolling(15).corr(delta_volume)
        factor = 1 - corr.abs()
        raw = mean_ret * factor
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_271_tanh(df, window=55):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        mean_ret = ret.rolling(5).mean()
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        corr = delta_close.rolling(15).corr(delta_volume)
        factor = 1 - corr.abs()
        raw = mean_ret * factor
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_271_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        mean_ret = ret.rolling(5).mean()
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        corr = delta_close.rolling(15).corr(delta_volume)
        factor = 1 - corr.abs()
        raw = mean_ret * factor
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_271_sign(df, window=35):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        mean_ret = ret.rolling(5).mean()
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        corr = delta_close.rolling(15).corr(delta_volume)
        factor = 1 - corr.abs()
        raw = mean_ret * factor
        normalized = np.sign(raw) * raw.rolling(window).rank(pct=True).replace(0, np.nan).fillna(0)
        normalized = (normalized * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_271_wf(df, window=40, p1=0.1):
        p2 = window
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = close.pct_change()
        mean_ret = ret.rolling(5).mean()
        delta_close = close.diff(1)
        delta_volume = volume.diff(1)
        corr = delta_close.rolling(15).corr(delta_volume)
        factor = 1 - corr.abs()
        raw = mean_ret * factor
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_full_base_272_rank(df, window=10, corr_window=40):
        # Tính rolling low và high
        rolling_low = df['low'].rolling(window).min()
        rolling_high = df['high'].rolling(window).max()
        # Tính raw signal
        raw = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        # Tính return và delayed return
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        # Tính correlation giữa ret và delay_ret
        corr = ret.rolling(corr_window).corr(delay_ret).replace(np.nan, 0)
        # Tính raw signal cuối cùng
        raw_signal = raw * (1 - corr)
        # Chuẩn hóa Rolling Rank (A)
        param = max(window, corr_window)
        signal = (raw_signal.rolling(param).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_272_tanh(df, window=10, corr_window=40):
        # Tính rolling low và high
        rolling_low = df['low'].rolling(window).min()
        rolling_high = df['high'].rolling(window).max()
        # Tính raw signal
        raw = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        # Tính return và delayed return
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        # Tính correlation
        corr = ret.rolling(corr_window).corr(delay_ret).replace(np.nan, 0)
        raw_signal = raw * (1 - corr)
        # Chuẩn hóa Dynamic Tanh (B)
        param = max(window, corr_window)
        std = raw_signal.rolling(param).std().replace(0, np.nan).ffill().fillna(1e-8)
        signal = np.tanh(raw_signal / std)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_272_zscore(df, window=10, corr_window=1):
        # Tính rolling low và high
        rolling_low = df['low'].rolling(window).min()
        rolling_high = df['high'].rolling(window).max()
        # Tính raw signal
        raw = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        # Tính return và delayed return
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        # Tính correlation
        corr = ret.rolling(corr_window).corr(delay_ret).replace(np.nan, 0)
        raw_signal = raw * (1 - corr)
        # Chuẩn hóa Rolling Z-Score (C)
        param = max(window, corr_window)
        mean = raw_signal.rolling(param).mean()
        std = raw_signal.rolling(param).std().replace(0, np.nan).ffill().fillna(1e-8)
        signal = ((raw_signal - mean) / std).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_272_sign(df, window=70, corr_window=1):
        # Tính rolling low và high
        rolling_low = df['low'].rolling(window).min()
        rolling_high = df['high'].rolling(window).max()
        # Tính raw signal
        raw = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        # Tính return và delayed return
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        # Tính correlation
        corr = ret.rolling(corr_window).corr(delay_ret).replace(np.nan, 0)
        raw_signal = raw * (1 - corr)
        # Chuẩn hóa Sign/Binary Soft (D)
        signal = np.sign(raw_signal)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_full_base_272_wf(df, window=10, corr_window=1, p1=0.05, p2=40):
        # Tính rolling low và high
        rolling_low = df['low'].rolling(window).min()
        rolling_high = df['high'].rolling(window).max()
        # Tính raw signal
        raw = (df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-8)
        # Tính return và delayed return
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        # Tính correlation
        corr = ret.rolling(corr_window).corr(delay_ret).replace(np.nan, 0)
        raw_signal = raw * (1 - corr)
        # Hardcode p1 và p2 vào để giới hạn 2 tham số chính ở ngoài
        # Chuẩn hóa Winsorized Fisher (E)
        low_quant = raw_signal.rolling(p2).quantile(p1)
        high_quant = raw_signal.rolling(p2).quantile(1 - p1)
        winsorized = raw_signal.clip(lower=low_quant, upper=high_quant)
        # Tính Fisher Transform
        adj = ((winsorized - low_quant) / (high_quant - low_quant + 1e-9)) * 1.98 - 0.99
        adj = adj.clip(-0.99, 0.99)
        signal = np.arctanh(adj)
        signal = signal.fillna(0)
        return signal