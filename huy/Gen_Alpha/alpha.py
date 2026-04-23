import pandas as pd
import numpy as np

class Alpha:
    pass


    @staticmethod
    def alpha_quanta_001_rank(df, window=20, percentile=0.1):
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(percentile)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Rolling Rank (Case A) - removes noise, uniform distribution
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_tanh(df, window=20, percentile=0.1):
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(percentile)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Dynamic Tanh (Case B) - preserves magnitude
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-9))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_zscore(df, window=100, percentile=0.1):
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(percentile)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Rolling Z-Score/Clip (Case C) - for spread/oscillator
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-9)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_sign(df, window=60, percentile=0.1):
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(percentile)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Sign/Binary Soft (Case D) - pure breakout/trend
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_001_wf(df, window=10, percentile=0.1):
        # Raw calculation
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        volume_series = df.get('matchingVolume', df.get('volume', 1))
        volume_percentile = volume_series.rolling(window).quantile(percentile)
        sign_term = np.sign(volume_series - volume_percentile)
        raw = price_term * sign_term
        # Normalization: Winsorized Fisher (Case E) - heavy tails, preserve distribution
        p1 = 0.05  # Hardcoded winsorization percentile
        p2 = 60    # Hardcoded rolling window for quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(low, high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_002_rank(df, window=30, factor=5):
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range = (high - low) / (open_price.replace(0, np.nan))
        volume_mean = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        volume_zscore = (volume - volume_mean) / volume_std
        raw = price_range * volume_zscore * factor
        raw = raw.ffill().fillna(0)
        normalized = (raw.rolling(window=window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_002_tanh(df, window=100, factor=40):
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range = (high - low) / (open_price.replace(0, np.nan))
        volume_mean = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        volume_zscore = (volume - volume_mean) / volume_std
        raw = price_range * volume_zscore * factor
        raw = raw.ffill().fillna(0)
        rolling_std = raw.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_002_zscore(df, window=90, factor=5):
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range = (high - low) / (open_price.replace(0, np.nan))
        volume_mean = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        volume_zscore = (volume - volume_mean) / volume_std
        raw = price_range * volume_zscore * factor
        raw = raw.ffill().fillna(0)
        rolling_mean = raw.rolling(window=window, min_periods=1).mean()
        rolling_std = raw.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_002_sign(df, window=80, factor=1):
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range = (high - low) / (open_price.replace(0, np.nan))
        volume_mean = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        volume_zscore = (volume - volume_mean) / volume_std
        raw = price_range * volume_zscore * factor
        raw = raw.ffill().fillna(0)
        normalized = np.sign(raw)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_002_wf(df, window=60, factor=40):
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        price_range = (high - low) / (open_price.replace(0, np.nan))
        volume_mean = volume.rolling(window=window, min_periods=1).mean()
        volume_std = volume.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        volume_zscore = (volume - volume_mean) / volume_std
        raw = price_range * volume_zscore * factor
        raw = raw.ffill().fillna(0)
        p1 = 0.05
        p2 = window * 2
        low_bound = raw.rolling(window=p2, min_periods=1).quantile(p1)
        high_bound = raw.rolling(window=p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_003_rank(df, window=15):
        # Logic gốc: SIGN(($open - DELAY($close, 1))/DELAY($close, 1)) * TS_ZSCORE($volume, 10)
        # Trường hợp A: Rolling Rank cho TS_ZSCORE, giữ nguyên SIGN.
        # Xử lý volume: TS_ZSCORE có thể có outliers, dùng rank để chuẩn hóa về phân phối đồng đều.
        # Tính phần SIGN: (open - close_shift) / close_shift
        close_shift = df['close'].shift(1)
        sign_raw = (df['open'] - close_shift) / (close_shift + 1e-9)
        sign_part = np.sign(sign_raw)
        # Tính TS_ZSCORE(volume, window): (volume - rolling_mean) / rolling_std
        vol_mean = df['matchingVolume'].rolling(window).mean()
        vol_std = df['matchingVolume'].rolling(window).std()
        vol_zscore = (df['matchingVolume'] - vol_mean) / (vol_std + 1e-9)
        # Chuẩn hóa vol_zscore bằng rolling rank (pct) về [-1,1]
        vol_rank = (vol_zscore.rolling(window).rank(pct=True) * 2) - 1
        # Kết hợp: sign_part * vol_rank
        raw = sign_part * vol_rank
        # Fill NaN do shift và rolling
        raw = raw.ffill().fillna(0)
        # Đảm bảo trong [-1,1]
        return raw.clip(-1, 1)

    @staticmethod
    def alpha_quanta_003_tanh(df, window=75):
        # Trường hợp B: Dynamic Tanh cho TS_ZSCORE, giữ nguyên SIGN.
        # Mục tiêu: giữ lại cường độ của TS_ZSCORE.
        close_shift = df['close'].shift(1)
        sign_raw = (df['open'] - close_shift) / (close_shift + 1e-9)
        sign_part = np.sign(sign_raw)
        # Tính TS_ZSCORE(volume, window)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        vol_std = df['matchingVolume'].rolling(window).std()
        vol_zscore = (df['matchingVolume'] - vol_mean) / (vol_std + 1e-9)
        # Chuẩn hóa bằng tanh: np.tanh(raw / rolling_std) nhưng ở đây raw đã là z-score, nên dùng tanh trực tiếp trên z-score để giữ cường độ.
        # Công thức B: np.tanh(raw / raw.rolling(param).std()). Ở đây raw là vol_zscore, param là window.
        # Tuy nhiên, vol_zscore đã được chuẩn hóa theo std, nên có thể dùng tanh(vol_zscore / scale) với scale là độ lệch chuẩn của vol_zscore.
        scale = vol_zscore.rolling(window).std()
        vol_norm = np.tanh(vol_zscore / (scale + 1e-9))
        raw = sign_part * vol_norm
        raw = raw.ffill().fillna(0)
        return raw.clip(-1, 1)

    @staticmethod
    def alpha_quanta_003_zscore(df, window=100):
        # Trường hợp C: Rolling Z-Score/Clip cho toàn bộ biểu thức.
        # Tính raw theo công thức gốc: sign * TS_ZSCORE(volume, window)
        close_shift = df['close'].shift(1)
        sign_raw = (df['open'] - close_shift) / (close_shift + 1e-9)
        sign_part = np.sign(sign_raw)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        vol_std = df['matchingVolume'].rolling(window).std()
        vol_zscore = (df['matchingVolume'] - vol_mean) / (vol_std + 1e-9)
        raw = sign_part * vol_zscore
        # Chuẩn hóa raw bằng rolling z-score và clip: ((raw - mean)/std).clip(-1,1)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        normalized = ((raw - raw_mean) / (raw_std + 1e-9)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_003_sign(df, window=10):
        # Trường hợp D: Sign/Binary Soft - chỉ lấy dấu của toàn bộ biểu thức.
        # Tính raw như gốc.
        close_shift = df['close'].shift(1)
        sign_raw = (df['open'] - close_shift) / (close_shift + 1e-9)
        sign_part = np.sign(sign_raw)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        vol_std = df['matchingVolume'].rolling(window).std()
        vol_zscore = (df['matchingVolume'] - vol_mean) / (vol_std + 1e-9)
        raw = sign_part * vol_zscore
        # Lấy dấu: np.sign(raw)
        signal = np.sign(raw)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_003_wf(df, window=100):
        # Trường hợp E: Winsorized Fisher với p1=0.1, p2=window (hardcode).
        # Tính raw như gốc.
        close_shift = df['close'].shift(1)
        sign_raw = (df['open'] - close_shift) / (close_shift + 1e-9)
        sign_part = np.sign(sign_raw)
        vol_mean = df['matchingVolume'].rolling(window).mean()
        vol_std = df['matchingVolume'].rolling(window).std()
        vol_zscore = (df['matchingVolume'] - vol_mean) / (vol_std + 1e-9)
        raw = sign_part * vol_zscore
        # Hardcode p1=0.1, p2=window
        p1 = 0.1
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_004_rank(df, window=70, factor=3):
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa A: Rolling Rank để loại bỏ nhiễu, phân phối đồng nhất.
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa A: Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_004_tanh(df, window=70, factor=1):
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa B: Dynamic Tanh để giữ lại cường độ (magnitude).
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa B: Dynamic Tanh
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0).clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_004_zscore(df, window=70, factor=1):
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa C: Rolling Z-Score/Clip cho các công thức Spread/Oscillator.
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa C: Rolling Z-Score với clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_quanta_004_sign(df, window=40, factor=5):
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa D: Sign/Binary Soft cho Breakout/Trend Following.
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa D: Sign
        normalized = np.sign(raw)
        return normalized.fillna(0) * factor

    @staticmethod
    def alpha_quanta_004_wf(df, window=30, factor=1):
        # Logic gốc: (volume / max_volume_25) * ((high - open)/open - (open - low)/open)
        # Chuẩn hóa E: Winsorized Fisher cho dữ liệu có đuôi nặng.
        # Xử lý volume: Công thức gốc dùng volume tuyệt đối, áp dụng log1p để giảm skew.
        volume = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        max_vol = volume.rolling(window).max()
        volume_ratio = volume / (max_vol + 1e-8)
        price_term = (df['high'] - df['open']) / df['open'] - (df['open'] - df['low']) / df['open']
        raw = volume_ratio * price_term
        # Chuẩn hóa E: Winsorized Fisher
        # Hardcode tham số phụ: quantile threshold p1=0.1, rolling window p2=window*2
        p1 = 0.1
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_005_rank(df, window=10, factor=0.1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        percentile = volume.rolling(window).quantile(factor)
        condition = volume > percentile
        raw = pd.Series(np.where(condition, np.sign(high - open_), np.sign(open_ - low)), index=df.index)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_005_tanh(df, window=40, factor=0.9):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        percentile = volume.rolling(window).quantile(factor)
        condition = volume > percentile
        raw = pd.Series(np.where(condition, np.sign(high - open_), np.sign(open_ - low)), index=df.index)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_005_zscore(df, window=30, factor=0.1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        percentile = volume.rolling(window).quantile(factor)
        condition = volume > percentile
        raw = pd.Series(np.where(condition, np.sign(high - open_), np.sign(open_ - low)), index=df.index)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_005_sign(df, window=100, factor=0.9):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        percentile = volume.rolling(window).quantile(factor)
        condition = volume > percentile
        raw = pd.Series(np.where(condition, np.sign(high - open_), np.sign(open_ - low)), index=df.index)
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_005_wf(df, window=100, factor=0.7):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        percentile = volume.rolling(window).quantile(factor)
        condition = volume > percentile
        raw = pd.Series(np.where(condition, np.sign(high - open_), np.sign(open_ - low)), index=df.index)
        p1 = 0.05
        p2 = 60
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_006_rank(df, window=95):
        # Raw calculation
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))) - np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).mean()) / \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).std() + 1e-8)
        raw = pd.Series(raw, index=df.index)
        # Rolling Rank normalization (Case A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_006_tanh(df, window=70):
        # Raw calculation
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))) - np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).mean()) / \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).std() + 1e-8)
        raw = pd.Series(raw, index=df.index)
        # Dynamic Tanh normalization (Case B)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_006_zscore(df, window=50):
        # Raw calculation
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))) - np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).mean()) / \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).std() + 1e-8)
        raw = pd.Series(raw, index=df.index)
        # Rolling Z-Score/Clip normalization (Case C)
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        normalized = zscore.clip(-1, 1)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_006_sign(df):
        # Raw calculation
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))) - np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).mean()) / \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).std() + 1e-8)
        raw = pd.Series(raw, index=df.index)
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_006_wf(df, window=80):
        # Raw calculation
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))) - np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).mean()) / \
              (np.log1p(df.get('matchingVolume', df.get('volume', 1))).rolling(5).std() + 1e-8)
        raw = pd.Series(raw, index=df.index)
        # Winsorized Fisher normalization (Case E) with hardcoded p1=0.05, p2=window
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_007_rank(df, window=95):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Part 1: (open - delay(close,1)) / (delay(close,1) + 1e-8)
        delayed_close = close_price.shift(1)
        part1_raw = (open_price - delayed_close) / (delayed_close + 1e-8)

        # Part 2: log(volume + 1)
        part2_raw = np.log1p(volume)

        # Combine raw signals
        raw_signal = part1_raw * part2_raw

        # Rolling rank normalization (Case A)
        normalized = (raw_signal.rolling(window).rank(pct=True) * 2) - 1

        # Fill NaN
        normalized = normalized.fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_007_tanh(df, window=100):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Part 1: (open - delay(close,1)) / (delay(close,1) + 1e-8)
        delayed_close = close_price.shift(1)
        part1_raw = (open_price - delayed_close) / (delayed_close + 1e-8)

        # Part 2: log(volume + 1)
        part2_raw = np.log1p(volume)

        # Combine raw signals
        raw_signal = part1_raw * part2_raw

        # Dynamic tanh normalization (Case B)
        rolling_std = raw_signal.rolling(window).std()
        normalized = np.tanh(raw_signal / rolling_std.replace(0, np.nan))

        # Fill NaN
        normalized = normalized.fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_007_zscore(df, window=95):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Part 1: (open - delay(close,1)) / (delay(close,1) + 1e-8)
        delayed_close = close_price.shift(1)
        part1_raw = (open_price - delayed_close) / (delayed_close + 1e-8)

        # Part 2: log(volume + 1)
        part2_raw = np.log1p(volume)

        # Combine raw signals
        raw_signal = part1_raw * part2_raw

        # Rolling z-score normalization (Case C)
        rolling_mean = raw_signal.rolling(window).mean()
        rolling_std = raw_signal.rolling(window).std()
        normalized = ((raw_signal - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-1, 1)

        # Fill NaN
        normalized = normalized.fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_007_sign(df):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Part 1: (open - delay(close,1)) / (delay(close,1) + 1e-8)
        delayed_close = close_price.shift(1)
        part1_raw = (open_price - delayed_close) / (delayed_close + 1e-8)

        # Part 2: log(volume + 1)
        part2_raw = np.log1p(volume)

        # Combine raw signals
        raw_signal = part1_raw * part2_raw

        # Sign/binary normalization (Case D)
        normalized = np.sign(raw_signal)

        # Fill NaN
        normalized = normalized.fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_007_wf(df, window=90, factor=0.1):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Part 1: (open - delay(close,1)) / (delay(close,1) + 1e-8)
        delayed_close = close_price.shift(1)
        part1_raw = (open_price - delayed_close) / (delayed_close + 1e-8)

        # Part 2: log(volume + 1)
        part2_raw = np.log1p(volume)

        # Combine raw signals
        raw_signal = part1_raw * part2_raw

        # Winsorized Fisher normalization (Case E)
        # Hardcode p1 = factor (quantile threshold), p2 = window (rolling window)
        p1 = factor
        p2 = window

        low = raw_signal.rolling(p2).quantile(p1)
        high = raw_signal.rolling(p2).quantile(1 - p1)
        winsorized = raw_signal.clip(lower=low, upper=high, axis=0)

        # Fisher Transform approximation
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)

        # Fill NaN
        normalized = normalized.fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_008_rank(df, window=60):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Time-series mean and std of open
        open_mean = open_price.rolling(window).mean()
        open_std = open_price.rolling(window).std()

        # First component: normalized open
        comp1 = (open_price - open_mean) / (open_std + 1e-8)

        # Second component: correlation between open return and log volume
        open_return = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        log_volume = np.log1p(volume)
        comp2 = open_return.rolling(window).corr(log_volume)

        # Raw alpha
        raw = comp1 * comp2

        # Rolling rank normalization (Case A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_008_tanh(df, window=100):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Time-series mean and std of open
        open_mean = open_price.rolling(window).mean()
        open_std = open_price.rolling(window).std()

        # First component: normalized open
        comp1 = (open_price - open_mean) / (open_std + 1e-8)

        # Second component: correlation between open return -and log volume
        open_return = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        log_volume = np.log1p(volume)
        comp2 = open_return.rolling(window).corr(log_volume)

        # Raw alpha
        raw = comp1 * comp2

        # Dynamic tanh normalization (Case B)
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_008_zscore(df, window=100):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Time-series mean and std of open
        open_mean = open_price.rolling(window).mean()
        open_std = open_price.rolling(window).std()

        # First component: normalized open
        comp1 = (open_price - open_mean) / (open_std + 1e-8)

        # Second component: correlation between open return -and log volume
        open_return = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        log_volume = np.log1p(volume)
        comp2 = open_return.rolling(window).corr(log_volume)

        # Raw alpha
        raw = comp1 * comp2

        # Rolling z-score normalization (Case C)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        normalized = ((raw - raw_mean) / (raw_std + 1e-8)).clip(-1, 1)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_008_sign(df, window=80):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Time-series mean and std of open
        open_mean = open_price.rolling(window).mean()
        open_std = open_price.rolling(window).std()

        # First component: normalized open
        comp1 = (open_price - open_mean) / (open_std + 1e-8)

        # Second component: correlation between open return -and log volume
        open_return = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        log_volume = np.log1p(volume)
        comp2 = open_return.rolling(window).corr(log_volume)

        # Raw alpha
        raw = comp1 * comp2

        # Sign normalization (Case D)
        normalized = np.sign(raw)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_008_wf(df, window=100):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Time-series mean and std of open
        open_mean = open_price.rolling(window).mean()
        open_std = open_price.rolling(window).std()

        # First component: normalized open
        comp1 = (open_price - open_mean) / (open_std + 1e-8)

        # Second component: correlation between open return -and log volume
        open_return = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        log_volume = np.log1p(volume)
        comp2 = open_return.rolling(window).corr(log_volume)

        # Raw alpha
        raw = comp1 * comp2

        # Winsorized Fisher normalization (Case E)
        # Hardcoded parameters: quantile=0.05, winsor_window=window*2
        quantile = 0.05
        winsor_window = window * 2

        low = raw.rolling(winsor_window).quantile(quantile)
        high = raw.rolling(winsor_window).quantile(1 - quantile)

        winsorized = raw.clip(lower=low, upper=high, axis=0)

        # Fisher transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_009_rank(df, window=50):
        # Logic gốc: Chuẩn hóa sự thay đổi của open so với close trước đó, trừ đi trung bình động và chia cho độ lệch chuẩn động.
        # Phương pháp A (Rolling Rank): Loại bỏ nhiễu, đưa về phân phối đồng nhất.
        raw = (df['open'].diff(1) / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        z = ((raw - raw_mean) / (raw_std + 1e-8)).ffill()
        signal = (z.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_009_tanh(df, window=45):
        # Logic gốc: Chuẩn hóa sự thay đổi của open so với close trước đó, trừ đi trung bình động và chia cho độ lệch chuẩn động.
        # Phương pháp B (Dynamic Tanh): Giữ lại cường độ (magnitude) của tín hiệu.
        raw = (df['open'].diff(1) / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        z = ((raw - raw_mean) / (raw_std + 1e-8)).ffill()
        signal = np.tanh(z / (z.rolling(window).std() + 1e-8))
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_009_zscore(df, window=50):
        # Logic gốc: Chuẩn hóa sự thay đổi của open so với close trước đó, trừ đi trung bình động và chia cho độ lệch chuẩn động.
        # Phương pháp C (Rolling Z-Score/Clip): Phù hợp cho các công thức tính Spread, Basis hoặc Oscillator.
        raw = (df['open'].diff(1) / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        z = ((raw - raw_mean) / (raw_std + 1e-8)).ffill()
        signal = z.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_009_sign(df, window=90):
        # Logic gốc: Chuẩn hóa sự thay đổi của open so với close trước đó, trừ đi trung bình động và chia cho độ lệch chuẩn động.
        # Phương pháp D (Sign/Binary Soft): Dùng cho Breakout hoặc Trend Following thuần túy.
        raw = (df['open'].diff(1) / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        z = ((raw - raw_mean) / (raw_std + 1e-8)).ffill()
        signal = np.sign(z)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_009_wf(df, window=60):
        # Logic gốc: Chuẩn hóa sự thay đổi của open so với close trước đó, trừ đi trung bình động và chia cho độ lệch chuẩn động.
        # Phương pháp E (Winsorized Fisher): Xử lý đuôi nặng, giữ lại cấu trúc phân phối.
        # Hardcode tham số phụ: quantile_thresh=0.05, winsor_window=20
        quantile_thresh = 0.05
        winsor_window = 20
        raw = (df['open'].diff(1) / df['close'].shift(1)).replace([np.inf, -np.inf], np.nan)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        z = ((raw - raw_mean) / (raw_std + 1e-8)).ffill()
        low = z.rolling(winsor_window).quantile(quantile_thresh)
        high = z.rolling(winsor_window).quantile(1 - quantile_thresh)
        winsorized = z.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_010_rank(df, window=95):
        # Raw calculation: standardized log volume * normalized high-low range
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        vol_z = (log_vol - log_vol.rolling(window).mean()) / (log_vol.rolling(window).std() + 1e-8)
        hl_range = df['high'] - df['low']
        hl_norm = hl_range / (hl_range.rolling(window).mean() + 1e-8)
        raw = vol_z * hl_norm
        # Rolling Rank normalization (Case A)
        param = window
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).astype(float).clip(-1, 1)

    @staticmethod
    def alpha_quanta_010_tanh(df, window=100):
        # Raw calculation: standardized log volume * normalized high-low range
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        vol_z = (log_vol - log_vol.rolling(window).mean()) / (log_vol.rolling(window).std() + 1e-8)
        hl_range = df['high'] - df['low']
        hl_norm = hl_range / (hl_range.rolling(window).mean() + 1e-8)
        raw = vol_z * hl_norm
        # Dynamic Tanh normalization (Case B)
        param = window
        normalized = np.tanh(raw / (raw.rolling(param).std() + 1e-8))
        return -normalized.fillna(0).astype(float).clip(-1, 1)

    @staticmethod
    def alpha_quanta_010_zscore(df, window=30):
        # Raw calculation: standardized log volume * normalized high-low range
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        vol_z = (log_vol - log_vol.rolling(window).mean()) / (log_vol.rolling(window).std() + 1e-8)
        hl_range = df['high'] - df['low']
        hl_norm = hl_range / (hl_range.rolling(window).mean() + 1e-8)
        raw = vol_z * hl_norm
        # Rolling Z-Score/Clip normalization (Case C)
        param = window
        normalized = ((raw - raw.rolling(param).mean()) / (raw.rolling(param).std() + 1e-8)).clip(-1, 1)
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_010_sign(df, window=30):
        # Raw calculation: standardized log volume * normalized high-low range
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        vol_z = (log_vol - log_vol.rolling(window).mean()) / (log_vol.rolling(window).std() + 1e-8)
        hl_range = df['high'] - df['low']
        hl_norm = hl_range / (hl_range.rolling(window).mean() + 1e-8)
        raw = vol_z * hl_norm
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_010_wf(df, window=80):
        # Raw calculation: standardized log volume * normalized high-low range
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        vol_z = (log_vol - log_vol.rolling(window).mean()) / (log_vol.rolling(window).std() + 1e-8)
        hl_range = df['high'] - df['low']
        hl_norm = hl_range / (hl_range.rolling(window).mean() + 1e-8)
        raw = vol_z * hl_norm
        # Winsorized Fisher normalization (Case E)
        p1 = 0.05  # Hardcoded quantile threshold
        p2 = window  # Rolling window for quantile calculation
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).astype(float).clip(-1, 1)

    @staticmethod
    def alpha_quanta_011_rank(df, window=55):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        raw = np.sign(open_price - delay_close) * (np.abs(close_price - open_price) / (np.abs(close_price - open_price).rolling(window).mean() + 1e-8))
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_011_tanh(df, window=55):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        raw = np.sign(open_price - delay_close) * (np.abs(close_price - open_price) / (np.abs(close_price - open_price).rolling(window).mean() + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_011_zscore(df, window=55):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        raw = np.sign(open_price - delay_close) * (np.abs(close_price - open_price) / (np.abs(close_price - open_price).rolling(window).mean() + 1e-8))
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std()
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_011_sign(df, window=85):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        raw = np.sign(open_price - delay_close) * (np.abs(close_price - open_price) / (np.abs(close_price - open_price).rolling(window).mean() + 1e-8))
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_011_wf(df, window=100):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        raw = np.sign(open_price - delay_close) * (np.abs(close_price - open_price) / (np.abs(close_price - open_price).rolling(window).mean() + 1e-8))
        p1 = 0.05
        p2 = window * 5
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_012_rank(df, window=100, factor=1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_hl = high - low
        log_vol = np.log1p(volume)
        mean_range = range_hl.rolling(window).mean()
        mean_log_vol = log_vol.rolling(window).mean()
        raw = (range_hl / (mean_range + 1e-8)) / (log_vol / (mean_log_vol + 1e-8))
        raw = pd.Series(raw, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_012_tanh(df, window=50, factor=1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_hl = high - low
        log_vol = np.log1p(volume)
        mean_range = range_hl.rolling(window).mean()
        mean_log_vol = log_vol.rolling(window).mean()
        raw = (range_hl / (mean_range + 1e-8)) / (log_vol / (mean_log_vol + 1e-8))
        raw = pd.Series(raw, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        std_raw = raw.rolling(window).std()
        normalized = np.tanh(raw / (std_raw + 1e-8))
        return -normalized.clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_012_zscore(df, window=80, factor=1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_hl = high - low
        log_vol = np.log1p(volume)
        mean_range = range_hl.rolling(window).mean()
        mean_log_vol = log_vol.rolling(window).mean()
        raw = (range_hl / (mean_range + 1e-8)) / (log_vol / (mean_log_vol + 1e-8))
        raw = pd.Series(raw, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        normalized = ((raw - mean_raw) / (std_raw + 1e-8)).clip(-1, 1)
        return normalized * factor

    @staticmethod
    def alpha_quanta_012_sign(df, window=20, factor=20):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_hl = high - low
        log_vol = np.log1p(volume)
        mean_range = range_hl.rolling(window).mean()
        mean_log_vol = log_vol.rolling(window).mean()
        raw = (range_hl / (mean_range + 1e-8)) / (log_vol / (mean_log_vol + 1e-8))
        raw = pd.Series(raw, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        normalized = np.sign(raw)
        return normalized.clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_012_wf(df, window=40, factor=1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_hl = high - low
        log_vol = np.log1p(volume)
        mean_range = range_hl.rolling(window).mean()
        mean_log_vol = log_vol.rolling(window).mean()
        raw = (range_hl / (mean_range + 1e-8)) / (log_vol / (mean_log_vol + 1e-8))
        raw = pd.Series(raw, index=df.index)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        p1 = 0.05
        p2 = window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return -normalized.clip(-1, 1) * factor

    @staticmethod
    def alpha_quanta_013_rank(df, window=40, param=20):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_ratio = volume.pct_change()
        raw = (volume_ratio - volume_ratio.rolling(window).mean()) / (volume_ratio.rolling(window).std() + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_013_tanh(df, window=60, param=7):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_ratio = volume.pct_change()
        raw = (volume_ratio - volume_ratio.rolling(window).mean()) / (volume_ratio.rolling(window).std() + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = np.tanh(raw / raw.rolling(param).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_013_zscore(df, window=60, param=3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_ratio = volume.pct_change()
        raw = (volume_ratio - volume_ratio.rolling(window).mean()) / (volume_ratio.rolling(window).std() + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = ((raw - raw.rolling(param).mean()) / raw.rolling(param).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_013_sign(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_ratio = volume.pct_change()
        raw = (volume_ratio - volume_ratio.rolling(window).mean()) / (volume_ratio.rolling(window).std() + 1e-8)
        raw = raw.ffill().fillna(0)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_013_wf(df, window=90, param=3):
        p1 = 0.05
        p2 = param
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_ratio = volume.pct_change()
        raw = (volume_ratio - volume_ratio.rolling(window).mean()) / (volume_ratio.rolling(window).std() + 1e-8)
        raw = raw.ffill().fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_014_rank(df, window=2, sub_window=100):
        # Raw calculation
        open_price = df['open']
        close_delay1 = df['close'].shift(1)
        high = df['high']
        low = df['low']
        high_low_delay1 = (high - low).shift(1)
        std_delay_close = df['close'].shift(1).rolling(window).std()
        raw = np.sign(open_price / close_delay1 - 1) * (high_low_delay1 / (std_delay_close + 1e-8))
        # Rolling Rank normalization
        normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_014_tanh(df, window=2, sub_window=60):
        # Raw calculation
        open_price = df['open']
        close_delay1 = df['close'].shift(1)
        high = df['high']
        low = df['low']
        high_low_delay1 = (high - low).shift(1)
        std_delay_close = df['close'].shift(1).rolling(window).std()
        raw = np.sign(open_price / close_delay1 - 1) * (high_low_delay1 / (std_delay_close + 1e-8))
        # Dynamic Tanh normalization
        rolling_std = raw.rolling(sub_window).std()
        normalized = np.tanh(raw / (rolling_std + 1e-8))
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_014_zscore(df, window=2, sub_window=40):
        # Raw calculation
        open_price = df['open']
        close_delay1 = df['close'].shift(1)
        high = df['high']
        low = df['low']
        high_low_delay1 = (high - low).shift(1)
        std_delay_close = df['close'].shift(1).rolling(window).std()
        raw = np.sign(open_price / close_delay1 - 1) * (high_low_delay1 / (std_delay_close + 1e-8))
        # Rolling Z-Score/Clip normalization
        rolling_mean = raw.rolling(sub_window).mean()
        rolling_std = raw.rolling(sub_window).std()
        normalized = ((raw - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_014_sign(df, window=70):
        # Raw calculation
        open_price = df['open']
        close_delay1 = df['close'].shift(1)
        high = df['high']
        low = df['low']
        high_low_delay1 = (high - low).shift(1)
        std_delay_close = df['close'].shift(1).rolling(window).std()
        raw = np.sign(open_price / close_delay1 - 1) * (high_low_delay1 / (std_delay_close + 1e-8))
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_014_wf(df, window=2, sub_window=40):
        # Raw calculation
        open_price = df['open']
        close_delay1 = df['close'].shift(1)
        high = df['high']
        low = df['low']
        high_low_delay1 = (high - low).shift(1)
        std_delay_close = df['close'].shift(1).rolling(window).std()
        raw = np.sign(open_price / close_delay1 - 1) * (high_low_delay1 / (std_delay_close + 1e-8))
        # Winsorized Fisher normalization (hardcoded p1=0.05, p2=sub_window)
        p1 = 0.05
        p2 = sub_window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_015_rank(df, window=90, factor=30):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        prev_hl_range = (high - low).shift(1)
        raw = (open_price / prev_close - 1) * prev_hl_range
        mean_prev_hl = prev_hl_range.rolling(window).mean()
        raw = raw / (mean_prev_hl + 1e-8)
        raw = raw * factor
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.ffill().fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_015_tanh(df, window=90, factor=5):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        prev_hl_range = (high - low).shift(1)
        raw = (open_price / prev_close - 1) * prev_hl_range
        mean_prev_hl = prev_hl_range.rolling(window).mean()
        raw = raw / (mean_prev_hl + 1e-8)
        raw = raw * factor
        std_raw = raw.rolling(window).std()
        normalized = np.tanh(raw / (std_raw + 1e-8))
        normalized = normalized.ffill().fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_015_zscore(df, window=100, factor=5):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        prev_hl_range = (high - low).shift(1)
        raw = (open_price / prev_close - 1) * prev_hl_range
        mean_prev_hl = prev_hl_range.rolling(window).mean()
        raw = raw / (mean_prev_hl + 1e-8)
        raw = raw * factor
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        normalized = ((raw - mean_raw) / (std_raw + 1e-8)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_015_sign(df, window=70, factor=40):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        prev_hl_range = (high - low).shift(1)
        raw = (open_price / prev_close - 1) * prev_hl_range
        mean_prev_hl = prev_hl_range.rolling(window).mean()
        raw = raw / (mean_prev_hl + 1e-8)
        raw = raw * factor
        normalized = np.sign(raw)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_015_wf(df, window=10, factor=3):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        prev_close = close.shift(1)
        prev_hl_range = (high - low).shift(1)
        raw = (open_price / prev_close - 1) * prev_hl_range
        mean_prev_hl = prev_hl_range.rolling(window).mean()
        raw = raw / (mean_prev_hl + 1e-8)
        raw = raw * factor
        p1 = 0.05
        p2 = 20
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound)
        scaled = ((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        normalized = normalized.ffill().fillna(0)
        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_016_rank(df, window=5):
        # Raw calculation
        hl = df['high'] - df['low']
        hl_delay = hl.shift(1)
        hl_mean = hl_delay.rolling(window).mean()
        ret = df['close'].pct_change()
        ret_delay = ret.shift(1)
        ret_std = ret_delay.rolling(window).std()
        open_ratio = df['open'] / df['close'].shift(1) - 1
        raw = ((hl_delay - hl_mean) / (ret_std + 1e-8)) * open_ratio
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_016_tanh(df, window=50):
        # Raw calculation
        hl = df['high'] - df['low']
        hl_delay = hl.shift(1)
        hl_mean = hl_delay.rolling(window).mean()
        ret = df['close'].pct_change()
        ret_delay = ret.shift(1)
        ret_std = ret_delay.rolling(window).std()
        open_ratio = df['open'] / df['close'].shift(1) - 1
        raw = ((hl_delay - hl_mean) / (ret_std + 1e-8)) * open_ratio
        # Dynamic Tanh normalization
        raw_std = raw.rolling(window).std()
        normalized = np.tanh(raw / (raw_std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_016_zscore(df, window=85):
        # Raw calculation
        hl = df['high'] - df['low']
        hl_delay = hl.shift(1)
        hl_mean = hl_delay.rolling(window).mean()
        ret = df['close'].pct_change()
        ret_delay = ret.shift(1)
        ret_std = ret_delay.rolling(window).std()
        open_ratio = df['open'] / df['close'].shift(1) - 1
        raw = ((hl_delay - hl_mean) / (ret_std + 1e-8)) * open_ratio
        # Rolling Z-Score/Clip normalization
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        normalized = ((raw - raw_mean) / (raw_std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_016_sign(df, window=45):
        # Raw calculation
        hl = df['high'] - df['low']
        hl_delay = hl.shift(1)
        hl_mean = hl_delay.rolling(window).mean()
        ret = df['close'].pct_change()
        ret_delay = ret.shift(1)
        ret_std = ret_delay.rolling(window).std()
        open_ratio = df['open'] / df['close'].shift(1) - 1
        raw = ((hl_delay - hl_mean) / (ret_std + 1e-8)) * open_ratio
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_016_wf(df, window=35):
        # Raw calculation
        hl = df['high'] - df['low']
        hl_delay = hl.shift(1)
        hl_mean = hl_delay.rolling(window).mean()
        ret = df['close'].pct_change()
        ret_delay = ret.shift(1)
        ret_std = ret_delay.rolling(window).std()
        open_ratio = df['open'] / df['close'].shift(1) - 1
        raw = ((hl_delay - hl_mean) / (ret_std + 1e-8)) * open_ratio
        # Winsorized Fisher normalization (hardcoded p1=0.05, p2=window*2)
        p1 = 0.05
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_017_rank(df, window=60, sub_window=5):
        # Raw calculation
        hl_diff = df['high'] - df['low']
        hl_diff_lag1 = hl_diff.shift(1)
        hl_mean = hl_diff_lag1.rolling(window).mean()
        open_close_ratio = df['open'] / df['close'].shift(1) - 1
        ret = df['close'].pct_change()
        ret_lag1_mean = ret.shift(1).rolling(window).mean()
        raw = (hl_diff_lag1 / (hl_mean + 1e-8)) * (open_close_ratio - ret_lag1_mean)
        # Rolling Rank normalization
        normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_017_tanh(df, window=80, factor=1):
        # Raw calculation
        hl_diff = df['high'] - df['low']
        hl_diff_lag1 = hl_diff.shift(1)
        hl_mean = hl_diff_lag1.rolling(window).mean()
        open_close_ratio = df['open'] / df['close'].shift(1) - 1
        ret = df['close'].pct_change()
        ret_lag1_mean = ret.shift(1).rolling(window).mean()
        raw = (hl_diff_lag1 / (hl_mean + 1e-8)) * (open_close_ratio - ret_lag1_mean)
        # Dynamic Tanh normalization
        std = raw.rolling(window).std()
        normalized = np.tanh(factor * raw / (std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_017_zscore(df, window=2, clip_window=100):
        # Raw calculation
        hl_diff = df['high'] - df['low']
        hl_diff_lag1 = hl_diff.shift(1)
        hl_mean = hl_diff_lag1.rolling(window).mean()
        open_close_ratio = df['open'] / df['close'].shift(1) - 1
        ret = df['close'].pct_change()
        ret_lag1_mean = ret.shift(1).rolling(window).mean()
        raw = (hl_diff_lag1 / (hl_mean + 1e-8)) * (open_close_ratio - ret_lag1_mean)
        # Rolling Z-Score with Clip
        mean = raw.rolling(clip_window).mean()
        std = raw.rolling(clip_window).std()
        zscore = (raw - mean) / (std + 1e-8)
        normalized = zscore.clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_017_sign(df, window=80, delta=5):
        # Raw calculation
        hl_diff = df['high'] - df['low']
        hl_diff_lag1 = hl_diff.shift(1)
        hl_mean = hl_diff_lag1.rolling(window).mean()
        open_close_ratio = df['open'] / df['close'].shift(1) - 1
        ret = df['close'].pct_change()
        ret_lag1_mean = ret.shift(1).rolling(window).mean()
        raw = (hl_diff_lag1 / (hl_mean + 1e-8)) * (open_close_ratio - ret_lag1_mean)
        # Sign/Binary Soft normalization
        normalized = np.sign(raw - delta)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_017_wf(df, window=2, winsor_window=70):
        # Raw calculation
        hl_diff = df['high'] - df['low']
        hl_diff_lag1 = hl_diff.shift(1)
        hl_mean = hl_diff_lag1.rolling(window).mean()
        open_close_ratio = df['open'] / df['close'].shift(1) - 1
        ret = df['close'].pct_change()
        ret_lag1_mean = ret.shift(1).rolling(window).mean()
        raw = (hl_diff_lag1 / (hl_mean + 1e-8)) * (open_close_ratio - ret_lag1_mean)
        # Winsorized Fisher normalization
        p1 = 0.05  # Hardcoded quantile
        low = raw.rolling(winsor_window).quantile(p1)
        high = raw.rolling(winsor_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform approximation
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_018_rank(df, window=100):
        # Raw calculation
        ret = df['open'] / df['close'].shift(1) - 1
        hl_diff = df['high'] - df['low']
        hl_diff_lag = hl_diff.shift(1)
        hl_median = hl_diff_lag.rolling(window=window, min_periods=1).median()
        raw = np.sign(ret) * (hl_diff_lag / (hl_median + 1e-8))
        # Rolling Rank normalization
        normalized = (raw.rolling(window=window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_018_tanh(df, window=15):
        # Raw calculation
        ret = df['open'] / df['close'].shift(1) - 1
        hl_diff = df['high'] - df['low']
        hl_diff_lag = hl_diff.shift(1)
        hl_median = hl_diff_lag.rolling(window=window, min_periods=1).median()
        raw = np.sign(ret) * (hl_diff_lag / (hl_median + 1e-8))
        # Dynamic Tanh normalization
        std = raw.rolling(window=window, min_periods=1).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_018_zscore(df, window=5):
        # Raw calculation
        ret = df['open'] / df['close'].shift(1) - 1
        hl_diff = df['high'] - df['low']
        hl_diff_lag = hl_diff.shift(1)
        hl_median = hl_diff_lag.rolling(window=window, min_periods=1).median()
        raw = np.sign(ret) * (hl_diff_lag / (hl_median + 1e-8))
        # Rolling Z-Score/Clip normalization
        mean = raw.rolling(window=window, min_periods=1).mean()
        std = raw.rolling(window=window, min_periods=1).std()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_018_sign(df, window=15):
        # Raw calculation
        ret = df['open'] / df['close'].shift(1) - 1
        hl_diff = df['high'] - df['low']
        hl_diff_lag = hl_diff.shift(1)
        hl_median = hl_diff_lag.rolling(window=window, min_periods=1).median()
        raw = np.sign(ret) * (hl_diff_lag / (hl_median + 1e-8))
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_018_wf(df, window=100, p1=0.1):
        # Raw calculation
        ret = df['open'] / df['close'].shift(1) - 1
        hl_diff = df['high'] - df['low']
        hl_diff_lag = hl_diff.shift(1)
        hl_median = hl_diff_lag.rolling(window=window, min_periods=1).median()
        raw = np.sign(ret) * (hl_diff_lag / (hl_median + 1e-8))
        # Winsorized Fisher normalization
        low = raw.rolling(window=window, min_periods=1).quantile(p1)
        high = raw.rolling(window=window, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_019_rank(df, window=85):
        raw = abs(df['open'] / df['close'].shift(1) - 1) / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_019_tanh(df, window=5):
        raw = abs(df['open'] / df['close'].shift(1) - 1) / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_019_zscore(df, window=15):
        raw = abs(df['open'] / df['close'].shift(1) - 1) / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std()
        normalized = z.clip(-1, 1)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_019_sign(df, window=50):
        raw = abs(df['open'] / df['close'].shift(1) - 1) / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        normalized = np.sign(raw - raw.rolling(window).mean())
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_019_wf(df, window=50, p1=0.9):
        raw = abs(df['open'] / df['close'].shift(1) - 1) / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_020_rank(df, window=65):
        # Raw: SIGN(open / DELAY(close, 1) - 1) * (volume / (TS_MEAN(volume, 10) + 1e-8) - 1)
        # ZSCORE gốc được thay bằng Rolling Rank (Case A) để chuẩn hóa về [-1,1] và loại bỏ outliers.
        # Volume: Sử dụng log1p vì tỷ lệ volume có thể có độ lệch cao.
        # Xử lý missing data: ffill cho giá, fillna(0) cho signal trung lập.
        # Tối đa 2 tham số: window (cho rank) và window_volume (cố định =10).
        open_price = df['open'].ffill()
        close = df['close'].ffill()
        volume = df.get('matchingVolume', df.get('volume', 1)).ffill()
        log_volume = np.log1p(volume)
        # Tính phần giá: sign(open / delay(close,1) - 1)
        price_ratio = open_price / close.shift(1) - 1
        price_signal = np.sign(price_ratio)
        # Tính phần volume: volume / (TS_MEAN(volume,10) + 1e-8) - 1
        # Sử dụng log volume để giảm skew, nhưng vẫn giữ tỷ lệ so với trung bình.
        volume_mean = log_volume.rolling(10).mean()
        volume_ratio = log_volume / (volume_mean + 1e-8) - 1
        # Raw alpha
        raw = price_signal * volume_ratio
        # Chuẩn hóa Case A: Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_020_tanh(df, window=75):
        # Raw: SIGN(open / DELAY(close, 1) - 1) * (volume / (TS_MEAN(volume, 10) + 1e-8) - 1)
        # Chuẩn hóa Case B: Dynamic Tanh để giữ lại cường độ (magnitude) của tín hiệu.
        # Volume: Sử dụng log1p để giảm skew.
        open_price = df['open'].ffill()
        close = df['close'].ffill()
        volume = df.get('matchingVolume', df.get('volume', 1)).ffill()
        log_volume = np.log1p(volume)
        price_ratio = open_price / close.shift(1) - 1
        price_signal = np.sign(price_ratio)
        volume_mean = log_volume.rolling(10).mean()
        volume_ratio = log_volume / (volume_mean + 1e-8) - 1
        raw = price_signal * volume_ratio
        # Chuẩn hóa Case B: Tanh(raw / rolling_std)
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_020_zscore(df, window=100):
        # Raw: SIGN(open / DELAY(close, 1) - 1) * (volume / (TS_MEAN(volume, 10) + 1e-8) - 1)
        # Chuẩn hóa Case C: Rolling Z-Score + Clip, phù hợp cho spread/oscillator.
        # Volume: Sử dụng log1p.
        open_price = df['open'].ffill()
        close = df['close'].ffill()
        volume = df.get('matchingVolume', df.get('volume', 1)).ffill()
        log_volume = np.log1p(volume)
        price_ratio = open_price / close.shift(1) - 1
        price_signal = np.sign(price_ratio)
        volume_mean = log_volume.rolling(10).mean()
        volume_ratio = log_volume / (volume_mean + 1e-8) - 1
        raw = price_signal * volume_ratio
        # Chuẩn hóa Case C: Z-Score clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        zscore = (raw - mean) / (std + 1e-8)
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_020_sign(df, window_volume=5):
        # Raw: SIGN(open / DELAY(close, 1) - 1) * (volume / (TS_MEAN(volume, 10) + 1e-8) - 1)
        # Chuẩn hóa Case D: Sign/Binary Soft, chỉ giữ hướng (directional).
        # Volume: Sử dụng log1p.
        # window_volume là tham số cho TS_MEAN, window cho sign không cần vì sign là tức thời.
        open_price = df['open'].ffill()
        close = df['close'].ffill()
        volume = df.get('matchingVolume', df.get('volume', 1)).ffill()
        log_volume = np.log1p(volume)
        price_ratio = open_price / close.shift(1) - 1
        price_signal = np.sign(price_ratio)
        volume_mean = log_volume.rolling(window_volume).mean()
        volume_ratio = log_volume / (volume_mean + 1e-8) - 1
        raw = price_signal * volume_ratio
        # Chuẩn hóa Case D: np.sign
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_020_wf(df, window=80, factor=0.1):
        # Raw: SIGN(open / DELAY(close, 1) - 1) * (volume / (TS_MEAN(volume, 10) + 1e-8) - 1)
        # Chuẩn hóa Case E: Winsorized Fisher, xử lý heavy tails.
        # Volume: Sử dụng log1p.
        # Tham số: window (cho rolling quantile), factor (phần trăm winsorize, cố định nếu cần).
        # Tối đa 2 tham số: window và factor.
        open_price = df['open'].ffill()
        close = df['close'].ffill()
        volume = df.get('matchingVolume', df.get('volume', 1)).ffill()
        log_volume = np.log1p(volume)
        price_ratio = open_price / close.shift(1) - 1
        price_signal = np.sign(price_ratio)
        volume_mean = log_volume.rolling(10).mean()
        volume_ratio = log_volume / (volume_mean + 1e-8) - 1
        raw = price_signal * volume_ratio
        # Winsorize
        low = raw.rolling(window).quantile(factor)
        high = raw.rolling(window).quantile(1 - factor)
        winsorized = raw.clip(low, high)
        # Fisher Transform
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_021_rank(df, window=15, sub_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (open_ / close.shift(1) - 1).abs()
        delta_vol = volume.diff(1)
        denom = volume + 1e-8
        ratio = delta_vol / denom
        # Rolling correlation
        x = raw.rolling(window).apply(lambda s: s.corr(ratio), raw=True)
        # Normalization: Rolling Rank (Case A) - robust to outliers, uniform distribution
        signal = (x.rolling(sub_window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_021_tanh(df, window=15, sub_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (open_ / close.shift(1) - 1).abs()
        delta_vol = volume.diff(1)
        denom = volume + 1e-8
        ratio = delta_vol / denom
        # Rolling correlation
        x = raw.rolling(window).apply(lambda s: s.corr(ratio), raw=True)
        # Normalization: Dynamic Tanh (Case B) - preserve magnitude
        rolling_std = x.rolling(sub_window).std()
        signal = np.tanh(x / rolling_std.replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_021_zscore(df, window=15, sub_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (open_ / close.shift(1) - 1).abs()
        delta_vol = volume.diff(1)
        denom = volume + 1e-8
        ratio = delta_vol / denom
        # Rolling correlation
        x = raw.rolling(window).apply(lambda s: s.corr(ratio), raw=True)
        # Normalization: Rolling Z-Score/Clip (Case C) - for spread/oscillator
        rolling_mean = x.rolling(sub_window).mean()
        rolling_std = x.rolling(sub_window).std()
        signal = ((x - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_021_sign(df, window=15, sub_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (open_ / close.shift(1) - 1).abs()
        delta_vol = volume.diff(1)
        denom = volume + 1e-8
        ratio = delta_vol / denom
        # Rolling correlation
        x = raw.rolling(window).apply(lambda s: s.corr(ratio), raw=True)
        # Normalization: Sign/Binary Soft (Case D) - pure breakout/trend
        signal = np.sign(x)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_021_wf(df, window=15, sub_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (open_ / close.shift(1) - 1).abs()
        delta_vol = volume.diff(1)
        denom = volume + 1e-8
        ratio = delta_vol / denom
        # Rolling correlation
        x = raw.rolling(window).apply(lambda s: s.corr(ratio), raw=True)
        # Normalization: Winsorized Fisher (Case E) - heavy tails, preserve distribution
        p1 = 0.05  # hardcoded quantile
        p2 = sub_window  # rolling window
        low = x.rolling(p2).quantile(p1)
        high = x.rolling(p2).quantile(1 - p1)
        winsorized = x.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_022_rank(df, window=45):
        # Raw: ZSCORE of regression slope of log(volume+1) vs abs(open/prev_close-1) over 20 days
        # Context: Measures sensitivity of volume to absolute overnight return, normalized.
        # Normalization A (Rolling Rank): For uniform distribution, robust to outliers.
        # Use log1p for volume due to skew.
        prev_close = df['close'].shift(1)
        abs_return = np.abs(df['open'] / prev_close - 1).replace([np.inf, -np.inf], np.nan)
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        # Rolling regression slope: cov(x,y)/var(x)
        cov = log_vol.rolling(window).cov(abs_return)
        var = abs_return.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        # Rolling rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_022_tanh(df, window=95):
        # Raw: ZSCORE of regression slope of log(volume+1) vs abs(open/prev_close-1) over 20 days
        # Context: Measures sensitivity of volume to absolute overnight return, normalized.
        # Normalization B (Dynamic Tanh): Preserve magnitude, robust to volatility.
        # Use log1p for volume due to skew.
        prev_close = df['close'].shift(1)
        abs_return = np.abs(df['open'] / prev_close - 1).replace([np.inf, -np.inf], np.nan)
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        cov = log_vol.rolling(window).cov(abs_return)
        var = abs_return.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        # Dynamic tanh normalization
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_022_zscore(df, window=45):
        # Raw: ZSCORE of regression slope of log(volume+1) vs abs(open/prev_close-1) over 20 days
        # Context: Measures sensitivity of volume to absolute overnight return, normalized.
        # Normalization C (Rolling Z-Score/Clip): Direct z-score as per original, clipped.
        # Use log1p for volume due to skew.
        prev_close = df['close'].shift(1)
        abs_return = np.abs(df['open'] / prev_close - 1).replace([np.inf, -np.inf], np.nan)
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        cov = log_vol.rolling(window).cov(abs_return)
        var = abs_return.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        # Rolling z-score (original ZSCORE)
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        normalized = z.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_022_sign(df, window=40):
        # Raw: ZSCORE of regression slope of log(volume+1) vs abs(open/prev_close-1) over 20 days
        # Context: Measures sensitivity of volume to absolute overnight return, normalized.
        # Normalization D (Sign/Binary Soft): Pure direction of sensitivity.
        # Use log1p for volume due to skew.
        prev_close = df['close'].shift(1)
        abs_return = np.abs(df['open'] / prev_close - 1).replace([np.inf, -np.inf], np.nan)
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        cov = log_vol.rolling(window).cov(abs_return)
        var = abs_return.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        # Sign normalization
        normalized = np.sign(raw)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_022_wf(df, window=60):
        # Raw: ZSCORE of regression slope of log(volume+1) vs abs(open/prev_close-1) over 20 days
        # Context: Measures sensitivity of volume to absolute overnight return, normalized.
        # Normalization E (Winsorized Fisher): Handle heavy tails, preserve distribution shape.
        # Use log1p for volume due to skew.
        # Hardcode winsorization parameters: quantile=0.05, lookback=60
        p1 = 0.05
        p2 = 60
        prev_close = df['close'].shift(1)
        abs_return = np.abs(df['open'] / prev_close - 1).replace([np.inf, -np.inf], np.nan)
        log_vol = np.log1p(df.get('matchingVolume', df.get('volume', 1)))
        cov = log_vol.rolling(window).cov(abs_return)
        var = abs_return.rolling(window).var().replace(0, np.nan)
        slope = cov / var
        raw = slope
        # Winsorization
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_023_rank(df, window=80):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ratio = open_ / close.shift(1)
        mean_ratio = ratio.rolling(window).mean()
        sign_raw = np.sign(ratio - mean_ratio)
        volume_ratio = volume / (volume.rolling(window).mean() + 1e-8) - 1
        raw = sign_raw * volume_ratio
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_023_tanh(df, window=5):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ratio = open_ / close.shift(1)
        mean_ratio = ratio.rolling(window).mean()
        sign_raw = np.sign(ratio - mean_ratio)
        volume_ratio = volume / (volume.rolling(window).mean() + 1e-8) - 1
        raw = sign_raw * volume_ratio
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_023_zscore(df, window=80):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ratio = open_ / close.shift(1)
        mean_ratio = ratio.rolling(window).mean()
        sign_raw = np.sign(ratio - mean_ratio)
        volume_ratio = volume / (volume.rolling(window).mean() + 1e-8) - 1
        raw = sign_raw * volume_ratio
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std()
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_023_sign(df, window=5):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ratio = open_ / close.shift(1)
        mean_ratio = ratio.rolling(window).mean()
        sign_raw = np.sign(ratio - mean_ratio)
        volume_ratio = volume / (volume.rolling(window).mean() + 1e-8) - 1
        raw = sign_raw * volume_ratio
        normalized = np.sign(raw)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_023_wf(df, window=75):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ratio = open_ / close.shift(1)
        mean_ratio = ratio.rolling(window).mean()
        sign_raw = np.sign(ratio - mean_ratio)
        volume_ratio = volume / (volume.rolling(window).mean() + 1e-8) - 1
        raw = sign_raw * volume_ratio
        p1 = 0.05
        p2 = window * 2
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_024_rank(df, window=15):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (close - open_) / (open_ + 1e-8) * volume
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        slope = slope.ffill()
        normalized = (slope.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_024_tanh(df, window=5):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (close - open_) / (open_ + 1e-8) * volume
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        slope = slope.ffill()
        normalized = np.tanh(slope / slope.rolling(window).std().replace(0, np.nan))
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_024_zscore(df, window=15):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (close - open_) / (open_ + 1e-8) * volume
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        slope = slope.ffill()
        normalized = (slope - slope.rolling(window).mean()) / slope.rolling(window).std().replace(0, np.nan)
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_024_sign(df, window=15):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (close - open_) / (open_ + 1e-8) * volume
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        slope = slope.ffill()
        normalized = np.sign(slope)
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_024_wf(df, window=5):
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (close - open_) / (open_ + 1e-8) * volume
        days = pd.Series(np.arange(len(df)), index=df.index)
        slope = raw.rolling(window).cov(days) / days.rolling(window).var().replace(0, np.nan)
        slope = slope.ffill()
        p1 = 0.05
        p2 = 20
        low = slope.rolling(p2).quantile(p1)
        high = slope.rolling(p2).quantile(1 - p1)
        winsorized = slope.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_025_rank(df, window=20):
        # Raw calculation: correlation between (open - prev_close)/prev_close and volume over rolling window
        prev_close = df['close'].shift(1)
        ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        vol = df.get('matchingVolume', df.get('volume', 1))
        # Use vectorized rolling correlation
        raw = ret.rolling(window).corr(vol)
        # Normalization A: Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_025_tanh(df, window=15):
        # Raw calculation: correlation between (open - prev_close)/prev_close and volume over rolling window
        prev_close = df['close'].shift(1)
        ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        vol = df.get('matchingVolume', df.get('volume', 1))
        # Use vectorized rolling correlation
        raw = ret.rolling(window).corr(vol)
        # Normalization B: Dynamic Tanh
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_025_zscore(df, window=20):
        # Raw calculation: correlation between (open - prev_close)/prev_close and volume over rolling window
        prev_close = df['close'].shift(1)
        ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        vol = df.get('matchingVolume', df.get('volume', 1))
        # Use vectorized rolling correlation
        raw = ret.rolling(window).corr(vol)
        # Normalization C: Rolling Z-Score/Clip
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_025_sign(df, window=15):
        # Raw calculation: correlation between (open - prev_close)/prev_close and volume over rolling window
        prev_close = df['close'].shift(1)
        ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        vol = df.get('matchingVolume', df.get('volume', 1))
        # Use vectorized rolling correlation
        raw = ret.rolling(window).corr(vol)
        # Normalization D: Sign/Binary Soft
        normalized = np.sign(raw)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_025_wf(df, window=20, p1=0.1):
        # Raw calculation: correlation between (open - prev_close)/prev_close and volume over rolling window
        prev_close = df['close'].shift(1)
        ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        vol = df.get('matchingVolume', df.get('volume', 1))
        # Use vectorized rolling correlation
        raw = ret.rolling(window).corr(vol)
        # Normalization E: Winsorized Fisher
        p2 = window  # using same window for rolling quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_026_rank(df, window=80, delta_window=1):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume - volume.shift(delta_window)
        mean_vol = volume.rolling(window).mean()
        raw = delta_vol / (mean_vol + 1e-8) * 1.0 / (high - low + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).astype(np.float64)

    @staticmethod
    def alpha_quanta_026_tanh(df, window=40, delta_window=4):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume - volume.shift(delta_window)
        mean_vol = volume.rolling(window).mean()
        raw = delta_vol / (mean_vol + 1e-8) * 1.0 / (high - low + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0).astype(np.float64)

    @staticmethod
    def alpha_quanta_026_zscore(df, window=40, delta_window=4):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume - volume.shift(delta_window)
        mean_vol = volume.rolling(window).mean()
        raw = delta_vol / (mean_vol + 1e-8) * 1.0 / (high - low + 1e-8)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0).astype(np.float64)

    @staticmethod
    def alpha_quanta_026_sign(df, window=10, delta_window=7):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume - volume.shift(delta_window)
        mean_vol = volume.rolling(window).mean()
        raw = delta_vol / (mean_vol + 1e-8) * 1.0 / (high - low + 1e-8)
        normalized = np.sign(raw)
        return -normalized.fillna(0).astype(np.float64)

    @staticmethod
    def alpha_quanta_026_wf(df, window=40, delta_window=3):
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume - volume.shift(delta_window)
        mean_vol = volume.rolling(window).mean()
        raw = delta_vol / (mean_vol + 1e-8) * 1.0 / (high - low + 1e-8)
        p1 = 0.05
        p2 = 20
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        scaled = ((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return -normalized.fillna(0).astype(np.float64)

    @staticmethod
    def alpha_quanta_027_rank(df, window=6, sub_window=6):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        def alpha_quanta_027_rank(y, w):
            cov = y.rolling(w).cov(days)
            var = days.rolling(w).var().replace(0, np.nan)
            return cov / var
        beta_close = regbeta(close, window)
        beta_vol = regbeta(volume, sub_window)
        raw = beta_close - beta_vol
        raw = raw.fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_027_tanh(df, window=6, sub_window=6):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        def alpha_quanta_027_tanh(y, w):
            cov = y.rolling(w).cov(days)
            var = days.rolling(w).var().replace(0, np.nan)
            return cov / var
        beta_close = regbeta(close, window)
        beta_vol = regbeta(volume, sub_window)
        raw = beta_close - beta_vol
        raw = raw.fillna(0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_027_zscore(df, window=6, sub_window=6):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        def alpha_quanta_027_zscore(y, w):
            cov = y.rolling(w).cov(days)
            var = days.rolling(w).var().replace(0, np.nan)
            return cov / var
        beta_close = regbeta(close, window)
        beta_vol = regbeta(volume, sub_window)
        raw = beta_close - beta_vol
        raw = raw.fillna(0)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_027_sign(df, window=6, sub_window=6):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        def alpha_quanta_027_sign(y, w):
            cov = y.rolling(w).cov(days)
            var = days.rolling(w).var().replace(0, np.nan)
            return cov / var
        beta_close = regbeta(close, window)
        beta_vol = regbeta(volume, sub_window)
        raw = beta_close - beta_vol
        raw = raw.fillna(0)
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_027_wf(df, window=6, sub_window=6):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        days = pd.Series(np.arange(len(df)), index=df.index)
        def alpha_quanta_027_wf(y, w):
            cov = y.rolling(w).cov(days)
            var = days.rolling(w).var().replace(0, np.nan)
            return cov / var
        beta_close = regbeta(close, window)
        beta_vol = regbeta(volume, sub_window)
        raw = beta_close - beta_vol
        raw = raw.fillna(0)
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_028_rank(df, window=90):
        # Raw alpha: (($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)) / (TS_STD($close, 15) + 1e-8) * $volume / (TS_MEAN($volume, 15) + 1e-8)
        # Context: Price momentum normalized by volatility and volume scaling.
        # Normalization A (Rolling Rank): For uniform distribution and outlier robustness.
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_close = close.shift(1)
        price_change = (open_ - delay_close) / (delay_close + 1e-8)
        close_std = close.rolling(window=window, min_periods=1).std() + 1e-8
        volume_mean = volume.rolling(window=window, min_periods=1).mean() + 1e-8
        raw = price_change / close_std * volume / volume_mean
        raw = pd.Series(raw, index=df.index)
        raw_rank = raw.rolling(window=window, min_periods=1).rank(pct=True)
        signal = (raw_rank * 2) - 1
        signal = signal.fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_028_tanh(df, window=100):
        # Raw alpha: (($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)) / (TS_STD($close, 15) + 1e-8) * $volume / (TS_MEAN($volume, 15) + 1e-8)
        # Context: Price momentum normalized by volatility and volume scaling.
        # Normalization B (Dynamic Tanh): Preserve magnitude and intensity.
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_close = close.shift(1)
        price_change = (open_ - delay_close) / (delay_close + 1e-8)
        close_std = close.rolling(window=window, min_periods=1).std() + 1e-8
        volume_mean = volume.rolling(window=window, min_periods=1).mean() + 1e-8
        raw = price_change / close_std * volume / volume_mean
        raw = pd.Series(raw, index=df.index)
        raw_std = raw.rolling(window=window, min_periods=1).std() + 1e-8
        signal = np.tanh(raw / raw_std)
        signal = signal.fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_028_zscore(df, window=100):
        # Raw alpha: (($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)) / (TS_STD($close, 15) + 1e-8) * $volume / (TS_MEAN($volume, 15) + 1e-8)
        # Context: Price momentum normalized by volatility and volume scaling.
        # Normalization C (Rolling Z-Score/Clip): For spread/oscillator-like signals.
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_close = close.shift(1)
        price_change = (open_ - delay_close) / (delay_close + 1e-8)
        close_std = close.rolling(window=window, min_periods=1).std() + 1e-8
        volume_mean = volume.rolling(window=window, min_periods=1).mean() + 1e-8
        raw = price_change / close_std * volume / volume_mean
        raw = pd.Series(raw, index=df.index)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        raw_std = raw.rolling(window=window, min_periods=1).std() + 1e-8
        signal = (raw - raw_mean) / raw_std
        signal = signal.fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_028_sign(df, window=55):
        # Raw alpha: (($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)) / (TS_STD($close, 15) + 1e-8) * $volume / (TS_MEAN($volume, 15) + 1e-8)
        # Context: Price momentum normalized by volatility and volume scaling.
        # Normalization D (Sign/Binary Soft): For pure breakout/trend following.
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_close = close.shift(1)
        price_change = (open_ - delay_close) / (delay_close + 1e-8)
        close_std = close.rolling(window=window, min_periods=1).std() + 1e-8
        volume_mean = volume.rolling(window=window, min_periods=1).mean() + 1e-8
        raw = price_change / close_std * volume / volume_mean
        raw = pd.Series(raw, index=df.index)
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_028_wf(df, window=80, factor=0.9):
        # Raw alpha: (($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)) / (TS_STD($close, 15) + 1e-8) * $volume / (TS_MEAN($volume, 15) + 1e-8)
        # Context: Price momentum normalized by volatility and volume scaling.
        # Normalization E (Winsorized Fisher): For heavy-tailed data preserving distribution structure.
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_close = close.shift(1)
        price_change = (open_ - delay_close) / (delay_close + 1e-8)
        close_std = close.rolling(window=window, min_periods=1).std() + 1e-8
        volume_mean = volume.rolling(window=window, min_periods=1).mean() + 1e-8
        raw = price_change / close_std * volume / volume_mean
        raw = pd.Series(raw, index=df.index)
        p1 = factor
        p2 = window
        low = raw.rolling(window=p2, min_periods=1).quantile(p1)
        high = raw.rolling(window=p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_029_rank(df, window=70):
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close_price.shift(1)
        price_change = open_price - prev_close
        sign_price = pd.Series(np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)), index=df.index)
        volume_mean = volume.rolling(window).mean()
        volume_diff = volume - volume_mean
        volume_ratio = volume_diff / (volume_mean + 1e-8)
        close_std = close_price.rolling(window).std()
        raw = sign_price * volume_ratio / (close_std + 1e-8)
        raw_rank = raw.rolling(window).rank(pct=True)
        normalized = (raw_rank * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_029_tanh(df, window=90):
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close_price.shift(1)
        price_change = open_price - prev_close
        sign_price = pd.Series(np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)), index=df.index)
        volume_mean = volume.rolling(window).mean()
        volume_diff = volume - volume_mean
        volume_ratio = volume_diff / (volume_mean + 1e-8)
        close_std = close_price.rolling(window).std()
        raw = sign_price * volume_ratio / (close_std + 1e-8)
        raw_std = raw.rolling(window).std()
        normalized = np.tanh(raw / (raw_std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_029_zscore(df, window=90):
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close_price.shift(1)
        price_change = open_price - prev_close
        sign_price = pd.Series(np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)), index=df.index)
        volume_mean = volume.rolling(window).mean()
        volume_diff = volume - volume_mean
        volume_ratio = volume_diff / (volume_mean + 1e-8)
        close_std = close_price.rolling(window).std()
        raw = sign_price * volume_ratio / (close_std + 1e-8)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        zscore = (raw - raw_mean) / (raw_std + 1e-8)
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_029_sign(df, window=5):
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close_price.shift(1)
        price_change = open_price - prev_close
        sign_price = pd.Series(np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)), index=df.index)
        volume_mean = volume.rolling(window).mean()
        volume_diff = volume - volume_mean
        volume_ratio = volume_diff / (volume_mean + 1e-8)
        close_std = close_price.rolling(window).std()
        raw = sign_price * volume_ratio / (close_std + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_029_wf(df, window=70, factor=0.1):
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close_price.shift(1)
        price_change = open_price - prev_close
        sign_price = pd.Series(np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0)), index=df.index)
        volume_mean = volume.rolling(window).mean()
        volume_diff = volume - volume_mean
        volume_ratio = volume_diff / (volume_mean + 1e-8)
        close_std = close_price.rolling(window).std()
        raw = sign_price * volume_ratio / (close_std + 1e-8)
        winsorize_window = max(20, window * 2)
        low = raw.rolling(winsorize_window).quantile(factor)
        high = raw.rolling(winsorize_window).quantile(1 - factor)
        winsorized = raw.clip(low, high)
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_030_rank(df, window=10, rank_window=90):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Avoid division by zero
        eps = 1e-8

        # Component 1: (open - previous_close) / previous_close
        prev_close = close.shift(1)
        comp1 = (open_ - prev_close) / (prev_close + eps)

        # Component 2: ((close - open) / open) * (volume / rolling_mean_volume)
        comp2 = ((close - open_) / (open_ + eps)) * (volume / (volume.rolling(window).mean() + eps))

        # Time-series correlation over window
        raw_corr = comp1.rolling(window).corr(comp2)

        # Rolling rank normalization (Case A)
        raw_rank = raw_corr.rolling(rank_window).rank(pct=True)
        signal = (raw_rank * 2) - 1

        # Forward fill and clip
        signal = signal.ffill().fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_030_tanh(df, window=7, std_window=100):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        eps = 1e-8

        prev_close = close.shift(1)
        comp1 = (open_ - prev_close) / (prev_close + eps)
        comp2 = ((close - open_) / (open_ + eps)) * (volume / (volume.rolling(window).mean() + eps))

        raw_corr = comp1.rolling(window).corr(comp2)

        # Dynamic Tanh normalization (Case B)
        rolling_std = raw_corr.rolling(std_window).std()
        normalized = raw_corr / (rolling_std + eps)
        signal = np.tanh(normalized)

        signal = signal.ffill().fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_030_zscore(df, window=30, z_window=10):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        eps = 1e-8

        prev_close = close.shift(1)
        comp1 = (open_ - prev_close) / (prev_close + eps)
        comp2 = ((close - open_) / (open_ + eps)) * (volume / (volume.rolling(window).mean() + eps))

        raw_corr = comp1.rolling(window).corr(comp2)

        # Rolling Z-Score with clip (Case C)
        rolling_mean = raw_corr.rolling(z_window).mean()
        rolling_std = raw_corr.rolling(z_window).std()
        z_score = (raw_corr - rolling_mean) / (rolling_std + eps)
        signal = z_score.clip(-1, 1)

        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_030_sign(df, window=5):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        eps = 1e-8

        prev_close = close.shift(1)
        comp1 = (open_ - prev_close) / (prev_close + eps)
        comp2 = ((close - open_) / (open_ + eps)) * (volume / (volume.rolling(window).mean() + eps))

        raw_corr = comp1.rolling(window).corr(comp2)

        # Sign/Binary Soft normalization (Case D)
        signal = np.sign(raw_corr)

        signal = signal.ffill().fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_030_wf(df, window=10, quantile_window=100):
        # Raw calculation
        open_ = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        eps = 1e-8

        prev_close = close.shift(1)
        comp1 = (open_ - prev_close) / (prev_close + eps)
        comp2 = ((close - open_) / (open_ + eps)) * (volume / (volume.rolling(window).mean() + eps))

        raw_corr = comp1.rolling(window).corr(comp2)

        # Winsorized Fisher normalization (Case E)
        # Hardcode quantile and window for the third parameter
        p1 = 0.05  # Hardcoded quantile threshold
        p2 = quantile_window  # Second parameter

        low = raw_corr.rolling(p2).quantile(p1)
        high = raw_corr.rolling(p2).quantile(1 - p1)
        winsorized = raw_corr.clip(lower=low, upper=high, axis=0)

        # Fisher Transform approximation
        scaled = ((winsorized - low) / (high - low + eps)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))

        signal = normalized.ffill().fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_031_rank(df, window=40):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        abs_change = np.abs(open_price - prev_close) / (prev_close + 1e-8)
        min_vol = volume.rolling(3).min()
        max_vol = volume.rolling(3).max()
        vol_range = (volume - min_vol) / (max_vol - min_vol + 1e-8)
        raw = abs_change / (vol_range + 1e-8)
        raw_ffilled = raw.ffill()
        normalized = (raw_ffilled.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_031_tanh(df, window=95):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        abs_change = np.abs(open_price - prev_close) / (prev_close + 1e-8)
        min_vol = volume.rolling(3).min()
        max_vol = volume.rolling(3).max()
        vol_range = (volume - min_vol) / (max_vol - min_vol + 1e-8)
        raw = abs_change / (vol_range + 1e-8)
        raw_ffilled = raw.ffill()
        std = raw_ffilled.rolling(window).std()
        normalized = np.tanh(raw_ffilled / (std + 1e-8))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_031_zscore(df, window=55):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        abs_change = np.abs(open_price - prev_close) / (prev_close + 1e-8)
        min_vol = volume.rolling(3).min()
        max_vol = volume.rolling(3).max()
        vol_range = (volume - min_vol) / (max_vol - min_vol + 1e-8)
        raw = abs_change / (vol_range + 1e-8)
        raw_ffilled = raw.ffill()
        mean = raw_ffilled.rolling(window).mean()
        std = raw_ffilled.rolling(window).std()
        zscore = (raw_ffilled - mean) / (std + 1e-8)
        normalized = zscore.clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_031_sign(df):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        abs_change = np.abs(open_price - prev_close) / (prev_close + 1e-8)
        min_vol = volume.rolling(3).min()
        max_vol = volume.rolling(3).max()
        vol_range = (volume - min_vol) / (max_vol - min_vol + 1e-8)
        raw = abs_change / (vol_range + 1e-8)
        raw_ffilled = raw.ffill()
        normalized = np.sign(raw_ffilled)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_031_wf(df, window=100, quantile=0.9):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        abs_change = np.abs(open_price - prev_close) / (prev_close + 1e-8)
        min_vol = volume.rolling(3).min()
        max_vol = volume.rolling(3).max()
        vol_range = (volume - min_vol) / (max_vol - min_vol + 1e-8)
        raw = abs_change / (vol_range + 1e-8)
        raw_ffilled = raw.ffill()
        low = raw_ffilled.rolling(window).quantile(quantile)
        high = raw_ffilled.rolling(window).quantile(1 - quantile)
        winsorized = raw_ffilled.clip(low, high)
        scaled = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_032_rank(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính giá trị raw
        price_change = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        price_zscore = (price_change - price_change.rolling(window).mean()) / price_change.rolling(window).std()
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        raw = np.sign(price_zscore) * volume_zscore

        # Chuẩn hóa Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_032_tanh(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính giá trị raw
        price_change = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        price_zscore = (price_change - price_change.rolling(window).mean()) / price_change.rolling(window).std()
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        raw = np.sign(price_zscore) * volume_zscore

        # Chuẩn hóa Dynamic Tanh
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_032_zscore(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính giá trị raw
        price_change = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        price_zscore = (price_change - price_change.rolling(window).mean()) / price_change.rolling(window).std()
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        raw = np.sign(price_zscore) * volume_zscore

        # Chuẩn hóa Rolling Z-Score/Clip
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_032_sign(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính giá trị raw
        price_change = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        price_zscore = (price_change - price_change.rolling(window).mean()) / price_change.rolling(window).std()
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        raw = np.sign(price_zscore) * volume_zscore

        # Chuẩn hóa Sign/Binary Soft
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_032_wf(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính giá trị raw
        price_change = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        price_zscore = (price_change - price_change.rolling(window).mean()) / price_change.rolling(window).std()
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
        raw = np.sign(price_zscore) * volume_zscore

        # Chuẩn hóa Winsorized Fisher
        p1 = 0.05  # Hardcoded quantile
        p2 = 20    # Hardcoded rolling window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_033_rank(df, window=40, sub_window=100):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # First term: TS_RANK((open - delay(close,1)) / (delay(close,1) + 1e-8), 6)
        term1_raw = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        term1_rank = term1_raw.rolling(window).rank(pct=True)

        # Second term: TS_RANK(delta(volume,1) / (delay(volume,1) + 1e-8), 6)
        term2_raw = (volume.diff(1)) / (volume.shift(1) + 1e-8)
        term2_rank = term2_raw.rolling(window).rank(pct=True)

        # Raw alpha
        raw_alpha = term1_rank - term2_rank

        # Normalization: Rolling Rank (Case A)
        normalized = (raw_alpha.rolling(sub_window).rank(pct=True) * 2) - 1

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_033_tanh(df, window=40, sub_window=60):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # First term: TS_RANK((open - delay(close,1)) / (delay(close,1) + 1e-8), 6)
        term1_raw = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        term1_rank = term1_raw.rolling(window).rank(pct=True)

        # Second term: TS_RANK(delta(volume,1) / (delay(volume,1) + 1e-8), 6)
        term2_raw = (volume.diff(1)) / (volume.shift(1) + 1e-8)
        term2_rank = term2_raw.rolling(window).rank(pct=True)

        # Raw alpha
        raw_alpha = term1_rank - term2_rank

        # Normalization: Dynamic Tanh (Case B)
        std_dev = raw_alpha.rolling(sub_window).std()
        normalized = pd.Series(np.tanh(raw_alpha / (std_dev + 1e-9)), index=df.index)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_033_zscore(df, window=7, sub_window=10):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # First term: TS_RANK((open - delay(close,1)) / (delay(close,1) + 1e-8), 6)
        term1_raw = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        term1_rank = term1_raw.rolling(window).rank(pct=True)

        # Second term: TS_RANK(delta(volume,1) / (delay(volume,1) + 1e-8), 6)
        term2_raw = (volume.diff(1)) / (volume.shift(1) + 1e-8)
        term2_rank = term2_raw.rolling(window).rank(pct=True)

        # Raw alpha
        raw_alpha = term1_rank - term2_rank

        # Normalization: Rolling Z-Score/Clip (Case C)
        rolling_mean = raw_alpha.rolling(sub_window).mean()
        rolling_std = raw_alpha.rolling(sub_window).std()
        normalized = ((raw_alpha - rolling_mean) / (rolling_std + 1e-9)).clip(-1, 1)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_033_sign(df, window=60):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # First term: TS_RANK((open - delay(close,1)) / (delay(close,1) + 1e-8), 6)
        term1_raw = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        term1_rank = term1_raw.rolling(window).rank(pct=True)

        # Second term: TS_RANK(delta(volume,1) / (delay(volume,1) + 1e-8), 6)
        term2_raw = (volume.diff(1)) / (volume.shift(1) + 1e-8)
        term2_rank = term2_raw.rolling(window).rank(pct=True)

        # Raw alpha
        raw_alpha = term1_rank - term2_rank

        # Normalization: Sign/Binary Soft (Case D)
        normalized = pd.Series(np.sign(raw_alpha), index=df.index)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_033_wf(df, window=3, sub_window=70):
        # Raw calculation
        open_price = df['open']
        close_price = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # First term: TS_RANK((open - delay(close,1)) / (delay(close,1) + 1e-8), 6)
        term1_raw = (open_price - close_price.shift(1)) / (close_price.shift(1) + 1e-8)
        term1_rank = term1_raw.rolling(window).rank(pct=True)

        # Second term: TS_RANK(delta(volume,1) / (delay(volume,1) + 1e-8), 6)
        term2_raw = (volume.diff(1)) / (volume.shift(1) + 1e-8)
        term2_rank = term2_raw.rolling(window).rank(pct=True)

        # Raw alpha
        raw_alpha = term1_rank - term2_rank

        # Normalization: Winsorized Fisher (Case E)
        # Hardcode parameters: p1=0.05, p2=sub_window
        p1 = 0.05
        p2 = sub_window

        low = raw_alpha.rolling(p2).quantile(p1)
        high = raw_alpha.rolling(p2).quantile(1 - p1)
        winsorized = raw_alpha.clip(lower=low, upper=high, axis=0)

        # Fisher Transform
        normalized = pd.Series(np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99), index=df.index)

        # Fill missing values
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_034_rank(df, window=50, sub_window=40):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        prev_volume = volume.shift(1)
        ret_open = (open_price - prev_close) / (prev_close + 1e-8)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (prev_volume - vol_mean) / (vol_std + 1e-8)
        raw = ret_open.rolling(sub_window).corr(vol_z)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.ffill().fillna(0).clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_034_tanh(df, window=80, sub_window=20):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        prev_volume = volume.shift(1)
        ret_open = (open_price - prev_close) / (prev_close + 1e-8)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (prev_volume - vol_mean) / (vol_std + 1e-8)
        raw = ret_open.rolling(sub_window).corr(vol_z)
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        normalized = normalized.ffill().fillna(0).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_034_zscore(df, window=100, sub_window=40):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        prev_volume = volume.shift(1)
        ret_open = (open_price - prev_close) / (prev_close + 1e-8)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (prev_volume - vol_mean) / (vol_std + 1e-8)
        raw = ret_open.rolling(sub_window).corr(vol_z)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_034_sign(df, window=20, sub_window=30):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        prev_volume = volume.shift(1)
        ret_open = (open_price - prev_close) / (prev_close + 1e-8)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (prev_volume - vol_mean) / (vol_std + 1e-8)
        raw = ret_open.rolling(sub_window).corr(vol_z)
        normalized = np.sign(raw)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_034_wf(df, window=50, sub_window=30):
        open_price = df['open']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        prev_close = close.shift(1)
        prev_volume = volume.shift(1)
        ret_open = (open_price - prev_close) / (prev_close + 1e-8)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (prev_volume - vol_mean) / (vol_std + 1e-8)
        raw = ret_open.rolling(sub_window).corr(vol_z)
        p1 = 0.05
        p2 = 100
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_035_rank(df, window=90, sub_window=10):
        # Raw calculation
        volume = df['matchingVolume']
        delayed_volume = volume.shift(1)
        volume_mean = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std()
        raw_volume_part = (delayed_volume - volume_mean) / (volume_std + 1e-8)

        open_price = df['open']
        close_price = df['close']
        delayed_close = close_price.shift(1)
        price_ratio = (open_price - delayed_close) / (delayed_close + 1e-8)
        price_ratio_mean = price_ratio.rolling(sub_window).mean()
        sign_part = np.sign(price_ratio_mean)

        raw = raw_volume_part * sign_part
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.ffill()

        # Case A: Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_035_tanh(df, window=40, sub_window=20):
        # Raw calculation
        volume = df['matchingVolume']
        delayed_volume = volume.shift(1)
        volume_mean = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std()
        raw_volume_part = (delayed_volume - volume_mean) / (volume_std + 1e-8)

        open_price = df['open']
        close_price = df['close']
        delayed_close = close_price.shift(1)
        price_ratio = (open_price - delayed_close) / (delayed_close + 1e-8)
        price_ratio_mean = price_ratio.rolling(sub_window).mean()
        sign_part = np.sign(price_ratio_mean)

        raw = raw_volume_part * sign_part
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.ffill()

        # Case B: Dynamic Tanh
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_035_zscore(df, window=50, sub_window=20):
        # Raw calculation
        volume = df['matchingVolume']
        delayed_volume = volume.shift(1)
        volume_mean = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std()
        raw_volume_part = (delayed_volume - volume_mean) / (volume_std + 1e-8)

        open_price = df['open']
        close_price = df['close']
        delayed_close = close_price.shift(1)
        price_ratio = (open_price - delayed_close) / (delayed_close + 1e-8)
        price_ratio_mean = price_ratio.rolling(sub_window).mean()
        sign_part = np.sign(price_ratio_mean)

        raw = raw_volume_part * sign_part
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.ffill()

        # Case C: Rolling Z-Score/Clip
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_035_sign(df, window=80, sub_window=20):
        # Raw calculation
        volume = df['matchingVolume']
        delayed_volume = volume.shift(1)
        volume_mean = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std()
        raw_volume_part = (delayed_volume - volume_mean) / (volume_std + 1e-8)

        open_price = df['open']
        close_price = df['close']
        delayed_close = close_price.shift(1)
        price_ratio = (open_price - delayed_close) / (delayed_close + 1e-8)
        price_ratio_mean = price_ratio.rolling(sub_window).mean()
        sign_part = np.sign(price_ratio_mean)

        raw = raw_volume_part * sign_part
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.ffill()

        # Case D: Sign/Binary Soft
        normalized = np.sign(raw)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_035_wf(df, window=70, sub_window=20):
        # Hardcoded parameters for Case E
        p1 = 0.05
        p2 = 30

        # Raw calculation
        volume = df['matchingVolume']
        delayed_volume = volume.shift(1)
        volume_mean = volume.rolling(window).mean()
        volume_std = volume.rolling(window).std()
        raw_volume_part = (delayed_volume - volume_mean) / (volume_std + 1e-8)

        open_price = df['open']
        close_price = df['close']
        delayed_close = close_price.shift(1)
        price_ratio = (open_price - delayed_close) / (delayed_close + 1e-8)
        price_ratio_mean = price_ratio.rolling(sub_window).mean()
        sign_part = np.sign(price_ratio_mean)

        raw = raw_volume_part * sign_part
        raw = raw.replace([np.inf, -np.inf], np.nan)
        raw = raw.ffill()

        # Case E: Winsorized Fisher
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.clip(-1, 1)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_036_rank(df, window=100, rank_window=3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        ts_rank = volume.shift(1).rolling(rank_window).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        raw = ts_rank * raw
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_036_tanh(df, window=90, rank_window=3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        ts_rank = volume.shift(1).rolling(rank_window).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        raw = ts_rank * raw
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_036_zscore(df, window=80, rank_window=1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        ts_rank = volume.shift(1).rolling(rank_window).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        raw = ts_rank * raw
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_036_sign(df, rank_window=65):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        ts_rank = volume.shift(1).rolling(rank_window).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        raw = ts_rank * raw
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_036_wf(df, window=100, rank_window=3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        ts_rank = volume.shift(1).rolling(rank_window).apply(lambda s: s.rank(pct=True).iloc[-1], raw=False)
        raw = ts_rank * raw
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_037_rank(df, window=80, sub_window=1):
        # Raw components
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close = df['close']
        # Component 1: (DELAY(volume,1) - TS_MEAN(volume,25)) / (TS_STD(volume,25)+1e-8)
        delayed_vol = volume.shift(1)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        comp1 = (delayed_vol - vol_mean) / (vol_std + 1e-8)
        # Component 2: TS_MEAN((open - DELAY(close,1)) / (DELAY(close,1)+1e-8), 3)
        delayed_close = close.shift(1)
        price_ratio = (open_ - delayed_close) / (delayed_close + 1e-8)
        comp2 = price_ratio.rolling(sub_window).mean()
        # TS_CORR over window
        raw = comp1.rolling(window).corr(comp2)
        # Rolling Rank normalization (Case A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_037_tanh(df, window=20, sub_window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close = df['close']
        delayed_vol = volume.shift(1)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        comp1 = (delayed_vol - vol_mean) / (vol_std + 1e-8)
        delayed_close = close.shift(1)
        price_ratio = (open_ - delayed_close) / (delayed_close + 1e-8)
        comp2 = price_ratio.rolling(sub_window).mean()
        raw = comp1.rolling(window).corr(comp2)
        # Dynamic Tanh normalization (Case B)
        rolling_std = raw.rolling(window).std()
        normalized = np.tanh(raw / (rolling_std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_037_zscore(df, window=20, sub_window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close = df['close']
        delayed_vol = volume.shift(1)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        comp1 = (delayed_vol - vol_mean) / (vol_std + 1e-8)
        delayed_close = close.shift(1)
        price_ratio = (open_ - delayed_close) / (delayed_close + 1e-8)
        comp2 = price_ratio.rolling(sub_window).mean()
        raw = comp1.rolling(window).corr(comp2)
        # Rolling Z-Score/Clip normalization (Case C)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_037_sign(df, window=50, sub_window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close = df['close']
        delayed_vol = volume.shift(1)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        comp1 = (delayed_vol - vol_mean) / (vol_std + 1e-8)
        delayed_close = close.shift(1)
        price_ratio = (open_ - delayed_close) / (delayed_close + 1e-8)
        comp2 = price_ratio.rolling(sub_window).mean()
        raw = comp1.rolling(window).corr(comp2)
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_037_wf(df, window=20, sub_window=10):
        # Hardcode Winsorization parameters: p1=0.05, p2=window
        p1 = 0.05
        p2 = window
        volume = df.get('matchingVolume', df.get('volume', 1))
        open_ = df['open']
        close = df['close']
        delayed_vol = volume.shift(1)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        comp1 = (delayed_vol - vol_mean) / (vol_std + 1e-8)
        delayed_close = close.shift(1)
        price_ratio = (open_ - delayed_close) / (delayed_close + 1e-8)
        comp2 = price_ratio.rolling(sub_window).mean()
        raw = comp1.rolling(window).corr(comp2)
        # Winsorized Fisher normalization (Case E)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_038_rank(df, window=40, sub_window=5):
        # Raw calculation
        volume = df.get('matchingVolume', df.get('volume', 1))
        delayed_volume = volume.shift(1)
        zscore_vol = (delayed_volume - delayed_volume.rolling(window).mean()) / delayed_volume.rolling(window).std().replace(0, np.nan)
        price_ratio = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw = zscore_vol * price_ratio
        # Rolling Rank normalization (Case A)
        normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_038_tanh(df, window=20, sub_window=30):
        # Raw calculation
        volume = df.get('matchingVolume', df.get('volume', 1))
        delayed_volume = volume.shift(1)
        zscore_vol = (delayed_volume - delayed_volume.rolling(window).mean()) / delayed_volume.rolling(window).std().replace(0, np.nan)
        price_ratio = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw = zscore_vol * price_ratio
        # Dynamic Tanh normalization (Case B)
        normalized = np.tanh(raw / raw.rolling(sub_window).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_038_zscore(df, window=80, sub_window=40):
        # Raw calculation
        volume = df.get('matchingVolume', df.get('volume', 1))
        delayed_volume = volume.shift(1)
        zscore_vol = (delayed_volume - delayed_volume.rolling(window).mean()) / delayed_volume.rolling(window).std().replace(0, np.nan)
        price_ratio = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw = zscore_vol * price_ratio
        # Rolling Z-Score/Clip normalization (Case C)
        normalized = ((raw - raw.rolling(sub_window).mean()) / raw.rolling(sub_window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_038_sign(df, window=75):
        # Raw calculation
        volume = df.get('matchingVolume', df.get('volume', 1))
        delayed_volume = volume.shift(1)
        zscore_vol = (delayed_volume - delayed_volume.rolling(window).mean()) / delayed_volume.rolling(window).std().replace(0, np.nan)
        price_ratio = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw = zscore_vol * price_ratio
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_038_wf(df, window=40, sub_window=30):
        # Raw calculation
        volume = df.get('matchingVolume', df.get('volume', 1))
        delayed_volume = volume.shift(1)
        zscore_vol = (delayed_volume - delayed_volume.rolling(window).mean()) / delayed_volume.rolling(window).std().replace(0, np.nan)
        price_ratio = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw = zscore_vol * price_ratio
        # Winsorized Fisher normalization (Case E)
        p1 = 0.05  # Hardcoded quantile
        p2 = sub_window  # Rolling window for winsorization
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_039_rank(df, window=30, rank_window=80):
        # Raw: Rank of rolling mean of open-to-prev-close relative change
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        # Normalization A: Rolling Rank
        signal = (raw_mean.rolling(window=rank_window, min_periods=1).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_039_tanh(df, window=30, scale_window=80):
        # Raw: Rolling mean of open-to-prev-close relative change
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        # Normalization B: Dynamic Tanh
        rolling_std = raw_mean.rolling(window=scale_window, min_periods=1).std()
        signal = np.tanh(raw_mean / (rolling_std + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_039_zscore(df, window=30, z_window=90):
        # Raw: Rolling mean of open-to-prev-close relative change
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        # Normalization C: Rolling Z-Score/Clip
        rolling_mean = raw_mean.rolling(window=z_window, min_periods=1).mean()
        rolling_std = raw_mean.rolling(window=z_window, min_periods=1).std()
        signal = ((raw_mean - rolling_mean) / (rolling_std + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_039_sign(df, window=20):
        # Raw: Rolling mean of open-to-prev-close relative change
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        # Normalization D: Sign/Binary Soft
        signal = np.sign(raw_mean)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_039_wf(df, window=30, winsor_window=90, quantile=0.05):
        # Raw: Rolling mean of open-to-prev-close relative change
        raw = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        raw_mean = raw.rolling(window=window, min_periods=1).mean()
        # Normalization E: Winsorized Fisher
        low = raw_mean.rolling(window=winsor_window, min_periods=1).quantile(quantile)
        high = raw_mean.rolling(window=winsor_window, min_periods=1).quantile(1 - quantile)
        winsorized = raw_mean.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_040_rank(df, window=85):
        raw = (df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw_rank = raw.rolling(window).rank(pct=True)
        signal = (raw_rank * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_040_tanh(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        std = raw.rolling(window).std()
        signal = np.tanh(raw / (std + 1e-9))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_040_zscore(df, window=50):
        raw = (df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        z = (raw - mean) / (std + 1e-9)
        signal = z.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_040_sign(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_040_wf(df, window=50, quantile=0.3):
        raw = (df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        low_bound = raw.rolling(window).quantile(quantile)
        high_bound = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_bound, upper=high_bound)
        scaled = ((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99
        signal = np.arctanh(scaled.clip(-0.99, 0.99))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_041_rank(df, window=60, corr_window=20):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))

        # Tính ($open - DELAY($close, 1)) / (DELAY($close, 1) + 1e-8)
        delayed_close = close.shift(1)
        numerator = open_price - delayed_close
        denominator = delayed_close + 1e-8
        raw_ratio = numerator / denominator

        # TS_MEAN với window
        ts_mean_ratio = raw_ratio.rolling(window).mean()

        # RANK của TS_MEAN
        rank_ts_mean = ts_mean_ratio.rolling(window).rank(pct=True)

        # Tính TS_CORR($high - $low, $volume, corr_window)
        high_low_spread = high - low
        ts_corr = high_low_spread.rolling(corr_window).corr(volume)

        # Kết hợp: RANK * TS_CORR
        raw_signal = rank_ts_mean * ts_corr

        # Chuẩn hóa theo TRƯỜNG HỢP A (Rolling Rank): Loại bỏ nhiễu/outliers
        normalized = (raw_signal.rolling(window).rank(pct=True) * 2) - 1

        # Xử lý NaN: forward fill cho time-series, fillna 0 ở cuối
        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_041_tanh(df, window=10, corr_window=5):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))

        delayed_close = close.shift(1)
        numerator = open_price - delayed_close
        denominator = delayed_close + 1e-8
        raw_ratio = numerator / denominator

        ts_mean_ratio = raw_ratio.rolling(window).mean()
        rank_ts_mean = ts_mean_ratio.rolling(window).rank(pct=True)

        high_low_spread = high - low
        ts_corr = high_low_spread.rolling(corr_window).corr(volume)

        raw_signal = rank_ts_mean * ts_corr

        # Chuẩn hóa theo TRƯỜNG HỢP B (Dynamic Tanh): Giữ lại cường độ
        rolling_std = raw_signal.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw_signal / rolling_std)

        normalized = normalized.ffill().fillna(0)

        return -normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_041_zscore(df, window=60, corr_window=10):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))

        delayed_close = close.shift(1)
        numerator = open_price - delayed_close
        denominator = delayed_close + 1e-8
        raw_ratio = numerator / denominator

        ts_mean_ratio = raw_ratio.rolling(window).mean()
        rank_ts_mean = ts_mean_ratio.rolling(window).rank(pct=True)

        high_low_spread = high - low
        ts_corr = high_low_spread.rolling(corr_window).corr(volume)

        raw_signal = rank_ts_mean * ts_corr

        # Chuẩn hóa theo TRƯỜNG HỢP C (Rolling Z-Score/Clip): Phù hợp cho Spread/Basis/Oscillator
        rolling_mean = raw_signal.rolling(window).mean()
        rolling_std = raw_signal.rolling(window).std().replace(0, np.nan)
        normalized = ((raw_signal - rolling_mean) / rolling_std).clip(-1, 1)

        normalized = normalized.ffill().fillna(0)

        return normalized

    @staticmethod
    def alpha_quanta_041_sign(df, window=60, corr_window=3):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))

        delayed_close = close.shift(1)
        numerator = open_price - delayed_close
        denominator = delayed_close + 1e-8
        raw_ratio = numerator / denominator

        ts_mean_ratio = raw_ratio.rolling(window).mean()
        rank_ts_mean = ts_mean_ratio.rolling(window).rank(pct=True)

        high_low_spread = high - low
        ts_corr = high_low_spread.rolling(corr_window).corr(volume)

        raw_signal = rank_ts_mean * ts_corr

        # Chuẩn hóa theo TRƯỜNG HỢP D (Sign/Binary Soft): Breakout/Trend Following
        normalized = np.sign(raw_signal)

        normalized = normalized.ffill().fillna(0)

        return -normalized

    @staticmethod
    def alpha_quanta_041_wf(df, window=60, corr_window=40):
        open_price = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))

        delayed_close = close.shift(1)
        numerator = open_price - delayed_close
        denominator = delayed_close + 1e-8
        raw_ratio = numerator / denominator

        ts_mean_ratio = raw_ratio.rolling(window).mean()
        rank_ts_mean = ts_mean_ratio.rolling(window).rank(pct=True)

        high_low_spread = high - low
        ts_corr = high_low_spread.rolling(corr_window).corr(volume)

        raw_signal = rank_ts_mean * ts_corr

        # Chuẩn hóa theo TRƯỜNG HỢP E (Winsorized Fisher): Xử lý heavy tails
        p1 = 0.05  # Hardcode quantile threshold
        p2 = window  # Hardcode rolling window
        low_bound = raw_signal.rolling(p2).quantile(p1)
        high_bound = raw_signal.rolling(p2).quantile(1 - p1)
        winsorized = raw_signal.clip(lower=low_bound, upper=high_bound, axis=0)

        # Fisher Transform
        scale = (winsorized - low_bound) / (high_bound - low_bound + 1e-9)
        scaled = scale * 1.98 - 0.99
        normalized = np.arctanh(scaled.clip(-0.99, 0.99))

        normalized = normalized.ffill().fillna(0)

        return normalized.clip(-1, 1)

    @staticmethod
    def alpha_quanta_042_rank(df, window=80, sub_window=30):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        base = (open_price - delay_close) / (delay_close + 1e-8)
        numerator = base.rolling(sub_window).mean()
        denominator = base.rolling(window).mean() + base.rolling(window).std()
        raw = numerator / (denominator + 1e-9)
        raw_ffilled = raw.ffill()
        normalized = (raw_ffilled.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_042_tanh(df, window=100, sub_window=20):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        base = (open_price - delay_close) / (delay_close + 1e-8)
        numerator = base.rolling(sub_window).mean()
        denominator = base.rolling(window).mean() + base.rolling(window).std()
        raw = numerator / (denominator + 1e-9)
        raw_ffilled = raw.ffill()
        std = raw_ffilled.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw_ffilled / (std + 1e-9))
        return normalized.fillna(0).clip(-1, 1)