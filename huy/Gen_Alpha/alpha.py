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

    @staticmethod
    def alpha_quanta_042_zscore(df, window=100, sub_window=30):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        base = (open_price - delay_close) / (delay_close + 1e-8)
        numerator = base.rolling(sub_window).mean()
        denominator = base.rolling(window).mean() + base.rolling(window).std()
        raw = numerator / (denominator + 1e-9)
        raw_ffilled = raw.ffill()
        mean = raw_ffilled.rolling(window).mean()
        std = raw_ffilled.rolling(window).std().replace(0, np.nan)
        zscore = (raw_ffilled - mean) / (std + 1e-9)
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_042_sign(df, window=60, sub_window=20):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        base = (open_price - delay_close) / (delay_close + 1e-8)
        numerator = base.rolling(sub_window).mean()
        denominator = base.rolling(window).mean() + base.rolling(window).std()
        raw = numerator / (denominator + 1e-9)
        raw_ffilled = raw.ffill()
        normalized = np.sign(raw_ffilled)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_042_wf(df, window=90, sub_window=30):
        open_price = df['open']
        close_price = df['close']
        delay_close = close_price.shift(1)
        base = (open_price - delay_close) / (delay_close + 1e-8)
        numerator = base.rolling(sub_window).mean()
        denominator = base.rolling(window).mean() + base.rolling(window).std()
        raw = numerator / (denominator + 1e-9)
        raw_ffilled = raw.ffill()
        p1 = 0.05
        p2 = window
        low = raw_ffilled.rolling(p2).quantile(p1)
        high = raw_ffilled.rolling(p2).quantile(1 - p1)
        winsorized = raw_ffilled.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_043_rank(df, window=90, sub_window=40):
        # Raw alpha calculation
        close_delay = df['close'].shift(1)
        price_ratio = (df['open'] - close_delay) / (close_delay + 1e-8)
        ts_mean_price = price_ratio.rolling(window).mean()
        rank_price = ts_mean_price.rolling(sub_window).rank(pct=True)

        ts_mean_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        zscore_volume = (ts_mean_volume - ts_mean_volume.rolling(sub_window).mean()) / ts_mean_volume.rolling(sub_window).std()

        raw = rank_price * zscore_volume
        raw = raw.fillna(0)

        # Case A: Rolling Rank normalization
        param = max(20, window, sub_window)
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_043_tanh(df, window=60, sub_window=30):
        # Raw alpha calculation
        close_delay = df['close'].shift(1)
        price_ratio = (df['open'] - close_delay) / (close_delay + 1e-8)
        ts_mean_price = price_ratio.rolling(window).mean()
        rank_price = ts_mean_price.rolling(sub_window).rank(pct=True)

        ts_mean_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        zscore_volume = (ts_mean_volume - ts_mean_volume.rolling(sub_window).mean()) / ts_mean_volume.rolling(sub_window).std()

        raw = rank_price * zscore_volume
        raw = raw.fillna(0)

        # Case B: Dynamic Tanh normalization
        param = max(20, window, sub_window)
        std_dev = raw.rolling(param).std().replace(0, 1e-9)
        normalized = np.tanh(raw / std_dev)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_043_zscore(df, window=40, sub_window=40):
        # Raw alpha calculation
        close_delay = df['close'].shift(1)
        price_ratio = (df['open'] - close_delay) / (close_delay + 1e-8)
        ts_mean_price = price_ratio.rolling(window).mean()
        rank_price = ts_mean_price.rolling(sub_window).rank(pct=True)

        ts_mean_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        zscore_volume = (ts_mean_volume - ts_mean_volume.rolling(sub_window).mean()) / ts_mean_volume.rolling(sub_window).std()

        raw = rank_price * zscore_volume
        raw = raw.fillna(0)

        # Case C: Rolling Z-Score/Clip normalization
        param = max(20, window, sub_window)
        rolling_mean = raw.rolling(param).mean()
        rolling_std = raw.rolling(param).std().replace(0, 1e-9)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_043_sign(df, window=60, sub_window=10):
        # Raw alpha calculation
        close_delay = df['close'].shift(1)
        price_ratio = (df['open'] - close_delay) / (close_delay + 1e-8)
        ts_mean_price = price_ratio.rolling(window).mean()
        rank_price = ts_mean_price.rolling(sub_window).rank(pct=True)

        ts_mean_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        zscore_volume = (ts_mean_volume - ts_mean_volume.rolling(sub_window).mean()) / ts_mean_volume.rolling(sub_window).std()

        raw = rank_price * zscore_volume
        raw = raw.fillna(0)

        # Case D: Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_043_wf(df, window=60, sub_window=40):
        # Raw alpha calculation
        close_delay = df['close'].shift(1)
        price_ratio = (df['open'] - close_delay) / (close_delay + 1e-8)
        ts_mean_price = price_ratio.rolling(window).mean()
        rank_price = ts_mean_price.rolling(sub_window).rank(pct=True)

        ts_mean_volume = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        zscore_volume = (ts_mean_volume - ts_mean_volume.rolling(sub_window).mean()) / ts_mean_volume.rolling(sub_window).std()

        raw = rank_price * zscore_volume
        raw = raw.fillna(0)

        # Case E: Winsorized Fisher normalization
        p1 = 0.05  # Hardcoded quantile parameter
        p2 = max(50, window, sub_window)  # Hardcoded rolling window

        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)

        # Fisher Transform approximation
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_044_rank(df, window=80, factor=0.3):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None)
        corr_window = int(window * factor) if int(window * factor) > 1 else 5
        corr = returns.rolling(corr_window).corr(volume)
        delay_corr = corr.shift(1)
        raw = ((open_price - close.shift(1)) / (close.shift(1) + 1e-8)) * (delay_corr ** 2)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_044_tanh(df, window=90, factor=0.7):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None)
        corr_window = int(window * factor) if int(window * factor) > 1 else 5
        corr = returns.rolling(corr_window).corr(volume)
        delay_corr = corr.shift(1)
        raw = ((open_price - close.shift(1)) / (close.shift(1) + 1e-8)) * (delay_corr ** 2)
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_044_zscore(df, window=60, factor=0.1):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None)
        corr_window = int(window * factor) if int(window * factor) > 1 else 5
        corr = returns.rolling(corr_window).corr(volume)
        delay_corr = corr.shift(1)
        raw = ((open_price - close.shift(1)) / (close.shift(1) + 1e-8)) * (delay_corr ** 2)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_044_sign(df, window=60, factor=0.9):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None)
        corr_window = int(window * factor) if int(window * factor) > 1 else 5
        corr = returns.rolling(corr_window).corr(volume)
        delay_corr = corr.shift(1)
        raw = ((open_price - close.shift(1)) / (close.shift(1) + 1e-8)) * (delay_corr ** 2)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_044_wf(df, window=90, factor=0.5):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None)
        corr_window = int(window * factor) if int(window * factor) > 1 else 5
        corr = returns.rolling(corr_window).corr(volume)
        delay_corr = corr.shift(1)
        raw = ((open_price - close.shift(1)) / (close.shift(1) + 1e-8)) * (delay_corr ** 2)
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_045_rank(df, window=100):
        # Raw calculation
        close_shift = df['close'].shift(1)
        price_change_ratio = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl_spread = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl_spread.rolling(window=20).corr(volume)
        corr_shift = corr_hl_vol.shift(1)
        pow_factor = corr_shift ** 3
        raw = price_change_ratio * pow_factor
        # Rolling Rank normalization (Case A)
        normalized = (raw.rolling(window=window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_045_tanh(df, window=100):
        # Raw calculation
        close_shift = df['close'].shift(1)
        price_change_ratio = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl_spread = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl_spread.rolling(window=20).corr(volume)
        corr_shift = corr_hl_vol.shift(1)
        pow_factor = corr_shift ** 3
        raw = price_change_ratio * pow_factor
        # Dynamic Tanh normalization (Case B)
        rolling_std = raw.rolling(window=window).std()
        normalized = np.tanh(raw / rolling_std.replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_045_zscore(df, window=95):
        # Raw calculation
        close_shift = df['close'].shift(1)
        price_change_ratio = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl_spread = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl_spread.rolling(window=20).corr(volume)
        corr_shift = corr_hl_vol.shift(1)
        pow_factor = corr_shift ** 3
        raw = price_change_ratio * pow_factor
        # Rolling Z-Score/Clip normalization (Case C)
        rolling_mean = raw.rolling(window=window).mean()
        rolling_std = raw.rolling(window=window).std()
        normalized = ((raw - rolling_mean) / rolling_std.replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_045_sign(df):
        # Raw calculation
        close_shift = df['close'].shift(1)
        price_change_ratio = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl_spread = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl_spread.rolling(window=20).corr(volume)
        corr_shift = corr_hl_vol.shift(1)
        pow_factor = corr_shift ** 3
        raw = price_change_ratio * pow_factor
        # Sign/Binary Soft normalization (Case D)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_045_wf(df, window=100, quantile=0.1):
        # Raw calculation
        close_shift = df['close'].shift(1)
        price_change_ratio = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl_spread = df['high'] - df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl_spread.rolling(window=20).corr(volume)
        corr_shift = corr_hl_vol.shift(1)
        pow_factor = corr_shift ** 3
        raw = price_change_ratio * pow_factor
        # Winsorized Fisher normalization (Case E)
        low_bound = raw.rolling(window=window).quantile(quantile)
        high_bound = raw.rolling(window=window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_bound, upper=high_bound)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_046_rank(df, window=50):
        # Raw calculation
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        sign_term = np.sign((open_ - close.shift(1)) / (close.shift(1) + 1e-8))
        corr_pos = returns.rolling(window).corr(volume)
        corr_neg = (-returns).rolling(window).corr(volume)
        diff_corr = corr_pos.shift(1) - corr_neg.shift(1)
        raw = sign_term * diff_corr
        # Normalization A: Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_046_tanh(df, window=60):
        # Raw calculation
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        sign_term = np.sign((open_ - close.shift(1)) / (close.shift(1) + 1e-8))
        corr_pos = returns.rolling(window).corr(volume)
        corr_neg = (-returns).rolling(window).corr(volume)
        diff_corr = corr_pos.shift(1) - corr_neg.shift(1)
        raw = sign_term * diff_corr
        # Normalization B: Dynamic Tanh
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_046_zscore(df, window=50):
        # Raw calculation
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        sign_term = np.sign((open_ - close.shift(1)) / (close.shift(1) + 1e-8))
        corr_pos = returns.rolling(window).corr(volume)
        corr_neg = (-returns).rolling(window).corr(volume)
        diff_corr = corr_pos.shift(1) - corr_neg.shift(1)
        raw = sign_term * diff_corr
        # Normalization C: Rolling Z-Score/Clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_046_sign(df, window=40):
        # Raw calculation
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        sign_term = np.sign((open_ - close.shift(1)) / (close.shift(1) + 1e-8))
        corr_pos = returns.rolling(window).corr(volume)
        corr_neg = (-returns).rolling(window).corr(volume)
        diff_corr = corr_pos.shift(1) - corr_neg.shift(1)
        raw = sign_term * diff_corr
        # Normalization D: Sign/Binary Soft
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_046_wf(df, window=60):
        # Raw calculation
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        returns = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
        sign_term = np.sign((open_ - close.shift(1)) / (close.shift(1) + 1e-8))
        corr_pos = returns.rolling(window).corr(volume)
        corr_neg = (-returns).rolling(window).corr(volume)
        diff_corr = corr_pos.shift(1) - corr_neg.shift(1)
        raw = sign_term * diff_corr
        # Normalization E: Winsorized Fisher
        p1 = 0.05  # Hardcoded quantile threshold
        p2 = window * 2  # Hardcoded rolling window for winsorization
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_047_rank(df, window=80):
        # Raw calculation
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_adj = np.log1p(vol) if vol.skew() > 5 else vol
        std_ret_vol = (ret / (vol_adj + 1e-9)).rolling(window).std()
        raw = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * std_ret_vol.shift(1)
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_047_tanh(df, window=95):
        # Raw calculation
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_adj = np.log1p(vol) if vol.skew() > 5 else vol
        std_ret_vol = (ret / (vol_adj + 1e-9)).rolling(window).std()
        raw = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * std_ret_vol.shift(1)
        # Dynamic Tanh normalization
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_047_zscore(df, window=85):
        # Raw calculation
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_adj = np.log1p(vol) if vol.skew() > 5 else vol
        std_ret_vol = (ret / (vol_adj + 1e-9)).rolling(window).std()
        raw = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * std_ret_vol.shift(1)
        # Rolling Z-Score/Clip normalization
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / (rolling_std + 1e-9)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_047_sign(df, window=80):
        # Raw calculation
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_adj = np.log1p(vol) if vol.skew() > 5 else vol
        std_ret_vol = (ret / (vol_adj + 1e-9)).rolling(window).std()
        raw = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * std_ret_vol.shift(1)
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_047_wf(df, window=100, factor=0.9):
        # Raw calculation
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_adj = np.log1p(vol) if vol.skew() > 5 else vol
        std_ret_vol = (ret / (vol_adj + 1e-9)).rolling(window).std()
        raw = ((df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)) * std_ret_vol.shift(1)
        # Winsorized Fisher normalization
        p1 = factor
        p2 = int(window * 1.5)  # Hardcoded third parameter
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_048_rank(df, window=100, factor=0.9):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_ret = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl = df['high'] - df['low']
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl.rolling(window).corr(vol)
        corr_ret_vol = ret.rolling(window).corr(vol)
        mean_ret_vol = (ret / (vol + 1)).rolling(window).mean()
        weighted = 0.4 * corr_hl_vol.shift(1) + 0.3 * corr_ret_vol.shift(1) + 0.3 * mean_ret_vol.shift(1)
        raw = open_ret * weighted
        # Case A: Rolling Rank
        normalized = (raw.rolling(int(window * factor)).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_048_tanh(df, window=50, factor=0.1):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_ret = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl = df['high'] - df['low']
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl.rolling(window).corr(vol)
        corr_ret_vol = ret.rolling(window).corr(vol)
        mean_ret_vol = (ret / (vol + 1)).rolling(window).mean()
        weighted = 0.4 * corr_hl_vol.shift(1) + 0.3 * corr_ret_vol.shift(1) + 0.3 * mean_ret_vol.shift(1)
        raw = open_ret * weighted
        # Case B: Dynamic Tanh
        normalized = np.tanh(raw / raw.rolling(int(window * factor)).std())
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_048_zscore(df, window=40, factor=0.3):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_ret = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl = df['high'] - df['low']
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl.rolling(window).corr(vol)
        corr_ret_vol = ret.rolling(window).corr(vol)
        mean_ret_vol = (ret / (vol + 1)).rolling(window).mean()
        weighted = 0.4 * corr_hl_vol.shift(1) + 0.3 * corr_ret_vol.shift(1) + 0.3 * mean_ret_vol.shift(1)
        raw = open_ret * weighted
        # Case C: Rolling Z-Score/Clip
        roll_mean = raw.rolling(int(window * factor)).mean()
        roll_std = raw.rolling(int(window * factor)).std()
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_048_sign(df, window=80):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_ret = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl = df['high'] - df['low']
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl.rolling(window).corr(vol)
        corr_ret_vol = ret.rolling(window).corr(vol)
        mean_ret_vol = (ret / (vol + 1)).rolling(window).mean()
        weighted = 0.4 * corr_hl_vol.shift(1) + 0.3 * corr_ret_vol.shift(1) + 0.3 * mean_ret_vol.shift(1)
        raw = open_ret * weighted
        # Case D: Sign/Binary Soft
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_048_wf(df, window=100, factor=0.9):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_ret = (df['open'] - close_shift) / (close_shift + 1e-8)
        hl = df['high'] - df['low']
        ret = df['close'].pct_change()
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_hl_vol = hl.rolling(window).corr(vol)
        corr_ret_vol = ret.rolling(window).corr(vol)
        mean_ret_vol = (ret / (vol + 1)).rolling(window).mean()
        weighted = 0.4 * corr_hl_vol.shift(1) + 0.3 * corr_ret_vol.shift(1) + 0.3 * mean_ret_vol.shift(1)
        raw = open_ret * weighted
        # Case E: Winsorized Fisher
        p1 = 0.1  # hardcoded quantile
        p2 = int(window * factor)  # rolling window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_049_rank(df, window=80, sub_window=5):
        # Raw calculation
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low

        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()

        # Stochastics-like component
        stoch = (close - min_low) / (max_high - min_low + 1e-8)

        # Volatility signal
        std_hl = hl_range.rolling(window).std()
        delta_std = std_hl.diff(1)
        sign_vol = np.sign(delta_std)

        raw = stoch * sign_vol

        # Normalization A: Rolling Rank
        normalized = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_049_tanh(df, window=30, sub_window=7):
        # Raw calculation
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low

        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()

        stoch = (close - min_low) / (max_high - min_low + 1e-8)

        std_hl = hl_range.rolling(window).std()
        delta_std = std_hl.diff(1)
        sign_vol = np.sign(delta_std)

        raw = stoch * sign_vol

        # Normalization B: Dynamic Tanh
        rolling_std = raw.rolling(sub_window).std().replace(0, np.nan)
        normalized = np.tanh(raw / rolling_std)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_049_zscore(df, window=80, sub_window=3):
        # Raw calculation
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low

        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()

        stoch = (close - min_low) / (max_high - min_low + 1e-8)

        std_hl = hl_range.rolling(window).std()
        delta_std = std_hl.diff(1)
        sign_vol = np.sign(delta_std)

        raw = stoch * sign_vol

        # Normalization C: Rolling Z-Score/Clip
        rolling_mean = raw.rolling(sub_window).mean()
        rolling_std = raw.rolling(sub_window).std().replace(0, np.nan)
        zscore = (raw - rolling_mean) / rolling_std
        normalized = zscore.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_049_sign(df, window=40):
        # Raw calculation
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low

        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()

        stoch = (close - min_low) / (max_high - min_low + 1e-8)

        std_hl = hl_range.rolling(window).std()
        delta_std = std_hl.diff(1)
        sign_vol = np.sign(delta_std)

        raw = stoch * sign_vol

        # Normalization D: Sign/Binary Soft
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_049_wf(df, window=90):
        # Hardcoded parameters for Winsorized Fisher
        p1 = 0.05  # quantile threshold
        p2 = 100   # rolling window for quantile

        # Raw calculation
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low

        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()

        stoch = (close - min_low) / (max_high - min_low + 1e-8)

        std_hl = hl_range.rolling(window).std()
        delta_std = std_hl.diff(1)
        sign_vol = np.sign(delta_std)

        raw = stoch * sign_vol

        # Normalization E: Winsorized Fisher
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)

        # Fisher Transform approximation
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_050_rank(df, window=25):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_high = high.rolling(window).mean()
        mean_low = low.rolling(window).mean()
        cond_up = close > mean_high
        cond_down = close < mean_low
        sum_up = (volume * cond_up).rolling(window).sum()
        sum_down = (volume * cond_down).rolling(window).sum()
        ratio = sum_up / (sum_down + 1)
        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()
        pos = (close - min_low) / (max_high - min_low + 1e-8)
        raw = ratio * pos
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.ffill().fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_050_tanh(df, window=75):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_high = high.rolling(window).mean()
        mean_low = low.rolling(window).mean()
        cond_up = close > mean_high
        cond_down = close < mean_low
        sum_up = (volume * cond_up).rolling(window).sum()
        sum_down = (volume * cond_down).rolling(window).sum()
        ratio = sum_up / (sum_down + 1)
        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()
        pos = (close - min_low) / (max_high - min_low + 1e-8)
        raw = ratio * pos
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.ffill().fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_050_zscore(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_high = high.rolling(window).mean()
        mean_low = low.rolling(window).mean()
        cond_up = close > mean_high
        cond_down = close < mean_low
        sum_up = (volume * cond_up).rolling(window).sum()
        sum_down = (volume * cond_down).rolling(window).sum()
        ratio = sum_up / (sum_down + 1)
        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()
        pos = (close - min_low) / (max_high - min_low + 1e-8)
        raw = ratio * pos
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        normalized = z.clip(-1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_050_sign(df, window=5):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_high = high.rolling(window).mean()
        mean_low = low.rolling(window).mean()
        cond_up = close > mean_high
        cond_down = close < mean_low
        sum_up = (volume * cond_up).rolling(window).sum()
        sum_down = (volume * cond_down).rolling(window).sum()
        ratio = sum_up / (sum_down + 1)
        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()
        pos = (close - min_low) / (max_high - min_low + 1e-8)
        raw = ratio * pos
        normalized = np.sign(raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_050_wf(df, window=40, quantile=0.9):
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_high = high.rolling(window).mean()
        mean_low = low.rolling(window).mean()
        cond_up = close > mean_high
        cond_down = close < mean_low
        sum_up = (volume * cond_up).rolling(window).sum()
        sum_down = (volume * cond_down).rolling(window).sum()
        ratio = sum_up / (sum_down + 1)
        min_low = low.rolling(window).min()
        max_high = high.rolling(window).max()
        pos = (close - min_low) / (max_high - min_low + 1e-8)
        raw = ratio * pos
        low_bound = raw.rolling(window).quantile(quantile)
        high_bound = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_051_rank(df, window=90, delta=5):
        close = df['close']
        low = df['low']
        high = df['high']
        hl_mean = (high - low).rolling(window).mean()
        numerator = close - low.rolling(window).mean()
        denominator = high.rolling(window).max() - low.rolling(window).mean() + 1e-8
        raw = (numerator / denominator) * hl_mean.pct_change(delta)
        raw = raw.fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_051_tanh(df, window=30, delta=20):
        close = df['close']
        low = df['low']
        high = df['high']
        hl_mean = (high - low).rolling(window).mean()
        numerator = close - low.rolling(window).mean()
        denominator = high.rolling(window).max() - low.rolling(window).mean() + 1e-8
        raw = (numerator / denominator) * hl_mean.pct_change(delta)
        raw = raw.fillna(0)
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std).fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_051_zscore(df, window=80, delta=20):
        close = df['close']
        low = df['low']
        high = df['high']
        hl_mean = (high - low).rolling(window).mean()
        numerator = close - low.rolling(window).mean()
        denominator = high.rolling(window).max() - low.rolling(window).mean() + 1e-8
        raw = (numerator / denominator) * hl_mean.pct_change(delta)
        raw = raw.fillna(0)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        zscore = ((raw - mean) / std).fillna(0)
        normalized = zscore.clip(-1, 1)
        return -normalized

    @staticmethod
    def alpha_quanta_051_sign(df, window=60, delta=30):
        close = df['close']
        low = df['low']
        high = df['high']
        hl_mean = (high - low).rolling(window).mean()
        numerator = close - low.rolling(window).mean()
        denominator = high.rolling(window).max() - low.rolling(window).mean() + 1e-8
        raw = (numerator / denominator) * hl_mean.pct_change(delta)
        raw = raw.fillna(0)
        normalized = np.sign(raw)
        return -normalized

    @staticmethod
    def alpha_quanta_051_wf(df, window=70, delta=30):
        close = df['close']
        low = df['low']
        high = df['high']
        hl_mean = (high - low).rolling(window).mean()
        numerator = close - low.rolling(window).mean()
        denominator = high.rolling(window).max() - low.rolling(window).mean() + 1e-8
        raw = (numerator / denominator) * hl_mean.pct_change(delta)
        raw = raw.fillna(0)
        p1 = 0.05
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_052_rank(df, window=15):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_diff = df['open'] - close_shift
        close_denom = close_shift + 1e-8
        term1 = open_diff / close_denom
        hl_range = df['high'] - df['low']
        hl_mean = hl_range.rolling(5).mean() + 1e-8
        term2 = (hl_range / hl_mean) - 1
        raw = term1 * term2
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_052_tanh(df, window=30):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_diff = df['open'] - close_shift
        close_denom = close_shift + 1e-8
        term1 = open_diff / close_denom
        hl_range = df['high'] - df['low']
        hl_mean = hl_range.rolling(5).mean() + 1e-8
        term2 = (hl_range / hl_mean) - 1
        raw = term1 * term2
        # Dynamic Tanh normalization
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_052_zscore(df, window=95):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_diff = df['open'] - close_shift
        close_denom = close_shift + 1e-8
        term1 = open_diff / close_denom
        hl_range = df['high'] - df['low']
        hl_mean = hl_range.rolling(5).mean() + 1e-8
        term2 = (hl_range / hl_mean) - 1
        raw = term1 * term2
        # Rolling Z-Score/Clip normalization
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        z = (raw - mean) / (std + 1e-8)
        normalized = z.clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_052_sign(df):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_diff = df['open'] - close_shift
        close_denom = close_shift + 1e-8
        term1 = open_diff / close_denom
        hl_range = df['high'] - df['low']
        hl_mean = hl_range.rolling(5).mean() + 1e-8
        term2 = (hl_range / hl_mean) - 1
        raw = term1 * term2
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_052_wf(df, sub_window=100):
        # Raw calculation
        close_shift = df['close'].shift(1)
        open_diff = df['open'] - close_shift
        close_denom = close_shift + 1e-8
        term1 = open_diff / close_denom
        hl_range = df['high'] - df['low']
        hl_mean = hl_range.rolling(5).mean() + 1e-8
        term2 = (hl_range / hl_mean) - 1
        raw = term1 * term2
        # Winsorized Fisher normalization
        p1 = 0.05  # Hardcoded quantile threshold
        p2 = sub_window  # Rolling window for quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_053_rank(df, window=45):
        raw = (abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1).rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / ((df['high'] - df['low']).rolling(window).mean() + 1e-8) - 1)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_053_tanh(df, window=70):
        raw = (abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1).rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / ((df['high'] - df['low']).rolling(window).mean() + 1e-8) - 1)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_053_zscore(df, window=60):
        raw = (abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1).rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / ((df['high'] - df['low']).rolling(window).mean() + 1e-8) - 1)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_053_sign(df, window=100):
        raw = (abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1).rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / ((df['high'] - df['low']).rolling(window).mean() + 1e-8) - 1)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_053_wf(df, window=50, quantile=0.1):
        raw = (abs(df['open'] - df['close'].shift(1)) / (df['close'].shift(1).rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / ((df['high'] - df['low']).rolling(window).mean() + 1e-8) - 1)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_054_rank(df, window=100):
        open = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        raw = (open - delay_close) / (delay_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = high - low
        cov = y.rolling(window).cov(days)
        var = days.rolling(window).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        normalized = (result.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_054_tanh(df, window=85):
        open = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        raw = (open - delay_close) / (delay_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = high - low
        cov = y.rolling(window).cov(days)
        var = days.rolling(window).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        normalized = np.tanh(result / result.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_054_zscore(df, window=85):
        open = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        raw = (open - delay_close) / (delay_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = high - low
        cov = y.rolling(window).cov(days)
        var = days.rolling(window).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        normalized = ((result - result.rolling(window).mean()) / result.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_054_sign(df, window=35):
        open = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        raw = (open - delay_close) / (delay_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = high - low
        cov = y.rolling(window).cov(days)
        var = days.rolling(window).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        normalized = np.sign(result)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_054_wf(df, window=70, p1=0.1):
        open = df['open']
        close = df['close']
        high = df['high']
        low = df['low']
        delay_close = close.shift(1)
        raw = (open - delay_close) / (delay_close + 1e-8)
        days = pd.Series(np.arange(len(df)), index=df.index)
        y = high - low
        cov = y.rolling(window).cov(days)
        var = days.rolling(window).var().replace(0, np.nan)
        reg = cov / var
        result = raw * reg
        p2 = window
        low_q = result.rolling(p2).quantile(p1)
        high_q = result.rolling(p2).quantile(1 - p1)
        winsorized = result.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_055_k(df, window=50):
        # Tính phần trăm thay đổi giá mở cửa so với giá đóng cửa phiên trước
        open_change = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)

        # Tính dải dao động (high - low) trượt
        high_low_range = df['high'] - df['low']

        # Tính các thống kê rolling cho high_low_range
        max_range = high_low_range.rolling(window=window, min_periods=1).max().replace(0, np.nan) + 1e-8
        min_range = high_low_range.rolling(window=window, min_periods=1).min()

        # Tính tỷ lệ dao động: (current_range / max_range) - (min_range / max_range)
        range_ratio = (high_low_range / max_range) - (min_range / max_range)

        # Kết hợp với dấu của open_change
        raw = np.sign(open_change) * range_ratio

        # Chuẩn hóa bằng Rolling Rank (Trường hợp A)
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1

        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_055_h(df, window=15):
        open_change = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        high_low_range = df['high'] - df['low']
        max_range = high_low_range.rolling(window=window, min_periods=1).max().replace(0, np.nan) + 1e-8
        min_range = high_low_range.rolling(window=window, min_periods=1).min()
        range_ratio = (high_low_range / max_range) - (min_range / max_range)
        raw = np.sign(open_change) * range_ratio
        std_raw = raw.rolling(window).std().replace(0, np.nan) + 1e-8
        normalized = np.tanh(raw / std_raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_055_e(df, window=20):
        open_change = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        high_low_range = df['high'] - df['low']
        max_range = high_low_range.rolling(window=window, min_periods=1).max().replace(0, np.nan) + 1e-8
        min_range = high_low_range.rolling(window=window, min_periods=1).min()
        range_ratio = (high_low_range / max_range) - (min_range / max_range)
        raw = np.sign(open_change) * range_ratio
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan) + 1e-8
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_055_n(df, window=75):
        open_change = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        high_low_range = df['high'] - df['low']
        max_range = high_low_range.rolling(window=window, min_periods=1).max().replace(0, np.nan) + 1e-8
        min_range = high_low_range.rolling(window=window, min_periods=1).min()
        range_ratio = (high_low_range / max_range) - (min_range / max_range)
        raw = np.sign(open_change) * range_ratio
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_055_r(df, window=50, quantile_factor=0.1):
        open_change = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        high_low_range = df['high'] - df['low']
        max_range = high_low_range.rolling(window=window, min_periods=1).max().replace(0, np.nan) + 1e-8
        min_range = high_low_range.rolling(window=window, min_periods=1).min()
        range_ratio = (high_low_range / max_range) - (min_range / max_range)
        raw = np.sign(open_change) * range_ratio
        low_percentile = raw.rolling(window).quantile(quantile_factor)
        high_percentile = raw.rolling(window).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low_percentile, upper=high_percentile, axis=0)
        normalized = np.arctanh(((winsorized - low_percentile) / (high_percentile - low_percentile + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_056_rank(df, window_volume=20, window_pctchange=1):
        close = df['close']
        volume = df['matchingVolume']
        mean_vol = volume.rolling(window=window_volume, min_periods=1).mean().fillna(volume)
        raw_volume = (volume / (mean_vol + 1e-8)) - 1
        pct_change = close.pct_change(window_pctchange, fill_method=None).fillna(0)
        direction = np.sign(pct_change).replace(0, 1)
        raw = raw_volume * direction
        signal = (raw.rolling(window=window_volume, min_periods=1).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_056_tanh(df, window_volume=10, window_pctchange=1):
        close = df['close']
        volume = df['matchingVolume']
        mean_vol = volume.rolling(window=window_volume, min_periods=1).mean().fillna(volume)
        raw_volume = (volume / (mean_vol + 1e-8)) - 1
        pct_change = close.pct_change(window_pctchange, fill_method=None).fillna(0)
        direction = np.sign(pct_change).replace(0, 1)
        raw = raw_volume * direction
        signal = np.tanh(raw / raw.rolling(window=window_volume, min_periods=1).std().replace(0, np.nan).ffill().fillna(1))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_056_zscore(df, window_volume=10, window_pctchange=1):
        close = df['close']
        volume = df['matchingVolume']
        mean_vol = volume.rolling(window=window_volume, min_periods=1).mean().fillna(volume)
        raw_volume = (volume / (mean_vol + 1e-8)) - 1
        pct_change = close.pct_change(window_pctchange, fill_method=None).fillna(0)
        direction = np.sign(pct_change).replace(0, 1)
        raw = raw_volume * direction
        rolling_mean = raw.rolling(window=window_volume, min_periods=1).mean().ffill()
        rolling_std = raw.rolling(window=window_volume, min_periods=1).std().replace(0, np.nan).ffill().fillna(1)
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_056_sign(df, window_volume=60, window_pctchange=1):
        close = df['close']
        volume = df['matchingVolume']
        mean_vol = volume.rolling(window=window_volume, min_periods=1).mean().fillna(volume)
        raw_volume = (volume / (mean_vol + 1e-8)) - 1
        pct_change = close.pct_change(window_pctchange, fill_method=None).fillna(0)
        direction = np.sign(pct_change).replace(0, 1)
        raw = raw_volume * direction
        signal = np.sign(raw).astype(float)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_056_wf(df, window_volume=40, window_pctchange=1, p1=0.05, p2=100):
        close = df['close']
        volume = df['matchingVolume']
        mean_vol = volume.rolling(window=window_volume, min_periods=1).mean().fillna(volume)
        raw_volume = (volume / (mean_vol + 1e-8)) - 1
        pct_change = close.pct_change(window_pctchange, fill_method=None).fillna(0)
        direction = np.sign(pct_change).replace(0, 1)
        raw = raw_volume * direction
        low = raw.rolling(window=p2, min_periods=1).quantile(p1).ffill()
        high = raw.rolling(window=p2, min_periods=1).quantile(1 - p1).ffill()
        winsorized = raw.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_057_k(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        rank_volume = volume.rolling(window).rank(pct=True).fillna(0.5)
        corr = rank_volume.rolling(window).corr(ret).fillna(0)
        raw = corr
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_057_h(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        rank_volume = volume.rolling(window).rank(pct=True).fillna(0.5)
        corr = rank_volume.rolling(window).corr(ret).fillna(0)
        raw = corr
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        result = np.tanh(raw / std)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_057_e(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        rank_volume = volume.rolling(window).rank(pct=True).fillna(0.5)
        corr = rank_volume.rolling(window).corr(ret).fillna(0)
        raw = corr
        mean = raw.rolling(window).mean().fillna(0)
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        result = ((raw - mean) / std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_057_y(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        rank_volume = volume.rolling(window).rank(pct=True).fillna(0.5)
        corr = rank_volume.rolling(window).corr(ret).fillna(0)
        raw = corr
        result = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_057_r(df, window=20, sub_window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret = df['close'].pct_change().fillna(0)
        rank_volume = volume.rolling(window).rank(pct=True).fillna(0.5)
        corr = rank_volume.rolling(window).corr(ret).fillna(0)
        raw = corr
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1).fillna(method='ffill').fillna(raw.min())
        high = raw.rolling(p2).quantile(1 - p1).fillna(method='ffill').fillna(raw.max())
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_058_k(df, window=55):
        pch_vol = df['matchingVolume'].pct_change(3).replace([np.inf, -np.inf], np.nan)
        pch_close = df['close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        raw = (pch_vol / (pch_vol.rolling(window).std() + 1e-8)) * np.sign(pch_close)
        return (raw.rolling(window).rank(pct=True) * 2) - 1

    @staticmethod
    def alpha_quanta_058_h(df, window=60):
        pch_vol = df['matchingVolume'].pct_change(3).replace([np.inf, -np.inf], np.nan)
        pch_close = df['close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        raw = (pch_vol / (pch_vol.rolling(window).std() + 1e-8)) * np.sign(pch_close)
        return np.tanh(raw / raw.rolling(window).std())

    @staticmethod
    def alpha_quanta_058_e(df, window=75):
        pch_vol = df['matchingVolume'].pct_change(3).replace([np.inf, -np.inf], np.nan)
        pch_close = df['close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        raw = (pch_vol / (pch_vol.rolling(window).std() + 1e-8)) * np.sign(pch_close)
        return ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)

    @staticmethod
    def alpha_quanta_058_y(df, window=65):
        pch_vol = df['matchingVolume'].pct_change(3).replace([np.inf, -np.inf], np.nan)
        pch_close = df['close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        raw = (pch_vol / (pch_vol.rolling(window).std() + 1e-8)) * np.sign(pch_close)
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_058_r(df, window=40, p1=0.1, p2=20):
        pch_vol = df['matchingVolume'].pct_change(3).replace([np.inf, -np.inf], np.nan)
        pch_close = df['close'].pct_change(5).replace([np.inf, -np.inf], np.nan)
        raw = (pch_vol / (pch_vol.rolling(window).std() + 1e-8)) * np.sign(pch_close)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_059_9(df, window=95):
        # Calculate volume ratio
        median_volume = df['matchingVolume'].rolling(window).median()
        volume_ratio = (df['matchingVolume'] / (median_volume + 1e-8)) - 1
        # Calculate returns
        returns = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Sum returns over 10 periods
        cum_returns = returns.rolling(10).sum()
        # Raw signal
        raw = volume_ratio * cum_returns
        # Standardization C: Rolling Z-Score (Clip)
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        signal = ((raw - raw_mean) / (raw_std + 1e-9)).clip(-1, 1)
        # Fill NaN with 0
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_059_9(df, window=25):
        # Calculate raw signal
        median_volume = df['matchingVolume'].rolling(window).median()
        volume_ratio = (df['matchingVolume'] / (median_volume + 1e-8)) - 1
        cum_returns = (df['close'].pct_change().replace([np.inf, -np.inf], 0)).rolling(10).sum()
        raw = volume_ratio * cum_returns
        # Standardization A: Rolling Rank
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_059_9(df, window=25):
        # Calculate raw signal
        median_volume = df['matchingVolume'].rolling(window).median()
        volume_ratio = (df['matchingVolume'] / (median_volume + 1e-8)) - 1
        cum_returns = (df['close'].pct_change().replace([np.inf, -np.inf], 0)).rolling(10).sum()
        raw = volume_ratio * cum_returns
        # Standardization B: Dynamic Tanh
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_059_9(df, window=25):
        # Calculate raw signal
        median_volume = df['matchingVolume'].rolling(window).median()
        volume_ratio = (df['matchingVolume'] / (median_volume + 1e-8)) - 1
        cum_returns = (df['close'].pct_change().replace([np.inf, -np.inf], 0)).rolling(10).sum()
        raw = volume_ratio * cum_returns
        # Standardization D: Sign/Binary Soft
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_059_9(df, window=25, sub_window=10):
        # Calculate raw signal
        median_volume = df['matchingVolume'].rolling(window).median()
        volume_ratio = (df['matchingVolume'] / (median_volume + 1e-8)) - 1
        cum_returns = (df['close'].pct_change().replace([np.inf, -np.inf], 0)).rolling(10).sum()
        raw = volume_ratio * cum_returns
        # Standardization E: Winsorized Fisher
        low = raw.rolling(sub_window).quantile(0.1)
        high = raw.rolling(sub_window).quantile(0.9)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_059_k(df, window=100):
        median_vol = df['matchingVolume'].rolling(window=window, min_periods=5).median()
        vol_ratio = (df['matchingVolume'] / (median_vol + 1e-8)) - 1
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        cumprod_ret = ret.rolling(window=10).apply(lambda x: pd.Series(x).prod(), raw=True).ffill()
        raw = vol_ratio * cumprod_ret
        normalized = (raw.rolling(window=window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_059_rank(df, window=60, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        volume_ratio = volume / (volume.rolling(window, min_periods=1).median() + 1e-8) - 1
        returns = close.pct_change().fillna(0)
        momentum = returns.rolling(sub_window, min_periods=1).sum()
        raw = volume_ratio * momentum
        normalized = (raw.rolling(window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_059_tanh(df, window=50, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        volume_ratio = volume / (volume.rolling(window, min_periods=1).median() + 1e-8) - 1
        returns = close.pct_change().fillna(0)
        momentum = returns.rolling(sub_window, min_periods=1).sum()
        raw = volume_ratio * momentum
        normalized = np.tanh(raw / (raw.rolling(window, min_periods=1).std() + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_059_zscore(df, window=50, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        volume_ratio = volume / (volume.rolling(window, min_periods=1).median() + 1e-8) - 1
        returns = close.pct_change().fillna(0)
        momentum = returns.rolling(sub_window, min_periods=1).sum()
        raw = volume_ratio * momentum
        roll_mean = raw.rolling(window, min_periods=1).mean()
        roll_std = raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_059_sign(df, window=30, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        volume_ratio = volume / (volume.rolling(window, min_periods=1).median() + 1e-8) - 1
        returns = close.pct_change().fillna(0)
        momentum = returns.rolling(sub_window, min_periods=1).sum()
        raw = volume_ratio * momentum
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_059_wf(df, window=50, sub_window=1, p1=0.05):
        close = df['close']
        volume = df['matchingVolume']
        volume_ratio = volume / (volume.rolling(window, min_periods=1).median() + 1e-8) - 1
        returns = close.pct_change().fillna(0)
        momentum = returns.rolling(sub_window, min_periods=1).sum()
        raw = volume_ratio * momentum
        low = raw.rolling(window, min_periods=1).quantile(p1)
        high = raw.rolling(window, min_periods=1).quantile(1 - p1)
        winsorized = raw.copy()
        winsorized = winsorized.where(winsorized >= low, low)
        winsorized = winsorized.where(winsorized <= high, high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_060_rank(df, window=5):
        volume_ratio = (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)) - 1
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        raw = volume_ratio * (rsi - 50)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_060_tanh(df, window=5):
        volume_ratio = (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)) - 1
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        raw = volume_ratio * (rsi - 50)
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_060_zscore(df, window=5):
        volume_ratio = (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)) - 1
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        raw = volume_ratio * (rsi - 50)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_060_sign(df, window=50):
        volume_ratio = (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)) - 1
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        raw = volume_ratio * (rsi - 50)
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_060_wf(df, window=80, quantile=0.1):
        volume_ratio = (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)) - 1
        delta = df['close'].diff()
        gain = delta.clip(lower=0).rolling(window).mean()
        loss = (-delta).clip(lower=0).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        raw = volume_ratio * (rsi - 50)
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_061_rank(df, window=90):
        sub_window = 5
        close_shift1 = df['close'].shift(1)
        ret_close = (df['close'] - close_shift1) / (close_shift1 + 1e-8)
        open_shift1 = close_shift1
        open_ret = (df['open'] - open_shift1) / (open_shift1 + 1e-8)
        ts_mean = open_ret.rolling(sub_window).mean()
        ts_std = ret_close.rolling(window).std() + 1e-8
        raw = ts_mean / ts_std
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_061_tanh(df, window=5):
        sub_window = 5
        close_shift1 = df['close'].shift(1)
        ret_close = (df['close'] - close_shift1) / (close_shift1 + 1e-8)
        open_shift1 = close_shift1
        open_ret = (df['open'] - open_shift1) / (open_shift1 + 1e-8)
        ts_mean = open_ret.rolling(sub_window).mean()
        ts_std = ret_close.rolling(window).std() + 1e-8
        raw = ts_mean / ts_std
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_061_zscore(df, window=90):
        sub_window = 5
        close_shift1 = df['close'].shift(1)
        ret_close = (df['close'] - close_shift1) / (close_shift1 + 1e-8)
        open_shift1 = close_shift1
        open_ret = (df['open'] - open_shift1) / (open_shift1 + 1e-8)
        ts_mean = open_ret.rolling(sub_window).mean()
        ts_std = ret_close.rolling(window).std() + 1e-8
        raw = ts_mean / ts_std
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_061_sign(df, window=45):
        sub_window = 5
        close_shift1 = df['close'].shift(1)
        ret_close = (df['close'] - close_shift1) / (close_shift1 + 1e-8)
        open_shift1 = close_shift1
        open_ret = (df['open'] - open_shift1) / (open_shift1 + 1e-8)
        ts_mean = open_ret.rolling(sub_window).mean()
        ts_std = ret_close.rolling(window).std() + 1e-8
        raw = ts_mean / ts_std
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_061_wf(df, window=50):
        sub_window = 5
        p1 = 0.05
        p2 = window
        close_shift1 = df['close'].shift(1)
        ret_close = (df['close'] - close_shift1) / (close_shift1 + 1e-8)
        open_shift1 = close_shift1
        open_ret = (df['open'] - open_shift1) / (open_shift1 + 1e-8)
        ts_mean = open_ret.rolling(sub_window).mean()
        ts_std = ret_close.rolling(window).std() + 1e-8
        raw = ts_mean / ts_std
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_062_rank(df, window=20):
        # Compute overnight return -(open - prev close)/prev close
        prev_close = df['close'].shift(1)
        overnight_ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        # Compute intraday return (close - open)/open
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        # Rolling correlation between overnight and intraday returns
        corr = overnight_ret.rolling(window).corr(intraday_ret)
        sign_corr = np.sign(corr)
        # Rolling std of overnight return for normalization
        std_overnight = overnight_ret.rolling(20).std() + 1e-8
        raw = sign_corr / std_overnight
        # Standardization: Rolling rank scaled to [-1, 1]
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_062_tanh(df, window=100):
        prev_close = df['close'].shift(1)
        overnight_ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = overnight_ret.rolling(window).corr(intraday_ret)
        sign_corr = np.sign(corr)
        std_overnight = overnight_ret.rolling(20).std() + 1e-8
        raw = sign_corr / std_overnight
        # Dynamic tanh normalization
        rolling_std = raw.rolling(window).std() + 1e-8
        signal = np.tanh(raw / rolling_std)
        return signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_062_zscore(df, window=100):
        prev_close = df['close'].shift(1)
        overnight_ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = overnight_ret.rolling(window).corr(intraday_ret)
        sign_corr = np.sign(corr)
        std_overnight = overnight_ret.rolling(20).std() + 1e-8
        raw = sign_corr / std_overnight
        # Rolling Z-score normalization
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std() + 1e-8
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_062_sign(df, window=80):
        prev_close = df['close'].shift(1)
        overnight_ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = overnight_ret.rolling(window).corr(intraday_ret)
        sign_corr = np.sign(corr)
        std_overnight = overnight_ret.rolling(20).std() + 1e-8
        raw = sign_corr / std_overnight
        # Sign/Binary soft: just sign
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_062_wf(df, window=100, p1=0.1):
        prev_close = df['close'].shift(1)
        overnight_ret = (df['open'] - prev_close) / (prev_close + 1e-8)
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = overnight_ret.rolling(window).corr(intraday_ret)
        sign_corr = np.sign(corr)
        std_overnight = overnight_ret.rolling(20).std() + 1e-8
        raw = sign_corr / std_overnight
        # Winsorized Fisher transform
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_063_rank(df, window1=10, window2=60):
        log_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        mean_log_ret = log_ret.rolling(window1).mean()
        zscore_mean = (mean_log_ret - mean_log_ret.rolling(window2).mean()) / mean_log_ret.rolling(window2).std().replace(0, np.nan)
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        std_ret = ret.rolling(window2).std()
        zscore_std = (std_ret - std_ret.rolling(window2).mean()) / std_ret.rolling(window2).std().replace(0, np.nan)
        raw = zscore_mean * (1 - zscore_std)
        norm = (raw.rolling(window1).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_063_tanh(df, window1=30, window2=40):
        log_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        mean_log_ret = log_ret.rolling(window1).mean()
        zscore_mean = (mean_log_ret - mean_log_ret.rolling(window2).mean()) / mean_log_ret.rolling(window2).std().replace(0, np.nan)
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        std_ret = ret.rolling(window2).std()
        zscore_std = (std_ret - std_ret.rolling(window2).mean()) / std_ret.rolling(window2).std().replace(0, np.nan)
        raw = zscore_mean * (1 - zscore_std)
        norm = np.tanh(raw / raw.rolling(window1).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_063_zscore(df, window1=10, window2=10):
        log_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        mean_log_ret = log_ret.rolling(window1).mean()
        zscore_mean = (mean_log_ret - mean_log_ret.rolling(window2).mean()) / mean_log_ret.rolling(window2).std().replace(0, np.nan)
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        std_ret = ret.rolling(window2).std()
        zscore_std = (std_ret - std_ret.rolling(window2).mean()) / std_ret.rolling(window2).std().replace(0, np.nan)
        raw = zscore_mean * (1 - zscore_std)
        norm = ((raw - raw.rolling(window1).mean()) / raw.rolling(window1).std().replace(0, np.nan)).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_063_sign(df, window1=30, window2=80):
        log_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        mean_log_ret = log_ret.rolling(window1).mean()
        zscore_mean = (mean_log_ret - mean_log_ret.rolling(window2).mean()) / mean_log_ret.rolling(window2).std().replace(0, np.nan)
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        std_ret = ret.rolling(window2).std()
        zscore_std = (std_ret - std_ret.rolling(window2).mean()) / std_ret.rolling(window2).std().replace(0, np.nan)
        raw = zscore_mean * (1 - zscore_std)
        norm = pd.Series(np.sign(raw), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_063_wf(df, window1=10, window2=20, quantile=0.1):
        log_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        mean_log_ret = log_ret.rolling(window1).mean()
        zscore_mean = (mean_log_ret - mean_log_ret.rolling(window2).mean()) / mean_log_ret.rolling(window2).std().replace(0, np.nan)
        ret = (df['close'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        std_ret = ret.rolling(window2).std()
        zscore_std = (std_ret - std_ret.rolling(window2).mean()) / std_ret.rolling(window2).std().replace(0, np.nan)
        raw = zscore_mean * (1 - zscore_std)
        low = raw.rolling(window1).quantile(quantile)
        high = raw.rolling(window1).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_064_rank(df, window_rank=30, window_std=40):
        ret_lag1 = df['close'].diff() / df['close'].shift(1).replace(0, np.nan)
        ret_lag1 = ret_lag1.fillna(0)
        open_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan))
        open_ret = open_ret.fillna(0)
        ts_mean = open_ret.rolling(window_rank).mean()
        ts_std = ret_lag1.rolling(window_std).std().replace(0, np.nan)
        raw = ts_mean / (ts_std + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = raw.rolling(window_rank).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_064_tanh(df, window_rank=20, window_std=50):
        ret_lag1 = df['close'].diff() / df['close'].shift(1).replace(0, np.nan)
        ret_lag1 = ret_lag1.fillna(0)
        open_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan))
        open_ret = open_ret.fillna(0)
        ts_mean = open_ret.rolling(window_rank).mean()
        ts_std = ret_lag1.rolling(window_std).std().replace(0, np.nan)
        raw = ts_mean / (ts_std + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = np.tanh(raw / raw.rolling(window_rank).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_064_zscore(df, window_rank=30, window_std=10):
        ret_lag1 = df['close'].diff() / df['close'].shift(1).replace(0, np.nan)
        ret_lag1 = ret_lag1.fillna(0)
        open_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan))
        open_ret = open_ret.fillna(0)
        ts_mean = open_ret.rolling(window_rank).mean()
        ts_std = ret_lag1.rolling(window_std).std().replace(0, np.nan)
        raw = ts_mean / (ts_std + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        norm = (raw - raw.rolling(window_rank).mean()) / raw.rolling(window_rank).std().replace(0, np.nan)
        signal = norm.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_064_sign(df, window_rank=20, window_std=60):
        ret_lag1 = df['close'].diff() / df['close'].shift(1).replace(0, np.nan)
        ret_lag1 = ret_lag1.fillna(0)
        open_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan))
        open_ret = open_ret.fillna(0)
        ts_mean = open_ret.rolling(window_rank).mean()
        ts_std = ret_lag1.rolling(window_std).std().replace(0, np.nan)
        raw = ts_mean / (ts_std + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_064_wf(df, window_rank=30, window_std=80, p1=0.05, p2=20):
        ret_lag1 = df['close'].diff() / df['close'].shift(1).replace(0, np.nan)
        ret_lag1 = ret_lag1.fillna(0)
        open_ret = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan))
        open_ret = open_ret.fillna(0)
        ts_mean = open_ret.rolling(window_rank).mean()
        ts_std = ret_lag1.rolling(window_std).std().replace(0, np.nan)
        raw = ts_mean / (ts_std + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        winsorized = winsorized.fillna(0)
        low_filled = low.fillna(0)
        high_filled = high.fillna(0)
        normalized = np.arctanh(((winsorized - low_filled) / (high_filled - low_filled + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], 0).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_065_rank(df, window=5):
        # Tính returns: (open - close_delay) / (close_delay + 1e-8)
        close_delay = df['close'].shift(1)
        returns_open = (df['open'] - close_delay) / (close_delay + 1e-8)
        # Tính intraday returns: (close - open) / (open + 1e-8)
        returns_intra = (df['close'] - df['open']) / (df['open'] + 1e-8)
        # Tính rolling correlation
        corr = returns_open.rolling(window).corr(returns_intra)
        # Chuẩn hóa về [-1, 1] dùng Rolling Rank
        raw = corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_065_tanh(df, window=40):
        close_delay = df['close'].shift(1)
        returns_open = (df['open'] - close_delay) / (close_delay + 1e-8)
        returns_intra = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = returns_open.rolling(window).corr(returns_intra)
        # Chuẩn hóa dùng Dynamic Tanh
        normalized = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_065_zscore(df, window=5):
        close_delay = df['close'].shift(1)
        returns_open = (df['open'] - close_delay) / (close_delay + 1e-8)
        returns_intra = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = returns_open.rolling(window).corr(returns_intra)
        # Chuẩn hóa dùng Rolling Z-Score / Clip
        normalized = ((corr - corr.rolling(window).mean()) / corr.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_065_sign(df, window=80):
        close_delay = df['close'].shift(1)
        returns_open = (df['open'] - close_delay) / (close_delay + 1e-8)
        returns_intra = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = returns_open.rolling(window).corr(returns_intra)
        # Chuẩn hóa dùng Sign/Binary Soft
        normalized = np.sign(corr)
        return -pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_065_wf(df, window=30, p1=0.1, p2=20):
        # p2 là window cho rolling quantile, p1 là percentile để cắt
        close_delay = df['close'].shift(1)
        returns_open = (df['open'] - close_delay) / (close_delay + 1e-8)
        returns_intra = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = returns_open.rolling(window).corr(returns_intra)
        raw = corr
        # Winsorized Fisher
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_066_rank(df, window=15):
        close = df['close']
        volume = df['matchingVolume']
        delta = close.diff(1)
        delay = close.shift(1)
        raw = np.log(volume.rolling(window).mean() + 1) / (delta.div(delay + 1e-8).rolling(window).std() + 1)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_066_tanh(df, window=55):
        close = df['close']
        volume = df['matchingVolume']
        delta = close.diff(1)
        delay = close.shift(1)
        raw = np.log(volume.rolling(window).mean() + 1) / (delta.div(delay + 1e-8).rolling(window).std() + 1)
        normalized = np.tanh(raw / raw.rolling(window).std())
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_066_zscore(df, window=15):
        close = df['close']
        volume = df['matchingVolume']
        delta = close.diff(1)
        delay = close.shift(1)
        raw = np.log(volume.rolling(window).mean() + 1) / (delta.div(delay + 1e-8).rolling(window).std() + 1)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_066_sign(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        delta = close.diff(1)
        delay = close.shift(1)
        raw = np.log(volume.rolling(window).mean() + 1) / (delta.div(delay + 1e-8).rolling(window).std() + 1)
        normalized = np.sign(raw)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_066_wf(df, window=20, winsor_quantile=0.7):
        close = df['close']
        volume = df['matchingVolume']
        delta = close.diff(1)
        delay = close.shift(1)
        raw = np.log(volume.rolling(window).mean() + 1) / (delta.div(delay + 1e-8).rolling(window).std() + 1)
        p = winsor_quantile
        low = raw.rolling(window).quantile(p)
        high = raw.rolling(window).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_067_rank(df, window=15):
        delay_close = df['close'].shift(1).replace(0, np.nan).ffill()
        intraday_range = (df['high'] - df['low']) / (delay_close + 1e-8)
        open_pct = (df['open'] - delay_close).abs() / (delay_close + 1e-8)
        ratio = open_pct / (intraday_range + 1e-8)
        raw = ratio.rolling(window).mean()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_067_tanh(df, window=20):
        delay_close = df['close'].shift(1).replace(0, np.nan).ffill()
        intraday_range = (df['high'] - df['low']) / (delay_close + 1e-8)
        open_pct = (df['open'] - delay_close).abs() / (delay_close + 1e-8)
        ratio = open_pct / (intraday_range + 1e-8)
        raw = ratio.rolling(window).mean()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_067_zscore(df, window=30):
        delay_close = df['close'].shift(1).replace(0, np.nan).ffill()
        intraday_range = (df['high'] - df['low']) / (delay_close + 1e-8)
        open_pct = (df['open'] - delay_close).abs() / (delay_close + 1e-8)
        ratio = open_pct / (intraday_range + 1e-8)
        raw = ratio.rolling(window).mean()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_067_sign(df, window=90):
        delay_close = df['close'].shift(1).replace(0, np.nan).ffill()
        intraday_range = (df['high'] - df['low']) / (delay_close + 1e-8)
        open_pct = (df['open'] - delay_close).abs() / (delay_close + 1e-8)
        ratio = open_pct / (intraday_range + 1e-8)
        raw = ratio.rolling(window).mean()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_067_wf(df, window=30, quantile=0.1):
        delay_close = df['close'].shift(1).replace(0, np.nan).ffill()
        intraday_range = (df['high'] - df['low']) / (delay_close + 1e-8)
        open_pct = (df['open'] - delay_close).abs() / (delay_close + 1e-8)
        ratio = open_pct / (intraday_range + 1e-8)
        raw = ratio.rolling(window).mean()
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_068_zscore(df, window=5):
        # Tính toán các thành phần cơ bản tránh chia cho 0
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        close_shift = close.shift(1)
        open_shift = open_price.shift(1)

        # ROC của close (phần tử a: (open - delay(close,1)) / delay(close,1))
        a = (open_price - close_shift) / (close_shift + 1e-8)

        # ROC của close trong bar (phần tử b: (close - open) / open)
        b = (close - open_price) / (open_price + 1e-8)

        # Tính rolling correlation giữa a và b với window
        a_mean = a.rolling(window, min_periods=window).mean()
        b_mean = b.rolling(window, min_periods=window).mean()
        cov_ab = (a * b).rolling(window, min_periods=window).mean() - a_mean * b_mean
        var_a = (a**2).rolling(window, min_periods=window).mean() - a_mean**2
        var_b = (b**2).rolling(window, min_periods=window).mean() - b_mean**2
        corr_ab = cov_ab / ((var_a * var_b + 1e-8)**0.5 + 1e-8)

        # TS_STD của Delta(close) / delay(close,1)
        delta_close = close.diff(1)
        return_series = delta_close / (close_shift + 1e-8)
        ts_std = return_series.rolling(window, min_periods=window).std()

        # LOG của TS_MEAN(volume, 15)
        log_vol = np.log(volume.rolling(window, min_periods=window).mean() + 1)

        # Công thức chính: corr * (ts_std + 1) / (log_vol + 1e-8)
        raw = corr_ab * (ts_std + 1) / (log_vol + 1e-8)

        # Chuẩn hóa Rolling Z-Score (Trường hợp C - phù hợp với spread/oscillator)
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -result

    # Tên_Alpha: quanta_068 (gốc không có hậu tố, nhưng theo quy tắc cần 5 phiên bản. Tuy nhiên YC chỉ có 1 công thức gốc nên tạm thời chỉ output 1 version).

    @staticmethod
    def alpha_quanta_068_rank(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret_open = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        ret_close = (close - open_) / (open_ + 1e-8)
        corr = ret_open.rolling(window).corr(ret_close)
        delta_close = close.diff(1) / (close.shift(1) + 1e-8)
        std = delta_close.rolling(window).std()
        log_vol = np.log(volume.rolling(window).mean() + 1)
        raw = corr * (std + 1) / (log_vol + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal[signal.isna()] = 0
        return -signal

    @staticmethod
    def alpha_quanta_068_tanh(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret_open = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        ret_close = (close - open_) / (open_ + 1e-8)
        corr = ret_open.rolling(window).corr(ret_close)
        delta_close = close.diff(1) / (close.shift(1) + 1e-8)
        std = delta_close.rolling(window).std()
        log_vol = np.log(volume.rolling(window).mean() + 1)
        raw = corr * (std + 1) / (log_vol + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        signal[signal.isna()] = 0
        return -signal

    @staticmethod
    def alpha_quanta_068_sign(df, window=80):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret_open = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        ret_close = (close - open_) / (open_ + 1e-8)
        corr = ret_open.rolling(window).corr(ret_close)
        delta_close = close.diff(1) / (close.shift(1) + 1e-8)
        std = delta_close.rolling(window).std()
        log_vol = np.log(volume.rolling(window).mean() + 1)
        raw = corr * (std + 1) / (log_vol + 1e-8)
        signal = np.sign(raw)
        signal[signal.isna()] = np.sign(0)
        return -signal

    @staticmethod
    def alpha_quanta_068_wf(df, window=30, p1=0.7):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        ret_open = (open_ - close.shift(1)) / (close.shift(1) + 1e-8)
        ret_close = (close - open_) / (open_ + 1e-8)
        corr = ret_open.rolling(window).corr(ret_close)
        delta_close = close.diff(1) / (close.shift(1) + 1e-8)
        std = delta_close.rolling(window).std()
        log_vol = np.log(volume.rolling(window).mean() + 1)
        raw = corr * (std + 1) / (log_vol + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized[normalized.isna()] = 0
        return -normalized

    @staticmethod
    def alpha_quanta_069_rank(df, window=20):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * df['matchingVolume'] / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal

    @staticmethod
    def alpha_quanta_069_tanh(df, window=100):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * df['matchingVolume'] / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal

    @staticmethod
    def alpha_quanta_069_zscore(df, window=50):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * df['matchingVolume'] / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_069_sign(df):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * df['matchingVolume'] / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_069_wf(df, p1=0.1, p2=60):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) * df['matchingVolume'] / (df['matchingVolume'].rolling(5).mean() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_070_rank(df, window=15):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        signal = np.sign(mean_raw) * (1 - std_raw)
        # Chuẩn hóa Rolling Rank (trường hợp A)
        normalized = signal.rolling(window).rank(pct=True) * 2 - 1
        normalized = normalized.ffill().fillna(0)
        return -pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_070_tanh(df, window=15):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        signal = np.sign(mean_raw) * (1 - std_raw)
        # Chuẩn hóa Dynamic Tanh (trường hợp B)
        normalized = np.tanh(signal / (signal.rolling(window).std() + 1e-8))
        normalized = normalized.ffill().fillna(0)
        return -pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_070_zscore(df, window=100):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        signal = np.sign(mean_raw) * (1 - std_raw)
        # Chuẩn hóa Rolling Z-Score (trường hợp C)
        normalized = ((signal - signal.rolling(window).mean()) / signal.rolling(window).std()).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_070_sign(df, window=15):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        signal = np.sign(mean_raw) * (1 - std_raw)
        # Chuẩn hóa Sign/Binary Soft (trường hợp D)
        normalized = np.sign(signal)
        normalized = normalized.ffill().fillna(0)
        return pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_070_wf(df, window=40, p1=0.1, p2=20):
        raw = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        signal = np.sign(mean_raw) * (1 - std_raw)
        # Chuẩn hóa Winsorized Fisher (trường hợp E) - hardcode p1, p2
        low_quant = signal.rolling(p2).quantile(p1)
        high_quant = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low_quant, upper=high_quant, axis=0)
        normalized = np.arctanh(((winsorized - low_quant) / (high_quant - low_quant + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_071_1(df, window=40):
        # Tính chỉ báo Stochastic (STC) trung gian
        stc = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        # Delta(1) của STC
        delta_stc = stc.diff(1)
        # Mối tương quan cuộn giữa STC và Volume
        # Tạo các mảng tương đương cho hệ số tương quan dạng rolling
        mean_stc = stc.rolling(window).mean()
        mean_vol = df['matchingVolume'].rolling(window).mean()
        cov = (stc * df['matchingVolume']).rolling(window).mean() - mean_stc * mean_vol
        std_stc = stc.rolling(window).std().replace(0, np.nan)
        std_vol = df['matchingVolume'].rolling(window).std().replace(0, np.nan)
        corr = cov / (std_stc * std_vol + 1e-9)
        # Tín hiệu thô: delta_stc * corr
        raw = delta_stc * corr
        # Chuẩn hóa sử dụng Rolling Rank (Trường hợp A)
        raw_rank = raw.rolling(window).rank(pct=True)
        # Xử lý giá trị thiếu bằng ffill
        signal = raw_rank.ffill().fillna(0)
        # Chuẩn hóa về dải [-1, 1]
        signal = (signal * 2) - 1
        # Xử lý các giá trị vô hạn còn lại
        signal = signal.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_071_k(df, window_rank=30, window_corr=7):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).diff()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', 1)))
        signal = (corr.rolling(window_rank).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_071_h(df, window_std=90, window_corr=30):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).diff()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', 1)))
        signal = np.tanh(corr / (corr.rolling(window_std).std().replace(0, np.nan).ffill() + 1e-8))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_071_e(df, window=90, window_corr=7):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).diff()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan).ffill()
        signal = ((corr - mean) / (std + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_071_y(df, window_corr=95):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).diff()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', 1)))
        signal = np.sign(corr)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_071_r(df, p1=0.1, window_corr=90, p2=20):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).diff()
        corr = raw.rolling(window_corr).corr(df.get('matchingVolume', df.get('volume', 1)))
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        ratio = (winsorized - low) / (high - low + 1e-9)
        signal = np.arctanh(ratio * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_072_rank(df, window=10):
        close = df['close']
        high = df['high']
        low = df['low']
        # Avoid division by zero
        eps = 1e-8
        # TS_CORR part
        x = close / (high + eps)
        y = close / (low + eps)
        # Rolling correlation
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov_xy = (x * y).rolling(window).mean() - x_mean * y_mean
        var_x = (x**2).rolling(window).mean() - x_mean**2
        var_y = (y**2).rolling(window).mean() - y_mean**2
        corr = cov_xy / (var_x * var_y + eps).clip(None, 0)  # Ensure non-negative denominator (odd but for safety)
        corr = corr.fillna(0).replace([np.inf, -np.inf], 0)
        # TS_MEAN part
        ratio = (close - low) / (high - low + eps)
        mean_ratio = ratio.rolling(window).mean()
        # Multiply
        raw = corr * mean_ratio
        # Normalize A: Rolling Rank
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_072_tanh(df, window=60):
        close = df['close']
        high = df['high']
        low = df['low']
        eps = 1e-8
        x = close / (high + eps)
        y = close / (low + eps)
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov_xy = (x * y).rolling(window).mean() - x_mean * y_mean
        var_x = (x**2).rolling(window).mean() - x_mean**2
        var_y = (y**2).rolling(window).mean() - y_mean**2
        corr = cov_xy / (var_x * var_y + eps).clip(None, 0)
        corr = corr.fillna(0).replace([np.inf, -np.inf], 0)
        ratio = (close - low) / (high - low + eps)
        mean_ratio = ratio.rolling(window).mean()
        raw = corr * mean_ratio
        # Normalize B: Dynamic Tanh
        std_raw = raw.rolling(window).std()
        signal = np.tanh(raw / (std_raw + eps))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_072_zscore(df, window=40):
        close = df['close']
        high = df['high']
        low = df['low']
        eps = 1e-8
        x = close / (high + eps)
        y = close / (low + eps)
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov_xy = (x * y).rolling(window).mean() - x_mean * y_mean
        var_x = (x**2).rolling(window).mean() - x_mean**2
        var_y = (y**2).rolling(window).mean() - y_mean**2
        corr = cov_xy / (var_x * var_y + eps).clip(None, 0)
        corr = corr.fillna(0).replace([np.inf, -np.inf], 0)
        ratio = (close - low) / (high - low + eps)
        mean_ratio = ratio.rolling(window).mean()
        raw = corr * mean_ratio
        # Normalize C: Rolling Z-Score
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        z = (raw - mean_raw) / (std_raw + eps)
        signal = z.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_072_sign(df, window=60):
        close = df['close']
        high = df['high']
        low = df['low']
        eps = 1e-8
        x = close / (high + eps)
        y = close / (low + eps)
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov_xy = (x * y).rolling(window).mean() - x_mean * y_mean
        var_x = (x**2).rolling(window).mean() - x_mean**2
        var_y = (y**2).rolling(window).mean() - y_mean**2
        corr = cov_xy / (var_x * var_y + eps).clip(None, 0)
        corr = corr.fillna(0).replace([np.inf, -np.inf], 0)
        ratio = (close - low) / (high - low + eps)
        mean_ratio = ratio.rolling(window).mean()
        raw = corr * mean_ratio
        # Normalize D: Sign/Binary
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_072_wf(df, window=70, p1=0.3):
        close = df['close']
        high = df['high']
        low = df['low']
        p2 = window
        eps = 1e-9
        # Compute raw
        x = close / (high + 1e-8)
        y = close / (low + 1e-8)
        x_mean = x.rolling(window).mean()
        y_mean = y.rolling(window).mean()
        cov_xy = (x * y).rolling(window).mean() - x_mean * y_mean
        var_x = (x**2).rolling(window).mean() - x_mean**2
        var_y = (y**2).rolling(window).mean() - y_mean**2
        corr = cov_xy / (var_x * var_y + 1e-8).clip(None, 0)
        corr = corr.fillna(0).replace([np.inf, -np.inf], 0)
        ratio = (close - low) / (high - low + 1e-8)
        mean_ratio = ratio.rolling(window).mean()
        raw = corr * mean_ratio
        # Winsorized Fisher
        low_p = raw.rolling(p2).quantile(p1)
        high_p = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_p, upper=high_p, axis=0)
        norm_val = ((winsorized - low_p) / (high_p - low_p + eps)) * 1.98 - 0.99
        norm_val = norm_val.clip(-0.99 + eps, 0.99 - eps)
        signal = np.arctanh(norm_val)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_073_rank(df, window_rank=20):
        # Tính toán phần trăm chênh lệch chuẩn hóa
        hilo = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Tính delta 2 bước và 1 bước
        delta_2 = hilo.diff(2)
        delta_1 = hilo.diff(1)

        # Tính raw signal
        raw = delta_2 - delta_1

        # Rolling rank và chuẩn hóa về [-1, 1]
        ranked = (raw.rolling(window_rank).rank(pct=True) * 2) - 1

        # Xử lý volume: log1p để giảm skew
        vol_log = np.log1p(df['volume'].fillna(0))

        # Volume ratio với rolling mean
        vol_ratio = df['volume'] / (df['volume'].rolling(3).mean() + 1e-8)

        # Kết hợp và scale
        signal = ranked * vol_ratio

        # Fill NaN và chuẩn hóa
        signal = signal.fillna(0)
        signal = (signal - signal.expanding().mean()) / (signal.expanding().std() + 1e-8)
        signal = signal.clip(-1, 1)

        return signal.bfill().fillna(0)

    @staticmethod
    def alpha_quanta_073_tanh(df, window_tanh=20):
        # Tính toán phần trăm chênh lệch chuẩn hóa
        hilo = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Tính delta 2 bước và 1 bước
        delta_2 = hilo.diff(2)
        delta_1 = hilo.diff(1)

        # Tính raw signal
        raw = delta_2 - delta_1

        # Dynamic Tanh normalization
        signal = np.tanh(raw / (raw.rolling(window_tanh).std() + 1e-8))

        # Xử lý volume: log1p để giảm skew
        vol = np.log1p(df['volume'].fillna(0))

        # Volume ratio với rolling mean
        vol_ratio = df['volume'] / (df['volume'].rolling(3).mean() + 1e-8)

        # Kết hợp và trọng số hóa
        signal = signal * vol_ratio

        # Chuẩn hóa cuối cùng về [-1, 1]
        signal = signal.fillna(0)
        signal = (signal - signal.expanding().mean()) / (signal.expanding().std() + 1e-8)
        signal = signal.clip(-1, 1)

        return signal.bfill().fillna(0)

    @staticmethod
    def alpha_quanta_073_zscore(df, window_z=20):
        # Tính toán phần trăm chênh lệch chuẩn hóa
        hilo = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Tính delta 2 bước và 1 bước
        delta_2 = hilo.diff(2)
        delta_1 = hilo.diff(1)

        # Tính raw signal (oscillator-type)
        raw = delta_2 - delta_1

        # Rolling Z-Score normalization
        mean = raw.rolling(window_z).mean()
        std = raw.rolling(window_z).std()
        signal = ((raw - mean) / (std + 1e-8)).clip(-1, 1)

        # Xử lý volume
        vol = np.log1p(df['volume'].fillna(0))
        vol_ratio = df['volume'] / (df['volume'].rolling(3).mean() + 1e-8)

        # Kết hợp
        signal = signal * vol_ratio

        # Chuẩn hóa cuối
        signal = signal.fillna(0)
        signal = (signal - signal.expanding().mean()) / (signal.expanding().std() + 1e-8)
        signal = signal.clip(-1, 1)

        return signal.bfill().fillna(0)

    @staticmethod
    def alpha_quanta_073_sign(df, window_sign=20):
        # Tính toán phần trăm chênh lệch chuẩn hóa
        hilo = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Tính delta 2 bước và 1 bước
        delta_2 = hilo.diff(2)
        delta_1 = hilo.diff(1)

        # Tính raw signal
        raw = delta_2 - delta_1

        # Sign/Binary Soft normalization (breakout/trend following)
        # Sử dụng rolling mean để làm mịn trước khi lấy dấu
        smoothed = raw.rolling(window_sign).mean()
        signal = np.sign(smoothed)

        # Xử lý volume
        vol = np.log1p(df['volume'].fillna(0))
        vol_ratio = df['volume'] / (df['volume'].rolling(3).mean() + 1e-8)

        # Kết hợp
        signal = signal * vol_ratio

        # Chuẩn hóa cuối
        signal = signal.fillna(0)
        signal = (signal - signal.expanding().mean()) / (signal.expanding().std() + 1e-8)
        signal = signal.clip(-1, 1)

        return signal.bfill().fillna(0)

    @staticmethod
    def alpha_quanta_073_wf(df, window_fish=20, quantile_factor=0.1):
        # Tính toán phần trăm chênh lệch chuẩn hóa
        hilo = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)

        # Tính delta 2 bước và 1 bước
        delta_2 = hilo.diff(2)
        delta_1 = hilo.diff(1)

        # Tính raw signal
        raw = delta_2 - delta_1

        # Winsorized Fisher normalization
        low = raw.rolling(window_fish).quantile(quantile_factor)
        high = raw.rolling(window_fish).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)

        # Xử lý volume
        vol = np.log1p(df['volume'].fillna(0))
        vol_ratio = df['volume'] / (df['volume'].rolling(3).mean() + 1e-8)

        # Kết hợp
        signal = signal * vol_ratio

        # Chuẩn hóa cuối cùng về [-1, 1]
        signal = signal.fillna(0)
        signal = (signal - signal.expanding().mean()) / (signal.expanding().std() + 1e-8)
        signal = signal.clip(-1, 1)

        return signal.bfill().fillna(0)

    @staticmethod
    def alpha_quanta_074_rank(df, window=100):
        ret = df['close'].diff() / df['close']
        mean_ret = ret.rolling(window).mean()
        delay_ret = ret.shift(window)
        mean_delay = delay_ret.rolling(window).mean()
        ratio = mean_ret / (mean_delay + 1e-8)
        raw = ratio.rolling(window).rank(pct=True) * 2 - 1
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_074_tanh(df, window=100):
        ret = df['close'].diff() / df['close']
        mean_ret = ret.rolling(window).mean()
        delay_ret = ret.shift(window)
        mean_delay = delay_ret.rolling(window).mean()
        ratio = mean_ret / (mean_delay + 1e-8)
        raw = np.tanh(ratio / ratio.rolling(window).std().replace(0, np.nan))
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_074_zscore(df, window=95):
        ret = df['close'].diff() / df['close']
        mean_ret = ret.rolling(window).mean()
        delay_ret = ret.shift(window)
        mean_delay = delay_ret.rolling(window).mean()
        ratio = mean_ret / (mean_delay + 1e-8)
        z = (ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan)
        raw = z.clip(-1, 1)
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_074_sign(df, window=100):
        ret = df['close'].diff() / df['close']
        mean_ret = ret.rolling(window).mean()
        delay_ret = ret.shift(window)
        mean_delay = delay_ret.rolling(window).mean()
        ratio = mean_ret / (mean_delay + 1e-8)
        raw = np.sign(ratio)
        return pd.Series(raw, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_074_wf(df, window=80, winsor_quantile=0.7):
        ret = df['close'].diff() / df['close']
        mean_ret = ret.rolling(window).mean()
        delay_ret = ret.shift(window)
        mean_delay = delay_ret.rolling(window).mean()
        ratio = mean_ret / (mean_delay + 1e-8)
        low = ratio.rolling(window).quantile(winsor_quantile)
        high = ratio.rolling(window).quantile(1 - winsor_quantile)
        winsorized = ratio.clip(lower=low, upper=high, axis=0)
        norm_factor = (winsorized - low) / (high - low + 1e-9)
        raw = np.arctanh(norm_factor * 1.98 - 0.99)
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_075_rank(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_lag = volume.shift(1)
        delta_volume_ratio = (volume - volume_lag) / (volume + 1e-8)
        ret = close.pct_change()
        cov = ret.rolling(window).cov(delta_volume_ratio)
        var_ret = ret.rolling(window).var()
        var_vol = delta_volume_ratio.rolling(window).var()
        std_ret = np.sqrt(var_ret)
        std_vol = np.sqrt(var_vol)
        corr = cov / (std_ret * std_vol + 1e-9)
        corr_ranked = (corr.rolling(window).rank(pct=True) * 2) - 1
        return corr_ranked.fillna(0)

    @staticmethod
    def alpha_quanta_075_tanh(df, window=15):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_lag = volume.shift(1)
        delta_volume_ratio = (volume - volume_lag) / (volume + 1e-8)
        ret = close.pct_change()
        cov = ret.rolling(window).cov(delta_volume_ratio)
        var_ret = ret.rolling(window).var()
        var_vol = delta_volume_ratio.rolling(window).var()
        std_ret = np.sqrt(var_ret)
        std_vol = np.sqrt(var_vol)
        corr = cov / (std_ret * std_vol + 1e-9)
        result = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_075_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_lag = volume.shift(1)
        delta_volume_ratio = (volume - volume_lag) / (volume + 1e-8)
        ret = close.pct_change()
        cov = ret.rolling(window).cov(delta_volume_ratio)
        var_ret = ret.rolling(window).var()
        var_vol = delta_volume_ratio.rolling(window).var()
        std_ret = np.sqrt(var_ret)
        std_vol = np.sqrt(var_vol)
        corr = cov / (std_ret * std_vol + 1e-9)
        mean_corr = corr.rolling(window).mean()
        std_corr = corr.rolling(window).std().replace(0, np.nan)
        result = ((corr - mean_corr) / std_corr).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_075_sign(df, window=15):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_lag = volume.shift(1)
        delta_volume_ratio = (volume - volume_lag) / (volume + 1e-8)
        ret = close.pct_change()
        cov = ret.rolling(window).cov(delta_volume_ratio)
        var_ret = ret.rolling(window).var()
        var_vol = delta_volume_ratio.rolling(window).var()
        std_ret = np.sqrt(var_ret)
        std_vol = np.sqrt(var_vol)
        corr = cov / (std_ret * std_vol + 1e-9)
        result = np.sign(corr)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_075_wf(df, window=20, sub_window=50):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_lag = volume.shift(1)
        delta_volume_ratio = (volume - volume_lag) / (volume + 1e-8)
        ret = close.pct_change()
        cov = ret.rolling(window).cov(delta_volume_ratio)
        var_ret = ret.rolling(window).var()
        var_vol = delta_volume_ratio.rolling(window).var()
        std_ret = np.sqrt(var_ret)
        std_vol = np.sqrt(var_vol)
        corr = cov / (std_ret * std_vol + 1e-9)
        low = corr.rolling(sub_window).quantile(0.25)
        high = corr.rolling(sub_window).quantile(0.75)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_076_k(df, window=90):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(5).std().rolling(10).mean() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_076_h(df, window=80):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(5).std().rolling(10).mean() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_076_e(df, window=5):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(5).std().rolling(10).mean() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_076_y(df):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(5).std().rolling(10).mean() + 1e-8)
        normalized = np.sign(raw)
        # window param giữ để đồng bộ, không dùng trong công thức
        return pd.Series(normalized, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_076_r(df, window=90, p1=0.1):
        raw = (df['high'] - df['low']).rolling(5).std() / ((df['high'] - df['low']).rolling(5).std().rolling(10).mean() + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_077_k(df, window=15):
        raw = df['close'].pct_change()
        pos_count = (raw > 0).rolling(window=window).sum()
        ratio = pos_count / window
        normalized = (ratio.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_077_h(df, window=20):
        raw = df['close'].pct_change()
        pos_count = (raw > 0).rolling(window=window).sum()
        ratio = pos_count / window
        normalized = np.tanh(ratio / ratio.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_077_e(df, window=15):
        raw = df['close'].pct_change()
        pos_count = (raw > 0).rolling(window=window).sum()
        ratio = pos_count / window
        mean = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        normalized = ((ratio - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_077_y(df, window=15):
        raw = df['close'].pct_change()
        pos_count = (raw > 0).rolling(window=window).sum()
        ratio = pos_count / window
        normalized = np.sign(ratio - 0.5)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_077_r(df, window=30, p1=0.1):
        p2 = window
        raw = df['close'].pct_change()
        pos_count = (raw > 0).rolling(window=window).sum()
        ratio = pos_count / window
        low = ratio.rolling(p2).quantile(p1)
        high = ratio.rolling(p2).quantile(1 - p1)
        winsorized = ratio.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_078_rank(df, window=25):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_078_tanh(df, window=15):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_078_zscore(df, window=25):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_078_sign(df, window=30):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_078_wf(df, window=40, winsor_quantile=0.3):
        ret = df['close'].pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(window).std() + 1e-8)
        p1 = winsor_quantile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_079_k(df, window=100):
        volume = df.get('matchingVolume', df['close'] * 0.2)
        volume_mean = volume.rolling(window=5).mean()
        raw_volume_ratio = volume / (volume_mean + 1e-8)
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        sign_delay = np.sign(delay_ret)
        raw = raw_volume_ratio * sign_delay
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_079_h(df, window=15):
        volume = df.get('matchingVolume', df['close'] * 0.2)
        volume_mean = volume.rolling(window=5).mean()
        raw_volume_ratio = volume / (volume_mean + 1e-8)
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        sign_delay = np.sign(delay_ret)
        raw = raw_volume_ratio * sign_delay
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_079_e(df, window=45):
        volume = df.get('matchingVolume', df['close'] * 0.2)
        volume_mean = volume.rolling(window=5).mean()
        raw_volume_ratio = volume / (volume_mean + 1e-8)
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        sign_delay = np.sign(delay_ret)
        raw = raw_volume_ratio * sign_delay
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_079_n(df):
        volume = df.get('matchingVolume', df['close'] * 0.2)
        volume_mean = volume.rolling(window=5).mean()
        raw_volume_ratio = volume / (volume_mean + 1e-8)
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        sign_delay = np.sign(delay_ret)
        raw = raw_volume_ratio * sign_delay
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_079_r(df, window=20):
        volume = df.get('matchingVolume', df['close'] * 0.2)
        volume_mean = volume.rolling(window=5).mean()
        raw_volume_ratio = volume / (volume_mean + 1e-8)
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        sign_delay = np.sign(delay_ret)
        raw = raw_volume_ratio * sign_delay
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        # Ensure range [-1,1]
        signal = signal.clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_080_rank(df, window_rank=65):
        # Assuming $return -is daily return: (close / close.shift(1)) - 1
        ret = df['close'].pct_change()
        # DELAY($return, 1) means shift forward by 1 -> look at previous return
        condition = ret.shift(1) < 0
        vol = df['matchingVolume']
        vol_mean = vol.rolling(window=window_rank).mean()
        # Replace 0 with NaN to avoid division by zero, fill later
        vol_mean_safe = vol_mean.replace(0, np.nan)
        raw = vol / (vol_mean_safe + 1e-8)
        # Apply condition: where previous return < 0, keep raw, else 0
        signal = np.where(condition, raw, 0)
        signal = pd.Series(signal, index=df.index)
        # Rolling Rank normalization (case A)
        norm = (signal.rolling(window=window_rank).rank(pct=True) * 2) - 1
        norm = norm.ffill().fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_080_tanh(df, window_tanh=70):
        ret = df['close'].pct_change()
        condition = ret.shift(1) < 0
        vol = df['matchingVolume']
        vol_mean = vol.rolling(window=window_tanh).mean().replace(0, np.nan)
        raw = vol / (vol_mean + 1e-8)
        signal = np.where(condition, raw, 0)
        signal = pd.Series(signal, index=df.index)
        # Dynamic Tanh normalization (case B)
        std = signal.rolling(window=window_tanh).std()
        norm = np.tanh(signal / (std + 1e-8))
        norm = norm.ffill().fillna(0)
        return -norm

    @staticmethod
    def alpha_quanta_080_zscore(df, window_z=70):
        ret = df['close'].pct_change()
        condition = ret.shift(1) < 0
        vol = df['matchingVolume']
        vol_mean = vol.rolling(window=window_z).mean().replace(0, np.nan)
        raw = vol / (vol_mean + 1e-8)
        signal = np.where(condition, raw, 0)
        signal = pd.Series(signal, index=df.index)
        # Rolling Z-Score/Clip normalization (case C)
        mean = signal.rolling(window=window_z).mean()
        std = signal.rolling(window=window_z).std()
        norm = ((signal - mean) / (std + 1e-8)).clip(-1, 1)
        norm = norm.ffill().fillna(0)
        return -norm

    @staticmethod
    def alpha_quanta_080_sign(df, window_sign=20):
        ret = df['close'].pct_change()
        condition = ret.shift(1) < 0
        vol = df['matchingVolume']
        vol_mean = vol.rolling(window=window_sign).mean().replace(0, np.nan)
        raw = vol / (vol_mean + 1e-8)
        signal = np.where(condition, raw, 0)
        signal = pd.Series(signal, index=df.index)
        # Sign/Binary Soft normalization (case D)
        norm = np.sign(signal)
        norm = norm.ffill().fillna(0)
        return -norm

    @staticmethod
    def alpha_quanta_080_wf(df, window_fisher=40, quantile_p=0.3):
        ret = df['close'].pct_change()
        condition = ret.shift(1) < 0
        vol = df['matchingVolume']
        vol_mean = vol.rolling(window=window_fisher).mean().replace(0, np.nan)
        raw = vol / (vol_mean + 1e-8)
        signal = np.where(condition, raw, 0)
        signal = pd.Series(signal, index=df.index)
        # Winsorized Fisher normalization (case E)
        low = signal.rolling(window=window_fisher).quantile(quantile_p)
        high = signal.rolling(window=window_fisher).quantile(1 - quantile_p)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        # Fisher Transform to [-1, 1]
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        # Handle infinity cases from arctanh
        norm = norm.replace([np.inf, -np.inf], np.nan)
        norm = norm.ffill().fillna(0)
        return -norm

    @staticmethod
    def alpha_quanta_081_rank(df, window=5):
        # Phân tích: Tính tương quan trượt giữa delta volume và delta close
        # Trường hợp A (Rolling Rank): Phù hợp để loại bỏ nhiễu và outliers, đưa về phân phối đồng nhất
        # Xử lý: Volume được log1p để giảm skew, không dùng dữ liệu tương lai, ffill NaN
        vol = df['matchingVolume'].astype(float)
        close = df['close'].astype(float)
        delta_vol = vol.diff(1)
        delta_close = close.diff(1)
        raw = delta_vol.rolling(window).corr(delta_close)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_081_tanh(df, window=10):
        # Phân tích: Tính tương quan trượt giữa delta volume và delta close
        # Trường hợp B (Dynamic Tanh): Giữ lại cường độ (magnitude) của tương quan
        # Xử lý: Volume được log1p để giảm skew, không dùng dữ liệu tương lai, ffill NaN
        vol = df['matchingVolume'].astype(float)
        close = df['close'].astype(float)
        delta_vol = vol.diff(1)
        delta_close = close.diff(1)
        raw = delta_vol.rolling(window).corr(delta_close)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_081_zscore(df, window=5):
        # Phân tích: Tính tương quan trượt giữa delta volume và delta close
        # Trường hợp C (Rolling Z-Score/Clip): Phù hợp với basis/spread/oscillator, tương quan là dạng spread
        # Xử lý: Volume được log1p để giảm skew, không dùng dữ liệu tương lai, ffill NaN
        vol = df['matchingVolume'].astype(float)
        close = df['close'].astype(float)
        delta_vol = vol.diff(1)
        delta_close = close.diff(1)
        raw = delta_vol.rolling(window).corr(delta_close)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_) / std_).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_081_sign(df, window=10):
        # Phân tích: Tính tương quan trượt giữa delta volume và delta close
        # Trường hợp D (Sign/Binary Soft): Dùng cho breakout hoặc trend following thuần túy, chỉ lấy hướng
        # Xử lý: Volume được log1p để giảm skew, không dùng dữ liệu tương lai, ffill NaN
        vol = df['matchingVolume'].astype(float)
        close = df['close'].astype(float)
        delta_vol = vol.diff(1)
        delta_close = close.diff(1)
        raw = delta_vol.rolling(window).corr(delta_close)
        signal = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_081_wf(df, p1=0.3, p2=10):
        # Phân tích: Tính tương quan trượt giữa delta volume và delta close
        # Trường hợp E (Winsorized Fisher): Xử lý heavy tails, outliers cực đoan, giữ cấu trúc phân phối
        # Xử lý: Volume được log1p để giảm skew, không dùng dữ liệu tương lai, ffill NaN
        vol = df['matchingVolume'].astype(float)
        close = df['close'].astype(float)
        delta_vol = vol.diff(1)
        delta_close = close.diff(1)
        raw = delta_vol.rolling(p2).corr(delta_close)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_082_rank(df, window=45):
        volume = df.get('matchingVolume', df.get('volume', df.get('matchingVolume', 1)))
        raw = volume.rolling(3).sum() / (volume.rolling(window).sum() + 1e-8)
        ret = df['close'].pct_change().shift(1)
        raw = raw * np.sign(ret)
        result = raw.rolling(window).rank(pct=True) * 2 - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_082_tanh(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', df.get('matchingVolume', 1)))
        raw = volume.rolling(3).sum() / (volume.rolling(window).sum() + 1e-8)
        ret = df['close'].pct_change().shift(1)
        raw = raw * np.sign(ret)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_082_zscore(df, window=40):
        volume = df.get('matchingVolume', df.get('volume', df.get('matchingVolume', 1)))
        raw = volume.rolling(3).sum() / (volume.rolling(window).sum() + 1e-8)
        ret = df['close'].pct_change().shift(1)
        raw = raw * np.sign(ret)
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean_) / std_).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_082_sign(df, window=55):
        volume = df.get('matchingVolume', df.get('volume', df.get('matchingVolume', 1)))
        raw = volume.rolling(3).sum() / (volume.rolling(window).sum() + 1e-8)
        ret = df['close'].pct_change().shift(1)
        raw = raw * np.sign(ret)
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_082_wf(df, window=60, q=0.1):
        volume = df.get('matchingVolume', df.get('volume', df.get('matchingVolume', 1)))
        raw = volume.rolling(3).sum() / (volume.rolling(window).sum() + 1e-8)
        ret = df['close'].pct_change().shift(1)
        raw = raw * np.sign(ret)
        low = raw.rolling(window).quantile(q)
        high = raw.rolling(window).quantile(1 - q)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_083_rank(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_rank = volume.rolling(window).rank(pct=True)
        volume_rank_norm = volume_rank * 2 - 1
        price = df['close']
        ret = price.pct_change()
        ret_delayed = ret.shift(1)
        sign = np.sign(ret_delayed)
        raw = volume_rank_norm * sign
        return raw

    @staticmethod
    def alpha_quanta_083_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        ret = df['close'].pct_change()
        ret_delayed = ret.shift(1)
        sign = np.sign(ret_delayed)
        raw = volume_log * sign
        return np.tanh(raw / raw.rolling(window).std())

    @staticmethod
    def alpha_quanta_083_zscore(df, window=90):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        ret = df['close'].pct_change()
        ret_delayed = ret.shift(1)
        sign = np.sign(ret_delayed)
        raw = volume_log * sign
        ma = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        z = (raw - ma) / std.replace(0, np.nan)
        return z.clip(-1, 1)

    @staticmethod
    def alpha_quanta_083_sign(df, window=20):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_rank = volume.rolling(window).rank(pct=True)
        volume_rank_norm = volume_rank * 2 - 1
        ret = df['close'].pct_change()
        ret_delayed = ret.shift(1)
        sign = np.sign(ret_delayed)
        raw = volume_rank_norm * sign
        return np.sign(raw)

    @staticmethod
    def alpha_quanta_083_wf(df, p1=0.7, p2=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        ret = df['close'].pct_change()
        ret_delayed = ret.shift(1)
        sign = np.sign(ret_delayed)
        raw = volume_log * sign
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_084_k(df, window=30, sub_window=10):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        cov = ret.rolling(window).cov(mean_ret)
        var = mean_ret.rolling(window).var().replace(0, np.nan)
        corr = cov / var
        vol_rank = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_084_h(df, window=30, sub_window=7):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        cov = ret.rolling(window).cov(mean_ret)
        var = mean_ret.rolling(window).var().replace(0, np.nan)
        corr = cov / var
        vol_rank = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_084_p(df, window=90, sub_window=20):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        cov = ret.rolling(window).cov(mean_ret)
        var = mean_ret.rolling(window).var().replace(0, np.nan)
        corr = cov / var
        vol_rank = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        signal = zscore.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_084_y(df, window=30, sub_window=40):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        cov = ret.rolling(window).cov(mean_ret)
        var = mean_ret.rolling(window).var().replace(0, np.nan)
        corr = cov / var
        vol_rank = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_084_r(df, window=30, p1=0.7, p2=60, sub_window=20):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        cov = ret.rolling(window).cov(mean_ret)
        var = mean_ret.rolling(window).var().replace(0, np.nan)
        corr = cov / var
        vol_rank = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_085_rank(df, window=35):
        mean_return = df['close'].pct_change().rolling(window).mean()
        std_return = df['close'].pct_change().rolling(window).std()
        raw = (mean_return / std_return.replace(0, np.nan)).fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        volume_factor = 1 - df['matchingVolume'].rolling(window).mean().rank(pct=True)
        signal = signal * volume_factor
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_085_tanh(df, window=15):
        mean_return = df['close'].pct_change().rolling(window).mean()
        std_return = df['close'].pct_change().rolling(window).std()
        raw = (mean_return / std_return.replace(0, np.nan)).fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        volume_factor = 1 - df['matchingVolume'].rolling(window).mean().rank(pct=True)
        signal = signal * volume_factor
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_085_zscore(df, window=5):
        mean_return = df['close'].pct_change().rolling(window).mean()
        std_return = df['close'].pct_change().rolling(window).std()
        raw = (mean_return / std_return.replace(0, np.nan)).fillna(0)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        volume_factor = 1 - df['matchingVolume'].rolling(window).mean().rank(pct=True)
        signal = signal * volume_factor
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_085_sign(df, window=20):
        mean_return = df['close'].pct_change().rolling(window).mean()
        std_return = df['close'].pct_change().rolling(window).std()
        raw = (mean_return / std_return.replace(0, np.nan)).fillna(0)
        signal = np.sign(raw)
        volume_factor = 1 - df['matchingVolume'].rolling(window).mean().rank(pct=True)
        signal = signal * volume_factor
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_085_wf(df, window=40, p1=0.3):
        mean_return = df['close'].pct_change().rolling(window).mean()
        std_return = df['close'].pct_change().rolling(window).std()
        raw = (mean_return / std_return.replace(0, np.nan)).fillna(0)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        volume_factor = 1 - df['matchingVolume'].rolling(window).mean().rank(pct=True)
        signal = signal * volume_factor
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_086_rank(df, window=100, vol_window=40):
        high_low = df['high'] - df['low']
        mean_hl = high_low.rolling(window).mean()
        corr = high_low.rolling(window).corr(mean_hl)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean()
        vol_rank = vol_mean.rolling(vol_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_086_tanh(df, window=60, vol_window=40):
        high_low = df['high'] - df['low']
        mean_hl = high_low.rolling(window).mean()
        corr = high_low.rolling(window).corr(mean_hl)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean()
        vol_rank = vol_mean.rolling(vol_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / (std.replace(0, np.nan)))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_086_zscore(df, window=50, vol_window=40):
        high_low = df['high'] - df['low']
        mean_hl = high_low.rolling(window).mean()
        corr = high_low.rolling(window).corr(mean_hl)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean()
        vol_rank = vol_mean.rolling(vol_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / (std.replace(0, np.nan))).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_086_sign(df, window=60, vol_window=30):
        high_low = df['high'] - df['low']
        mean_hl = high_low.rolling(window).mean()
        corr = high_low.rolling(window).corr(mean_hl)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean()
        vol_rank = vol_mean.rolling(vol_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_086_wf(df, window=100, vol_window=7, p1=0.05, p2=20):
        high_low = df['high'] - df['low']
        mean_hl = high_low.rolling(window).mean()
        corr = high_low.rolling(window).corr(mean_hl)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean()
        vol_rank = vol_mean.rolling(vol_window).rank(pct=True)
        raw = corr * (1 - vol_rank)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_087_rank(df, window=20, vol_window=5):
        # Calculate returns
        ret = df['close'].pct_change()
        # Calculate TS_MEAN of returns over window
        mean_ret = ret.rolling(window=window).mean()
        # Calculate median of the rolling means over the same window
        median_mean_ret = mean_ret.rolling(window=window).median()
        # Absolute difference
        raw = (mean_ret - median_mean_ret).abs()
        # Calculate volume mean over vol_window
        vol_mean = np.log1p(df['matchingVolume']).rolling(window=vol_window).mean()
        # Rank volume (1 = low, 0 = high? adjust: we want low volume to contribute positively)
        vol_rank = vol_mean.rank(pct=True)
        # Multiply raw by (1 - rank) so high raw and low volume gives high signal
        raw = raw * (1 - vol_rank)
        # Rolling Rank normalization (Case A)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_087_tanh(df, window=100, vol_window=30):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window=window).mean()
        median_mean_ret = mean_ret.rolling(window=window).median()
        raw = (mean_ret - median_mean_ret).abs()
        vol_mean = np.log1p(df['matchingVolume']).rolling(window=vol_window).mean()
        vol_rank = vol_mean.rank(pct=True)
        raw = raw * (1 - vol_rank)
        # Dynamic Tanh normalization (Case B)
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_087_zscore(df, window=30, vol_window=7):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window=window).mean()
        median_mean_ret = mean_ret.rolling(window=window).median()
        raw = (mean_ret - median_mean_ret).abs()
        vol_mean = np.log1p(df['matchingVolume']).rolling(window=vol_window).mean()
        vol_rank = vol_mean.rank(pct=True)
        raw = raw * (1 - vol_rank)
        # Rolling Z-Score/Clip (Case C)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_087_sign(df, window=10, vol_window=3):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window=window).mean()
        median_mean_ret = mean_ret.rolling(window=window).median()
        raw = (mean_ret - median_mean_ret).abs()
        vol_mean = np.log1p(df['matchingVolume']).rolling(window=vol_window).mean()
        vol_rank = vol_mean.rank(pct=True)
        raw = raw * (1 - vol_rank)
        # Sign/Binary Soft (Case D)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_087_wf(df, window=20, vol_window=7, p1=0.05, p2=60):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window=window).mean()
        median_mean_ret = mean_ret.rolling(window=window).median()
        raw = (mean_ret - median_mean_ret).abs()
        vol_mean = np.log1p(df['matchingVolume']).rolling(window=vol_window).mean()
        vol_rank = vol_mean.rank(pct=True)
        raw = raw * (1 - vol_rank)
        # Winsorized Fisher (Case E)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_088_rank(df, window=100, vol_window=5):
        ret = df['close'].pct_change().fillna(0)
        std_ret = ret.rolling(window).std()
        rolling_corr = ret.rolling(window).corr(std_ret).fillna(0)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean().fillna(0)
        vol_rank = (vol_mean.rolling(window).rank(pct=True) * 2) - 1
        signal = rolling_corr * (1 - vol_rank)
        signal = (signal.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_088_tanh(df, window=80, vol_window=1):
        ret = df['close'].pct_change().fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        rolling_corr = ret.rolling(window).corr(std_ret).fillna(0)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean().fillna(0)
        vol_norm = (vol_mean - vol_mean.rolling(window).mean()) / vol_mean.rolling(window).std().replace(0, np.nan)
        vol_rank = (vol_norm.rolling(window).rank(pct=True) * 2) - 1
        raw = rolling_corr * (1 - vol_rank)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_088_zscore(df, window=70, vol_window=10):
        ret = df['close'].pct_change().fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        rolling_corr = ret.rolling(window).corr(std_ret).fillna(0)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean().fillna(0)
        vol_z = (vol_mean - vol_mean.rolling(window).mean()) / vol_mean.rolling(window).std().replace(0, np.nan)
        vol_rank = vol_z.rolling(window).rank(pct=True)
        raw = rolling_corr * (1 - vol_rank)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_088_sign(df, window=90, vol_window=7):
        ret = df['close'].pct_change().fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        rolling_corr = ret.rolling(window).corr(std_ret).fillna(0)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean().fillna(0)
        vol_rank = vol_mean.rolling(window).rank(pct=True)
        raw = rolling_corr * (1 - vol_rank)
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_088_wf(df, window=100, vol_window=20, p1=0.05):
        p2 = window
        ret = df['close'].pct_change().fillna(0)
        std_ret = ret.rolling(window).std().fillna(0)
        rolling_corr = ret.rolling(window).corr(std_ret).fillna(0)
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(vol_window).mean().fillna(0)
        vol_rank = vol_mean.rolling(window).rank(pct=True)
        raw = rolling_corr * (1 - vol_rank)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_089_rank(df, window=30):
        raw = df['matchingVolume'].diff(1).rolling(window).corr(df['matchingVolume'].diff(1).shift(1))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_089_tanh(df, window=50):
        raw = df['matchingVolume'].diff(1).rolling(window).corr(df['matchingVolume'].diff(1).shift(1))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_089_zscore(df, window=30):
        raw = df['matchingVolume'].diff(1).rolling(window).corr(df['matchingVolume'].diff(1).shift(1))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_089_sign(df, window=95):
        raw = df['matchingVolume'].diff(1).rolling(window).corr(df['matchingVolume'].diff(1).shift(1))
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_089_wf(df, window=30, p1=0.7):
        raw = df['matchingVolume'].diff(1).rolling(window).corr(df['matchingVolume'].diff(1).shift(1))
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_090_rank(df, window=100):
        ret = df['close'].pct_change()
        pos_ret = ret.clip(0, None)
        neg_ret = ret.clip(None, 0)
        pos_ret_lag1 = pos_ret.shift(1)
        neg_ret_lag1 = neg_ret.shift(1)
        corr_pos = pos_ret.rolling(window).corr(pos_ret_lag1)
        corr_neg = neg_ret.rolling(window).corr(neg_ret_lag1)
        raw = corr_pos - corr_neg
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_090_tanh(df, window=70):
        ret = df['close'].pct_change()
        pos_ret = ret.clip(0, None)
        neg_ret = ret.clip(None, 0)
        pos_ret_lag1 = pos_ret.shift(1)
        neg_ret_lag1 = neg_ret.shift(1)
        corr_pos = pos_ret.rolling(window).corr(pos_ret_lag1)
        corr_neg = neg_ret.rolling(window).corr(neg_ret_lag1)
        raw = corr_pos - corr_neg
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_090_zscore(df, window=100):
        ret = df['close'].pct_change()
        pos_ret = ret.clip(0, None)
        neg_ret = ret.clip(None, 0)
        pos_ret_lag1 = pos_ret.shift(1)
        neg_ret_lag1 = neg_ret.shift(1)
        corr_pos = pos_ret.rolling(window).corr(pos_ret_lag1)
        corr_neg = neg_ret.rolling(window).corr(neg_ret_lag1)
        raw = corr_pos - corr_neg
        raw_ma = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - raw_ma) / raw_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_090_sign(df, window=70):
        ret = df['close'].pct_change()
        pos_ret = ret.clip(0, None)
        neg_ret = ret.clip(None, 0)
        pos_ret_lag1 = pos_ret.shift(1)
        neg_ret_lag1 = neg_ret.shift(1)
        corr_pos = pos_ret.rolling(window).corr(pos_ret_lag1)
        corr_neg = neg_ret.rolling(window).corr(neg_ret_lag1)
        raw = corr_pos - corr_neg
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_090_wf(df, window=30, sub_window=40):
        ret = df['close'].pct_change()
        pos_ret = ret.clip(0, None)
        neg_ret = ret.clip(None, 0)
        pos_ret_lag1 = pos_ret.shift(1)
        neg_ret_lag1 = neg_ret.shift(1)
        corr_pos = pos_ret.rolling(window).corr(pos_ret_lag1)
        corr_neg = neg_ret.rolling(window).corr(neg_ret_lag1)
        raw = corr_pos - corr_neg
        low = raw.rolling(sub_window).quantile(0.1)
        high = raw.rolling(sub_window).quantile(0.9)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        ratio = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        ratio = ratio.clip(-0.9999, 0.9999)
        normalized = np.arctanh(ratio)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_091_k(df, window1=70, window2=10):
        range_ = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window1).mean().replace(0, np.nan).ffill()
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window1).mean().replace(0, np.nan).ffill()
        raw = range_.rolling(window2).corr(vol_ratio)
        normalized = (raw.rolling(window1).rank(pct=True) * 2) - 1
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_091_h(df, window1=40, window2=3):
        range_ = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window1).mean().replace(0, np.nan).ffill()
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window1).mean().replace(0, np.nan).ffill()
        raw = range_.rolling(window2).corr(vol_ratio)
        normalized = np.tanh(raw / raw.rolling(window1).std().replace(0, np.nan).ffill())
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_091_p(df, window1=50, window2=7):
        range_ = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window1).mean().replace(0, np.nan).ffill()
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window1).mean().replace(0, np.nan).ffill()
        raw = range_.rolling(window2).corr(vol_ratio)
        normalized = ((raw - raw.rolling(window1).mean()) / raw.rolling(window1).std().replace(0, np.nan).ffill()).clip(-1, 1)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_091_y(df, window1=50, window2=5):
        range_ = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window1).mean().replace(0, np.nan).ffill()
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window1).mean().replace(0, np.nan).ffill()
        raw = range_.rolling(window2).corr(vol_ratio)
        normalized = np.sign(raw)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_091_r(df, window1=90, window2=7, p1=0.05):
        range_ = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window1).mean().replace(0, np.nan).ffill()
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window1).mean().replace(0, np.nan).ffill()
        raw = range_.rolling(window2).corr(vol_ratio)
        low = raw.rolling(window1).quantile(p1)
        high = raw.rolling(window1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_092_rank(df, window=100, delay=2):
        close = df['close'].ffill()
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).ffill()
        std = close.rolling(window=10).std().replace(0, np.nan).ffill()
        delay_std = std.shift(delay)
        slope = returns.rolling(window).cov(delay_std) / delay_std.rolling(window).var().replace(0, np.nan)
        raw = slope
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        raw = raw.fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_092_tanh(df, window=70, delay=4):
        close = df['close'].ffill()
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).ffill()
        std = close.rolling(window=10).std().replace(0, np.nan).ffill()
        delay_std = std.shift(delay)
        slope = returns.rolling(window).cov(delay_std) / delay_std.rolling(window).var().replace(0, np.nan)
        raw = slope
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_092_zscore(df, window=30, delay=3):
        close = df['close'].ffill()
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).ffill()
        std = close.rolling(window=10).std().replace(0, np.nan).ffill()
        delay_std = std.shift(delay)
        slope = returns.rolling(window).cov(delay_std) / delay_std.rolling(window).var().replace(0, np.nan)
        raw = slope
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_092_sign(df, window=40, delay=5):
        close = df['close'].ffill()
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).ffill()
        std = close.rolling(window=10).std().replace(0, np.nan).ffill()
        delay_std = std.shift(delay)
        slope = returns.rolling(window).cov(delay_std) / delay_std.rolling(window).var().replace(0, np.nan)
        raw = slope
        normalized = np.sign(raw)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_092_wf(df, window=40, delay=7, p1=0.05):
        close = df['close'].ffill()
        returns = close.pct_change().replace([np.inf, -np.inf], np.nan).ffill()
        std = close.rolling(window=10).std().replace(0, np.nan).ffill()
        delay_std = std.shift(delay)
        slope = returns.rolling(window).cov(delay_std) / delay_std.rolling(window).var().replace(0, np.nan)
        raw = slope
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_093_3(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        hl_range = high - low
        avg_hl = hl_range.rolling(window=window).mean()
        std_close = close.rolling(window=window).std()
        raw = avg_hl / (std_close + 1e-8)
        std_raw = raw.rolling(window=window).std()
        mean_raw = raw.rolling(window=window).mean()
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_093_rank(df, window=25):
        raw = (df['high'] - df['low']).rolling(window).mean() / (df['close'].rolling(window).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal

    @staticmethod
    def alpha_quanta_093_tanh(df, window=20):
        raw = (df['high'] - df['low']).rolling(window).mean() / (df['close'].rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal

    @staticmethod
    def alpha_quanta_093_zscore(df, window=30):
        raw = (df['high'] - df['low']).rolling(window).mean() / (df['close'].rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_093_sign(df, window=85):
        raw = (df['high'] - df['low']).rolling(window).mean() / (df['close'].rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_093_wf(df, window=30, sub_window=40):
        raw = (df['high'] - df['low']).rolling(window).mean() / (df['close'].rolling(window).std() + 1e-8)
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_094_rank(df, window=85):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', 1)
        raw = ((open_ - close.shift(1)) / (close.rolling(5).std() + 1e-8)) * np.sign(volume / (volume.rolling(5).mean() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_094_tanh(df, window=5):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', 1)
        raw = ((open_ - close.shift(1)) / (close.rolling(5).std() + 1e-8)) * np.sign(volume / (volume.rolling(5).mean() + 1e-8))
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_094_zscore(df, window=95):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', 1)
        raw = ((open_ - close.shift(1)) / (close.rolling(5).std() + 1e-8)) * np.sign(volume / (volume.rolling(5).mean() + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_094_sign(df, window=75):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', 1)
        raw = ((open_ - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign(volume / (volume.rolling(window).mean() + 1e-8))
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_094_wf(df, p2=30):
        close = df['close']
        open_ = df['open']
        volume = df.get('matchingVolume', 1)
        raw = ((open_ - close.shift(1)) / (close.rolling(5).std() + 1e-8)) * np.sign(volume / (volume.rolling(5).mean() + 1e-8))
        low = raw.rolling(p2).quantile(0.05)
        high = raw.rolling(p2).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_095_k(df, window=30):
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_095_h(df, window=25):
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_095_e(df, window=65):
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_095_t(df, window=85):
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_095_r(df, window=70, tail_percentile=0.7):
        vol = df.get('matchingVolume', df.get('volume', 1))
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std() + 1e-8)
        p2 = min(len(df), max(2, int(window * 2)))
        low = raw.rolling(p2).quantile(tail_percentile)
        high = raw.rolling(p2).quantile(1 - tail_percentile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_096_k(df, window=30):
        close = df['close']
        mean = close.rolling(window).mean()
        max_val = close.rolling(window).max()
        min_val = close.rolling(window).min()
        raw = (close - mean) / (max_val - min_val + 1e-8)
        sign = pd.Series(np.where(raw > 0, 1.0, np.where(raw < 0, -1.0, 0.0)), index=df.index)
        signal = (sign.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(method='ffill').fillna(0)

    @staticmethod
    def alpha_quanta_096_h(df, window=10):
        close = df['close']
        mean = close.rolling(window).mean()
        max_val = close.rolling(window).max()
        min_val = close.rolling(window).min()
        raw = (close - mean) / (max_val - min_val + 1e-8)
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return signal.fillna(method='ffill').fillna(0)

    @staticmethod
    def alpha_quanta_096_e(df, window=10):
        close = df['close']
        mean = close.rolling(window).mean()
        max_val = close.rolling(window).max()
        min_val = close.rolling(window).min()
        raw = (close - mean) / (max_val - min_val + 1e-8)
        z = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        signal = z.clip(-1, 1)
        return signal.fillna(method='ffill').fillna(0)

    @staticmethod
    def alpha_quanta_096_y(df, window=5):
        close = df['close']
        mean = close.rolling(window).mean()
        max_val = close.rolling(window).max()
        min_val = close.rolling(window).min()
        raw = (close - mean) / (max_val - min_val + 1e-8)
        signal = pd.Series(np.sign(raw), index=df.index)
        return signal.fillna(method='ffill').fillna(0)

    @staticmethod
    def alpha_quanta_096_r(df, window=10, p1=0.1):
        p2 = window
        close = df['close']
        mean = close.rolling(window).mean()
        max_val = close.rolling(window).max()
        min_val = close.rolling(window).min()
        raw = (close - mean) / (max_val - min_val + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(method='ffill').fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_097_k(df, window=65):
        close = df['close']
        volume = df['matchingVolume']
        open_p = df['open']
        delta_close = close.diff() / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        std = close.rolling(window).std()
        delay_close = close.shift(1)
        raw = corr * ((open_p - delay_close) / (std + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_097_h(df, window=65):
        close = df['close']
        volume = df['matchingVolume']
        open_p = df['open']
        delta_close = close.diff() / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        std = close.rolling(window).std()
        delay_close = close.shift(1)
        raw = corr * ((open_p - delay_close) / (std + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_097_e(df, window=65):
        close = df['close']
        volume = df['matchingVolume']
        open_p = df['open']
        delta_close = close.diff() / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        std = close.rolling(window).std()
        delay_close = close.shift(1)
        raw = corr * ((open_p - delay_close) / (std + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill()
        mean = raw.rolling(window).mean()
        std_series = raw.rolling(window).std()
        signal = ((raw - mean) / std_series.replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_097_y(df, window=45):
        close = df['close']
        volume = df['matchingVolume']
        open_p = df['open']
        delta_close = close.diff() / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        std = close.rolling(window).std()
        delay_close = close.shift(1)
        raw = corr * ((open_p - delay_close) / (std + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill()
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_097_r(df, window=80, p1=0.7):
        close = df['close']
        volume = df['matchingVolume']
        open_p = df['open']
        delta_close = close.diff() / (close + 1e-8)
        delta_volume = volume.diff() / (volume + 1e-8)
        corr = delta_close.rolling(window).corr(delta_volume)
        std = close.rolling(window).std()
        delay_close = close.shift(1)
        raw = corr * ((open_p - delay_close) / (std + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill()
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_098_rank(df, window=90):
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        delta_open = open_price - close.shift(1)
        std_close = close.rolling(window).std()
        raw = (delta_open / (std_close + 1e-8)) * (volume / (volume.rolling(window).mean() + 1e-8))
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_098_tanh(df, window=100):
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        delta_open = open_price - close.shift(1)
        std_close = close.rolling(window).std()
        raw = (delta_open / (std_close + 1e-8)) * (volume / (volume.rolling(window).mean() + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_098_zscore(df, window=100):
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        delta_open = open_price - close.shift(1)
        std_close = close.rolling(window).std()
        raw = (delta_open / (std_close + 1e-8)) * (volume / (volume.rolling(window).mean() + 1e-8))
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        normalized = ((raw - mean_raw) / (std_raw + 1e-8)).clip(-1, 1)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_098_sign(df, window=80):
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        delta_open = open_price - close.shift(1)
        std_close = close.rolling(window).std()
        raw = (delta_open / (std_close + 1e-8)) * (volume / (volume.rolling(window).mean() + 1e-8))
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_098_wf(df, window=60, quantile=0.9):
        close = df['close']
        open_price = df['open']
        volume = df['matchingVolume']
        delta_open = open_price - close.shift(1)
        std_close = close.rolling(window).std()
        raw = (delta_open / (std_close + 1e-8)) * (volume / (volume.rolling(window).mean() + 1e-8))
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_099_rank(df, window=90):
        close_shifted = df['close'].shift(1)
        raw = (df['open'] - close_shifted) / close_shifted
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vol_ma = vol_ratio.rolling(window).mean()
        signal = rank * vol_ma
        signal = (signal - signal.rolling(window).mean()) / signal.rolling(window).std().replace(0, np.nan)
        signal = signal.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_099_tanh(df, window=85):
        close_shifted = df['close'].shift(1)
        raw = (df['open'] - close_shifted) / close_shifted
        raw = raw / raw.rolling(window).std().replace(0, np.nan)
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vol_ma = vol_ratio.rolling(window).mean()
        signal = rank * vol_ma
        signal = np.tanh(signal / signal.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_099_zscore(df, window=5):
        close_shifted = df['close'].shift(1)
        raw = (df['open'] - close_shifted) / close_shifted
        rank = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        rank = rank.clip(-1, 1)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vol_ma = vol_ratio.rolling(window).mean()
        signal = rank * vol_ma
        signal = (signal - signal.rolling(window).mean()) / signal.rolling(window).std().replace(0, np.nan)
        signal = signal.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_099_sign(df, window=90):
        close_shifted = df['close'].shift(1)
        raw = (df['open'] - close_shifted) / close_shifted
        rank = np.sign(raw)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vol_ma = vol_ratio.rolling(window).mean()
        signal = rank * vol_ma
        signal = (signal - signal.rolling(window).mean()) / signal.rolling(window).std().replace(0, np.nan)
        signal = signal.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_099_wf(df, window=60, p1=0.7):
        close_shifted = df['close'].shift(1)
        raw = (df['open'] - close_shifted) / close_shifted
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()
        vol_ma = vol_ratio.rolling(window).mean()
        signal = rank * vol_ma
        low = signal.rolling(window).quantile(p1)
        high = signal.rolling(window).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_100_rank(df, window=100):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_mean = vol.rolling(window).mean()
        vol_ratio = vol / (vol_mean + 1e-8)
        raw = np.sign(vol_ratio - vol_ratio.shift(1)) * vol_ratio
        alpha = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -alpha.fillna(0)

    @staticmethod
    def alpha_quanta_100_tanh(df, window=50):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_mean = vol.rolling(window).mean()
        vol_ratio = vol / (vol_mean + 1e-8)
        raw = np.sign(vol_ratio - vol_ratio.shift(1)) * vol_ratio
        alpha = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -alpha.fillna(0)

    @staticmethod
    def alpha_quanta_100_zscore(df, window=100):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_mean = vol.rolling(window).mean()
        vol_ratio = vol / (vol_mean + 1e-8)
        raw = np.sign(vol_ratio - vol_ratio.shift(1)) * vol_ratio
        alpha = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -alpha.fillna(0)

    @staticmethod
    def alpha_quanta_100_sign(df, window=45):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_mean = vol.rolling(window).mean()
        vol_ratio = vol / (vol_mean + 1e-8)
        raw = np.sign(vol_ratio - vol_ratio.shift(1)) * vol_ratio
        alpha = np.sign(raw)
        return alpha.fillna(0)

    @staticmethod
    def alpha_quanta_100_wf(df, window_rank=60, quantile=0.1):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_mean = vol.rolling(window_rank).mean()
        vol_ratio = vol / (vol_mean + 1e-8)
        raw = np.sign(vol_ratio - vol_ratio.shift(1)) * vol_ratio
        low = raw.rolling(window_rank).quantile(quantile)
        high = raw.rolling(window_rank).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_101_rank(df, window=5):
        close = df['close']
        open = df['open']
        ret_today = (close - open) / (open + 1e-5)
        retro_yest = (open - close.shift(1)) / (close.shift(1) + 1e-5)
        corr = ret_today.rolling(window).corr(retro_yest).fillna(0).replace([np.inf, -np.inf], 0)
        raw = corr
        param = window
        result = (raw.rolling(param).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_101_tanh(df, window=5):
        close = df['close']
        open = df['open']
        ret_today = (close - open) / (open + 1e-5)
        retro_yest = (open - close.shift(1)) / (close.shift(1) + 1e-5)
        corr = ret_today.rolling(window).corr(retro_yest).fillna(0).replace([np.inf, -np.inf], 0)
        raw = corr
        param = window
        result = np.tanh(raw / raw.rolling(param).std().replace(0, np.nan)).fillna(0)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_101_zscore(df, window=5):
        close = df['close']
        open = df['open']
        ret_today = (close - open) / (open + 1e-5)
        retro_yest = (open - close.shift(1)) / (close.shift(1) + 1e-5)
        corr = ret_today.rolling(window).corr(retro_yest).fillna(0).replace([np.inf, -np.inf], 0)
        raw = corr
        param = window
        result = ((raw - raw.rolling(param).mean()) / raw.rolling(param).std().replace(0, np.nan)).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_101_sign(df, window=5):
        close = df['close']
        open = df['open']
        ret_today = (close - open) / (open + 1e-5)
        retro_yest = (open - close.shift(1)) / (close.shift(1) + 1e-5)
        corr = ret_today.rolling(window).corr(retro_yest).fillna(0).replace([np.inf, -np.inf], 0)
        result = np.sign(corr)
        return -pd.Series(result, index=df.index)

    @staticmethod
    def alpha_quanta_101_wf(df, window=20, sub_window=7):
        close = df['close']
        open = df['open']
        ret_today = (close - open) / (open + 1e-5)
        retro_yest = (open - close.shift(1)) / (close.shift(1) + 1e-5)
        corr = ret_today.rolling(window).corr(retro_yest).fillna(0).replace([np.inf, -np.inf], 0)
        raw = corr
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        ratio = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(ratio * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_102_rank(df, window=60):
        # Tính log return đơn giản hơn: close shift 1
        close_shift1 = df['close'].shift(1)
        # Tính tỷ lệ biến động giá
        price_change = (df['open'] - close_shift1) / (close_shift1 + 1e-9)
        # Tính tỷ lệ khối lượng với MA 20
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(20).mean() + 1e-9)
        # Tín hiệu dấu
        sign_price = np.sign(price_change)
        # Kết hợp
        raw = sign_price * vol_ratio
        # Rolling mean
        alpha_raw = raw.rolling(window).mean()
        # Chuẩn hóa Rolling Rank
        alpha_rank = (alpha_raw.rolling(window).rank(pct=True) * 2) - 1
        return alpha_rank

    @staticmethod
    def alpha_quanta_102_tanh(df, window=25):
        close_shift1 = df['close'].shift(1)
        price_change = (df['open'] - close_shift1) / (close_shift1 + 1e-9)
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(20).mean() + 1e-9)
        sign_price = np.sign(price_change)
        raw = sign_price * vol_ratio
        alpha_raw = raw.rolling(window).mean()
        # Chuẩn hóa Dynamic Tanh
        alpha_tanh = np.tanh(alpha_raw / (alpha_raw.rolling(window).std() + 1e-9))
        return alpha_tanh

    @staticmethod
    def alpha_quanta_102_zscore(df, window=60):
        close_shift1 = df['close'].shift(1)
        price_change = (df['open'] - close_shift1) / (close_shift1 + 1e-9)
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(20).mean() + 1e-9)
        sign_price = np.sign(price_change)
        raw = sign_price * vol_ratio
        alpha_raw = raw.rolling(window).mean()
        # Chuẩn hóa Rolling Z-Score
        mean = alpha_raw.rolling(window).mean()
        std = alpha_raw.rolling(window).std()
        alpha_zscore = ((alpha_raw - mean) / (std + 1e-9)).clip(-1, 1)
        return alpha_zscore

    @staticmethod
    def alpha_quanta_102_sign(df, window=15):
        close_shift1 = df['close'].shift(1)
        price_change = (df['open'] - close_shift1) / (close_shift1 + 1e-9)
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(20).mean() + 1e-9)
        sign_price = np.sign(price_change)
        raw = sign_price * vol_ratio
        alpha_raw = raw.rolling(window).mean()
        # Chuẩn hóa Sign/Binary Soft
        alpha_sign = np.sign(alpha_raw)
        return alpha_sign

    @staticmethod
    def alpha_quanta_102_wf(df, window=60, p1=0.1):
        close_shift1 = df['close'].shift(1)
        price_change = (df['open'] - close_shift1) / (close_shift1 + 1e-9)
        vol_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(20).mean() + 1e-9)
        sign_price = np.sign(price_change)
        raw = sign_price * vol_ratio
        alpha_raw = raw.rolling(window).mean()
        # Chuẩn hóa Winsorized Fisher
        low = alpha_raw.rolling(window).quantile(p1)
        high = alpha_raw.rolling(window).quantile(1 - p1)
        winsorized = alpha_raw.clip(lower=low, upper=high, axis=0)
        numerator = winsorized - low
        denominator = high - low + 1e-9
        normalized = np.arctanh((numerator / denominator) * 1.98 - 0.99)
        # Xử lý inf do arctanh
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_103_3(df, window=15):
        # Calculate price return
        price_return -= (df['open'] - df['close'].shift(1)) / (df['close'].shift(1).replace(0, np.nan) + 1e-8)
        # Z-score of price return over window
        zscore_price = (price_return - price_return.rolling(window).mean()) / price_return.rolling(window).std().replace(0, np.nan)
        # Sign of zscore_price for binary directionality
        sign_price = np.sign(zscore_price)
        # Volume ratio
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].shift(1).replace(0, np.nan) + 1e-8)
        # Z-score of volume ratio over window
        zscore_vol = (volume_ratio - volume_ratio.rolling(window).mean()) / volume_ratio.rolling(window).std().replace(0, np.nan)
        # Combine: sign * volume zscore
        raw = sign_price * zscore_vol
        # Normalize using Rolling Z-Score / Clip (Case C)
        norm = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        signal = norm.clip(-1, 1).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_103_rank(df, window=95):
        raw1 = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        z1 = (raw1 - raw1.rolling(window).mean()) / raw1.rolling(window).std().replace(0, np.nan)
        raw2 = df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8)
        z2 = (raw2 - raw2.rolling(window).mean()) / raw2.rolling(window).std().replace(0, np.nan)
        raw = pd.Series(np.sign(z1), index=df.index) * z2
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_103_tanh(df, window=5):
        raw1 = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        z1 = (raw1 - raw1.rolling(window).mean()) / raw1.rolling(window).std().replace(0, np.nan)
        raw2 = df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8)
        z2 = (raw2 - raw2.rolling(window).mean()) / raw2.rolling(window).std().replace(0, np.nan)
        raw = pd.Series(np.sign(z1), index=df.index) * z2
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_103_zscore(df, window=15):
        raw1 = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        z1 = (raw1 - raw1.rolling(window).mean()) / raw1.rolling(window).std().replace(0, np.nan)
        raw2 = df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8)
        z2 = (raw2 - raw2.rolling(window).mean()) / raw2.rolling(window).std().replace(0, np.nan)
        raw = pd.Series(np.sign(z1), index=df.index) * z2
        ma_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - ma_raw) / std_raw).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_103_sign(df, window=45):
        raw1 = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        z1 = (raw1 - raw1.rolling(window).mean()) / raw1.rolling(window).std().replace(0, np.nan)
        raw2 = df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8)
        z2 = (raw2 - raw2.rolling(window).mean()) / raw2.rolling(window).std().replace(0, np.nan)
        raw = pd.Series(np.sign(z1), index=df.index) * z2
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_103_wf(df, window=50, winsor_quantile=0.9):
        raw1 = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        z1 = (raw1 - raw1.rolling(window).mean()) / raw1.rolling(window).std().replace(0, np.nan)
        raw2 = df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8)
        z2 = (raw2 - raw2.rolling(window).mean()) / raw2.rolling(window).std().replace(0, np.nan)
        raw = pd.Series(np.sign(z1), index=df.index) * z2
        low = raw.rolling(window).quantile(winsor_quantile)
        high = raw.rolling(window).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_104_rank(df, window=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        open_ = df['open']
        raw = np.sign(open_ - close.shift(1)) * ((volume - volume.rolling(5).mean()) / (volume.rolling(5).mean() + 1e-8))
        raw = raw.ffill().fillna(0)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_104_tanh(df, window=65):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        open_ = df['open']
        raw = np.sign(open_ - close.shift(1)) * ((volume - volume.rolling(5).mean()) / (volume.rolling(5).mean() + 1e-8))
        raw = raw.ffill().fillna(0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_104_zscore(df, window=75):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        open_ = df['open']
        raw = np.sign(open_ - close.shift(1)) * ((volume - volume.rolling(5).mean()) / (volume.rolling(5).mean() + 1e-8))
        raw = raw.ffill().fillna(0)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_104_sign(df, param2=50):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        open_ = df['open']
        raw = np.sign(open_ - close.shift(1)) * ((volume - volume.rolling(param2).mean()) / (volume.rolling(param2).mean() + 1e-8))
        raw = raw.ffill().fillna(0)
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_104_wf(df, p1=0.1, p2=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        open_ = df['open']
        raw = np.sign(open_ - close.shift(1)) * ((volume - volume.rolling(5).mean()) / (volume.rolling(5).mean() + 1e-8))
        raw = raw.ffill().fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_105_rank(df, window=30):
        volume_ratio = df['close'] * df.get('matchingVolume', df.get('volume', 1)) / (df['close'] * df.get('matchingVolume', df.get('volume', 1)).rolling(window, min_periods=1).mean() + 1e-8)
        raw = volume_ratio * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_105_tanh(df, window=40):
        volume_ratio = df['close'] * df.get('matchingVolume', df.get('volume', 1)) / (df['close'] * df.get('matchingVolume', df.get('volume', 1)).rolling(window, min_periods=1).mean() + 1e-8)
        raw = volume_ratio * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std() + 1e-8)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_105_zscore(df, window=15):
        volume_ratio = df['close'] * df.get('matchingVolume', df.get('volume', 1)) / (df['close'] * df.get('matchingVolume', df.get('volume', 1)).rolling(window, min_periods=1).mean() + 1e-8)
        raw = volume_ratio * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_105_sign(df, window=55):
        volume_ratio = df['close'] * df.get('matchingVolume', df.get('volume', 1)) / (df['close'] * df.get('matchingVolume', df.get('volume', 1)).rolling(window, min_periods=1).mean() + 1e-8)
        raw = volume_ratio * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_105_wf(df, window=20, p1=0.1):
        p2 = window
        volume_ratio = df['close'] * df.get('matchingVolume', df.get('volume', 1)) / (df['close'] * df.get('matchingVolume', df.get('volume', 1)).rolling(window, min_periods=1).mean() + 1e-8)
        raw = volume_ratio * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_106_k(df, window=15):
        raw = df['close'].rolling(window).corr(df['volume'] / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan), df.get('matchingVolume', df.get('volume', 1)))
        norm = pd.Series(((raw.rolling(window).rank(pct=True) * 2) - 1), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_106_h(df, window=15):
        raw = df['close'].rolling(window).corr(df['volume'] / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan), df.get('matchingVolume', df.get('volume', 1)))
        norm = pd.Series(np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_106_e(df, window=15):
        raw = df['close'].rolling(window).corr(df['volume'] / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan), df.get('matchingVolume', df.get('volume', 1)))
        norm = pd.Series(((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_106_y(df, window=15):
        raw = df['close'].rolling(window).corr(df['volume'] / df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan), df.get('matchingVolume', df.get('volume', 1)))
        norm = pd.Series(np.sign(raw), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_106_r(df, window_rank=15, window_winsor=30):
        raw = df['close'].rolling(15).corr(df['volume'] / df.get('matchingVolume', df.get('volume', 1)).rolling(15).mean().replace(0, np.nan), df.get('matchingVolume', df.get('volume', 1)))
        low = raw.rolling(window_winsor).quantile(0.05)
        high = raw.rolling(window_winsor).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = pd.Series(np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_107_rank(df, window=20):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_ret = ret.rolling(window, min_periods=1).mean()
        vol_pctchg = volume.pct_change(periods=window)
        raw = np.sign(mean_ret) * vol_pctchg
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_107_tanh(df, window=45):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_ret = ret.rolling(window, min_periods=1).mean()
        vol_pctchg = volume.pct_change(periods=window)
        raw = np.sign(mean_ret) * vol_pctchg
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_107_zscore(df, window=45):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_ret = ret.rolling(window, min_periods=1).mean()
        vol_pctchg = volume.pct_change(periods=window)
        raw = np.sign(mean_ret) * vol_pctchg
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_107_sign(df, window=20):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_ret = ret.rolling(window, min_periods=1).mean()
        vol_pctchg = volume.pct_change(periods=window)
        raw = np.sign(mean_ret) * vol_pctchg
        normalized = np.sign(raw)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_107_wf(df, window=20, p1=0.7):
        ret = df['close'].pct_change()
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_ret = ret.rolling(window, min_periods=1).mean()
        vol_pctchg = volume.pct_change(periods=window)
        raw = np.sign(mean_ret) * vol_pctchg
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_108_rank(df, window=100, sub_window=1):
        close = df['close']
        open = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = ((open - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * (volume / (volume.rolling(sub_window).mean() + 1e-8))
        signal = (raw.rolling(5).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_108_tanh(df, window=20, sub_window=6):
        close = df['close']
        open = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = ((open - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * (volume / (volume.rolling(sub_window).mean() + 1e-8))
        signal = np.tanh(raw / raw.rolling(10).std().replace(0, np.nan).fillna(1))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_108_zscore(df, window=30, sub_window=4):
        close = df['close']
        open = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = ((open - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * (volume / (volume.rolling(sub_window).mean() + 1e-8))
        signal = ((raw - raw.rolling(20).mean()) / raw.rolling(20).std().replace(0, np.nan).fillna(1)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_108_sign(df, window=80, sub_window=2):
        close = df['close']
        open = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = ((open - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * (volume / (volume.rolling(sub_window).mean() + 1e-8))
        signal = np.sign(raw)
        return signal.fillna(0).astype(float)

    @staticmethod
    def alpha_quanta_108_wf(df, window=50, sub_window=5):
        close = df['close']
        open = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = ((open - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * (volume / (volume.rolling(sub_window).mean() + 1e-8))
        p1 = 0.05
        p2 = 20
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_109_rank(df, window=80):
        ret = df['open'].diff() / df['open']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = ret.rolling(5).corr(vol)
        sign_corr = np.sign(corr_5)
        raw = sign_corr * corr_5 / (df['close'].sub(df['open']).div(df['open']).rolling(window).std().add(1e-8))
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_109_tanh(df, window=90):
        ret = df['open'].diff() / df['open']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = ret.rolling(5).corr(vol)
        sign_corr = np.sign(corr_5)
        raw = sign_corr * corr_5 / (df['close'].sub(df['open']).div(df['open']).rolling(window).std().add(1e-8))
        norm = np.tanh(raw / raw.abs().rolling(window).mean().add(1e-8))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_109_zscore(df, window=60):
        ret = df['open'].diff() / df['open']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = ret.rolling(5).corr(vol)
        sign_corr = np.sign(corr_5)
        raw = sign_corr * corr_5 / (df['close'].sub(df['open']).div(df['open']).rolling(window).std().add(1e-8))
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().add(1e-8)).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_109_sign(df, window=70):
        ret = df['open'].diff() / df['open']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = ret.rolling(5).corr(vol)
        sign_corr = np.sign(corr_5)
        raw = sign_corr * corr_5 / (df['close'].sub(df['open']).div(df['open']).rolling(window).std().add(1e-8))
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_109_wf(df, window=20, sub_window=20):
        ret = df['open'].diff() / df['open']
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr_5 = ret.rolling(5).corr(vol)
        sign_corr = np.sign(corr_5)
        raw = sign_corr * corr_5 / (df['close'].sub(df['open']).div(df['open']).rolling(window).std().add(1e-8))
        low = raw.rolling(sub_window).quantile(0.1)
        high = raw.rolling(sub_window).quantile(0.9)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_110_rank(df, window=40):
        # Tính log return đơn giản
        ret = np.log(df['close'] / df['close'].shift(1))
        # Tổng return 20 kỳ (TS_SUM)
        sum_ret = ret.rolling(window=window).sum()
        # Median của tổng return 20 kỳ
        median_sum_ret = sum_ret.rolling(window=window).median()
        # Raw signal = sum_ret - median_sum_ret
        raw = sum_ret - median_sum_ret
        # Chuẩn hóa Rolling Rank (A): loại bỏ outlier, đưa về phân phối đều [-1,1]
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_110_tanh(df, window=30):
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_ret = ret.rolling(window=window).sum()
        median_sum_ret = sum_ret.rolling(window=window).median()
        raw = sum_ret - median_sum_ret
        # Chuẩn hóa Dynamic Tanh (B): giữ cường độ tín hiệu
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std_raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_110_zscore(df, window=40):
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_ret = ret.rolling(window=window).sum()
        median_sum_ret = sum_ret.rolling(window=window).median()
        raw = sum_ret - median_sum_ret
        # Chuẩn hóa Z-Score/Clip (C): phù hợp với oscillator
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_110_sign(df, window=35):
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_ret = ret.rolling(window=window).sum()
        median_sum_ret = sum_ret.rolling(window=window).median()
        raw = sum_ret - median_sum_ret
        # Chuẩn hóa Sign (D): tín hiệu binary direction
        signal = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_110_wf(df, window=30, p1=0.1):
        ret = np.log(df['close'] / df['close'].shift(1))
        sum_ret = ret.rolling(window=window).sum()
        median_sum_ret = sum_ret.rolling(window=window).median()
        raw = sum_ret - median_sum_ret
        # Chuẩn hóa Winsorized Fisher (E): xử lý heavy tails
        p2 = window  # hardcode rolling window p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        ratio = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        ratio = ratio.clip(-0.99, 0.99)
        signal = np.arctanh(ratio)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_111_k(df, window=55):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(10).mean()) / (volume.diff().rolling(10).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_111_h(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(10).mean()) / (volume.diff().rolling(10).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_111_e(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(10).mean()) / (volume.diff().rolling(10).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_111_y(df):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(10).mean()) / (volume.diff().rolling(10).std() + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_111_r(df, window=100, p1=0.9):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(10).mean()) / (volume.diff().rolling(10).std() + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_112_rank(df, window=60):
        hilo = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).std().replace(0, np.nan)
        corr = hilo.rolling(window).cov(volume_z) / (hilo.rolling(window).std() * volume_z.rolling(window).std()).replace(0, np.nan)
        raw = corr.fillna(0)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_112_tanh(df, window=5):
        hilo = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).std().replace(0, np.nan)
        corr = hilo.rolling(window).cov(volume_z) / (hilo.rolling(window).std() * volume_z.rolling(window).std()).replace(0, np.nan)
        raw = corr.fillna(0)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_112_zscore(df, window=65):
        hilo = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).std().replace(0, np.nan)
        corr = hilo.rolling(window).cov(volume_z) / (hilo.rolling(window).std() * volume_z.rolling(window).std()).replace(0, np.nan)
        raw = corr.fillna(0)
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_112_sign(df, window=5):
        hilo = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(window).std().replace(0, np.nan)
        corr = hilo.rolling(window).cov(volume_z) / (hilo.rolling(window).std() * volume_z.rolling(window).std()).replace(0, np.nan)
        raw = corr.fillna(0)
        result = np.sign(raw)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_112_wf(df, window=80, p1=0.9):
        p2 = window
        hilo = (df['high'] - df['low']) / df['close'].replace(0, np.nan)
        volume_z = (df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p2).mean()) / df.get('matchingVolume', df.get('volume', 1)).rolling(p2).std().replace(0, np.nan)
        corr = hilo.rolling(p2).cov(volume_z) / (hilo.rolling(p2).std() * volume_z.rolling(p2).std()).replace(0, np.nan)
        raw = corr.fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_113_k(df, window_rank=35):
        raw = np.sign((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * df['close'].pct_change().rolling(window=8).corr(df['matchingVolume'].diff())
        raw = raw.fillna(0)
        normalized = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_113_h(df, window=5):
        raw = np.sign((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * df['close'].pct_change().rolling(window=8).corr(df['matchingVolume'].diff())
        raw = raw.fillna(0)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_113_p(df, window=100):
        raw = np.sign((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * df['close'].pct_change().rolling(window=8).corr(df['matchingVolume'].diff())
        raw = raw.fillna(0)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_113_y(df):
        raw = np.sign((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * df['close'].pct_change().rolling(window=8).corr(df['matchingVolume'].diff())
        raw = raw.fillna(0)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_113_r(df, p1=0.1, p2=100):
        raw = np.sign((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * df['close'].pct_change().rolling(window=8).corr(df['matchingVolume'].diff())
        raw = raw.fillna(0)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_114_rank(df, window=70):
        volume = df.get('matchingVolume', df.get('volume', 1))
        max_vol = volume.rolling(window, min_periods=1).max()
        ratio = volume / (max_vol + 1e-8)
        sign = np.sign(ratio)
        corr = volume.rolling(window, min_periods=1).corr(df['close'])
        raw = sign * corr
        result = (raw.rolling(window, min_periods=1).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_114_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        max_vol = volume.rolling(window, min_periods=1).max()
        ratio = volume / (max_vol + 1e-8)
        sign = np.sign(ratio)
        corr = volume.rolling(window, min_periods=1).corr(df['close'])
        raw = sign * corr
        result = np.tanh(raw / raw.rolling(window, min_periods=1).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_114_zscore(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        max_vol = volume.rolling(window, min_periods=1).max()
        ratio = volume / (max_vol + 1e-8)
        sign = np.sign(ratio)
        corr = volume.rolling(window, min_periods=1).corr(df['close'])
        raw = sign * corr
        mean_ = raw.rolling(window, min_periods=1).mean()
        std_ = raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        result = ((raw - mean_) / std_).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_114_sign(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        max_vol = volume.rolling(window, min_periods=1).max()
        ratio = volume / (max_vol + 1e-8)
        sign = np.sign(ratio)
        corr = volume.rolling(window, min_periods=1).corr(df['close'])
        raw = sign * corr
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_114_wf(df, window=70, p1=0.3):
        volume = df.get('matchingVolume', df.get('volume', 1))
        max_vol = volume.rolling(window, min_periods=1).max()
        ratio = volume / (max_vol + 1e-8)
        sign = np.sign(ratio)
        corr = volume.rolling(window, min_periods=1).corr(df['close'])
        raw = sign * corr
        low = raw.rolling(window, min_periods=1).quantile(p1)
        high = raw.rolling(window, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_115_rank(df, window_vol=20):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * (df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).mean() + df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).std() + 1e-8))
        signal = (raw.rolling(window_vol).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_115_tanh(df, window_vol=15):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * (df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).mean() + df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).std() + 1e-8))
        signal = np.tanh(raw / (raw.rolling(window_vol).std() + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_115_zscore(df, window_vol=15):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * (df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).mean() + df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).std() + 1e-8))
        mean_r = raw.rolling(window_vol).mean()
        std_r = raw.rolling(window_vol).std()
        signal = ((raw - mean_r) / (std_r + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_115_sign(df, window_vol=45):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * (df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).mean() + df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).std() + 1e-8))
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_115_wf(df, window_vol=40, p1=0.1):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * (df.get('matchingVolume', df.get('volume', 1)) / (df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).mean() + df.get('matchingVolume', df.get('volume', 1)).rolling(window_vol).std() + 1e-8))
        p2 = max(window_vol, 10)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_116_rank(df, window=100):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_delay = volume.shift(1)
        vol_corr = volume.rolling(window).corr(vol_delay)
        ret = df['close'].pct_change()
        ret_mean = ret.rolling(window).mean()
        raw = vol_corr * ret_mean
        # Rolling Rank normalization
        norm = raw.rolling(window).rank(pct=True) * 2 - 1
        return norm.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_116_tanh(df, window=90):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_delay = volume.shift(1)
        vol_corr = volume.rolling(window).corr(vol_delay)
        ret = df['close'].pct_change()
        ret_mean = ret.rolling(window).mean()
        raw = vol_corr * ret_mean
        # Dynamic Tanh normalization
        std = raw.rolling(window).std().replace(0, np.nan)
        norm = np.tanh(raw / std)
        return norm.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_116_zscore(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_delay = volume.shift(1)
        vol_corr = volume.rolling(window).corr(vol_delay)
        ret = df['close'].pct_change()
        ret_mean = ret.rolling(window).mean()
        raw = vol_corr * ret_mean
        # Rolling Z-Score normalization
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        norm = ((raw - mean) / std).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_116_sign(df, window=70):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_delay = volume.shift(1)
        vol_corr = volume.rolling(window).corr(vol_delay)
        ret = df['close'].pct_change()
        ret_mean = ret.rolling(window).mean()
        raw = vol_corr * ret_mean
        # Sign normalization
        norm = np.sign(raw)
        return pd.Series(norm, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_116_wf(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_delay = volume.shift(1)
        vol_corr = volume.rolling(window).corr(vol_delay)
        ret = df['close'].pct_change()
        ret_mean = ret.rolling(window).mean()
        raw = vol_corr * ret_mean
        # Winsorized Fisher normalization
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        eps = 1e-9
        norm = np.arctanh(((winsorized - low) / (high - low + eps)) * 1.98 - 0.99)
        return -norm.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_117_rank(df, window=80):
        close_volume_corr = df['close'].rolling(window).corr(df['matchingVolume'].fillna(0).replace(0, 1e-8))
        rank_corr = close_volume_corr.rolling(window).rank(pct=True) * 2 - 1
        price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_position = price_position.rolling(window).mean()
        raw = rank_corr * mean_position
        signal = (raw.rolling(window).rank(pct=True) * 2 - 1).fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_117_tanh(df, window=70):
        close_volume_corr = df['close'].rolling(window).corr(df['matchingVolume'].fillna(0).replace(0, 1e-8))
        rank_corr = close_volume_corr.rolling(window).rank(pct=True) * 2 - 1
        price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_position = price_position.rolling(window).mean()
        raw = rank_corr * mean_position
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std_raw).fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_117_zscore(df, window=75):
        close_volume_corr = df['close'].rolling(window).corr(df['matchingVolume'].fillna(0).replace(0, 1e-8))
        rank_corr = close_volume_corr.rolling(window).rank(pct=True) * 2 - 1
        price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_position = price_position.rolling(window).mean()
        raw = rank_corr * mean_position
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_117_sign(df, window=75):
        close_volume_corr = df['close'].rolling(window).corr(df['matchingVolume'].fillna(0).replace(0, 1e-8))
        rank_corr = close_volume_corr.rolling(window).rank(pct=True) * 2 - 1
        price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_position = price_position.rolling(window).mean()
        raw = rank_corr * mean_position
        signal = np.sign(raw).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_117_wf(df, window=90, p1=0.7):
        close_volume_corr = df['close'].rolling(window).corr(df['matchingVolume'].fillna(0).replace(0, 1e-8))
        rank_corr = close_volume_corr.rolling(window).rank(pct=True) * 2 - 1
        price_position = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        mean_position = price_position.rolling(window).mean()
        raw = rank_corr * mean_position
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_118_rank(df, window_rank=80, delta=2):
        # Xử lý volume: log1p để giảm skew
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        # Tính delta của open
        open_delta = df['open'].diff(delta)
        sign_delta = np.sign(open_delta)
        # Tính correlation giữa open/close ratio và volume
        ratio = df['open'] / (df['close'] + 1e-8)
        # Tính rolling covariance và variance
        cov = ratio.rolling(window_rank).cov(volume_log)
        var_ratio = ratio.rolling(window_rank).var()
        var_vol = volume_log.rolling(window_rank).var()
        # Tính correlation: cov / (std1 * std2)
        corr = cov / (np.sqrt(var_ratio * var_vol) + 1e-8)
        # Ghép với sign_delta
        raw = sign_delta * corr
        # Chuẩn hóa Rolling Rank (A) vì dữ liệu correlation binary và sign
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        signal = signal.fillna(method='ffill').fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_118_tanh(df, window_tanh=10, delta=6):
        # Xử lý volume: log1p để giảm skew
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        # Tính delta của open
        open_delta = df['open'].diff(delta)
        sign_delta = np.sign(open_delta)
        # Tính correlation giữa open/close ratio và volume
        ratio = df['open'] / (df['close'] + 1e-8)
        cov = ratio.rolling(window_tanh).cov(volume_log)
        var_ratio = ratio.rolling(window_tanh).var()
        var_vol = volume_log.rolling(window_tanh).var()
        corr = cov / (np.sqrt(var_ratio * var_vol) + 1e-8)
        raw = sign_delta * corr
        # Chuẩn hóa Dynamic Tanh (B) để giữ magnitude
        rolling_std = raw.rolling(window_tanh).std().replace(0, np.nan)
        signal = np.tanh(raw / rolling_std)
        signal = signal.fillna(method='ffill').fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_118_zscore(df, window_z=10, delta=5):
        # Xử lý volume: log1p để giảm skew
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        # Tính delta của open
        open_delta = df['open'].diff(delta)
        sign_delta = np.sign(open_delta)
        # Tính correlation giữa open/close ratio và volume
        ratio = df['open'] / (df['close'] + 1e-8)
        cov = ratio.rolling(window_z).cov(volume_log)
        var_ratio = ratio.rolling(window_z).var()
        var_vol = volume_log.rolling(window_z).var()
        corr = cov / (np.sqrt(var_ratio * var_vol) + 1e-8)
        raw = sign_delta * corr
        # Chuẩn hóa Rolling Z-Score/Clip (C) vì là oscillator-like
        rolling_mean = raw.rolling(window_z).mean()
        rolling_std = raw.rolling(window_z).std().replace(0, np.nan)
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        signal = signal.fillna(method='ffill').fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_118_sign(df, window_corr=50, delta=2):
        # Xử lý volume: log1p để giảm skew
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        # Tính delta của open
        open_delta = df['open'].diff(delta)
        sign_delta = np.sign(open_delta)
        # Tính correlation giữa open/close ratio và volume
        ratio = df['open'] / (df['close'] + 1e-8)
        cov = ratio.rolling(window_corr).cov(volume_log)
        var_ratio = ratio.rolling(window_corr).var()
        var_vol = volume_log.rolling(window_corr).var()
        corr = cov / (np.sqrt(var_ratio * var_vol) + 1e-8)
        raw = sign_delta * corr
        # Chuẩn hóa Sign/Binary Soft (D) vì raw đã mang dấu và correlation
        raw_sign = np.sign(raw)
        # Làm mịn nhẹ bằng rolling mean để soft
        smoothed = raw_sign.rolling(window_corr).mean().fillna(method='ffill').fillna(0)
        signal = smoothed.clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_118_wf(df, window_fisher=10, delta=5, quantile=0.1):
        # Xử lý volume: log1p để giảm skew
        volume = df.get('matchingVolume', df.get('volume', 1))
        volume_log = np.log1p(volume)
        # Tính delta của open
        open_delta = df['open'].diff(delta)
        sign_delta = np.sign(open_delta)
        # Tính correlation giữa open/close ratio và volume
        ratio = df['open'] / (df['close'] + 1e-8)
        cov = ratio.rolling(window_fisher).cov(volume_log)
        var_ratio = ratio.rolling(window_fisher).var()
        var_vol = volume_log.rolling(window_fisher).var()
        corr = cov / (np.sqrt(var_ratio * var_vol) + 1e-8)
        raw = sign_delta * corr
        # Chuẩn hóa Winsorized Fisher (E) vì data correlation có heavy tails
        p1 = quantile
        p2 = window_fisher
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(method='ffill').fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_119_k(df, window=30):
        close = df['close']
        open_ = df['open']
        # No lookahead: shift close to get previous close
        prev_close = close.shift(1)
        raw = (open_ - prev_close).abs() / (close.rolling(window).std() + 1e-8)
        # Rolling rank normalization to [-1, 1]
        rank = raw.rolling(window).rank(pct=True) * 2 - 1
        return -rank.fillna(0)

    @staticmethod
    def alpha_quanta_119_h(df, window=5):
        close = df['close']
        open_ = df['open']
        prev_close = close.shift(1)
        raw = (open_ - prev_close).abs() / (close.rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_119_e(df, window=60):
        close = df['close']
        open_ = df['open']
        prev_close = close.shift(1)
        raw = (open_ - prev_close).abs() / (close.rolling(window).std() + 1e-8)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_119_n(df, window=60):
        close = df['close']
        open_ = df['open']
        prev_close = close.shift(1)
        raw = (open_ - prev_close).abs() / (close.rolling(window).std() + 1e-8)
        # For breakout-like signal: direction is based on open vs prev_close, magnitude normalized
        direction = np.sign(open_ - prev_close)
        # Combine with raw magnitude, but constrain to [-1,1]
        signal = direction * np.clip(raw, 0, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_119_r(df, p1=0.9, p2=100):
        close = df['close']
        open_ = df['open']
        prev_close = close.shift(1)
        raw = (open_ - prev_close).abs() / (close.rolling(p2).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_120_k(df, window=95):
        raw = abs(df['open'] - df['close'].shift(1)) / (df['high'].sub(df['low']).rolling(window).mean() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_120_h(df, window=5):
        raw = abs(df['open'] - df['close'].shift(1)) / (df['high'].sub(df['low']).rolling(window).mean() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_120_e(df, window=25):
        raw = abs(df['open'] - df['close'].shift(1)) / (df['high'].sub(df['low']).rolling(window).mean() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_120_y(df, window=85):
        raw = abs(df['open'] - df['close'].shift(1)) / (df['high'].sub(df['low']).rolling(window).mean() + 1e-8)
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_120_r(df, window=6, sub_window=100):
        raw = abs(df['open'] - df['close'].shift(1)) / (df['high'].sub(df['low']).rolling(window).mean() + 1e-8)
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_121_k(df, window=90):
        raw = np.sign(df['open'] - df['close'].shift(1)) * (df['close'] / df['close'].shift(1) - 1).rolling(10).mean()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_121_h(df, window=5):
        raw = np.sign(df['open'] - df['close'].shift(1)) * (df['close'] / df['close'].shift(1) - 1).rolling(10).mean()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_121_p(df, window=85):
        raw = np.sign(df['open'] - df['close'].shift(1)) * (df['close'] / df['close'].shift(1) - 1).rolling(10).mean()
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_121_y(df, window=35):
        raw = np.sign(df['open'] - df['close'].shift(1)) * (df['close'] / df['close'].shift(1) - 1).rolling(window).mean()
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_121_r(df, p1=0.1, p2=100):
        raw = np.sign(df['open'] - df['close'].shift(1)) * (df['close'] / df['close'].shift(1) - 1).rolling(10).mean()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_122_rank(df, window=30):
        raw = (df['open'] - df['close'].shift(1)).rolling(window).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0, raw=True) * (df['close'].rolling(window).apply(lambda x: (x[-1] - x.mean()) / x.std() if x.std() != 0 else 0, raw=True))
        # Use Rolling Z-Score/Clip (Method C)
        norm_raw = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -norm_raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_122_tanh(df, window=60):
        open_close_diff = df['open'] - df['close'].shift(1)
        zscore_open_diff = (open_close_diff - open_close_diff.rolling(window).mean()) / open_close_diff.rolling(window).std().replace(0, np.nan)
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std().replace(0, np.nan)
        raw = zscore_open_diff * zscore_close
        # Dynamic Tanh (Method B)
        norm_raw = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -norm_raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_122_zscore(df, window=30):
        open_close_diff = df['open'] - df['close'].shift(1)
        zscore_open_diff = (open_close_diff - open_close_diff.rolling(window).mean()) / open_close_diff.rolling(window).std().replace(0, np.nan)
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std().replace(0, np.nan)
        raw = zscore_open_diff * zscore_close
        # Rolling Z-Score/Clip (Method C)
        norm_raw = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -norm_raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_122_sign(df, window=5):
        open_close_diff = df['open'] - df['close'].shift(1)
        zscore_open_diff = (open_close_diff - open_close_diff.rolling(window).mean()) / open_close_diff.rolling(window).std().replace(0, np.nan)
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std().replace(0, np.nan)
        raw = zscore_open_diff * zscore_close
        # Sign/Binary Soft (Method D)
        norm_raw = np.sign(raw)
        return -norm_raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_122_wf(df, window=30, quantile=0.9):
        open_close_diff = df['open'] - df['close'].shift(1)
        zscore_open_diff = (open_close_diff - open_close_diff.rolling(window).mean()) / open_close_diff.rolling(window).std().replace(0, np.nan)
        zscore_close = (df['close'] - df['close'].rolling(window).mean()) / df['close'].rolling(window).std().replace(0, np.nan)
        raw = zscore_open_diff * zscore_close
        # Winsorized Fisher (Method E)
        p1 = quantile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_123_rank(df, window=60):
        ret = df['open'] - df['close'].shift(1)
        std5 = df['close'].rolling(5).std() + 1e-8
        raw = ret / std5 - (df['close'] / df['open'] - 1).rolling(5).mean() / std5
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_123_tanh(df, window=5):
        ret = df['open'] - df['close'].shift(1)
        std5 = df['close'].rolling(5).std() + 1e-8
        raw = ret / std5 - (df['close'] / df['open'] - 1).rolling(5).mean() / std5
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_123_zscore(df, window=5):
        ret = df['open'] - df['close'].shift(1)
        std5 = df['close'].rolling(5).std() + 1e-8
        raw = ret / std5 - (df['close'] / df['open'] - 1).rolling(5).mean() / std5
        mean_w = raw.rolling(window).mean()
        std_w = raw.rolling(window).std()
        normalized = ((raw - mean_w) / std_w).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_123_sign(df):
        ret = df['open'] - df['close'].shift(1)
        std5 = df['close'].rolling(5).std() + 1e-8
        raw = ret / std5 - (df['close'] / df['open'] - 1).rolling(5).mean() / std5
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_123_wf(df, window=30, p1=0.7):
        ret = df['open'] - df['close'].shift(1)
        std5 = df['close'].rolling(5).std() + 1e-8
        raw = ret / std5 - (df['close'] / df['open'] - 1).rolling(5).mean() / std5
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_124_rank(df, window=80):
        # Tính numerator1 = (high - low)
        numerator1 = df['high'] - df['low']
        # Tính denominator1 = open + close
        denominator1 = df['open'] + df['close'] + 1e-8
        # Tính x = numerator1 / denominator1  (xấp xỉ spread ratio)
        x = numerator1 / denominator1
        # Tính y = (close - open) / (open + 1e-8) (return -mở cửa)
        y = (df['close'] - df['open']) / (df['open'] + 1e-8)
        # Tính covariance rolling
        x_mean = x.rolling(window=window).mean()
        y_mean = y.rolling(window=window).mean()
        # Sử dụng nump để tính covariance và variance hiệu quả dạng vectorized
        # Tính rolling covariance bằng cách lấy tích lệch chia window
        # Dùng pd.Series với ngăn NaN bằng cách trượt
        # Cách thủ công: (x - x_mean)*(y - y_mean) rolling mean
        cov = ((x - x_mean) * (y - y_mean)).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_x = ((x - x_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_y = ((y - y_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        # Tránh chia 0
        denom = np.sqrt(var_x * var_y).replace(0, np.nan)
        ts_corr = cov / denom
        # Xử lý NaN (dữ liệu chuỗi thời gian - forward fill)
        ts_corr = ts_corr.ffill().fillna(0)
        # Chuẩn hóa về [-1, 1] bằng cách giữ nguyên vì correlation đã trong [-1,1]
        return ts_corr.clip(-1, 1)

    @staticmethod
    def alpha_quanta_124_tanh(df, window=30):
        numerator1 = df['high'] - df['low']
        denominator1 = df['open'] + df['close'] + 1e-8
        x = numerator1 / denominator1
        y = (df['close'] - df['open']) / (df['open'] + 1e-8)
        x_mean = x.rolling(window=window).mean()
        y_mean = y.rolling(window=window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_x = ((x - x_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_y = ((y - y_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        denom = np.sqrt(var_x * var_y).replace(0, np.nan)
        ts_corr = cov / denom
        ts_corr = ts_corr.ffill().fillna(0)
        # Dynamic Tanh normalization: giữ magnitude sau arctanh-like
        # dùng np.tanh trực tiếp trên raw correlation (đã trong [-1,1]), giữ nguyên cường độ
        return -np.tanh(ts_corr * 2)

    @staticmethod
    def alpha_quanta_124_zscore(df, window=75):
        numerator1 = df['high'] - df['low']
        denominator1 = df['open'] + df['close'] + 1e-8
        x = numerator1 / denominator1
        y = (df['close'] - df['open']) / (df['open'] + 1e-8)
        x_mean = x.rolling(window=window).mean()
        y_mean = y.rolling(window=window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_x = ((x - x_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_y = ((y - y_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        denom = np.sqrt(var_x * var_y).replace(0, np.nan)
        ts_corr = cov / denom
        ts_corr = ts_corr.ffill().fillna(0)
        # Rolling Z-score normalization để ổn định phân phối
        z = (ts_corr - ts_corr.rolling(window=window*2).mean()) / ts_corr.rolling(window=window*2).std().replace(0, np.nan)
        z = z.ffill().fillna(0)
        return -z.clip(-1, 1)

    @staticmethod
    def alpha_quanta_124_sign(df, window=5):
        numerator1 = df['high'] - df['low']
        denominator1 = df['open'] + df['close'] + 1e-8
        x = numerator1 / denominator1
        y = (df['close'] - df['open']) / (df['open'] + 1e-8)
        x_mean = x.rolling(window=window).mean()
        y_mean = y.rolling(window=window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_x = ((x - x_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_y = ((y - y_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        denom = np.sqrt(var_x * var_y).replace(0, np.nan)
        ts_corr = cov / denom
        ts_corr = ts_corr.ffill().fillna(0)
        # Binary sign soft: dấu của correlation, giữ break-out xu hướng
        return np.sign(ts_corr)

    @staticmethod
    def alpha_quanta_124_wf(df, window=50, p1=0.1):
        numerator1 = df['high'] - df['low']
        denominator1 = df['open'] + df['close'] + 1e-8
        x = numerator1 / denominator1
        y = (df['close'] - df['open']) / (df['open'] + 1e-8)
        x_mean = x.rolling(window=window).mean()
        y_mean = y.rolling(window=window).mean()
        cov = ((x - x_mean) * (y - y_mean)).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_x = ((x - x_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        var_y = ((y - y_mean) ** 2).rolling(window=window, min_periods=window).sum() / (window - 1)
        denom = np.sqrt(var_x * var_y).replace(0, np.nan)
        ts_corr = cov / denom
        ts_corr = ts_corr.ffill().fillna(0)
        # Winsorized Fisher Transform: giảm tác động của dữ liệu có đuôi nặng
        p2_fixed = window * 3  # fixed window cho quantile
        low = ts_corr.rolling(p2_fixed).quantile(p1)
        high = ts_corr.rolling(p2_fixed).quantile(1 - p1)
        winsorized = ts_corr.clip(lower=low, upper=high, axis=0)
        # Chuẩn hóa về [-1,1] bằng Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -normalized

    @staticmethod
    def alpha_quanta_125_rank(df, window=15):
        body_range = (df['open'] - df['close']).abs()
        total_range = (df['high'] - df['low']) + 1e-8
        ratio1 = body_range / total_range
        avg_ratio1 = ratio1.rolling(window=window).mean()
        hl_range = (df['high'] - df['low'])
        ratio2 = hl_range / (df['close'] + 1e-8)
        avg_ratio2 = ratio2.rolling(window=window).mean()
        raw = avg_ratio1 / (avg_ratio2 + 1e-8)
        signal = raw.rolling(window=window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_125_tanh(df, window=35):
        body_range = (df['open'] - df['close']).abs()
        total_range = (df['high'] - df['low']) + 1e-8
        ratio1 = body_range / total_range
        avg_ratio1 = ratio1.rolling(window=window).mean()
        hl_range = (df['high'] - df['low'])
        ratio2 = hl_range / (df['close'] + 1e-8)
        avg_ratio2 = ratio2.rolling(window=window).mean()
        raw = avg_ratio1 / (avg_ratio2 + 1e-8)
        signal = np.tanh(raw / raw.rolling(window=window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_125_zscore(df, window=25):
        body_range = (df['open'] - df['close']).abs()
        total_range = (df['high'] - df['low']) + 1e-8
        ratio1 = body_range / total_range
        avg_ratio1 = ratio1.rolling(window=window).mean()
        hl_range = (df['high'] - df['low'])
        ratio2 = hl_range / (df['close'] + 1e-8)
        avg_ratio2 = ratio2.rolling(window=window).mean()
        raw = avg_ratio1 / (avg_ratio2 + 1e-8)
        rolling_mean = raw.rolling(window=window).mean()
        rolling_std = raw.rolling(window=window).std()
        zscore = (raw - rolling_mean) / rolling_std
        signal = zscore.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_125_sign(df, window=50):
        body_range = (df['open'] - df['close']).abs()
        total_range = (df['high'] - df['low']) + 1e-8
        ratio1 = body_range / total_range
        avg_ratio1 = ratio1.rolling(window=window).mean()
        hl_range = (df['high'] - df['low'])
        ratio2 = hl_range / (df['close'] + 1e-8)
        avg_ratio2 = ratio2.rolling(window=window).mean()
        raw = avg_ratio1 / (avg_ratio2 + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_125_wf(df, window=10, p=0.3):
        body_range = (df['open'] - df['close']).abs()
        total_range = (df['high'] - df['low']) + 1e-8
        ratio1 = body_range / total_range
        avg_ratio1 = ratio1.rolling(window=window).mean()
        hl_range = (df['high'] - df['low'])
        ratio2 = hl_range / (df['close'] + 1e-8)
        avg_ratio2 = ratio2.rolling(window=window).mean()
        raw = avg_ratio1 / (avg_ratio2 + 1e-8)
        low = raw.rolling(window=window).quantile(p)
        high = raw.rolling(window=window).quantile(1 - p)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_126_k(df, w=80, sub_w=40):
        hv = (df['high'] - df['low']) / (df['close'] + 1e-8)
        corr_val = hv.rolling(w).corr(df.get('matchingVolume', df.get('volume', 1))).fillna(0)
        std_val = hv.rolling(sub_w).std().fillna(0)
        raw = corr_val - std_val
        norm = (raw.rolling(sub_w).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_126_h(df, w=80, sub_w=1):
        hv = (df['high'] - df['low']) / (df['close'] + 1e-8)
        corr_val = hv.rolling(w).corr(df.get('matchingVolume', df.get('volume', 1))).fillna(0)
        std_val = hv.rolling(sub_w).std().fillna(0)
        raw = corr_val - std_val
        norm = np.tanh(raw / raw.rolling(sub_w).std().replace(0, np.nan).fillna(method='ffill').fillna(1))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_126_e(df, w=80, sub_w=20):
        hv = (df['high'] - df['low']) / (df['close'] + 1e-8)
        corr_val = hv.rolling(w).corr(df.get('matchingVolume', df.get('volume', 1))).fillna(0)
        std_val = hv.rolling(sub_w).std().fillna(0)
        raw = corr_val - std_val
        norm = ((raw - raw.rolling(sub_w).mean()) / raw.rolling(sub_w).std().replace(0, np.nan)).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_126_n(df, w=90, sub_w=3):
        hv = (df['high'] - df['low']) / (df['close'] + 1e-8)
        corr_val = hv.rolling(w).corr(df.get('matchingVolume', df.get('volume', 1))).fillna(0)
        std_val = hv.rolling(sub_w).std().fillna(0)
        raw = corr_val - std_val
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_126_r(df, w=100, p=0.9):
        hv = (df['high'] - df['low']) / (df['close'] + 1e-8)
        corr_val = hv.rolling(w).corr(df.get('matchingVolume', df.get('volume', 1))).fillna(0)
        std_val = hv.rolling(w).std().fillna(0)
        raw = corr_val - std_val
        lo = raw.rolling(w).quantile(p)
        hi = raw.rolling(w).quantile(1 - p)
        clipped = raw.clip(lower=lo, upper=hi, axis=0)
        fraction = (clipped - lo) / (hi - lo + 1e-9)
        norm = np.arctanh(fraction * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_127_k(df, window_rank_8=85):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        raw = (high - low) / (volume + 1e-8)
        volume_change = volume.diff(1).fillna(0)
        vol_sign = np.sign(volume_change)
        raw_rank = (raw.rolling(window_rank_8).rank(pct=True) * 2) - 1
        signal = raw_rank * vol_sign
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_127_h(df, window_std_8=5):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        raw = (high - low) / (volume + 1e-8)
        volume_change = df['matchingVolume'].diff(1).fillna(0)
        vol_sign = np.sign(volume_change)
        raw_std = raw.rolling(window_std_8).std().replace(0, np.nan)
        norm = np.tanh(raw / raw_std)
        signal = norm * vol_sign
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_127_p(df, window_z_8=40):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        raw = (high - low) / (volume + 1e-8)
        volume_change = df['matchingVolume'].diff(1).fillna(0)
        vol_sign = np.sign(volume_change)
        mean = raw.rolling(window_z_8).mean()
        std = raw.rolling(window_z_8).std().replace(0, np.nan)
        z = ((raw - mean) / std).clip(-1, 1)
        signal = z * vol_sign
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_127_y(df):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        raw = (high - low) / (volume + 1e-8)
        volume_change = df['matchingVolume'].diff(1).fillna(0)
        vol_sign = np.sign(volume_change)
        signal_direction = np.sign(raw)
        signal = signal_direction * vol_sign
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_127_r(df, window_quantile_8=60, factor_quantile_0_05=0.1):
        high = df['high']
        low = df['low']
        volume = df['matchingVolume']
        raw = (high - low) / (volume + 1e-8)
        volume_change = df['matchingVolume'].diff(1).fillna(0)
        vol_sign = np.sign(volume_change)
        low_q = raw.rolling(window_quantile_8).quantile(factor_quantile_0_05)
        high_q = raw.rolling(window_quantile_8).quantile(1 - factor_quantile_0_05)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        signal = normalized * vol_sign
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_128_rank(df, window=90):
        raw = np.sign(df['open'] - df['close']) * (df['open'] - df['close']).div(df['high'] - df['low'] + 1e-8).rolling(window).mean()
        normalized = (raw.rolling(window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_128_tanh(df, window=85):
        raw = np.sign(df['open'] - df['close']) * (df['open'] - df['close']).div(df['high'] - df['low'] + 1e-8).rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = np.tanh(raw / std.replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_128_zscore(df, window=90):
        raw = np.sign(df['open'] - df['close']) * (df['open'] - df['close']).div(df['high'] - df['low'] + 1e-8).rolling(window).mean()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        normalized = ((raw - mean) / std.replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_128_sign(df, window=85):
        raw = np.sign(df['open'] - df['close']) * (df['open'] - df['close']).div(df['high'] - df['low'] + 1e-8).rolling(window).mean()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_128_wf(df, window=90, quantile_thresh=0.3):
        raw = np.sign(df['open'] - df['close']) * (df['open'] - df['close']).div(df['high'] - df['low'] + 1e-8).rolling(window).mean()
        p1 = quantile_thresh
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_129_rank(df, window=5):
        inv_close = 1.0 / df['close']
        raw = df['close'] - inv_close.rolling(window).mean()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_129_tanh(df, window=95):
        inv_close = 1.0 / df['close']
        raw = df['close'] - inv_close.rolling(window).mean()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_129_zscore(df, window=5):
        inv_close = 1.0 / df['close']
        raw = df['close'] - inv_close.rolling(window).mean()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_129_sign(df, window=30):
        inv_close = 1.0 / df['close']
        raw = df['close'] - inv_close.rolling(window).mean()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_129_wf(df, window_p2=5):
        p1 = 0.1
        inv_close = 1.0 / df['close']
        raw = df['close'] - inv_close.rolling(window_p2).mean()
        low = raw.rolling(window_p2).quantile(p1)
        high = raw.rolling(window_p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_130_e(df, window=70):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol = np.log1p(vol)
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std().replace(0, np.nan) + 1e-8)
        signal = raw.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_130_h(df, window=100):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol = np.log1p(vol)
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std().replace(0, np.nan) + 1e-8)
        signal = np.tanh(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_130_k(df, window=45):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol = np.log1p(vol)
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std().replace(0, np.nan) + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_130_y(df, window=50):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol = np.log1p(vol)
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std().replace(0, np.nan) + 1e-8)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_130_r(df, window=100, p1=0.7):
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol = np.log1p(vol)
        raw = (vol - vol.rolling(window).mean()) / (vol.rolling(window).std().replace(0, np.nan) + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_131_rank(df, p1=10):
        raw = -((((1 / df['close']) - ((1 / df['close'])).rolling(p1).mean()) / ((1 / df['close'])).rolling(p1).std()).fillna(0)) * ((df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p1).mean()) / (df.get('matchingVolume', df.get('volume', 1)).rolling(p1).std() + 1e-8)).fillna(0)
        normalized = (raw.rolling(p1).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_131_tanh(df, p1=5):
        raw = -((((1 / df['close']) - ((1 / df['close'])).rolling(p1).mean()) / ((1 / df['close'])).rolling(p1).std()).fillna(0)) * ((df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p1).mean()) / (df.get('matchingVolume', df.get('volume', 1)).rolling(p1).std() + 1e-8)).fillna(0)
        normalized = np.tanh(raw / raw.rolling(p1).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_131_zscore(df, p1=5):
        raw = -((((1 / df['close']) - ((1 / df['close'])).rolling(p1).mean()) / ((1 / df['close'])).rolling(p1).std()).fillna(0)) * ((df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p1).mean()) / (df.get('matchingVolume', df.get('volume', 1)).rolling(p1).std() + 1e-8)).fillna(0)
        normalized = ((raw - raw.rolling(p1).mean()) / raw.rolling(p1).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_131_sign(df, p1=60):
        raw = -((((1 / df['close']) - ((1 / df['close'])).rolling(p1).mean()) / ((1 / df['close'])).rolling(p1).std()).fillna(0)) * ((df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p1).mean()) / (df.get('matchingVolume', df.get('volume', 1)).rolling(p1).std() + 1e-8)).fillna(0)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_131_wf(df, p1=6, p2=10):
        raw = -((((1 / df['close']) - ((1 / df['close'])).rolling(p2).mean()) / ((1 / df['close'])).rolling(p2).std()).fillna(0)) * ((df.get('matchingVolume', df.get('volume', 1)) - df.get('matchingVolume', df.get('volume', 1)).rolling(p2).mean()) / (df.get('matchingVolume', df.get('volume', 1)).rolling(p2).std() + 1e-8)).fillna(0)
        low = raw.rolling(p2).quantile(p1 / 100.0)
        high = raw.rolling(p2).quantile(1 - p1 / 100.0)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_132_k(df, window=20):
        inv_close = np.sign(df['close'].diff()) * df['close']
        delta_volume = df['volume'].diff()
        y = inv_close
        x = delta_volume
        y_ma = y.rolling(window).mean()
        x_ma = x.rolling(window).mean()
        num = ((y - y_ma) * (x - x_ma)).rolling(window).sum()
        den = np.sqrt(((y - y_ma)**2).rolling(window).sum()) * np.sqrt(((x - x_ma)**2).rolling(window).sum())
        raw = num / den.replace(0, np.nan)
        raw = raw.fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_132_h(df, window=20):
        inv_close = np.sign(df['close'].diff()) * df['close']
        delta_volume = df['volume'].diff()
        y = inv_close
        x = delta_volume
        y_ma = y.rolling(window).mean()
        x_ma = x.rolling(window).mean()
        num = ((y - y_ma) * (x - x_ma)).rolling(window).sum()
        den = np.sqrt(((y - y_ma)**2).rolling(window).sum()) * np.sqrt(((x - x_ma)**2).rolling(window).sum())
        raw = num / den.replace(0, np.nan)
        raw = raw.fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_132_p(df, window=20):
        inv_close = np.sign(df['close'].diff()) * df['close']
        delta_volume = df['volume'].diff()
        y = inv_close
        x = delta_volume
        y_ma = y.rolling(window).mean()
        x_ma = x.rolling(window).mean()
        num = ((y - y_ma) * (x - x_ma)).rolling(window).sum()
        den = np.sqrt(((y - y_ma)**2).rolling(window).sum()) * np.sqrt(((x - x_ma)**2).rolling(window).sum())
        raw = num / den.replace(0, np.nan)
        raw = raw.fillna(0)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_132_t(df, window=20):
        inv_close = np.sign(df['close'].diff()) * df['close']
        delta_volume = df['volume'].diff()
        y = inv_close
        x = delta_volume
        y_ma = y.rolling(window).mean()
        x_ma = x.rolling(window).mean()
        num = ((y - y_ma) * (x - x_ma)).rolling(window).sum()
        den = np.sqrt(((y - y_ma)**2).rolling(window).sum()) * np.sqrt(((x - x_ma)**2).rolling(window).sum())
        raw = num / den.replace(0, np.nan)
        raw = raw.fillna(0)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_132_r(df, window=20, p1=0.05):
        inv_close = np.sign(df['close'].diff()) * df['close']
        delta_volume = df['volume'].diff()
        y = inv_close
        x = delta_volume
        y_ma = y.rolling(window).mean()
        x_ma = x.rolling(window).mean()
        num = ((y - y_ma) * (x - x_ma)).rolling(window).sum()
        den = np.sqrt(((y - y_ma)**2).rolling(window).sum()) * np.sqrt(((x - x_ma)**2).rolling(window).sum())
        raw = num / den.replace(0, np.nan)
        raw = raw.fillna(0)
        p1 = p1 if p1 else 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized.replace([np.inf, -np.inf], np.nan).fillna(0)

    @staticmethod
    def alpha_quanta_133_rank(df, window_rank=90):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_v = volume.rolling(5).mean()
        std_v = volume.rolling(5).std()
        raw = (volume - mean_v) / (std_v + 1e-8)
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_133_tanh(df, window_std=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_v = volume.rolling(5).mean()
        std_v = volume.rolling(5).std()
        raw = (volume - mean_v) / (std_v + 1e-8)
        signal = np.tanh(raw / raw.rolling(window_std).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_133_zscore(df, window_z=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_v = volume.rolling(5).mean()
        std_v = volume.rolling(5).std()
        raw = (volume - mean_v) / (std_v + 1e-8)
        signal = ((raw - raw.rolling(window_z).mean()) / raw.rolling(window_z).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_133_sign(df, window_sign=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_v = volume.rolling(window_sign).mean()
        std_v = volume.rolling(window_sign).std()
        raw = (volume - mean_v) / (std_v + 1e-8)
        signal = pd.Series(np.sign(raw), index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_133_wf(df, p1=0.9, p2=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_v = volume.rolling(5).mean()
        std_v = volume.rolling(5).std()
        raw = (volume - mean_v) / (std_v + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_134_rank(df, window=50):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = vol.diff(1) / (vol + 1e-8)
        corr = ret.rolling(5).corr(vol_delta).fillna(0)
        rank = corr.rolling(window).rank(pct=True).fillna(0.5) * 2 - 1
        return rank.fillna(0)

    @staticmethod
    def alpha_quanta_134_tanh(df, factor=5):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = vol.diff(1) / (vol + 1e-8)
        corr = ret.rolling(5).corr(vol_delta).fillna(0)
        raw = corr * factor
        return np.tanh(raw).fillna(0)

    @staticmethod
    def alpha_quanta_134_zscore(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = vol.diff(1) / (vol + 1e-8)
        corr = ret.rolling(5).corr(vol_delta).fillna(0)
        mean = corr.rolling(window).mean().fillna(0)
        std = corr.rolling(window).std().fillna(1)
        zscore = ((corr - mean) / std).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_134_sign(df, window=75):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = vol.diff(1) / (vol + 1e-8)
        corr = ret.rolling(5).corr(vol_delta).fillna(0)
        smoothing = corr.rolling(window).mean().fillna(0)
        return np.sign(smoothing).fillna(0)

    @staticmethod
    def alpha_quanta_134_wf(df, window_rank=30, p1=0.1):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_delta = vol.diff(1) / (vol + 1e-8)
        corr = ret.rolling(5).corr(vol_delta).fillna(0)
        low = corr.rolling(window_rank).quantile(p1).fillna(0)
        high = corr.rolling(window_rank).quantile(1 - p1).fillna(0)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_135_rank(df, window=45):
        high = df['high']
        low = df['low']
        close = df['close']
        raw_range = (high - low) / (high.rolling(window).mean() - low.rolling(window).mean() + 1e-8)
        raw_log = np.log(raw_range + 1e-8)
        std_close = close.rolling(window).std() + 1e-8
        raw = raw_log / std_close
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_135_tanh(df, window=55):
        high = df['high']
        low = df['low']
        close = df['close']
        raw_range = (high - low) / (high.rolling(window).mean() - low.rolling(window).mean() + 1e-8)
        raw_log = np.log(raw_range + 1e-8)
        std_close = close.rolling(window).std() + 1e-8
        raw = raw_log / std_close
        norm = np.tanh(raw / raw.rolling(window).std())
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_135_zscore(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        raw_range = (high - low) / (high.rolling(window).mean() - low.rolling(window).mean() + 1e-8)
        raw_log = np.log(raw_range + 1e-8)
        std_close = close.rolling(window).std() + 1e-8
        raw = raw_log / std_close
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_135_sign(df, window=90):
        high = df['high']
        low = df['low']
        close = df['close']
        raw_range = (high - low) / (high.rolling(window).mean() - low.rolling(window).mean() + 1e-8)
        raw_log = np.log(raw_range + 1e-8)
        std_close = close.rolling(window).std() + 1e-8
        raw = raw_log / std_close
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_135_wf(df, window=40, quantile=0.9):
        high = df['high']
        low = df['low']
        close = df['close']
        raw_range = (high - low) / (high.rolling(window).mean() - low.rolling(window).mean() + 1e-8)
        raw_log = np.log(raw_range + 1e-8)
        std_close = close.rolling(window).std() + 1e-8
        raw = raw_log / std_close
        p1 = quantile
        p2 = window
        low_bound = raw.rolling(p2).quantile(p1)
        high_bound = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        norm = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_136_rank(df, window=65):
        close = df['close']
        ret = close.pct_change()
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        raw = np.sign((close - low) / (high - low + 1e-8)) * ret.rolling(5).mean()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_136_tanh(df, window=75):
        close = df['close']
        ret = close.pct_change()
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        raw = np.sign((close - low) / (high - low + 1e-8)) * ret.rolling(5).mean()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_136_zscore(df, window=75):
        close = df['close']
        ret = close.pct_change()
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        raw = np.sign((close - low) / (high - low + 1e-8)) * ret.rolling(5).mean()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_136_sign(df, window=100):
        close = df['close']
        ret = close.pct_change()
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        raw = np.sign((close - low) / (high - low + 1e-8)) * ret.rolling(5).mean()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_136_wf(df, window=100, winsorize_quantile=0.1):
        close = df['close']
        ret = close.pct_change()
        low = close.rolling(window).min()
        high = close.rolling(window).max()
        raw = np.sign((close - low) / (high - low + 1e-8)) * ret.rolling(5).mean()
        low_w = raw.rolling(window).quantile(winsorize_quantile)
        high_w = raw.rolling(window).quantile(1 - winsorize_quantile)
        wins_w = raw.clip(lower=low_w, upper=high_w, axis=0)
        normalized = np.arctanh(((wins_w - low_w) / (high_w - low_w + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_137_rank(df, window=90, corr_window=40):
        close_med = df['close'] - df['close'].rolling(window).median()
        corr = close_med.rolling(corr_window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw = corr.abs()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_137_tanh(df, window=20, corr_window=40):
        close_med = df['close'] - df['close'].rolling(window).median()
        corr = close_med.rolling(corr_window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw = corr.abs()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_137_zscore(df, window=50, corr_window=30):
        close_med = df['close'] - df['close'].rolling(window).median()
        corr = close_med.rolling(corr_window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw = corr.abs()
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_137_sign(df, window=90, corr_window=20):
        close_med = df['close'] - df['close'].rolling(window).median()
        corr = close_med.rolling(corr_window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw = corr.abs()
        signal = np.sign(raw).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_137_wf(df, window=100, p1=0.3, p2=20):
        close_med = df['close'] - df['close'].rolling(window).median()
        corr = close_med.rolling(window).corr(df.get('matchingVolume', df.get('volume', 1)))
        raw = corr.abs().dropna()
        if len(raw) < p2:
            return pd.Series(0, index=df.index)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high)
        w_low = low
        w_high = high
        normalized = np.arctanh(((winsorized - w_low) / (w_high - w_low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_138_k(df, window=100):
        spread = df['high'] - df['low']
        spread_std = spread.rolling(window).std()
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        raw = (spread / (spread_std + 1e-8)) * volume_ratio
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_138_h(df, window=50):
        spread = df['high'] - df['low']
        spread_std = spread.rolling(window).std()
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        raw = (spread / (spread_std + 1e-8)) * volume_ratio
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_138_e(df, window=80):
        spread = df['high'] - df['low']
        spread_std = spread.rolling(window).std()
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        raw = (spread / (spread_std + 1e-8)) * volume_ratio
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_138_y(df, window=5):
        spread = df['high'] - df['low']
        spread_std = spread.rolling(window).std()
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        raw = (spread / (spread_std + 1e-8)) * volume_ratio
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_138_r(df, window=60, p1=0.7):
        spread = df['high'] - df['low']
        spread_std = spread.rolling(window).std()
        volume_ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        raw = (spread / (spread_std + 1e-8)) * volume_ratio
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_139_rank(df, window=50, sub_window=1):
        # TS_ZSCORE volume with rolling rank normalization
        volume = df['matchingVolume']
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        # Sign of mean return
        returns = df['close'].pct_change()
        mean_return = returns.rolling(sub_window).mean()
        sign_ret = np.sign(mean_return)
        raw = volume_zscore * sign_ret
        # Rolling rank normalization to [-1, 1]
        result = raw.rolling(window).rank(pct=True) * 2 - 1
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_139_tanh(df, window=30, sub_window=1):
        # TS_ZSCORE volume with dynamic tanh normalization
        volume = df['matchingVolume']
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        returns = df['close'].pct_change()
        mean_return = returns.rolling(sub_window).mean()
        sign_ret = np.sign(mean_return)
        raw = volume_zscore * sign_ret
        # Dynamic tanh
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_139_zscore(df, window=10, sub_window=1):
        # TS_ZSCORE volume with rolling z-score clip
        volume = df['matchingVolume']
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        returns = df['close'].pct_change()
        mean_return = returns.rolling(sub_window).mean()
        sign_ret = np.sign(mean_return)
        raw = volume_zscore * sign_ret
        # Rolling z-score clip
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - roll_mean) / roll_std).clip(-1, 1)
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_139_sign(df, window=10, sub_window=1):
        # TS_ZSCORE volume with sign/binary soft
        volume = df['matchingVolume']
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        returns = df['close'].pct_change()
        mean_return = returns.rolling(sub_window).mean()
        sign_ret = np.sign(mean_return)
        raw = volume_zscore * sign_ret
        # Sign normalization
        result = np.sign(raw)
        result = pd.Series(result, index=df.index).ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_139_wf(df, window=10, sub_window=1, winsor_pct=0.1):
        # TS_ZSCORE volume with winsorized fisher transform
        volume = df['matchingVolume']
        volume_zscore = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
        returns = df['close'].pct_change()
        mean_return = returns.rolling(sub_window).mean()
        sign_ret = np.sign(mean_return)
        raw = volume_zscore * sign_ret
        # Winsorized Fisher normalization
        p1 = winsor_pct
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = pd.Series(normalized, index=df.index).ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_140_rank(df, param=30):
        window = param
        _ret = df['close'].pct_change().fillna(0)
        raw = (_ret.rolling(5).mean() - _ret.rolling(20).mean()) / (_ret.rolling(10).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_140_tanh(df, param=10):
        window = param
        _ret = df['close'].pct_change().fillna(0)
        raw = (_ret.rolling(5).mean() - _ret.rolling(20).mean()) / (_ret.rolling(10).std() + 1e-8)
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        return signal.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_140_zscore(df, param=5):
        window = param
        _ret = df['close'].pct_change().fillna(0)
        raw = (_ret.rolling(5).mean() - _ret.rolling(20).mean()) / (_ret.rolling(10).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_140_sign(df, param=5):
        window = param
        _ret = df['close'].pct_change().fillna(0)
        raw = (_ret.rolling(5).mean() - _ret.rolling(20).mean()) / (_ret.rolling(10).std() + 1e-8)
        signal = np.sign(raw).fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_140_wf(df, p1=0.3, p2=30):
        _ret = df['close'].pct_change().fillna(0)
        raw = (_ret.rolling(5).mean() - _ret.rolling(20).mean()) / (_ret.rolling(10).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_141_rank(df, window=100):
        high_low_diff = df['high'] - df['low']
        raw = (high_low_diff.abs() / df['open']) / (high_low_diff.rolling(window).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_141_tanh(df, window=40):
        high_low_diff = df['high'] - df['low']
        raw = (high_low_diff.abs() / df['open']) / (high_low_diff.rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_141_zscore(df, window=5):
        high_low_diff = df['high'] - df['low']
        raw = (high_low_diff.abs() / df['open']) / (high_low_diff.rolling(window).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_141_sign(df, window=5):
        high_low_diff = df['high'] - df['low']
        raw = (high_low_diff.abs() / df['open']) / (high_low_diff.rolling(window).std() + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_141_wf(df, window=40, winsor_percentile=0.7):
        high_low_diff = df['high'] - df['low']
        raw = (high_low_diff.abs() / df['open']) / (high_low_diff.rolling(window).std() + 1e-8)
        p1 = winsor_percentile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_142_k(df, window=25):
        close = df['close'].astype(float)
        volume = df['matchingVolume'].astype(float)
        delta_close = close.diff(1)
        corr = delta_close.rolling(window).corr(volume)
        normalized = (corr.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_142_h(df, window=5):
        close = df['close'].astype(float)
        volume = df['matchingVolume'].astype(float)
        delta_close = close.diff(1)
        corr = delta_close.rolling(window).corr(volume)
        std = corr.rolling(window).std()
        normalized = np.tanh(corr / std.replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_142_e(df, window=5):
        close = df['close'].astype(float)
        volume = df['matchingVolume'].astype(float)
        delta_close = close.diff(1)
        corr = delta_close.rolling(window).corr(volume)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std()
        normalized = ((corr - mean) / std.replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_142_y(df, window=5):
        close = df['close'].astype(float)
        volume = df['matchingVolume'].astype(float)
        delta_close = close.diff(1)
        corr = delta_close.rolling(window).corr(volume)
        sign = pd.Series(np.sign(corr), index=df.index)
        return sign.fillna(0)

    @staticmethod
    def alpha_quanta_142_r(df, window=10, winsor_quantile=0.3):
        close = df['close'].astype(float)
        volume = df['matchingVolume'].astype(float)
        delta_close = close.diff(1)
        corr = delta_close.rolling(window).corr(volume)
        low = corr.rolling(window).quantile(winsor_quantile)
        high = corr.rolling(window).quantile(1 - winsor_quantile)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_143_rank(df, window=70):
        # Case A: Rolling Rank
        v = df.get('matchingVolume', df.get('volume', 1))
        c = df['close']
        h = df['high']
        l = df['low']
        v_zscore = (v - v.rolling(window).mean()) / v.rolling(window).std().replace(0, np.nan)
        v_zscore = v_zscore.ffill().fillna(0)
        v_rank = (v_zscore.rolling(window).rank(pct=True) * 2) - 1
        ret = c.pct_change()
        ret_mean_5 = ret.rolling(5).mean()
        ret_mean_20 = ret.rolling(20).mean()
        ret_std_10 = ret.rolling(10).std().replace(0, np.nan)
        ret_component = (ret_mean_5 - ret_mean_20) / (ret_std_10 + 1e-8)
        hl_range = h - l
        hl_std_10 = hl_range.rolling(10).std().replace(0, np.nan)
        hl_component = abs(hl_range) / (hl_std_10 + 1e-8)
        raw = v_rank + ret_component + hl_component
        raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_143_tanh(df, window=30):
        # Case B: Dynamic Tanh
        v = df.get('matchingVolume', df.get('volume', 1))
        c = df['close']
        h = df['high']
        l = df['low']
        v_zscore = (v - v.rolling(window).mean()) / v.rolling(window).std().replace(0, np.nan)
        v_zscore = v_zscore.ffill().fillna(0)
        v_tanh = np.tanh(v_zscore / (v_zscore.rolling(window).std().replace(0, np.nan) + 1e-8))
        ret = c.pct_change()
        ret_mean_5 = ret.rolling(5).mean()
        ret_mean_20 = ret.rolling(20).mean()
        ret_std_10 = ret.rolling(10).std().replace(0, np.nan)
        ret_component = (ret_mean_5 - ret_mean_20) / (ret_std_10 + 1e-8)
        hl_range = h - l
        hl_std_10 = hl_range.rolling(10).std().replace(0, np.nan)
        hl_component = abs(hl_range) / (hl_std_10 + 1e-8)
        raw = v_tanh + ret_component + hl_component
        sigma = raw.rolling(window).std().replace(0, np.nan)
        result = np.tanh(raw / (sigma + 1e-8))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_143_zscore(df, window=5):
        # Case C: Rolling Z-Score/Clip
        v = df.get('matchingVolume', df.get('volume', 1))
        c = df['close']
        h = df['high']
        l = df['low']
        v_zscore = (v - v.rolling(window).mean()) / v.rolling(window).std().replace(0, np.nan)
        v_zscore = v_zscore.ffill().fillna(0)
        v_zs = (v_zscore - v_zscore.rolling(window).mean()) / v_zscore.rolling(window).std().replace(0, np.nan)
        ret = c.pct_change()
        ret_mean_5 = ret.rolling(5).mean()
        ret_mean_20 = ret.rolling(20).mean()
        ret_std_10 = ret.rolling(10).std().replace(0, np.nan)
        ret_component = (ret_mean_5 - ret_mean_20) / (ret_std_10 + 1e-8)
        hl_range = h - l
        hl_std_10 = hl_range.rolling(10).std().replace(0, np.nan)
        hl_component = abs(hl_range) / (hl_std_10 + 1e-8)
        raw = v_zs + ret_component + hl_component
        raw = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        result = raw.clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_143_sign(df, window=55):
        # Case D: Sign/Binary Soft
        v = df.get('matchingVolume', df.get('volume', 1))
        c = df['close']
        h = df['high']
        l = df['low']
        v_zscore = (v - v.rolling(window).mean()) / v.rolling(window).std().replace(0, np.nan)
        v_zscore = v_zscore.ffill().fillna(0)
        v_sign = np.sign(v_zscore.rolling(window).mean()).fillna(0)
        ret = c.pct_change()
        ret_mean_5 = ret.rolling(5).mean()
        ret_mean_20 = ret.rolling(20).mean()
        ret_std_10 = ret.rolling(10).std().replace(0, np.nan)
        ret_component = (ret_mean_5 - ret_mean_20) / (ret_std_10 + 1e-8)
        hl_range = h - l
        hl_std_10 = hl_range.rolling(10).std().replace(0, np.nan)
        hl_component = abs(hl_range) / (hl_std_10 + 1e-8)
        raw = v_sign + ret_component + hl_component
        result = np.sign(raw).fillna(0)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_143_wf(df, window=80, p1=0.1):
        # Case E: Winsorized Fisher
        v = df.get('matchingVolume', df.get('volume', 1))
        c = df['close']
        h = df['high']
        l = df['low']
        v_zscore = (v - v.rolling(window).mean()) / v.rolling(window).std().replace(0, np.nan)
        v_zscore = v_zscore.ffill().fillna(0)
        v_win = v_zscore.clip(lower=v_zscore.rolling(20).quantile(p1), upper=v_zscore.rolling(20).quantile(1 - p1), axis=0)
        low_v = v_zscore.rolling(20).quantile(p1)
        high_v = v_zscore.rolling(20).quantile(1 - p1)
        v_fisher = np.arctanh(((v_win - low_v) / (high_v - low_v + 1e-9)) * 1.98 - 0.99).fillna(0)
        ret = c.pct_change()
        ret_mean_5 = ret.rolling(5).mean()
        ret_mean_20 = ret.rolling(20).mean()
        ret_std_10 = ret.rolling(10).std().replace(0, np.nan)
        ret_component = (ret_mean_5 - ret_mean_20) / (ret_std_10 + 1e-8)
        hl_range = h - l
        hl_std_10 = hl_range.rolling(10).std().replace(0, np.nan)
        hl_component = abs(hl_range) / (hl_std_10 + 1e-8)
        raw = v_fisher + ret_component + hl_component
        # Apply Fisher again on combined
        raw_win = raw.clip(lower=raw.rolling(20).quantile(p1), upper=raw.rolling(20).quantile(1 - p1), axis=0)
        low_raw = raw.rolling(20).quantile(p1)
        high_raw = raw.rolling(20).quantile(1 - p1)
        result = np.arctanh(((raw_win - low_raw) / (high_raw - low_raw + 1e-9)) * 1.98 - 0.99).fillna(0)
        return result.clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_144_rank(df, window=95):
        close = df['close']
        open = df['open']
        ret = close.pct_change()
        raw = (open - close.shift(1)).abs() / (ret.rolling(5).std() + 1e-8)
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_144_tanh(df, window=5):
        close = df['close']
        open = df['open']
        ret = close.pct_change()
        raw = (open - close.shift(1)).abs() / (ret.rolling(5).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_144_zscore(df, window=15):
        close = df['close']
        open = df['open']
        ret = close.pct_change()
        raw = (open - close.shift(1)).abs() / (ret.rolling(5).std() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_144_sign(df):
        close = df['close']
        open = df['open']
        ret = close.pct_change()
        raw = (open - close.shift(1)).abs() / (ret.rolling(5).std() + 1e-8)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_144_wf(df, window1=3, window2=10):
        close = df['close']
        open = df['open']
        ret = close.pct_change()
        raw = (open - close.shift(1)).abs() / (ret.rolling(window1).std() + 1e-8)
        low = raw.rolling(window2).quantile(0.05)
        high = raw.rolling(window2).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_145_rank(df, window=100):
        # Tính return
        close = df['close']
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        # Tính delta volume
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff(1)
        # Tính TS_CORR(ret, delta_vol, window)
        corr = ret.rolling(window).corr(delta_vol).replace([np.inf, -np.inf], np.nan)
        # SIGN(corr) * corr -> giữ nguyên dấu và độ lớn, clip sau
        raw = corr * np.sign(corr.fillna(0))
        # Chuẩn hóa Rolling Rank
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_145_tanh(df, window=50):
        # Tính return
        close = df['close']
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        # Tính delta volume
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff(1)
        # Tính TS_CORR(ret, delta_vol, window)
        corr = ret.rolling(window).corr(delta_vol).replace([np.inf, -np.inf], np.nan)
        # SIGN(corr) * corr -> giữ nguyên dấu và độ lớn
        raw = corr * np.sign(corr.fillna(0))
        # Chuẩn hóa Dynamic Tanh
        std = raw.rolling(window).std().replace(0, np.nan)
        result = np.tanh(raw / std)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_145_zscore(df, window=40):
        # Tính return
        close = df['close']
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        # Tính delta volume
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff(1)
        # Tính TS_CORR(ret, delta_vol, window)
        corr = ret.rolling(window).corr(delta_vol).replace([np.inf, -np.inf], np.nan)
        # SIGN(corr) * corr -> giữ nguyên dấu và độ lớn
        raw = corr * np.sign(corr.fillna(0))
        # Chuẩn hóa Rolling Z-Score/Clip
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_145_sign(df, window=65):
        # Tính return
        close = df['close']
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        # Tính delta volume
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff(1)
        # Tính TS_CORR(ret, delta_vol, window)
        corr = ret.rolling(window).corr(delta_vol).replace([np.inf, -np.inf], np.nan)
        # SIGN(corr) * corr -> giữ nguyên dấu và độ lớn, chỉ lấy dấu
        raw = corr * np.sign(corr.fillna(0))
        # Chuẩn hóa Sign/Binary Soft
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_145_wf(df, window=40, sub_window=40):
        # Tính return
        close = df['close']
        ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
        # Tính delta volume
        volume = df.get('matchingVolume', df.get('volume', 1))
        delta_vol = volume.diff(1)
        # Tính TS_CORR(ret, delta_vol, window)
        corr = ret.rolling(window).corr(delta_vol).replace([np.inf, -np.inf], np.nan)
        # SIGN(corr) * corr -> giữ nguyên dấu và độ lớn
        raw = corr * np.sign(corr.fillna(0))
        # Chuẩn hóa Winsorized Fisher
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_146_rank(df, window=100):
        # Tính spread giá (high - low)
        spread = df['high'] - df['low']
        # Tính trung bình khối lượng rolling
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan).ffill()
        # Ratio spread / volume
        ratio = spread / (vol_mean + 1e-8)
        # Tính Z-score của ratio với rolling mean và std
        mean = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        raw = (ratio - mean) / (std + 1e-8)
        # Chuẩn hóa Rolling Rank (Phương pháp A)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_146_tanh(df, window=80):
        # Tính spread giá (high - low)
        spread = df['high'] - df['low']
        # Tính trung bình khối lượng rolling
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan).ffill()
        # Ratio spread / volume
        ratio = spread / (vol_mean + 1e-8)
        # Tính Z-score của ratio với rolling mean và std
        mean = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        raw = (ratio - mean) / (std + 1e-8)
        # Chuẩn hóa Dynamic Tanh (Phương pháp B)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_146_zscore(df, window=65):
        # Tính spread giá (high - low)
        spread = df['high'] - df['low']
        # Tính trung bình khối lượng rolling
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan).ffill()
        # Ratio spread / volume
        ratio = spread / (vol_mean + 1e-8)
        # Tính Z-score của ratio với rolling mean và std
        mean = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        raw = (ratio - mean) / (std + 1e-8)
        # Chuẩn hóa Rolling Z-Score Clip (Phương pháp C)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_146_sign(df, window=30):
        # Tính spread giá (high - low)
        spread = df['high'] - df['low']
        # Tính trung bình khối lượng rolling
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window).mean().replace(0, np.nan).ffill()
        # Ratio spread / volume
        ratio = spread / (vol_mean + 1e-8)
        # Tính Z-score của ratio với rolling mean và std
        mean = ratio.rolling(window).mean()
        std = ratio.rolling(window).std().replace(0, np.nan)
        raw = (ratio - mean) / (std + 1e-8)
        # Chuẩn hóa Sign/Binary Soft (Phương pháp D)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_146_wf(df, window_rank=90, winsor_quantile=0.7):
        # Tính spread giá (high - low)
        spread = df['high'] - df['low']
        # Tính trung bình khối lượng rolling
        vol_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(window_rank).mean().replace(0, np.nan).ffill()
        # Ratio spread / volume
        ratio = spread / (vol_mean + 1e-8)
        # Tính Z-score của ratio với rolling mean và std
        mean = ratio.rolling(window_rank).mean()
        std = ratio.rolling(window_rank).std().replace(0, np.nan)
        raw = (ratio - mean) / (std + 1e-8)
        # Chuẩn hóa Winsorized Fisher (Phương pháp E)
        low = raw.rolling(window_rank).quantile(winsor_quantile)
        high = raw.rolling(window_rank).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = norm.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_147_rank(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct_close = close.pct_change(5)
        pct_volume = volume.pct_change(5)
        ts_rank_close = pct_close.rolling(window).rank(pct=True)
        ts_rank_volume = pct_volume.rolling(window).rank(pct=True)
        raw = ts_rank_close - ts_rank_volume
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_147_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct_close = close.pct_change(5)
        pct_volume = volume.pct_change(5)
        ts_rank_close = pct_close.rolling(window).rank(pct=True)
        ts_rank_volume = pct_volume.rolling(window).rank(pct=True)
        raw = ts_rank_close - ts_rank_volume
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_147_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct_close = close.pct_change(5)
        pct_volume = volume.pct_change(5)
        ts_rank_close = pct_close.rolling(window).rank(pct=True)
        ts_rank_volume = pct_volume.rolling(window).rank(pct=True)
        raw = ts_rank_close - ts_rank_volume
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_) / std_).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_147_sign(df, window=20):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct_close = close.pct_change(5)
        pct_volume = volume.pct_change(5)
        ts_rank_close = pct_close.rolling(window).rank(pct=True)
        ts_rank_volume = pct_volume.rolling(window).rank(pct=True)
        raw = ts_rank_close - ts_rank_volume
        signal = pd.Series(np.sign(raw), index=df.index)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_147_wf(df, window=30, quantile=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        pct_close = close.pct_change(5)
        pct_volume = volume.pct_change(5)
        ts_rank_close = pct_close.rolling(window).rank(pct=True)
        ts_rank_volume = pct_volume.rolling(window).rank(pct=True)
        raw = ts_rank_close - ts_rank_volume
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high)
        norm = (winsorized - low) / (high - low + 1e-9)
        signal = np.arctanh(norm * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_148_rank(df, window=50):
        """
        TRƯỜNG HỢP A (Rolling Rank): Chuẩn hóa output về phân phối đồng nhất, tốt cho việc loại nhiễu/outliers.
        """
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) * (ret.rolling(window).sum() / (ret.rolling(window).std() + 1e-8))
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_148_tanh(df, window=60):
        """
        TRƯỜNG HỢP B (Dynamic Tanh): Giữ lại cường độ tín hiệu (magnitude), phù hợp khi magnitude có ý nghĩa.
        """
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) * (ret.rolling(window).sum() / (ret.rolling(window).std() + 1e-8))
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_148_zscore(df, window=100):
        """
        TRƯỜNG HỢP C (Rolling Z-Score/Clip): Phù hợp cho các dạng oscillator/spread, đưa về z-score rồi clip (-1,1).
        """
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) * (ret.rolling(window).sum() / (ret.rolling(window).std() + 1e-8))
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std()
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_148_sign(df, window=20):
        """
        TRƯỜNG HỢP D (Sign/Binary Soft): Chỉ lấy hướng giao dịch thuần túy, phù hợp trend-following/breakout.
        """
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) * (ret.rolling(window).sum() / (ret.rolling(window).std() + 1e-8))
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_148_wf(df, window=90, quantile=0.7):
        """
        TRƯỜNG HỢP E (Winsorized Fisher): Xử lý đuôi nặng, outlier cực đoan nhưng giữ cấu trúc phân phối.
        Tham số quantile là tỷ lệ winsorization (mặc định 0.05 => cắt bỏ 5% mỗi đuôi).
        """
        ret = df['close'].pct_change()
        raw = np.sign(ret.rolling(window).sum()) * (ret.rolling(window).sum() / (ret.rolling(window).std() + 1e-8))
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        numerator = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(numerator * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_149_k(df, window=50):
        close = df['close']
        volume = df['matchingVolume'] if 'matchingVolume' in df else df.get('volume', 1)
        raw = ((df['open'] - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign((volume / (volume.rolling(window).mean() + 1e-8)) - 1)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_149_h(df, window=5):
        close = df['close']
        volume = df['matchingVolume'] if 'matchingVolume' in df else df.get('volume', 1)
        raw = ((df['open'] - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign((volume / (volume.rolling(window).mean() + 1e-8)) - 1)
        normalized = np.tanh(raw / raw.rolling(window).std())
        signal = normalized.fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_149_e(df, window=5):
        close = df['close']
        volume = df['matchingVolume'] if 'matchingVolume' in df else df.get('volume', 1)
        raw = ((df['open'] - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign((volume / (volume.rolling(window).mean() + 1e-8)) - 1)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = normalized.fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_149_y(df, window=5):
        close = df['close']
        volume = df['matchingVolume'] if 'matchingVolume' in df else df.get('volume', 1)
        raw = ((df['open'] - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign((volume / (volume.rolling(window).mean() + 1e-8)) - 1)
        normalized = np.sign(raw)
        signal = normalized.fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_149_r(df, window=60, p1=0.1, p2=20):
        close = df['close']
        volume = df['matchingVolume'] if 'matchingVolume' in df else df.get('volume', 1)
        raw = ((df['open'] - close.shift(1)) / (close.rolling(window).std() + 1e-8)) * np.sign((volume / (volume.rolling(window).mean() + 1e-8)) - 1)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_150_rank(df, window=55):
        # Rolling Rank normalization (Case A)
        # raw = DECAYLINEAR(TS_CORR(DELTA(volume, 1) / volume, (high - low) / (high + low), 15), 15)
        volume = df['matchingVolume']
        volume_delta = volume.diff(1)
        volume_ratio = volume_delta / (volume + 1e-8)
        spread = (df['high'] - df['low']) / (df['high'] + df['low'] + 1e-8)
        # Compute rolling correlation with covariance method
        corr_ts = (volume_ratio.rolling(window).cov(spread) / 
                   (volume_ratio.rolling(window).std() * spread.rolling(window).std() + 1e-8))
        corr_ts = corr_ts.fillna(0)
        # DECAYLINEAR: linear weighted moving average
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        decay_linear = corr_ts.rolling(window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == window else np.nan, raw=False
        )
        decay_linear = decay_linear.fillna(0)
        # Normalize: Rolling Rank
        signal = (decay_linear.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_150_tanh(df, window=95):
        # Dynamic Tanh normalization (Case B)
        volume = df['matchingVolume']
        volume_delta = volume.diff(1)
        volume_ratio = volume_delta / (volume + 1e-8)
        spread = (df['high'] - df['low']) / (df['high'] + df['low'] + 1e-8)
        corr_ts = (volume_ratio.rolling(window).cov(spread) / 
                   (volume_ratio.rolling(window).std() * spread.rolling(window).std() + 1e-8))
        corr_ts = corr_ts.fillna(0)
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        decay_linear = corr_ts.rolling(window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == window else np.nan, raw=False
        )
        decay_linear = decay_linear.fillna(0)
        std_val = decay_linear.rolling(window).std().replace(0, np.nan).fillna(1e-8)
        signal = np.tanh(decay_linear / std_val)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_150_zscore(df, window=60):
        # Rolling Z-Score normalization (Case C)
        volume = df['matchingVolume']
        volume_delta = volume.diff(1)
        volume_ratio = volume_delta / (volume + 1e-8)
        spread = (df['high'] - df['low']) / (df['high'] + df['low'] + 1e-8)
        corr_ts = (volume_ratio.rolling(window).cov(spread) / 
                   (volume_ratio.rolling(window).std() * spread.rolling(window).std() + 1e-8))
        corr_ts = corr_ts.fillna(0)
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        decay_linear = corr_ts.rolling(window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == window else np.nan, raw=False
        )
        decay_linear = decay_linear.fillna(0)
        mean_val = decay_linear.rolling(window).mean()
        std_val = decay_linear.rolling(window).std().replace(0, np.nan).fillna(1e-8)
        zscore = (decay_linear - mean_val) / std_val
        signal = zscore.clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_150_sign(df, window=10):
        # Sign/Binary Soft normalization (Case D)
        volume = df['matchingVolume']
        volume_delta = volume.diff(1)
        volume_ratio = volume_delta / (volume + 1e-8)
        spread = (df['high'] - df['low']) / (df['high'] + df['low'] + 1e-8)
        corr_ts = (volume_ratio.rolling(window).cov(spread) / 
                   (volume_ratio.rolling(window).std() * spread.rolling(window).std() + 1e-8))
        corr_ts = corr_ts.fillna(0)
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        decay_linear = corr_ts.rolling(window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == window else np.nan, raw=False
        )
        decay_linear = decay_linear.fillna(0)
        signal = np.sign(decay_linear)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_150_wf(df, window=60, p1=0.1):
        # Winsorized Fisher normalization (Case E)
        volume = df['matchingVolume']
        volume_delta = volume.diff(1)
        volume_ratio = volume_delta / (volume + 1e-8)
        spread = (df['high'] - df['low']) / (df['high'] + df['low'] + 1e-8)
        corr_ts = (volume_ratio.rolling(window).cov(spread) / 
                   (volume_ratio.rolling(window).std() * spread.rolling(window).std() + 1e-8))
        corr_ts = corr_ts.fillna(0)
        weights = np.arange(1, window + 1, dtype=float)
        weights /= weights.sum()
        decay_linear = corr_ts.rolling(window).apply(
            lambda x: np.dot(x, weights[:len(x)]) if len(x) == window else np.nan, raw=False
        )
        decay_linear = decay_linear.fillna(0)
        # Apply Winsorized Fisher
        p2 = window
        low_q = decay_linear.rolling(p2).quantile(p1)
        high_q = decay_linear.rolling(p2).quantile(1 - p1)
        winsorized = decay_linear.clip(lower=low_q, upper=high_q)
        norm_val = ((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99
        # Fisher Transform using arctanh
        signal = np.arctanh(np.clip(norm_val, -0.99, 0.99))
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_151_rank(df, window=70):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_ = high - low
        range_mean = range_.rolling(window, min_periods=1).mean().replace(0, np.nan)
        volume_mean = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
        raw = (range_ / range_mean) * (volume / volume_mean)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        z = (raw - raw.rolling(window, min_periods=1).mean()) / raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        signal = z.rolling(window, min_periods=1).rank(pct=True) * 2 - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_151_tanh(df, window=100):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_ = high - low
        range_mean = range_.rolling(window, min_periods=1).mean().replace(0, np.nan)
        volume_mean = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
        raw = (range_ / range_mean) * (volume / volume_mean)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        signal = np.tanh(raw / raw.rolling(window, min_periods=1).std().replace(0, np.nan))
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_151_zscore(df, window=50):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_ = high - low
        range_mean = range_.rolling(window, min_periods=1).mean().replace(0, np.nan)
        volume_mean = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
        raw = (range_ / range_mean) * (volume / volume_mean)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        raw_mean = raw.rolling(window, min_periods=1).mean()
        raw_std = raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        signal = ((raw - raw_mean) / raw_std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_151_sign(df, window=95):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_ = high - low
        range_mean = range_.rolling(window, min_periods=1).mean().replace(0, np.nan)
        volume_mean = volume.rolling(window, min_periods=1).mean().replace(0, np.nan)
        raw = (range_ / range_mean) * (volume / volume_mean)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        signal = pd.Series(np.sign(raw), index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_151_wf(df, p1=0.1, p2=10):
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        range_ = high - low
        range_mean = range_.rolling(10, min_periods=1).mean().replace(0, np.nan)
        volume_mean = volume.rolling(10, min_periods=1).mean().replace(0, np.nan)
        raw = (range_ / range_mean) * (volume / volume_mean)
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        low_q = raw.rolling(p2, min_periods=1).quantile(p1)
        high_q = raw.rolling(p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_152_k(df, window=5):
        raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        raw_abs = raw.abs()
        corr = raw.rolling(window).corr(raw_abs)
        normalized = (corr.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_152_h(df, window=5):
        raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        raw_abs = raw.abs()
        corr = raw.rolling(window).corr(raw_abs)
        denom = corr.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(corr / denom)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_152_p(df, window=5):
        raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        raw_abs = raw.abs()
        corr = raw.rolling(window).corr(raw_abs)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        normalized = ((corr - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_152_t(df, window=5):
        raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        raw_abs = raw.abs()
        corr = raw.rolling(window).corr(raw_abs)
        return np.sign(corr).fillna(0)

    @staticmethod
    def alpha_quanta_152_r(df, window=10, winsor_quantile=0.3):
        raw = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        raw_abs = raw.abs()
        corr = raw.rolling(window).corr(raw_abs)
        low = corr.rolling(window).quantile(winsor_quantile)
        high = corr.rolling(window).quantile(1 - winsor_quantile)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_153_rank(df, window=60, sub_window=30):
        ret = df['close'].pct_change()
        sign = np.sign(ret.rolling(window).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_ratio = vol.diff() / (vol + 1e-8)
        raw = sign * ((vol_ratio - vol_ratio.rolling(sub_window).mean()) / vol_ratio.rolling(sub_window).std())
        signal = (raw.rolling(sub_window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_153_tanh(df, window=10, sub_window=10):
        ret = df['close'].pct_change()
        sign = np.sign(ret.rolling(window).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_ratio = vol.diff() / (vol + 1e-8)
        raw = sign * ((vol_ratio - vol_ratio.rolling(sub_window).mean()) / vol_ratio.rolling(sub_window).std())
        signal = np.tanh(raw / raw.rolling(sub_window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_153_zscore(df, window=70, sub_window=30):
        ret = df['close'].pct_change()
        sign = np.sign(ret.rolling(window).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_ratio = vol.diff() / (vol + 1e-8)
        raw = sign * ((vol_ratio - vol_ratio.rolling(sub_window).mean()) / vol_ratio.rolling(sub_window).std())
        signal = ((raw - raw.rolling(sub_window).mean()) / raw.rolling(sub_window).std()).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_153_sign(df, window=20, sub_window=20):
        ret = df['close'].pct_change()
        sign = np.sign(ret.rolling(window).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_ratio = vol.diff() / (vol + 1e-8)
        raw = sign * ((vol_ratio - vol_ratio.rolling(sub_window).mean()) / vol_ratio.rolling(sub_window).std())
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_153_wf(df, p1=0.1, p2=20):
        ret = df['close'].pct_change()
        sign = np.sign(ret.rolling(p2).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        vol_ratio = vol.diff() / (vol + 1e-8)
        raw = sign * ((vol_ratio - vol_ratio.rolling(p2).mean()) / vol_ratio.rolling(p2).std())
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_154_k(df, window=35):
        raw = (df['high'] - df['low']).rolling(window).std() / (df['close'].rolling(window).std() + 1e-8)
        raw_s = raw.rolling(window).rank(pct=True) * 2 - 1
        return -raw_s

    @staticmethod
    def alpha_quanta_154_h(df, window=35):
        raw = (df['high'] - df['low']).rolling(window).std() / (df['close'].rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8))
        return -normalized

    @staticmethod
    def alpha_quanta_154_e(df, window=95):
        raw = (df['high'] - df['low']).rolling(window).std() / (df['close'].rolling(window).std() + 1e-8)
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_154_y(df, window=100):
        raw = (df['high'] - df['low']).rolling(window).std() / (df['close'].rolling(window).std() + 1e-8)
        normalized = np.sign(raw - raw.rolling(window).median())
        return normalized

    @staticmethod
    def alpha_quanta_154_r(df, window=100, p1=0.1, p2=40):
        raw = (df['high'] - df['low']).rolling(window).std() / (df['close'].rolling(window).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_155_rank(df, window=50):
        # Tính return
        close = df['close']
        ret = close.pct_change(fill_method=None).fillna(0)

        # Tính delta volume ratio
        volume = df['matchingVolume']
        vol_ratio = (volume - volume.shift(1)) / (volume + 1e-8)

        # Tính correlation giữa return và vol_ratio
        corr = ret.rolling(window, min_periods=1).corr(vol_ratio)

        # Tính std của return
        ret_std = ret.rolling(window, min_periods=1).std() + 1e-8

        # Tính mean sign của return
        ret_mean = ret.rolling(window, min_periods=1).mean()
        sign_mean = np.sign(ret_mean)

        # Raw signal
        raw = (corr / ret_std) * sign_mean

        # Chuẩn hóa rolling rank (A)
        result = (raw.rolling(window, min_periods=1).rank(pct=True) * 2) - 1
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_155_tanh(df, window=35):
        close = df['close']
        ret = close.pct_change(fill_method=None).fillna(0)
        volume = df['matchingVolume']
        vol_ratio = (volume - volume.shift(1)) / (volume + 1e-8)
        corr = ret.rolling(window, min_periods=1).corr(vol_ratio)
        ret_std = ret.rolling(window, min_periods=1).std() + 1e-8
        ret_mean = ret.rolling(window, min_periods=1).mean()
        sign_mean = np.sign(ret_mean)
        raw = (corr / ret_std) * sign_mean
        # Chuẩn hóa dynamic tanh (B)
        raw_std = raw.rolling(window, min_periods=1).std()
        result = np.tanh(raw / raw_std.replace(0, np.nan))
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_155_zscore(df, window=55):
        close = df['close']
        ret = close.pct_change(fill_method=None).fillna(0)
        volume = df['matchingVolume']
        vol_ratio = (volume - volume.shift(1)) / (volume + 1e-8)
        corr = ret.rolling(window, min_periods=1).corr(vol_ratio)
        ret_std = ret.rolling(window, min_periods=1).std() + 1e-8
        ret_mean = ret.rolling(window, min_periods=1).mean()
        sign_mean = np.sign(ret_mean)
        raw = (corr / ret_std) * sign_mean
        # Chuẩn hóa rolling z-score (C)
        raw_mean = raw.rolling(window, min_periods=1).mean()
        raw_std = raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        result = ((raw - raw_mean) / raw_std).clip(-1, 1)
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_155_sign(df, window=35):
        close = df['close']
        ret = close.pct_change(fill_method=None).fillna(0)
        volume = df['matchingVolume']
        vol_ratio = (volume - volume.shift(1)) / (volume + 1e-8)
        corr = ret.rolling(window, min_periods=1).corr(vol_ratio)
        ret_std = ret.rolling(window, min_periods=1).std() + 1e-8
        ret_mean = ret.rolling(window, min_periods=1).mean()
        sign_mean = np.sign(ret_mean)
        raw = (corr / ret_std) * sign_mean
        # Chuẩn hóa sign (D)
        result = np.sign(raw).astype(float)
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_155_wf(df, window=40, p1=0.1):
        close = df['close']
        ret = close.pct_change(fill_method=None).fillna(0)
        volume = df['matchingVolume']
        vol_ratio = (volume - volume.shift(1)) / (volume + 1e-8)
        corr = ret.rolling(window, min_periods=1).corr(vol_ratio)
        ret_std = ret.rolling(window, min_periods=1).std() + 1e-8
        ret_mean = ret.rolling(window, min_periods=1).mean()
        sign_mean = np.sign(ret_mean)
        raw = (corr / ret_std) * sign_mean
        # Chuẩn hóa winsorized fisher (E)
        low = raw.rolling(window, min_periods=1).quantile(p1)
        high = raw.rolling(window, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = result.fillna(0)
        return result

    @staticmethod
    def alpha_quanta_156_rank(df, window=100):
        w = window
        spread = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = spread.rolling(w).corr(volume)
        zscore = (corr - corr.rolling(w).mean()) / corr.rolling(w).std().replace(0, np.nan)
        raw = zscore.rolling(w).rank(pct=True) * 2 - 1
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_156_tanh(df, window=35):
        w = window
        spread = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = spread.rolling(w).corr(volume)
        zscore = (corr - corr.rolling(w).mean()) / corr.rolling(w).std().replace(0, np.nan)
        raw = np.tanh(zscore / zscore.rolling(w).std().replace(0, np.nan))
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_156_zscore(df, window=40):
        w = window
        spread = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = spread.rolling(w).corr(volume)
        zscore = (corr - corr.rolling(w).mean()) / corr.rolling(w).std().replace(0, np.nan)
        raw = ((zscore - zscore.rolling(w).mean()) / zscore.rolling(w).std().replace(0, np.nan)).clip(-1, 1)
        signal = raw.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_156_sign(df, window=70):
        w = window
        spread = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = spread.rolling(w).corr(volume)
        zscore = (corr - corr.rolling(w).mean()) / corr.rolling(w).std().replace(0, np.nan)
        raw = np.sign(zscore)
        signal = pd.Series(raw, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_156_wf(df, p1=0.7, p2=100):
        w = p2
        spread = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = spread.rolling(w).corr(volume)
        zscore = (corr - corr.rolling(w).mean()) / corr.rolling(w).std().replace(0, np.nan)
        low = zscore.rolling(p2).quantile(p1)
        high = zscore.rolling(p2).quantile(1 - p1)
        winsorized = zscore.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_157_rank(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(window).std().replace(0, np.nan)
        std_volume = volume.rolling(window).std().replace(0, np.nan)
        z_close = (close - close.rolling(window).mean()) / std_close
        z_volume = (volume - volume.rolling(window).mean()) / std_volume
        cov = z_close.rolling(window).cov(z_volume)
        var_close = z_close.rolling(window).var().replace(0, np.nan)
        var_volume = z_volume.rolling(window).var().replace(0, np.nan)
        corr = cov / np.sqrt(var_close * var_volume)
        raw = corr.rolling(window).rank(pct=True) * 2 - 1
        return raw.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_157_tanh(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(window).std().replace(0, np.nan)
        std_volume = volume.rolling(window).std().replace(0, np.nan)
        z_close = (close - close.rolling(window).mean()) / std_close
        z_volume = (volume - volume.rolling(window).mean()) / std_volume
        cov = z_close.rolling(window).cov(z_volume)
        var_close = z_close.rolling(window).var().replace(0, np.nan)
        var_volume = z_volume.rolling(window).var().replace(0, np.nan)
        corr = cov / np.sqrt(var_close * var_volume)
        raw = corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_157_zscore(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(window).std().replace(0, np.nan)
        std_volume = volume.rolling(window).std().replace(0, np.nan)
        z_close = (close - close.rolling(window).mean()) / std_close
        z_volume = (volume - volume.rolling(window).mean()) / std_volume
        cov = z_close.rolling(window).cov(z_volume)
        var_close = z_close.rolling(window).var().replace(0, np.nan)
        var_volume = z_volume.rolling(window).var().replace(0, np.nan)
        corr = cov / np.sqrt(var_close * var_volume)
        raw = corr
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_157_sign(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(window).std().replace(0, np.nan)
        std_volume = volume.rolling(window).std().replace(0, np.nan)
        z_close = (close - close.rolling(window).mean()) / std_close
        z_volume = (volume - volume.rolling(window).mean()) / std_volume
        cov = z_close.rolling(window).cov(z_volume)
        var_close = z_close.rolling(window).var().replace(0, np.nan)
        var_volume = z_volume.rolling(window).var().replace(0, np.nan)
        corr = cov / np.sqrt(var_close * var_volume)
        raw = corr
        normalized = np.sign(raw)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_157_wf(df, window=20, winsor_pct=0.1):
        close = df['close']
        volume = df['matchingVolume']
        std_close = close.rolling(window).std().replace(0, np.nan)
        std_volume = volume.rolling(window).std().replace(0, np.nan)
        z_close = (close - close.rolling(window).mean()) / std_close
        z_volume = (volume - volume.rolling(window).mean()) / std_volume
        cov = z_close.rolling(window).cov(z_volume)
        var_close = z_close.rolling(window).var().replace(0, np.nan)
        var_volume = z_volume.rolling(window).var().replace(0, np.nan)
        corr = cov / np.sqrt(var_close * var_volume)
        raw = corr
        p2 = max(int(0.1 * len(df)), 10)
        low = raw.rolling(p2).quantile(winsor_pct)
        high = raw.rolling(p2).quantile(1 - winsor_pct)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_158_rank(df, window=40, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        vol_mean = volume.rolling(sub_window).mean().replace(0, np.nan)
        volume_ratio = volume / (vol_mean + 1e-8) - 1
        volume_signal = np.sign(volume_ratio)
        volume_signal = pd.Series(volume_signal, index=df.index)
        close_std = close.rolling(window).std().replace(0, np.nan)
        raw_z = (close - close.rolling(window).mean()) / (close_std + 1e-8)
        raw = raw_z * volume_signal
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_158_tanh(df, window=10, sub_window=7):
        close = df['close']
        volume = df['matchingVolume']
        vol_mean = volume.rolling(sub_window).mean().replace(0, np.nan)
        volume_ratio = volume / (vol_mean + 1e-8) - 1
        volume_signal = np.sign(volume_ratio)
        volume_signal = pd.Series(volume_signal, index=df.index)
        close_std = close.rolling(window).std().replace(0, np.nan)
        raw_z = (close - close.rolling(window).mean()) / (close_std + 1e-8)
        raw = raw_z * volume_signal
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_158_zscore(df, window=10, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        vol_mean = volume.rolling(sub_window).mean().replace(0, np.nan)
        volume_ratio = volume / (vol_mean + 1e-8) - 1
        volume_signal = np.sign(volume_ratio)
        volume_signal = pd.Series(volume_signal, index=df.index)
        close_std = close.rolling(window).std().replace(0, np.nan)
        raw_z = (close - close.rolling(window).mean()) / (close_std + 1e-8)
        raw = raw_z * volume_signal
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_158_sign(df, window=10, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        vol_mean = volume.rolling(sub_window).mean().replace(0, np.nan)
        volume_ratio = volume / (vol_mean + 1e-8) - 1
        volume_signal = np.sign(volume_ratio)
        volume_signal = pd.Series(volume_signal, index=df.index)
        close_std = close.rolling(window).std().replace(0, np.nan)
        raw_z = (close - close.rolling(window).mean()) / (close_std + 1e-8)
        raw = raw_z * volume_signal
        normalized = np.sign(raw)
        normalized = pd.Series(normalized, index=df.index)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_158_wf(df, window=10, sub_window=1, p1=0.05):
        close = df['close']
        volume = df['matchingVolume']
        vol_mean = volume.rolling(sub_window).mean().replace(0, np.nan)
        volume_ratio = volume / (vol_mean + 1e-8) - 1
        volume_signal = np.sign(volume_ratio)
        volume_signal = pd.Series(volume_signal, index=df.index)
        close_std = close.rolling(window).std().replace(0, np.nan)
        raw_z = (close - close.rolling(window).mean()) / (close_std + 1e-8)
        raw = raw_z * volume_signal
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_159_rank(df, window=100):
        high_low = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean().add(1e-8)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)).astype(float) / df.get('matchingVolume', df.get('volume', 1)).astype(float).rolling(window=10).mean().add(1e-8)
        raw = high_low.rolling(window).corr(vol_ratio)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_159_tanh(df, window=5):
        high_low = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean().add(1e-8)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)).astype(float) / df.get('matchingVolume', df.get('volume', 1)).astype(float).rolling(window=10).mean().add(1e-8)
        raw = high_low.rolling(window).corr(vol_ratio)
        normalized = np.tanh(raw / raw.rolling(window).std().add(1e-8))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_159_zscore(df, window=45):
        high_low = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean().add(1e-8)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)).astype(float) / df.get('matchingVolume', df.get('volume', 1)).astype(float).rolling(window=10).mean().add(1e-8)
        raw = high_low.rolling(window).corr(vol_ratio)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().add(1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_159_sign(df, window=5):
        high_low = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean().add(1e-8)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)).astype(float) / df.get('matchingVolume', df.get('volume', 1)).astype(float).rolling(window=10).mean().add(1e-8)
        raw = high_low.rolling(window).corr(vol_ratio)
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_159_wf(df, window=80, p1=0.1):
        high_low = (df['high'] - df['low']) / (df['high'] - df['low']).rolling(window=10).mean().add(1e-8)
        vol_ratio = df.get('matchingVolume', df.get('volume', 1)).astype(float) / df.get('matchingVolume', df.get('volume', 1)).astype(float).rolling(window=10).mean().add(1e-8)
        raw = high_low.rolling(window).corr(vol_ratio)
        window2 = window
        low = raw.rolling(window2).quantile(p1)
        high = raw.rolling(window2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_160_rank(df, window=30):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        # Compute ZSCORE of close (window 20)
        mean_close_20 = close.rolling(window).mean()
        std_close_20 = close.rolling(window).std()
        zscore_close = (close - mean_close_20) / (std_close_20 + 1e-8)
        # Compute TS_CORR between standardized close and volume (window 5)
        mean_close_5 = close.rolling(5).mean()
        std_close_5 = close.rolling(5).std()
        std_close_5 = std_close_5.replace(0, np.nan)
        std_close_5 = std_close_5.ffill().fillna(1e-8)
        std_close_5 = std_close_5.replace(0, 1e-8)
        std_close_5 = std_close_5 + 1e-8
        zclose_5 = (close - mean_close_5) / std_close_5
        mean_vol_5 = volume.rolling(5).mean()
        std_vol_5 = volume.rolling(5).std()
        std_vol_5 = std_vol_5.replace(0, np.nan)
        std_vol_5 = std_vol_5.ffill().fillna(1e-8)
        std_vol_5 = std_vol_5.replace(0, 1e-8)
        std_vol_5 = std_vol_5 + 1e-8
        zvol_5 = (volume - mean_vol_5) / std_vol_5
        corr_close_vol_5 = zclose_5.rolling(5).corr(zvol_5)
        rank_corr = corr_close_vol_5.rank(pct=True) * 2 - 1
        # Compute TS_CORR between range and price change (window 5)
        range_ratio = (high - low) / (open_ + close + 1e-8)
        price_change_ratio = (close - open_) / (open_ + 1e-8)
        range_mean_5 = range_ratio.rolling(5).mean()
        range_std_5 = range_ratio.rolling(5).std()
        range_std_5 = range_std_5.replace(0, np.nan)
        range_std_5 = range_std_5.ffill().fillna(1e-8)
        range_std_5 = range_std_5.replace(0, 1e-8)
        range_std_5 = range_std_5 + 1e-8
        zrange_5 = (range_ratio - range_mean_5) / range_std_5
        price_change_mean_5 = price_change_ratio.rolling(5).mean()
        price_change_std_5 = price_change_ratio.rolling(5).std()
        price_change_std_5 = price_change_std_5.replace(0, np.nan)
        price_change_std_5 = price_change_std_5.ffill().fillna(1e-8)
        price_change_std_5 = price_change_std_5.replace(0, 1e-8)
        price_change_std_5 = price_change_std_5 + 1e-8
        zprice_change_5 = (price_change_ratio - price_change_mean_5) / price_change_std_5
        corr_range_price_5 = zrange_5.rolling(5).corr(zprice_change_5)
        # Multiply all components
        raw = rank_corr * zscore_close * corr_range_price_5
        # Normalize using Rolling Rank (Method A)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_160_tanh(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        # Compute ZSCORE of close (window 20)
        mean_close_20 = close.rolling(window).mean()
        std_close_20 = close.rolling(window).std()
        zscore_close = (close - mean_close_20) / (std_close_20 + 1e-8)
        # Compute TS_CORR between standardized close and volume (window 5)
        mean_close_5 = close.rolling(5).mean()
        std_close_5 = close.rolling(5).std()
        std_close_5 = std_close_5.replace(0, np.nan)
        std_close_5 = std_close_5.ffill().fillna(1e-8)
        std_close_5 = std_close_5.replace(0, 1e-8)
        std_close_5 = std_close_5 + 1e-8
        zclose_5 = (close - mean_close_5) / std_close_5
        mean_vol_5 = volume.rolling(5).mean()
        std_vol_5 = volume.rolling(5).std()
        std_vol_5 = std_vol_5.replace(0, np.nan)
        std_vol_5 = std_vol_5.ffill().fillna(1e-8)
        std_vol_5 = std_vol_5.replace(0, 1e-8)
        std_vol_5 = std_vol_5 + 1e-8
        zvol_5 = (volume - mean_vol_5) / std_vol_5
        corr_close_vol_5 = zclose_5.rolling(5).corr(zvol_5)
        rank_corr = corr_close_vol_5.rank(pct=True) * 2 - 1
        # Compute TS_CORR between range and price change (window 5)
        range_ratio = (high - low) / (open_ + close + 1e-8)
        price_change_ratio = (close - open_) / (open_ + 1e-8)
        range_mean_5 = range_ratio.rolling(5).mean()
        range_std_5 = range_ratio.rolling(5).std()
        range_std_5 = range_std_5.replace(0, np.nan)
        range_std_5 = range_std_5.ffill().fillna(1e-8)
        range_std_5 = range_std_5.replace(0, 1e-8)
        range_std_5 = range_std_5 + 1e-8
        zrange_5 = (range_ratio - range_mean_5) / range_std_5
        price_change_mean_5 = price_change_ratio.rolling(5).mean()
        price_change_std_5 = price_change_ratio.rolling(5).std()
        price_change_std_5 = price_change_std_5.replace(0, np.nan)
        price_change_std_5 = price_change_std_5.ffill().fillna(1e-8)
        price_change_std_5 = price_change_std_5.replace(0, 1e-8)
        price_change_std_5 = price_change_std_5 + 1e-8
        zprice_change_5 = (price_change_ratio - price_change_mean_5) / price_change_std_5
        corr_range_price_5 = zrange_5.rolling(5).corr(zprice_change_5)
        # Multiply all components
        raw = rank_corr * zscore_close * corr_range_price_5
        # Normalize using Dynamic Tanh (Method B)
        std_raw = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        normalized = np.tanh(raw / std_raw)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_160_zscore(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        # Compute ZSCORE of close (window 20)
        mean_close_20 = close.rolling(window).mean()
        std_close_20 = close.rolling(window).std()
        zscore_close = (close - mean_close_20) / (std_close_20 + 1e-8)
        # Compute TS_CORR between standardized close and volume (window 5)
        mean_close_5 = close.rolling(5).mean()
        std_close_5 = close.rolling(5).std()
        std_close_5 = std_close_5.replace(0, np.nan)
        std_close_5 = std_close_5.ffill().fillna(1e-8)
        std_close_5 = std_close_5.replace(0, 1e-8)
        std_close_5 = std_close_5 + 1e-8
        zclose_5 = (close - mean_close_5) / std_close_5
        mean_vol_5 = volume.rolling(5).mean()
        std_vol_5 = volume.rolling(5).std()
        std_vol_5 = std_vol_5.replace(0, np.nan)
        std_vol_5 = std_vol_5.ffill().fillna(1e-8)
        std_vol_5 = std_vol_5.replace(0, 1e-8)
        std_vol_5 = std_vol_5 + 1e-8
        zvol_5 = (volume - mean_vol_5) / std_vol_5
        corr_close_vol_5 = zclose_5.rolling(5).corr(zvol_5)
        rank_corr = corr_close_vol_5.rank(pct=True) * 2 - 1
        # Compute TS_CORR between range and price change (window 5)
        range_ratio = (high - low) / (open_ + close + 1e-8)
        price_change_ratio = (close - open_) / (open_ + 1e-8)
        range_mean_5 = range_ratio.rolling(5).mean()
        range_std_5 = range_ratio.rolling(5).std()
        range_std_5 = range_std_5.replace(0, np.nan)
        range_std_5 = range_std_5.ffill().fillna(1e-8)
        range_std_5 = range_std_5.replace(0, 1e-8)
        range_std_5 = range_std_5 + 1e-8
        zrange_5 = (range_ratio - range_mean_5) / range_std_5
        price_change_mean_5 = price_change_ratio.rolling(5).mean()
        price_change_std_5 = price_change_ratio.rolling(5).std()
        price_change_std_5 = price_change_std_5.replace(0, np.nan)
        price_change_std_5 = price_change_std_5.ffill().fillna(1e-8)
        price_change_std_5 = price_change_std_5.replace(0, 1e-8)
        price_change_std_5 = price_change_std_5 + 1e-8
        zprice_change_5 = (price_change_ratio - price_change_mean_5) / price_change_std_5
        corr_range_price_5 = zrange_5.rolling(5).corr(zprice_change_5)
        # Multiply all components
        raw = rank_corr * zscore_close * corr_range_price_5
        # Normalize using Rolling Z-Score (Method C)
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        normalized = ((raw - mean_raw) / std_raw).clip(-1, 1)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_160_sign(df, window=5):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        # Compute ZSCORE of close (window 20)
        mean_close_20 = close.rolling(window).mean()
        std_close_20 = close.rolling(window).std()
        zscore_close = (close - mean_close_20) / (std_close_20 + 1e-8)
        # Compute TS_CORR between standardized close and volume (window 5)
        mean_close_5 = close.rolling(5).mean()
        std_close_5 = close.rolling(5).std()
        std_close_5 = std_close_5.replace(0, np.nan)
        std_close_5 = std_close_5.ffill().fillna(1e-8)
        std_close_5 = std_close_5.replace(0, 1e-8)
        std_close_5 = std_close_5 + 1e-8
        zclose_5 = (close - mean_close_5) / std_close_5
        mean_vol_5 = volume.rolling(5).mean()
        std_vol_5 = volume.rolling(5).std()
        std_vol_5 = std_vol_5.replace(0, np.nan)
        std_vol_5 = std_vol_5.ffill().fillna(1e-8)
        std_vol_5 = std_vol_5.replace(0, 1e-8)
        std_vol_5 = std_vol_5 + 1e-8
        zvol_5 = (volume - mean_vol_5) / std_vol_5
        corr_close_vol_5 = zclose_5.rolling(5).corr(zvol_5)
        rank_corr = corr_close_vol_5.rank(pct=True) * 2 - 1
        # Compute TS_CORR between range and price change (window 5)
        range_ratio = (high - low) / (open_ + close + 1e-8)
        price_change_ratio = (close - open_) / (open_ + 1e-8)
        range_mean_5 = range_ratio.rolling(5).mean()
        range_std_5 = range_ratio.rolling(5).std()
        range_std_5 = range_std_5.replace(0, np.nan)
        range_std_5 = range_std_5.ffill().fillna(1e-8)
        range_std_5 = range_std_5.replace(0, 1e-8)
        range_std_5 = range_std_5 + 1e-8
        zrange_5 = (range_ratio - range_mean_5) / range_std_5
        price_change_mean_5 = price_change_ratio.rolling(5).mean()
        price_change_std_5 = price_change_ratio.rolling(5).std()
        price_change_std_5 = price_change_std_5.replace(0, np.nan)
        price_change_std_5 = price_change_std_5.ffill().fillna(1e-8)
        price_change_std_5 = price_change_std_5.replace(0, 1e-8)
        price_change_std_5 = price_change_std_5 + 1e-8
        zprice_change_5 = (price_change_ratio - price_change_mean_5) / price_change_std_5
        corr_range_price_5 = zrange_5.rolling(5).corr(zprice_change_5)
        # Multiply all components
        raw = rank_corr * zscore_close * corr_range_price_5
        # Normalize using Sign (Method D)
        normalized = np.sign(raw)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_160_wf(df, window=30, quantile=0.1):
        close = df['close']
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        open_ = df['open']
        # Compute ZSCORE of close (window 20)
        mean_close_20 = close.rolling(window).mean()
        std_close_20 = close.rolling(window).std()
        zscore_close = (close - mean_close_20) / (std_close_20 + 1e-8)
        # Compute TS_CORR between standardized close and volume (window 5)
        mean_close_5 = close.rolling(5).mean()
        std_close_5 = close.rolling(5).std()
        std_close_5 = std_close_5.replace(0, np.nan)
        std_close_5 = std_close_5.ffill().fillna(1e-8)
        std_close_5 = std_close_5.replace(0, 1e-8)
        std_close_5 = std_close_5 + 1e-8
        zclose_5 = (close - mean_close_5) / std_close_5
        mean_vol_5 = volume.rolling(5).mean()
        std_vol_5 = volume.rolling(5).std()
        std_vol_5 = std_vol_5.replace(0, np.nan)
        std_vol_5 = std_vol_5.ffill().fillna(1e-8)
        std_vol_5 = std_vol_5.replace(0, 1e-8)
        std_vol_5 = std_vol_5 + 1e-8
        zvol_5 = (volume - mean_vol_5) / std_vol_5
        corr_close_vol_5 = zclose_5.rolling(5).corr(zvol_5)
        rank_corr = corr_close_vol_5.rank(pct=True) * 2 - 1
        # Compute TS_CORR between range and price change (window 5)
        range_ratio = (high - low) / (open_ + close + 1e-8)
        price_change_ratio = (close - open_) / (open_ + 1e-8)
        range_mean_5 = range_ratio.rolling(5).mean()
        range_std_5 = range_ratio.rolling(5).std()
        range_std_5 = range_std_5.replace(0, np.nan)
        range_std_5 = range_std_5.ffill().fillna(1e-8)
        range_std_5 = range_std_5.replace(0, 1e-8)
        range_std_5 = range_std_5 + 1e-8
        zrange_5 = (range_ratio - range_mean_5) / range_std_5
        price_change_mean_5 = price_change_ratio.rolling(5).mean()
        price_change_std_5 = price_change_ratio.rolling(5).std()
        price_change_std_5 = price_change_std_5.replace(0, np.nan)
        price_change_std_5 = price_change_std_5.ffill().fillna(1e-8)
        price_change_std_5 = price_change_std_5.replace(0, 1e-8)
        price_change_std_5 = price_change_std_5 + 1e-8
        zprice_change_5 = (price_change_ratio - price_change_mean_5) / price_change_std_5
        corr_range_price_5 = zrange_5.rolling(5).corr(zprice_change_5)
        # Multiply all components
        raw = rank_corr * zscore_close * corr_range_price_5
        # Normalize using Winsorized Fisher (Method E)
        low_quant = raw.rolling(window * 5).quantile(quantile)
        high_quant = raw.rolling(window * 5).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_quant, upper=high_quant, axis=0)
        normalized = np.arctanh(((winsorized - low_quant) / (high_quant - low_quant + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_161_rank(df, window=35):
        close = df['close']
        open_ = df['open']
        return_ = close.pct_change()
        raw = (open_ - close.shift(1)) / close.shift(1)
        sign = np.sign(raw)
        std_ret = return_.rolling(window).std()
        exp_factor = np.exp(-np.abs(raw) / (std_ret + 1e-8))
        signal = sign * exp_factor
        normalized = (signal.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_161_tanh(df, window=60):
        close = df['close']
        open_ = df['open']
        raw = (open_ - close.shift(1)) / close.shift(1)
        sign = np.sign(raw)
        std_ret = close.pct_change().rolling(window).std()
        exp_factor = np.exp(-np.abs(raw) / (std_ret + 1e-8))
        signal = sign * exp_factor
        normalized = np.tanh(signal / signal.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_161_zscore(df, window=5):
        close = df['close']
        open_ = df['open']
        raw = (open_ - close.shift(1)) / close.shift(1)
        sign = np.sign(raw)
        std_ret = close.pct_change().rolling(window).std()
        exp_factor = np.exp(-np.abs(raw) / (std_ret + 1e-8))
        signal = sign * exp_factor
        rolling_mean = signal.rolling(window).mean()
        rolling_std = signal.rolling(window).std().replace(0, np.nan)
        normalized = ((signal - rolling_mean) / rolling_std).clip(-1, 1)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_161_sign(df, window=80):
        close = df['close']
        open_ = df['open']
        raw = (open_ - close.shift(1)) / close.shift(1)
        sign = np.sign(raw)
        std_ret = close.pct_change().rolling(window).std()
        exp_factor = np.exp(-np.abs(raw) / (std_ret + 1e-8))
        signal = sign * exp_factor
        normalized = np.sign(signal).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_161_wf(df, p1=0.1, p2=50):
        close = df['close']
        open_ = df['open']
        raw = (open_ - close.shift(1)) / close.shift(1)
        sign = np.sign(raw)
        std_ret = close.pct_change().rolling(p2).std()
        exp_factor = np.exp(-np.abs(raw) / (std_ret + 1e-8))
        signal = sign * exp_factor
        low = signal.rolling(p2).quantile(p1)
        high = signal.rolling(p2).quantile(1 - p1)
        winsorized = signal.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_162_rank(df, window=85):
        high_low = df['high'] - df['low']
        log_hl = np.log1p(high_low.clip(lower=0))
        volume = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(volume)
        corr = log_hl.rolling(window).corr(log_vol)
        raw = corr.fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal

    @staticmethod
    def alpha_quanta_162_tanh(df, window=5):
        high_low = df['high'] - df['low']
        log_hl = np.log1p(high_low.clip(lower=0))
        volume = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(volume)
        corr = log_hl.rolling(window).corr(log_vol)
        raw = corr.fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_162_zscore(df, window=35):
        high_low = df['high'] - df['low']
        log_hl = np.log1p(high_low.clip(lower=0))
        volume = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(volume)
        corr = log_hl.rolling(window).corr(log_vol)
        raw = corr.fillna(0)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_162_sign(df, window=10):
        high_low = df['high'] - df['low']
        log_hl = np.log1p(high_low.clip(lower=0))
        volume = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(volume)
        corr = log_hl.rolling(window).corr(log_vol)
        raw = corr.fillna(0)
        offset = 1e-9
        signal = np.where(raw > 0, 1, np.where(raw < 0, -1, 0))
        signal = pd.Series(signal, index=df.index)
        return -signal

    @staticmethod
    def alpha_quanta_162_wf(df, window=100, p1=0.9):
        high_low = df['high'] - df['low']
        log_hl = np.log1p(high_low.clip(lower=0))
        volume = df.get('matchingVolume', df.get('volume', 1))
        log_vol = np.log1p(volume)
        corr = log_hl.rolling(window).corr(log_vol)
        raw = corr.fillna(0)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_163_rank(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        close = df['close']
        # Tỷ lệ volume / range
        raw = volume / (high - low + 1e-8)
        # Rolling median
        rolling_median = raw.rolling(window).median()
        # Z-score của raw so với median (dùng median làm trung tâm thay vì mean)
        rolling_std = raw.rolling(window).std()
        z = (raw - rolling_median) / rolling_std.replace(0, np.nan)
        # Tính return và standard deviation của return
        ret = close.pct_change()
        ret_std = ret.rolling(window).std()
        # Sign của |return| - ret_std
        sign_factor = np.sign(np.abs(ret) - ret_std).fillna(0)
        # Tín hiệu raw
        signal_raw = z * sign_factor
        # Chuẩn hóa Rolling Rank (phương pháp A)
        signal = (signal_raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_163_tanh(df, window=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        close = df['close']
        raw = volume / (high - low + 1e-8)
        rolling_median = raw.rolling(window).median()
        rolling_std = raw.rolling(window).std()
        z = (raw - rolling_median) / rolling_std.replace(0, np.nan)
        ret = close.pct_change()
        ret_std = ret.rolling(window).std()
        sign_factor = np.sign(np.abs(ret) - ret_std).fillna(0)
        signal_raw = z * sign_factor
        # Chuẩn hóa Dynamic Tanh (phương pháp B)
        signal = np.tanh(signal_raw / signal_raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_163_zscore(df, window=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        close = df['close']
        raw = volume / (high - low + 1e-8)
        rolling_median = raw.rolling(window).median()
        rolling_std = raw.rolling(window).std()
        z = (raw - rolling_median) / rolling_std.replace(0, np.nan)
        ret = close.pct_change()
        ret_std = ret.rolling(window).std()
        sign_factor = np.sign(np.abs(ret) - ret_std).fillna(0)
        signal_raw = z * sign_factor
        # Chuẩn hóa Rolling Z-score Clip (phương pháp C)
        signal = ((signal_raw - signal_raw.rolling(window).mean()) / signal_raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_163_sign(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        close = df['close']
        raw = volume / (high - low + 1e-8)
        rolling_median = raw.rolling(window).median()
        rolling_std = raw.rolling(window).std()
        z = (raw - rolling_median) / rolling_std.replace(0, np.nan)
        ret = close.pct_change()
        ret_std = ret.rolling(window).std()
        sign_factor = np.sign(np.abs(ret) - ret_std).fillna(0)
        signal_raw = z * sign_factor
        # Chuẩn hóa Sign/Binary Soft (phương pháp D)
        signal = np.sign(signal_raw)
        return pd.Series(signal, index=df.index).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_163_wf(df, window=10, factor=0.7):
        volume = df.get('matchingVolume', df.get('volume', 1))
        high = df['high']
        low = df['low']
        close = df['close']
        raw = volume / (high - low + 1e-8)
        rolling_median = raw.rolling(window).median()
        rolling_std = raw.rolling(window).std()
        z = (raw - rolling_median) / rolling_std.replace(0, np.nan)
        ret = close.pct_change()
        ret_std = ret.rolling(window).std()
        sign_factor = np.sign(np.abs(ret) - ret_std).fillna(0)
        signal_raw = z * sign_factor
        # Chuẩn hóa Winsorized Fisher (phương pháp E)
        p1 = factor
        p2 = window
        low_quantile = signal_raw.rolling(p2).quantile(p1)
        high_quantile = signal_raw.rolling(p2).quantile(1 - p1)
        winsorized = signal_raw.clip(lower=low_quantile, upper=high_quantile, axis=0)
        normalized = np.arctanh(((winsorized - low_quantile) / (high_quantile - low_quantile + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_164_rank(df, window=25):
        ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        raw = np.sign(ret) * np.exp(-np.abs(ret) / (ret.rolling(window).std().replace(0, np.nan) + 1e-8))
        hl_vol_corr = np.log1p(df['high'] - df['low']).rolling(window).corr(np.log1p(df['matchingVolume'])) * raw
        han = hl_vol_corr.rolling(window*2).rank(pct=True)
        result = (han * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_164_tanh(df, window=30):
        ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        raw = np.sign(ret) * np.exp(-np.abs(ret) / (ret.rolling(window).std().replace(0, np.nan) + 1e-8))
        hl_vol_corr = np.log1p(df['high'] - df['low']).rolling(window).corr(np.log1p(df['matchingVolume'])) * raw
        result = np.tanh(hl_vol_corr / (hl_vol_corr.abs().rolling(window*2).mean().replace(0, np.nan) + 1e-8))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_164_zscore(df, window=80):
        ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        raw = np.sign(ret) * np.exp(-np.abs(ret) / (ret.rolling(window).std().replace(0, np.nan) + 1e-8))
        hl_vol_corr = np.log1p(df['high'] - df['low']).rolling(window).corr(np.log1p(df['matchingVolume'])) * raw
        roll_mean = hl_vol_corr.rolling(window*2).mean()
        roll_std = hl_vol_corr.rolling(window*2).std().replace(0, np.nan)
        result = ((hl_vol_corr - roll_mean) / (roll_std + 1e-8)).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_164_sign(df, window=75):
        ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        raw = np.sign(ret) * np.exp(-np.abs(ret) / (ret.rolling(window).std().replace(0, np.nan) + 1e-8))
        hl_vol_corr = np.log1p(df['high'] - df['low']).rolling(window).corr(np.log1p(df['matchingVolume'])) * raw
        result = pd.Series(np.sign(hl_vol_corr), index=df.index).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_164_wf(df, window=40, sub_window=40):
        ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).replace(0, np.nan)
        raw = np.sign(ret) * np.exp(-np.abs(ret) / (ret.rolling(window).std().replace(0, np.nan) + 1e-8))
        hl_vol_corr = np.log1p(df['high'] - df['low']).rolling(window).corr(np.log1p(df['matchingVolume'])) * raw
        low_quant = hl_vol_corr.rolling(sub_window).quantile(0.1)
        high_quant = hl_vol_corr.rolling(sub_window).quantile(0.9)
        winsorized = hl_vol_corr.clip(lower=low_quant, upper=high_quant, axis=0)
        numerator = (winsorized - low_quant) * 1.98
        denominator = (high_quant - low_quant).replace(0, np.nan) + 1e-9
        result = pd.Series(np.arctanh((numerator / denominator) - 0.99), index=df.index).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_165_rank(df, window=40):
        numerator = df['high'] - df['low']
        denominator = (df['close'] - df['open']).abs() + 1
        raw = numerator / denominator
        ts_mean = raw.rolling(window, min_periods=1).mean()
        rank_series = ts_mean.rolling(window, min_periods=1).rank(pct=True) * 2 - 1
        return -rank_series.fillna(0)

    @staticmethod
    def alpha_quanta_165_tanh(df, window=10):
        numerator = df['high'] - df['low']
        denominator = (df['close'] - df['open']).abs() + 1
        raw = numerator / denominator
        ts_mean = raw.rolling(window, min_periods=1).mean()
        std_val = ts_mean.rolling(window, min_periods=1).std()
        normalized = np.tanh(ts_mean / std_val.replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_165_zscore(df, window=40):
        numerator = df['high'] - df['low']
        denominator = (df['close'] - df['open']).abs() + 1
        raw = numerator / denominator
        ts_mean = raw.rolling(window, min_periods=1).mean()
        rolling_mean = ts_mean.rolling(window, min_periods=1).mean()
        rolling_std = ts_mean.rolling(window, min_periods=1).std()
        zscore = (ts_mean - rolling_mean) / rolling_std.replace(0, np.nan)
        return -zscore.clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_165_sign(df, window=50):
        numerator = df['high'] - df['low']
        denominator = (df['close'] - df['open']).abs() + 1
        raw = numerator / denominator
        ts_mean = raw.rolling(window, min_periods=1).mean()
        return np.sign(ts_mean).fillna(0)

    @staticmethod
    def alpha_quanta_165_wf(df, window=40, p1=0.7):
        p2 = window
        numerator = df['high'] - df['low']
        denominator = (df['close'] - df['open']).abs() + 1
        raw = numerator / denominator
        ts_mean = raw.rolling(window, min_periods=1).mean()
        low = ts_mean.rolling(p2, min_periods=1).quantile(p1)
        high = ts_mean.rolling(p2, min_periods=1).quantile(1 - p1)
        winsorized = ts_mean.clip(lower=low, upper=high, axis=0)
        range_val = high - low
        normalized = np.arctanh(((winsorized - low) / (range_val + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_166_rank(df, window=65):
        hilo = df['high'] - df['low']
        vol = df['matchingVolume']
        corr = hilo.rolling(window).corr(vol).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        raw = np.sign(corr) * hilo.rolling(window).mean().fillna(method='ffill').fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_166_tanh(df, window=50):
        hilo = df['high'] - df['low']
        vol = df['matchingVolume']
        corr = hilo.rolling(window).corr(vol).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        raw = np.sign(corr) * hilo.rolling(window).mean().fillna(method='ffill').fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).fillna(method='ffill').fillna(1))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_166_zscore(df, window=50):
        hilo = df['high'] - df['low']
        vol = df['matchingVolume']
        corr = hilo.rolling(window).corr(vol).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        raw = np.sign(corr) * hilo.rolling(window).mean().fillna(method='ffill').fillna(0)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_166_sign(df, window=5):
        hilo = df['high'] - df['low']
        vol = df['matchingVolume']
        corr = hilo.rolling(window).corr(vol).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        raw = np.sign(corr) * hilo.rolling(window).mean().fillna(method='ffill').fillna(0)
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_166_wf(df, window=50):
        hilo = df['high'] - df['low']
        vol = df['matchingVolume']
        corr = hilo.rolling(window).corr(vol).replace({np.nan: 0, np.inf: 0, -np.inf: 0})
        raw = np.sign(corr) * hilo.rolling(window).mean().fillna(method='ffill').fillna(0)
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        denom = (high - low + 1e-9)
        normalized = np.arctanh(((winsorized - low) / denom) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_167_rank(df, window=65):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(window).std() * (df['high'] - df['low']).rolling(window).mean()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_167_tanh(df, window=55):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(window).std() * (df['high'] - df['low']).rolling(window).mean()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_167_zscore(df, window=55):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(window).std() * (df['high'] - df['low']).rolling(window).mean()
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_167_sign(df, window=25):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(window).std() * (df['high'] - df['low']).rolling(window).mean()
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_167_wf(df, window=60, p1=0.7):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-9)).rolling(window).std() * (df['high'] - df['low']).rolling(window).mean()
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_168_k(df, window=100, sub_window=5):
        raw = (df['high'] - df['low']).rolling(window).mean()
        delta = raw.diff(sub_window)
        signal = (delta.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_168_h(df, window=60, sub_window=7):
        raw = (df['high'] - df['low']).rolling(window).mean()
        delta = raw.diff(sub_window)
        std = delta.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(delta / std)
        signal = signal.ffill().fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_168_p(df, window=100, sub_window=5):
        raw = (df['high'] - df['low']).rolling(window).mean()
        delta = raw.diff(sub_window)
        mean = delta.rolling(window).mean()
        std = delta.rolling(window).std().replace(0, np.nan)
        signal = ((delta - mean) / std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_168_y(df, window=100, sub_window=5):
        raw = (df['high'] - df['low']).rolling(window).mean()
        delta = raw.diff(sub_window)
        signal = np.sign(delta)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_168_r(df, window=60, sub_window=5, p1=0.05, p2=20):
        raw = (df['high'] - df['low']).rolling(window).mean()
        delta = raw.diff(sub_window)
        low = delta.rolling(p2).quantile(p1)
        high = delta.rolling(p2).quantile(1 - p1)
        winsorized = delta.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -pd.Series(normalized, index=df.index)

    @staticmethod
    def alpha_quanta_169_k(df, window=45):
        raw = df['close'] / (df['high'] - df['low']).rolling(window).mean() + 1
        std_close_ratio = raw.rolling(window).std()
        sign = pd.Series(np.sign(std_close_ratio), index=df.index)
        result = sign * (df['high'] - df['low']).rolling(window).mean()
        normalized = (result.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_169_h(df, window=50):
        raw = df['close'] / (df['high'] - df['low']).rolling(window).mean() + 1
        std_close_ratio = raw.rolling(window).std()
        sign = pd.Series(np.sign(std_close_ratio), index=df.index)
        result = sign * (df['high'] - df['low']).rolling(window).mean()
        normalized = np.tanh(result / result.rolling(window).std().replace(0, np.nan))
        normalized = normalized.ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_169_e(df, window=45):
        raw = df['close'] / (df['high'] - df['low']).rolling(window).mean() + 1
        std_close_ratio = raw.rolling(window).std()
        sign = pd.Series(np.sign(std_close_ratio), index=df.index)
        result = sign * (df['high'] - df['low']).rolling(window).mean()
        normalized = ((result - result.rolling(window).mean()) / result.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_169_y(df, window=90):
        raw = df['close'] / (df['high'] - df['low']).rolling(window).mean() + 1
        std_close_ratio = raw.rolling(window).std()
        sign = pd.Series(np.sign(std_close_ratio), index=df.index)
        result = sign * (df['high'] - df['low']).rolling(window).mean()
        normalized = np.sign(result)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_169_r(df, window=50, winsor_quantile=0.3):
        raw = df['close'] / (df['high'] - df['low']).rolling(window).mean() + 1
        std_close_ratio = raw.rolling(window).std()
        sign = pd.Series(np.sign(std_close_ratio), index=df.index)
        result = sign * (df['high'] - df['low']).rolling(window).mean()
        low = result.rolling(window).quantile(winsor_quantile)
        high = result.rolling(window).quantile(1 - winsor_quantile)
        winsorized = result.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_170_k(df, window=15):
        vol_5 = df['close'].rolling(5).mean()
        vol_10 = df['close'].rolling(10).mean()
        delta = vol_5 - vol_5.shift(1)
        raw = delta / (vol_10 + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal

    @staticmethod
    def alpha_quanta_170_h(df, window=100):
        vol_5 = df['close'].rolling(5).mean()
        vol_10 = df['close'].rolling(10).mean()
        delta = vol_5 - vol_5.shift(1)
        raw = delta / (vol_10 + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).fillna(1))
        return signal

    @staticmethod
    def alpha_quanta_170_e(df, window=5):
        vol_5 = df['close'].rolling(5).mean()
        vol_10 = df['close'].rolling(10).mean()
        delta = vol_5 - vol_5.shift(1)
        raw = delta / (vol_10 + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).fillna(1)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_170_y(df):
        vol_5 = df['close'].rolling(5).mean()
        vol_10 = df['close'].rolling(10).mean()
        delta = vol_5 - vol_5.shift(1)
        raw = delta / (vol_10 + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        signal = np.sign(raw)
        return signal

    @staticmethod
    def alpha_quanta_170_r(df, window=40, p1=0.1):
        vol_5 = df['close'].rolling(5).mean()
        vol_10 = df['close'].rolling(10).mean()
        delta = vol_5 - vol_5.shift(1)
        raw = delta / (vol_10 + 1e-8)
        raw = raw.replace([np.inf, -np.inf], 0).fillna(0)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], 0).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_171_rank(df, window=75):
        raw = (df['high'] - df['low']).rolling(window).corr((df['high'] - df['low']).shift(1))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_171_tanh(df, window=50):
        raw = (df['high'] - df['low']).rolling(window).corr((df['high'] - df['low']).shift(1))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_171_zscore(df, window=25):
        raw = (df['high'] - df['low']).rolling(window).corr((df['high'] - df['low']).shift(1))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_171_sign(df, window=60):
        raw = (df['high'] - df['low']).rolling(window).corr((df['high'] - df['low']).shift(1))
        signal = np.sign(raw)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_171_wf(df, window=30, quantile_low=0.3):
        raw = (df['high'] - df['low']).rolling(window).corr((df['high'] - df['low']).shift(1))
        low = raw.rolling(window).quantile(quantile_low)
        high = raw.rolling(window).quantile(1 - quantile_low)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_172_rank(df, window=35):
        delta_close = df['close'].diff()
        raw = (delta_close - delta_close.rolling(window).mean()) / (delta_close.rolling(window).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized

    @staticmethod
    def alpha_quanta_172_tanh(df, window=25):
        delta_close = df['close'].diff()
        raw = (delta_close - delta_close.rolling(window).mean()) / (delta_close.rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.abs().rolling(window).mean())
        return normalized

    @staticmethod
    def alpha_quanta_172_zscore(df, window=40, sub_window=20):
        delta_close = df['close'].diff()
        raw = (delta_close - delta_close.rolling(window).mean()) / (delta_close.rolling(window).std() + 1e-8)
        normalized = ((raw - raw.rolling(sub_window).mean()) / raw.rolling(sub_window).std()).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_172_sign(df, window=95):
        delta_close = df['close'].diff()
        raw = (delta_close - delta_close.rolling(window).mean()) / (delta_close.rolling(window).std() + 1e-8)
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_172_wf(df, window=60, p=0.1):
        delta_close = df['close'].diff()
        mid = (delta_close - delta_close.rolling(window).mean()) / (delta_close.rolling(window).std() + 1e-8)
        low = mid.rolling(window).quantile(p)
        high = mid.rolling(window).quantile(1 - p)
        winsorized = mid.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized

    @staticmethod
    def alpha_quanta_173_rank(df, window=20):
        pct_volume = df['matchingVolume'].pct_change()
        pct_close = df['close'].pct_change()
        rolling_corr = pct_volume.rolling(window=window).corr(pct_close)
        mean_corr = rolling_corr.rolling(window=window).mean()
        std_corr = rolling_corr.rolling(window=window).std()
        zscore = (rolling_corr - mean_corr) / std_corr.replace(0, np.nan)
        zscore = zscore.ffill().fillna(0)
        result = (zscore.rolling(window=window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_173_tanh(df, window=10):
        pct_volume = df['matchingVolume'].pct_change()
        pct_close = df['close'].pct_change()
        rolling_corr = pct_volume.rolling(window=window).corr(pct_close)
        mean_corr = rolling_corr.rolling(window=window).mean()
        std_corr = rolling_corr.rolling(window=window).std()
        zscore = (rolling_corr - mean_corr) / std_corr.replace(0, np.nan)
        zscore = zscore.ffill().fillna(0)
        result = np.tanh(zscore / zscore.rolling(window=window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_173_zscore(df, window=30):
        pct_volume = df['matchingVolume'].pct_change()
        pct_close = df['close'].pct_change()
        rolling_corr = pct_volume.rolling(window=window).corr(pct_close)
        mean_corr = rolling_corr.rolling(window=window).mean()
        std_corr = rolling_corr.rolling(window=window).std()
        zscore = (rolling_corr - mean_corr) / std_corr.replace(0, np.nan)
        zscore = zscore.ffill().fillna(0)
        result = ((zscore - zscore.rolling(window=window).mean()) / zscore.rolling(window=window).std().replace(0, np.nan)).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_173_sign(df, window=5):
        pct_volume = df['matchingVolume'].pct_change()
        pct_close = df['close'].pct_change()
        rolling_corr = pct_volume.rolling(window=window).corr(pct_close)
        mean_corr = rolling_corr.rolling(window=window).mean()
        std_corr = rolling_corr.rolling(window=window).std()
        zscore = (rolling_corr - mean_corr) / std_corr.replace(0, np.nan)
        zscore = zscore.ffill().fillna(0)
        result = np.sign(zscore)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_173_wf(df, window=10, sub_window=20):
        pct_volume = df['matchingVolume'].pct_change()
        pct_close = df['close'].pct_change()
        rolling_corr = pct_volume.rolling(window=window).corr(pct_close)
        mean_corr = rolling_corr.rolling(window=window).mean()
        std_corr = rolling_corr.rolling(window=window).std()
        zscore = (rolling_corr - mean_corr) / std_corr.replace(0, np.nan)
        zscore = zscore.ffill().fillna(0)
        p1 = 0.05
        low = zscore.rolling(window=sub_window).quantile(p1)
        high = zscore.rolling(window=sub_window).quantile(1 - p1)
        winsorized = zscore.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_174_rank(df, window=50):
        # Tính lợi nhuận mở cửa (Open return)
        open_return = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Lấy dấu của open_return
        sign_open = np.sign(open_return)
        # Tính trung bình động 8 kỳ của open_return
        ma_open = open_return.rolling(window=window).mean()
        # Nhân dấu với trung bình động
        raw = -sign_open * ma_open
        # Chuẩn hóa Rolling Rank
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        # Xử lý NaN
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_174_tanh(df, window=5):
        # Tính lợi nhuận mở cửa (Open return)
        open_return = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Lấy dấu của open_return
        sign_open = np.sign(open_return)
        # Tính trung bình động 8 kỳ của open_return
        ma_open = open_return.rolling(window=window).mean()
        # Nhân dấu với trung bình động
        raw = -sign_open * ma_open
        # Chuẩn hóa Dynamic Tanh
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        # Xử lý NaN
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_174_zscore(df, window=55):
        # Tính lợi nhuận mở cửa (Open return)
        open_return -= (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Lấy dấu của open_return
        sign_open = np.sign(open_return)
        # Tính trung bình động 8 kỳ của open_return
        ma_open = open_return.rolling(window=window).mean()
        # Nhân dấu với trung bình động
        raw = -sign_open * ma_open
        # Chuẩn hóa Rolling Z-Score/Clip
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        # Xử lý NaN
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_174_sign(df, window=5):
        # Tính lợi nhuận mở cửa (Open return)
        open_return = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Lấy dấu của open_return
        sign_open = np.sign(open_return)
        # Tính trung bình động 8 kỳ của open_return
        ma_open = open_return.rolling(window=window).mean()
        # Nhân dấu với trung bình động
        raw = -sign_open * ma_open
        # Chuẩn hóa Sign/Binary Soft
        normalized = np.sign(raw)
        # Xử lý NaN
        normalized = pd.Series(normalized, index=df.index).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_174_wf(df, window=70, sub_window=40):
        # Tính lợi nhuận mở cửa (Open return)
        open_return -= (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-8)
        # Lấy dấu của open_return
        sign_open = np.sign(open_return)
        # Tính trung bình động 8 kỳ của open_return
        ma_open = open_return.rolling(window=window).mean()
        # Nhân dấu với trung bình động
        raw = -sign_open * ma_open
        # Chuẩn hóa Winsorized Fisher
        p1 = 0.05
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Fisher Transform
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        # Xử lý NaN
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_175_rank(df, window=95):
        close = df['close']
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(window).std() + 1e-8
        raw = std_5 / std_20
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_175_tanh(df, window=40):
        close = df['close']
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(window).std() + 1e-8
        raw = std_5 / std_20
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_175_zscore(df, window=90):
        close = df['close']
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(window).std() + 1e-8
        raw = std_5 / std_20
        mean_ = raw.rolling(window).mean()
        std_ = raw.rolling(window).std()
        signal = ((raw - mean_) / std_).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_175_sign(df, window=85):
        close = df['close']
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(window).std() + 1e-8
        raw = std_5 / std_20
        signal = np.sign(raw)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_175_wf(df, window=100, p1=0.1):
        close = df['close']
        std_5 = close.rolling(5).std()
        std_20 = close.rolling(window).std() + 1e-8
        raw = std_5 / std_20
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_176_k(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        corr = close.rolling(window).corr(volume)
        sign_mean = np.sign(close.rolling(5).mean())
        raw = corr * sign_mean
        normalized = (raw.rolling(3).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_176_h(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        corr = close.rolling(window).corr(volume)
        sign_mean = np.sign(close.rolling(5).mean())
        raw = corr * sign_mean
        normalized = np.tanh(raw / raw.rolling(20).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_176_e(df, window=85):
        close = df['close']
        volume = df['matchingVolume']
        corr = close.rolling(window).corr(volume)
        sign_mean = np.sign(close.rolling(5).mean())
        raw = corr * sign_mean
        mean = raw.rolling(20).mean()
        std = raw.rolling(20).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_176_y(df, window=5):
        close = df['close']
        volume = df['matchingVolume']
        corr = close.rolling(window).corr(volume)
        sign_mean = np.sign(close.rolling(5).mean())
        raw = corr * sign_mean
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_176_r(df, window=80, p1=0.3, p2=30):
        close = df['close']
        volume = df['matchingVolume']
        corr = close.rolling(window).corr(volume)
        sign_mean = np.sign(close.rolling(5).mean())
        raw = corr * sign_mean
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_177_7(df, window=65):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_log = np.log1p(volume)
        mean_vol = vol_log.rolling(window).mean()
        std_vol = vol_log.rolling(window).std().replace(0, np.nan)
        raw = (vol_log - mean_vol) / (std_vol + 1e-8)
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -norm.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_177_rank(df, window=30):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_177_tanh(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_177_zscore(df, window=65):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_177_sign(df, window=85):
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
        signal = pd.Series(np.sign(raw), index=df.index)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_177_wf(df, window=40, p1=0.3):
        p2 = window
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = (volume - volume.rolling(window).mean()) / (volume.rolling(window).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_178_k(df, window=10):
        close = df['close']
        low = df['low']
        ts_min_low = low.rolling(window=window).min()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_min_low) / (ts_max_close - ts_min_low + 1e-8)
        signal = raw.rolling(window=window).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_178_h(df, window=5):
        close = df['close']
        low = df['low']
        ts_min_low = low.rolling(window=window).min()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_min_low) / (ts_max_close - ts_min_low + 1e-8)
        signal = np.tanh(raw / raw.rolling(window=window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_178_e(df, window=10):
        close = df['close']
        low = df['low']
        ts_min_low = low.rolling(window=window).min()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_min_low) / (ts_max_close - ts_min_low + 1e-8)
        mean = raw.rolling(window=window).mean()
        std = raw.rolling(window=window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_178_y(df, window=15):
        close = df['close']
        low = df['low']
        ts_min_low = low.rolling(window=window).min()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_min_low) / (ts_max_close - ts_min_low + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_178_r(df, window=10, winsor_quantile=0.1):
        close = df['close']
        low = df['low']
        ts_min_low = low.rolling(window=window).min()
        ts_max_close = close.rolling(window=window).max()
        raw = (close - ts_min_low) / (ts_max_close - ts_min_low + 1e-8)
        low_bound = raw.rolling(window=window).quantile(winsor_quantile)
        high_bound = raw.rolling(window=window).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low_bound, upper=high_bound, axis=0)
        normalized = np.arctanh(((winsorized - low_bound) / (high_bound - low_bound + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_179_k(df, window=10, sub_window=3):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        sum_ret = ret.rolling(window).sum()
        std_vol = volume.rolling(sub_window).std(ddof=0).replace(0, 1e-8)
        mean_vol = volume.rolling(sub_window).mean()
        lag_vol = volume.shift(1)
        vol_z = (lag_vol - mean_vol) / std_vol
        signal = sum_ret * np.sign(vol_z)
        raw = signal
        param = window
        normalized = (raw.rolling(param).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_179_h(df, window=10, sub_window=7):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        sum_ret = ret.rolling(window).sum()
        std_vol = volume.rolling(sub_window).std(ddof=0).replace(0, 1e-8)
        mean_vol = volume.rolling(sub_window).mean()
        lag_vol = volume.shift(1)
        vol_z = (lag_vol - mean_vol) / std_vol
        signal = sum_ret * np.sign(vol_z)
        raw = signal
        param = window
        normalized = np.tanh(raw / raw.rolling(param).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_179_e(df, window=10, sub_window=1):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        sum_ret = ret.rolling(window).sum()
        std_vol = volume.rolling(sub_window).std(ddof=0).replace(0, 1e-8)
        mean_vol = volume.rolling(sub_window).mean()
        lag_vol = volume.shift(1)
        vol_z = (lag_vol - mean_vol) / std_vol
        signal = sum_ret * np.sign(vol_z)
        raw = signal
        param = window
        rolling_mean = raw.rolling(param).mean()
        rolling_std = raw.rolling(param).std().replace(0, np.nan)
        normalized = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_179_y(df, window=100, sub_window=3):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        sum_ret = ret.rolling(window).sum()
        std_vol = volume.rolling(sub_window).std(ddof=0).replace(0, 1e-8)
        mean_vol = volume.rolling(sub_window).mean()
        lag_vol = volume.shift(1)
        vol_z = (lag_vol - mean_vol) / std_vol
        signal = sum_ret * np.sign(vol_z)
        raw = signal
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_179_r(df, window=10, sub_window=30, p1=0.05, p2=30):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change().fillna(0)
        sum_ret = ret.rolling(window).sum()
        std_vol = volume.rolling(sub_window).std(ddof=0).replace(0, 1e-8)
        mean_vol = volume.rolling(sub_window).mean()
        lag_vol = volume.shift(1)
        vol_z = (lag_vol - mean_vol) / std_vol
        signal = sum_ret * np.sign(vol_z)
        raw = signal
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        epsilon = 1e-9
        ratio = (winsorized - low) / (high - low + epsilon)
        ratio_clipped = ratio.clip(0, 1)
        normalized = np.arctanh(ratio_clipped * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_180_k(df, window=3, rank_window=10):
        ts_low = df['low'].rolling(window).min()
        raw = (df['close'] - ts_low) / (ts_low + 1e-8)
        ts_mean = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - ts_mean).replace(0, np.nan).ffill()
        raw = raw * sign
        signal = (raw.rolling(rank_window).rank(pct=True) * 2) - 1
        return signal

    @staticmethod
    def alpha_quanta_180_h(df, window=20, std_scale=20):
        ts_low = df['low'].rolling(window).min()
        raw = (df['close'] - ts_low) / (ts_low + 1e-8)
        ts_mean = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - ts_mean).replace(0, np.nan).ffill()
        raw = raw * sign
        signal = np.tanh(raw / raw.rolling(std_scale).std().replace(0, np.nan))
        return signal.filna(0)

    @staticmethod
    def alpha_quanta_180_e(df, window=20, z_window=40):
        ts_low = df['low'].rolling(window).min()
        raw = (df['close'] - ts_low) / (ts_low + 1e-8)
        ts_mean = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - ts_mean).replace(0, np.nan).ffill()
        raw = raw * sign
        signal = ((raw - raw.rolling(z_window).mean()) / raw.rolling(z_window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.filna(0)

    @staticmethod
    def alpha_quanta_180_y(df, window=20):
        ts_low = df['low'].rolling(window).min()
        raw = (df['close'] - ts_low) / (ts_low + 1e-8)
        ts_mean = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - ts_mean).replace(0, np.nan).ffill()
        raw = raw * sign
        signal = np.sign(raw)
        return signal.filna(0)

    @staticmethod
    def alpha_quanta_180_r(df, window=20, p1=0.05, p2=40):
        ts_low = df['low'].rolling(window).min()
        raw = (df['close'] - ts_low) / (ts_low + 1e-8)
        ts_mean = df['close'].rolling(window).mean()
        sign = np.sign(df['close'] - ts_mean).replace(0, np.nan).ffill()
        raw = raw * sign
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(low_q, high_q)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.filna(0)

    @staticmethod
    def alpha_quanta_181_k(df, window=5):
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        ts_max_high = high.rolling(window).max()
        sign_val = np.sign(close - ts_max_high)
        raw_alpha = sign_val * volume / (volume.rolling(window).mean() + 1e-8)
        ranked = raw_alpha.rolling(window).rank(pct=True)
        signal = (ranked * 2) - 1
        signal = signal.ffill().fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_181_h(df, window=5):
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        ts_max_high = high.rolling(window).max()
        sign_val = np.sign(close - ts_max_high)
        raw_alpha = sign_val * volume / (volume.rolling(window).mean() + 1e-8)
        std = raw_alpha.rolling(window).std()
        signal = np.tanh(raw_alpha / (std + 1e-9))
        signal = signal.ffill().fillna(0)
        return -pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_181_e(df, window=90):
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        ts_max_high = high.rolling(window).max()
        sign_val = np.sign(close - ts_max_high)
        raw_alpha = sign_val * volume / (volume.rolling(window).mean() + 1e-8)
        mean = raw_alpha.rolling(window).mean()
        std = raw_alpha.rolling(window).std()
        signal = ((raw_alpha - mean) / (std + 1e-9)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_181_y(df, window=25):
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        ts_max_high = high.rolling(window).max()
        sign_val = np.sign(close - ts_max_high)
        raw_alpha = sign_val * volume / (volume.rolling(window).mean() + 1e-8)
        signal = np.sign(raw_alpha)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_181_r(df, window=30, sub_window=30):
        close = df['close']
        high = df['high']
        volume = df['matchingVolume']
        ts_max_high = high.rolling(window).max()
        sign_val = np.sign(close - ts_max_high)
        raw_alpha = sign_val * volume / (volume.rolling(window).mean() + 1e-8)
        low = raw_alpha.rolling(sub_window).quantile(0.25)
        high_q = raw_alpha.rolling(sub_window).quantile(0.75)
        winsorized = raw_alpha.clip(lower=low, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high_q - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        signal = signal.clip(-1, 1)
        return pd.Series(signal, index=df.index)

    @staticmethod
    def alpha_quanta_182_rank(df, window=10):
        returns = df['close'].pct_change()
        ts_mean = returns.rolling(window).mean()
        delta_std = returns.rolling(window).std().diff(1)
        sign_delta = np.sign(-delta_std)
        raw = ts_mean * sign_delta
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_182_tanh(df, window=35):
        returns = df['close'].pct_change()
        ts_mean = returns.rolling(window).mean()
        delta_std = returns.rolling(window).std().diff(1)
        sign_delta = np.sign(-delta_std)
        raw = ts_mean * sign_delta
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_182_zscore(df, window=30):
        returns = df['close'].pct_change()
        ts_mean = returns.rolling(window).mean()
        delta_std = returns.rolling(window).std().diff(1)
        sign_delta = np.sign(-delta_std)
        raw = ts_mean * sign_delta
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_182_sign(df, window=30):
        returns = df['close'].pct_change()
        ts_mean = returns.rolling(window).mean()
        delta_std = returns.rolling(window).std().diff(1)
        sign_delta = np.sign(-delta_std)
        raw = ts_mean * sign_delta
        normalized = np.sign(raw)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_182_wf(df, window=60, winsor_percent=0.7):
        returns = df['close'].pct_change()
        ts_mean = returns.rolling(window).mean()
        delta_std = returns.rolling(window).std().diff(1)
        sign_delta = np.sign(-delta_std)
        raw = ts_mean * sign_delta
        low = raw.rolling(window).quantile(winsor_percent)
        high = raw.rolling(window).quantile(1 - winsor_percent)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_183_rank(df, window=5):
        high_low = df['high'] - df['low']
        close_low = df['close'] - df['low']
        corr = high_low.rolling(window).corr(close_low)
        if corr.isna().all():
            return pd.Series(0.0, index=df.index)
        mean_close_low = close_low.rolling(10).mean()
        raw = corr * mean_close_low
        # Rolling rank normalization
        rank_raw = raw.rolling(window).rank(pct=True) * 2 - 1
        return rank_raw.fillna(0)

    @staticmethod
    def alpha_quanta_183_tanh(df, window=5):
        high_low = df['high'] - df['low']
        close_low = df['close'] - df['low']
        corr = high_low.rolling(window).corr(close_low)
        if corr.isna().all():
            return -pd.Series(0.0, index=df.index)
        mean_close_low = close_low.rolling(10).mean()
        raw = corr * mean_close_low
        # Dynamic tanh normalization
        std_dev = raw.rolling(window).std()
        tanh_norm = np.tanh(raw / std_dev.replace(0, np.nan))
        return tanh_norm.fillna(0)

    @staticmethod
    def alpha_quanta_183_zscore(df, window=5):
        high_low = df['high'] - df['low']
        close_low = df['close'] - df['low']
        corr = high_low.rolling(window).corr(close_low)
        if corr.isna().all():
            return pd.Series(0.0, index=df.index)
        mean_close_low = close_low.rolling(10).mean()
        raw = corr * mean_close_low
        # Rolling z-score normalization
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std()
        zscore = ((raw - mean_raw) / std_raw.replace(0, np.nan)).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_183_sign(df, window=5):
        high_low = df['high'] - df['low']
        close_low = df['close'] - df['low']
        corr = high_low.rolling(window).corr(close_low)
        if corr.isna().all():
            return -pd.Series(0.0, index=df.index)
        mean_close_low = close_low.rolling(10).mean()
        raw = corr * mean_close_low
        # Sign normalization
        sign_norm = pd.Series(np.sign(raw), index=df.index)
        return sign_norm.fillna(0)

    @staticmethod
    def alpha_quanta_183_wf(df, window=10, p1=0.1):
        high_low = df['high'] - df['low']
        close_low = df['close'] - df['low']
        corr = high_low.rolling(window).corr(close_low)
        if corr.isna().all():
            return pd.Series(0.0, index=df.index)
        mean_close_low = close_low.rolling(10).mean()
        raw = corr * mean_close_low
        # Winsorized Fisher normalization
        low_q = raw.rolling(window).quantile(p1)
        high_q = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_184_rank(df, window=25):
        volume_ratio = (df['matchingVolume'] - df['matchingVolume'].rolling(window).mean()) / (df['matchingVolume'].rolling(window).std() + 1e-8)
        inv_close_std = 1.0 / (df['close'].rolling(window).std() + 1e-8)
        raw = volume_ratio * inv_close_std
        ranked = raw.rolling(window).rank(pct=True) * 2 - 1
        return -ranked.fillna(0)

    @staticmethod
    def alpha_quanta_184_tanh(df, window=50):
        volume_ratio = (df['matchingVolume'] - df['matchingVolume'].rolling(window).mean()) / (df['matchingVolume'].rolling(window).std() + 1e-8)
        inv_close_std = 1.0 / (df['close'].rolling(window).std() + 1e-8)
        raw = volume_ratio * inv_close_std
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_184_zscore(df, window=25):
        volume_ratio = (df['matchingVolume'] - df['matchingVolume'].rolling(window).mean()) / (df['matchingVolume'].rolling(window).std() + 1e-8)
        inv_close_std = 1.0 / (df['close'].rolling(window).std() + 1e-8)
        raw = volume_ratio * inv_close_std
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_184_sign(df, window=85):
        volume_ratio = (df['matchingVolume'] - df['matchingVolume'].rolling(window).mean()) / (df['matchingVolume'].rolling(window).std() + 1e-8)
        inv_close_std = 1.0 / (df['close'].rolling(window).std() + 1e-8)
        raw = volume_ratio * inv_close_std
        normalized = pd.Series(np.sign(raw), index=df.index)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_184_wf(df, window=30, p1=0.1):
        volume_ratio = (df['matchingVolume'] - df['matchingVolume'].rolling(window).mean()) / (df['matchingVolume'].rolling(window).std() + 1e-8)
        inv_close_std = 1.0 / (df['close'].rolling(window).std() + 1e-8)
        raw = volume_ratio * inv_close_std
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_185_k(df, window=4, volume_window=20):
        raw = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * np.sign(df['close'].diff().fillna(0).rolling(window).sum()) * ((df['matchingVolume'] < df['matchingVolume'].rolling(volume_window).mean())).astype(int)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_185_h(df, window=3, volume_window=40):
        raw = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * np.sign(df['close'].diff().fillna(0).rolling(window).sum()) * ((df['matchingVolume'] < df['matchingVolume'].rolling(volume_window).mean())).astype(int)
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / std)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_185_p(df, window=7, volume_window=80):
        raw = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * np.sign(df['close'].diff().fillna(0).rolling(window).sum()) * ((df['matchingVolume'] < df['matchingVolume'].rolling(volume_window).mean())).astype(int)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_185_y(df, window=1, volume_window=30):
        raw = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * np.sign(df['close'].diff().fillna(0).rolling(window).sum()) * ((df['matchingVolume'] < df['matchingVolume'].rolling(volume_window).mean())).astype(int)
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_185_r(df, window_rank=6, volume_window=100, winsor_quantile=0.05):
        raw = ((df['close'] - df['open']) / (df['open'] + 1e-8)) * np.sign(df['close'].diff().fillna(0).rolling(window_rank).sum()) * ((df['matchingVolume'] < df['matchingVolume'].rolling(volume_window).mean())).astype(int)
        low = raw.rolling(window_rank).quantile(winsor_quantile)
        high = raw.rolling(window_rank).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_186_rank(df, window=15):
        # Công thức: TS_CORR((close-open)/open, return, 10) * TS_MEAN(volume, 5)
        # Bản chất: Tương quan giữa intraday return -và close-to-close return, nhân với volume trung bình
        # Chọn chuẩn hóa A (Rolling Rank) vì volume có thể có outlier, rank giúp đồng nhất dữ liệu
        # Xử lý volume: dùng log1p vì volume quy mô lớn
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-9)
        ret = df['close'].pct_change().fillna(0)
        # Tính rolling correlation giữa intraday_ret và ret
        cov = intraday_ret.rolling(window, min_periods=window).cov(ret)
        var_intra = intraday_ret.rolling(window, min_periods=window).var()
        var_ret = ret.rolling(window, min_periods=window).var()
        corr = cov / (var_intra.mul(var_ret).replace(0, np.nan).apply(np.sqrt).replace(0, np.nan))
        corr = corr.fillna(0)
        # Volume trung bình 5 kỳ, log1p
        vol_mean = np.log1p(df['matchingVolume'].rolling(5, min_periods=3).mean())
        raw = corr * vol_mean
        # Rolling Rank normalization
        normalized = (raw.rolling(2*window, min_periods=window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_186_tanh(df, window=5):
        # Công thức: TS_CORR((close-open)/open, return, 10) * TS_MEAN(volume, 5)
        # Bản chất: Tương quan giữa intraday return -và close-to-close return, nhân với volume trung bình
        # Chọn chuẩn hóa B (Dynamic Tanh) để giữ cường độ tín hiệu
        # Xử lý volume: dùng log1p vì volume quy mô lớn
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-9)
        ret = df['close'].pct_change().fillna(0)
        # Tính rolling correlation
        cov = intraday_ret.rolling(window, min_periods=window).cov(ret)
        var_intra = intraday_ret.rolling(window, min_periods=window).var()
        var_ret = ret.rolling(window, min_periods=window).var()
        corr = cov / (var_intra.mul(var_ret).replace(0, np.nan).apply(np.sqrt).replace(0, np.nan))
        corr = corr.fillna(0)
        # Volume trung bình 5 kỳ, log1p
        vol_mean = np.log1p(df['matchingVolume'].rolling(5, min_periods=3).mean())
        raw = corr * vol_mean
        # Dynamic Tanh normalization
        std_raw = raw.rolling(2*window, min_periods=window).std()
        normalized = np.tanh(raw / (std_raw.replace(0, np.nan).fillna(1e-9)))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_186_zscore(df, window=15):
        # Công thức: TS_CORR((close-open)/open, return, 10) * TS_MEAN(volume, 5)
        # Bản chất: Tương quan giữa intraday return -và close-to-close return, nhân với volume trung bình
        # Chọn chuẩn hóa C (Rolling Z-Score/Clip) vì công thức có dạng spread/tương quan
        # Xử lý volume: dùng log1p vì volume quy mô lớn
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-9)
        ret = df['close'].pct_change().fillna(0)
        # Tính rolling correlation
        cov = intraday_ret.rolling(window, min_periods=window).cov(ret)
        var_intra = intraday_ret.rolling(window, min_periods=window).var()
        var_ret = ret.rolling(window, min_periods=window).var()
        corr = cov / (var_intra.mul(var_ret).replace(0, np.nan).apply(np.sqrt).replace(0, np.nan))
        corr = corr.fillna(0)
        # Volume trung bình 5 kỳ, log1p
        vol_mean = np.log1p(df['matchingVolume'].rolling(5, min_periods=3).mean())
        raw = corr * vol_mean
        # Rolling Z-Score normalization
        mean_raw = raw.rolling(2*window, min_periods=window).mean()
        std_raw = raw.rolling(2*window, min_periods=window).std()
        normalized = ((raw - mean_raw) / (std_raw.replace(0, np.nan).fillna(1e-9))).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_186_sign(df, window=5):
        # Công thức: TS_CORR((close-open)/open, return, 10) * TS_MEAN(volume, 5)
        # Bản chất: Tương quan giữa intraday return và close-to-close return, nhân với volume trung bình
        # Chọn chuẩn hóa D (Sign/Binary Soft) vì có thể dùng làm tín hiệu hướng
        # Xử lý volume: dùng log1p vì volume quy mô lớn
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-9)
        ret = df['close'].pct_change().fillna(0)
        # Tính rolling correlation
        cov = intraday_ret.rolling(window, min_periods=window).cov(ret)
        var_intra = intraday_ret.rolling(window, min_periods=window).var()
        var_ret = ret.rolling(window, min_periods=window).var()
        corr = cov / (var_intra.mul(var_ret).replace(0, np.nan).apply(np.sqrt).replace(0, np.nan))
        corr = corr.fillna(0)
        # Volume trung bình 5 kỳ, log1p
        vol_mean = np.log1p(df['matchingVolume'].rolling(5, min_periods=3).mean())
        raw = corr * vol_mean
        # Sign normalization
        normalized = pd.Series(np.sign(raw), index=df.index)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_186_wf(df, window=20, winsor_quantile=0.1):
        # Công thức: TS_CORR((close-open)/open, return, 10) * TS_MEAN(volume, 5)
        # Bản chất: Tương quan giữa intraday return -và close-to-close return, nhân với volume trung bình
        # Chọn chuẩn hóa E (Winsorized Fisher) để xử lý heavy tails từ tích corr*volume
        # Xử lý volume: dùng log1p vì volume quy mô lớn
        intraday_ret = (df['close'] - df['open']) / (df['open'] + 1e-9)
        ret = df['close'].pct_change().fillna(0)
        # Tính rolling correlation
        cov = intraday_ret.rolling(window, min_periods=window).cov(ret)
        var_intra = intraday_ret.rolling(window, min_periods=window).var()
        var_ret = ret.rolling(window, min_periods=window).var()
        corr = cov / (var_intra.mul(var_ret).replace(0, np.nan).apply(np.sqrt).replace(0, np.nan))
        corr = corr.fillna(0)
        # Volume trung bình 5 kỳ, log1p
        vol_mean = np.log1p(df['matchingVolume'].rolling(5, min_periods=3).mean())
        raw = corr * vol_mean
        # Winsorized Fisher normalization
        low = raw.rolling(2*window, min_periods=window).quantile(winsor_quantile)
        high = raw.rolling(2*window, min_periods=window).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Tránh division by zero
        denom = (high - low).replace(0, np.nan).fillna(1e-9)
        # Fisher Transform
        val = ((winsorized - low) / denom) * 1.98 - 0.99
        # Clip để tránh arctanh bị infinity
        val = val.clip(-0.99, 0.99)
        normalized = np.arctanh(val)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_187_rank(df, window=15):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw_intraday = (close - open_price) / (open_price + 1e-8)
        raw = abs(raw_intraday) * raw_intraday.rolling(3).sum() * (1 - volume / (volume.rolling(window).mean() + volume.rolling(window).std() + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        rolling_rank = raw.rolling(window).rank(pct=True) * 2 - 1
        return rolling_rank.fillna(0)

    @staticmethod
    def alpha_quanta_187_tanh(df, window=5):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw_intraday = (close - open_price) / (open_price + 1e-8)
        raw = abs(raw_intraday) * raw_intraday.rolling(3).sum() * (1 - volume / (volume.rolling(window).mean() + volume.rolling(window).std() + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        std_val = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        tanh_norm = np.tanh(raw / std_val)
        result = pd.Series(tanh_norm, index=df.index).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_187_zscore(df, window=5):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw_intraday = (close - open_price) / (open_price + 1e-8)
        raw = abs(raw_intraday) * raw_intraday.rolling(3).sum() * (1 - volume / (volume.rolling(window).mean() + volume.rolling(window).std() + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-8)
        zscore = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return zscore.fillna(0)

    @staticmethod
    def alpha_quanta_187_sign(df, window=100):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw_intraday = (close - open_price) / (open_price + 1e-8)
        raw = abs(raw_intraday) * raw_intraday.rolling(3).sum() * (1 - volume / (volume.rolling(window).mean() + volume.rolling(window).std() + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        sign_signal = np.sign(raw)
        result = pd.Series(sign_signal, index=df.index).fillna(0)
        return result

    @staticmethod
    def alpha_quanta_187_wf(df, window=70, p1=0.1):
        close = df['close']
        open_price = df['open']
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw_intraday = (close - open_price) / (open_price + 1e-8)
        raw = abs(raw_intraday) * raw_intraday.rolling(3).sum() * (1 - volume / (volume.rolling(window).mean() + volume.rolling(window).std() + 1e-8))
        raw = raw.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        denominator = (high - low).replace(0, np.nan).ffill().fillna(1e-9) + 1e-9
        normalized = np.arctanh(((winsorized - low) / denominator) * 1.98 - 0.99)
        result = pd.Series(normalized, index=df.index).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_188_rank(df, window=5):
        volume = pd.Series(np.where(df['matchingVolume'] > 2 * df['matchingVolume'].rolling(window).mean(), 1, 0), index=df.index)
        ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).max() + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        ret_sum = ret.rolling(7).sum()
        raw = ratio * ret_sum * volume
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_188_tanh(df, window=45):
        volume = pd.Series(np.where(df['matchingVolume'] > 2 * df['matchingVolume'].rolling(window).mean(), 1, 0), index=df.index)
        ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).max() + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        ret_sum = ret.rolling(7).sum()
        raw = ratio * ret_sum * volume
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_188_zscore(df, window=35):
        volume = pd.Series(np.where(df['matchingVolume'] > 2 * df['matchingVolume'].rolling(window).mean(), 1, 0), index=df.index)
        ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).max() + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        ret_sum = ret.rolling(7).sum()
        raw = ratio * ret_sum * volume
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_188_sign(df, window=30):
        volume = pd.Series(np.where(df['matchingVolume'] > 2 * df['matchingVolume'].rolling(window).mean(), 1, 0), index=df.index)
        ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).max() + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        ret_sum = ret.rolling(7).sum()
        raw = ratio * ret_sum * volume
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_188_wf(df, window=10, quantile_factor=0.9):
        volume = pd.Series(np.where(df['matchingVolume'] > 2 * df['matchingVolume'].rolling(window).mean(), 1, 0), index=df.index)
        ratio = df['matchingVolume'] / (df['matchingVolume'].rolling(window).max() + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        ret_sum = ret.rolling(7).sum()
        raw = ratio * ret_sum * volume
        low = raw.rolling(window).quantile(quantile_factor)
        high = raw.rolling(window).quantile(1 - quantile_factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_189_rank(df, window=40, sub_window=30):
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window=sub_window).mean()
        delta_mean_ret = mean_ret.diff().fillna(0)
        mean_range = high_low_range.rolling(window=window).mean()
        condition = (high_low_range > mean_range).astype(float)
        raw = high_low_range * delta_mean_ret * condition
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_189_tanh(df, window=10, sub_window=5):
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window=sub_window).mean()
        delta_mean_ret = mean_ret.diff().fillna(0)
        mean_range = high_low_range.rolling(window=window).mean()
        condition = (high_low_range > mean_range).astype(float)
        raw = high_low_range * delta_mean_ret * condition
        normalized = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_189_zscore(df, window=20, sub_window=7):
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window=sub_window).mean()
        delta_mean_ret = mean_ret.diff().fillna(0)
        mean_range = high_low_range.rolling(window=window).mean()
        condition = (high_low_range > mean_range).astype(float)
        raw = high_low_range * delta_mean_ret * condition
        normalized = ((raw - raw.rolling(window).mean()) / (raw.rolling(window).std() + 1e-8)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_189_sign(df, window=10, sub_window=30):
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window=sub_window).mean()
        delta_mean_ret = mean_ret.diff().fillna(0)
        mean_range = high_low_range.rolling(window=window).mean()
        condition = (high_low_range > mean_range).astype(float)
        raw = high_low_range * delta_mean_ret * condition
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_189_wf(df, window=10, sub_window=10, p1=0.1, p2=30):
        high_low_range = (df['high'] - df['low']) / (df['close'] + 1e-8)
        ret = df['close'].pct_change().fillna(0)
        mean_ret = ret.rolling(window=sub_window).mean()
        delta_mean_ret = mean_ret.diff().fillna(0)
        mean_range = high_low_range.rolling(window=window).mean()
        condition = (high_low_range > mean_range).astype(float)
        raw = high_low_range * delta_mean_ret * condition
        low = raw.rolling(window=p2).quantile(p1)
        high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_190_rank(df, window=30):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std().replace(0, np.nan))
        rank_series = raw.rolling(window).rank(pct=True) * 2 - 1
        return -rank_series.fillna(0)

    @staticmethod
    def alpha_quanta_190_tanh(df, window=40):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std().replace(0, np.nan))
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_190_zscore(df, window=80):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std().replace(0, np.nan))
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_190_sign(df, window=85):
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std().replace(0, np.nan))
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_190_wf(df, window=30, p1=0.3):
        p2 = window
        raw = (df['high'] - df['low']) / (df['close'].rolling(window).std().replace(0, np.nan))
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_191_rank(df, window=95):
        raw = (df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8) - 1) * (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_191_tanh(df, window=95):
        raw = (df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8) - 1) * (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-8))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_191_zscore(df, window=75):
        raw = (df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8) - 1) * (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / (raw.rolling(window).std() + 1e-8)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_191_sign(df, window=45):
        raw = (df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8) - 1) * (df['matchingVolume'] / (df['matchingVolume'].rolling(window).mean() + 1e-8))
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_191_wf(df, window_rank=10, factor=0.3):
        raw = (df['matchingVolume'] / (df['matchingVolume'].shift(1) + 1e-8) - 1) * (df['matchingVolume'] / (df['matchingVolume'].rolling(window_rank).mean() + 1e-8))
        low = raw.rolling(window_rank).quantile(factor)
        high = raw.rolling(window_rank).quantile(1 - factor)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_192_rank(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(5)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        raw = delta_close * np.maximum(0, vol_z)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_192_tanh(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(5)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        raw = delta_close * np.maximum(0, vol_z)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_192_zscore(df, window=10):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(5)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        raw = delta_close * np.maximum(0, vol_z)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_192_sign(df, window=80):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(5)
        vol_mean = volume.rolling(window).mean()
        vol_std = volume.rolling(window).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        raw = delta_close * np.maximum(0, vol_z)
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_192_wf(df, p1=0.7, p2=100):
        close = df['close']
        volume = df['matchingVolume']
        delta_close = close.diff(5)
        vol_mean = volume.rolling(p2).mean()
        vol_std = volume.rolling(p2).std()
        vol_z = (volume - vol_mean) / (vol_std + 1e-8)
        raw = delta_close * np.maximum(0, vol_z)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_193_rank(df, window=75):
        close = df['close']
        delta_close = close.diff(10)
        ts_std = close.rolling(10).std()
        raw = delta_close / (ts_std + 1e-8)
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_193_tanh(df, window=55):
        close = df['close']
        delta_close = close.diff(10)
        ts_std = close.rolling(10).std()
        raw = delta_close / (ts_std + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_193_zscore(df, window=5):
        close = df['close']
        delta_close = close.diff(10)
        ts_std = close.rolling(10).std()
        raw = delta_close / (ts_std + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_193_sign(df, window=20):
        close = df['close']
        delta_close = close.diff(window)
        ts_std = close.rolling(window).std()
        raw = delta_close / (ts_std + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_193_wf(df, p1=0.1, p2=100):
        close = df['close']
        delta_close = close.diff(10)
        ts_std = close.rolling(10).std()
        raw = delta_close / (ts_std + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        spread = high - low
        normalized = np.arctanh(((winsorized - low) / (spread + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_194_rank(df, window=30):
        raw = (df['open'] - df['close'].shift(1)).abs() / (df['close'].rolling(window).mean() + 1e-8)
        raw = raw * (1.0 / (df['close'].rolling(window).std() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_194_tanh(df, window=5):
        raw = (df['open'] - df['close'].shift(1)).abs() / (df['close'].rolling(window).mean() + 1e-8)
        raw = raw * (1.0 / (df['close'].rolling(window).std() + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_194_zscore(df, window=80):
        raw = (df['open'] - df['close'].shift(1)).abs() / (df['close'].rolling(window).mean() + 1e-8)
        raw = raw * (1.0 / (df['close'].rolling(window).std() + 1e-8))
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std()
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_194_sign(df, window=70):
        raw = (df['open'] - df['close'].shift(1)).abs() / (df['close'].rolling(window).mean() + 1e-8)
        raw = raw * (1.0 / (df['close'].rolling(window).std() + 1e-8))
        raw_rolling_mean = raw.rolling(window).mean()
        signal = np.where(raw > raw_rolling_mean, 1.0, -1.0)
        signal = pd.Series(signal, index=df.index)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_194_wf(df, window=70, percentile=0.9):
        raw = (df['open'] - df['close'].shift(1)).abs() / (df['close'].rolling(window).mean() + 1e-8)
        raw = raw * (1.0 / (df['close'].rolling(window).std() + 1e-8))
        low = raw.rolling(window).quantile(percentile)
        high = raw.rolling(window).quantile(1 - percentile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.replace([np.inf, -np.inf], np.nan).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_195_rank(df, window=5):
        raw = ((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).mean() / (((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).std() + 1e-8)
        signal = raw.rolling(window).rank(pct=True) * 2 - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_195_tanh(df, window=5):
        raw = ((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).mean() / (((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_195_zscore(df, window=90):
        raw = ((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).mean() / (((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_195_sign(df):
        raw = ((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).mean() / (((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).std() + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_195_wf(df, window_winsor=75):
        raw = ((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).mean() / (((df['high'] - df['low']) / (df['close'] + 1e-8)).rolling(5).std() + 1e-8)
        low = raw.rolling(window_winsor).quantile(0.1)
        high = raw.rolling(window_winsor).quantile(0.9)
        winsorized = raw.clip(lower=low, upper=high)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_196_k(df, window=100):
        ret = df['close'].pct_change()
        sign_ret = np.sign(ret.rolling(5).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = vol.rolling(window).corr(df['close'])
        raw = sign_ret * corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_196_h(df, window=100):
        ret = df['close'].pct_change()
        sign_ret = np.sign(ret.rolling(5).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = vol.rolling(window).corr(df['close'])
        raw = sign_ret * corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_196_e(df, window=100):
        ret = df['close'].pct_change()
        sign_ret = np.sign(ret.rolling(5).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = vol.rolling(window).corr(df['close'])
        raw = sign_ret * corr
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_196_y(df, window=95):
        ret = df['close'].pct_change()
        sign_ret = np.sign(ret.rolling(5).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = vol.rolling(window).corr(df['close'])
        raw = sign_ret * corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_196_r(df, window=100):
        ret = df['close'].pct_change()
        sign_ret = np.sign(ret.rolling(5).mean())
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = vol.rolling(window).corr(df['close'])
        raw = sign_ret * corr
        low_th = raw.rolling(window).quantile(0.05)
        high_th = raw.rolling(window).quantile(0.95)
        winsorized = raw.clip(lower=low_th, upper=high_th, axis=0)
        low_w = winsorized.rolling(window).min()
        high_w = winsorized.rolling(window).max()
        normalized = np.arctanh(((winsorized - low_w) / (high_w - low_w + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_197_rank(df, window=75):
        log_return = np.log(df['close'] / df['close'].shift(1))
        raw = np.sign(log_return.rolling(window).mean()) * (df['close'].rolling(window).rank(pct=True) * 2 - 1) / (log_return.rolling(window).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_197_tanh(df, window=70):
        log_return -= np.log(df['close'] / df['close'].shift(1))
        raw = np.sign(log_return.rolling(window).mean()) * (df['close'].rolling(window).rank(pct=True) * 2 - 1) / (log_return.rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_197_zscore(df, window=60):
        log_return = np.log(df['close'] / df['close'].shift(1))
        raw = np.sign(log_return.rolling(window).mean()) * (df['close'].rolling(window).rank(pct=True) * 2 - 1) / (log_return.rolling(window).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_197_sign(df, window=25):
        log_return -= np.log(df['close'] / df['close'].shift(1))
        raw = np.sign(log_return.rolling(window).mean()) * (df['close'].rolling(window).rank(pct=True) * 2 - 1) / (log_return.rolling(window).std() + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_197_wf(df, window=90, sub_window=10):
        log_return = np.log(df['close'] / df['close'].shift(1))
        raw = np.sign(log_return.rolling(window).mean()) * (df['close'].rolling(window).rank(pct=True) * 2 - 1) / (log_return.rolling(window).std() + 1e-8)
        raw_clean = raw.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        p1 = 0.05
        low = raw_clean.rolling(sub_window).quantile(p1)
        high = raw_clean.rolling(sub_window).quantile(1 - p1)
        winsorized = raw_clean.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_198_rank(df, window=35):
        ret = df['open'] / df['close'].shift(1) - 1
        raw = ret.rolling(window).std() / (ret.abs().rolling(window).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_198_tanh(df, window=20):
        ret = df['open'] / df['close'].shift(1) - 1
        raw = ret.rolling(window).std() / (ret.abs().rolling(window).mean() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_198_zscore(df, window=35):
        ret = df['open'] / df['close'].shift(1) - 1
        raw = ret.rolling(window).std() / (ret.abs().rolling(window).mean() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_198_sign(df, window=65):
        ret = df['open'] / df['close'].shift(1) - 1
        raw = ret.rolling(window).std() / (ret.abs().rolling(window).mean() + 1e-8)
        signal = np.sign(raw)
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_198_wf(df, window=90, quantile=0.1):
        ret = df['open'] / df['close'].shift(1) - 1
        raw = ret.rolling(window).std() / (ret.abs().rolling(window).mean() + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(quantile)
        high = raw.rolling(p2).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9) * 1.98 - 0.99).clip(-0.99, 0.99))
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_199_rank(df, window=55):
        # Tính raw alpha
        delay_close = df['close'].shift(1)
        raw = np.log(np.abs(df['open'] / delay_close - 1) + 1)
        denom = df['high'].rolling(8).mean() - df['low'].rolling(8).mean() + 1e-8
        raw = raw / denom
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = raw * np.log1p(volume)
        # Rolling Rank normalization
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_199_tanh(df, window=5):
        delay_close = df['close'].shift(1)
        raw = np.log(np.abs(df['open'] / delay_close - 1) + 1)
        denom = df['high'].rolling(8).mean() - df['low'].rolling(8).mean() + 1e-8
        raw = raw / denom
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = raw * np.log1p(volume)
        # Dynamic Tanh normalization
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        normalized = np.tanh(raw / std)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_199_zscore(df, window=70):
        delay_close = df['close'].shift(1)
        raw = np.log(np.abs(df['open'] / delay_close - 1) + 1)
        denom = df['high'].rolling(8).mean() - df['low'].rolling(8).mean() + 1e-8
        raw = raw / denom
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = raw * np.log1p(volume)
        # Rolling Z-Score normalization
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        normalized = ((raw - mean) / std).clip(-1, 1)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_199_sign(df):
        delay_close = df['close'].shift(1)
        raw = np.log(np.abs(df['open'] / delay_close - 1) + 1)
        denom = df['high'].rolling(8).mean() - df['low'].rolling(8).mean() + 1e-8
        raw = raw / denom
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = raw * np.log1p(volume)
        # Sign/Binary Soft normalization
        normalized = np.sign(raw)
        normalized = pd.Series(normalized, index=df.index).fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_199_wf(df, p1=0.9, p2=100):
        delay_close = df['close'].shift(1)
        raw = np.log(np.abs(df['open'] / delay_close - 1) + 1)
        denom = df['high'].rolling(8).mean() - df['low'].rolling(8).mean() + 1e-8
        raw = raw / denom
        volume = df.get('matchingVolume', df.get('volume', 1))
        raw = raw * np.log1p(volume)
        # Winsorized Fisher normalization (p1: percentile clip, p2: rolling window for quantile)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_200_k(df, window=30, sub_window=20):
        raw = df['open'] - df['close'].shift(1)
        denominator = df['high'] - df['low'] + 1e-8
        ratio = (raw.abs() / denominator).rolling(window=sub_window).mean()
        normalized = (ratio.rolling(window=window, min_periods=1).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_200_h(df, window=70, sub_window=10):
        raw = df['open'] - df['close'].shift(1)
        denominator = df['high'] - df['low'] + 1e-8
        ratio = (raw.abs() / denominator).rolling(window=sub_window).mean()
        normalized = np.tanh(ratio / ratio.rolling(window=window, min_periods=1).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_200_e(df, window=20, sub_window=20):
        raw = df['open'] - df['close'].shift(1)
        denominator = df['high'] - df['low'] + 1e-8
        ratio = (raw.abs() / denominator).rolling(window=sub_window).mean()
        mean_ = ratio.rolling(window=window, min_periods=1).mean()
        std_ = ratio.rolling(window=window, min_periods=1).std().replace(0, np.nan)
        normalized = ((ratio - mean_) / std_).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_200_n(df, sub_window=20):
        raw = df['open'] - df['close'].shift(1)
        denominator = df['high'] - df['low'] + 1e-8
        ratio = (raw.abs() / denominator).rolling(window=sub_window).mean()
        normalized = np.sign(ratio)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_200_r(df, sub_window=20, p1=0.3, p2=20):
        raw = df['open'] - df['close'].shift(1)
        denominator = df['high'] - df['low'] + 1e-8
        ratio = (raw.abs() / denominator).rolling(window=sub_window).mean()
        low = ratio.rolling(window=p2, min_periods=1).quantile(p1)
        high = ratio.rolling(window=p2, min_periods=1).quantile(1 - p1)
        winsorized = ratio.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_201_rank(df, window_rank=90):
        high_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = high_low.rolling(window_rank).corr(volume)
        sign = np.sign(corr)
        mean_vol = volume.rolling(5).mean()
        raw = sign * mean_vol
        normalized = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_201_tanh(df, window_std=10):
        high_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = high_low.rolling(window_std).corr(volume)
        sign = np.sign(corr)
        mean_vol = volume.rolling(5).mean()
        raw = sign * mean_vol
        normalized = np.tanh(raw / raw.rolling(window_std).std().replace(0, np.nan))
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_201_zscore(df, window_zscore=45):
        high_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = high_low.rolling(window_zscore).corr(volume)
        sign = np.sign(corr)
        mean_vol = volume.rolling(5).mean()
        raw = sign * mean_vol
        mean = raw.rolling(window_zscore).mean()
        std = raw.rolling(window_zscore).std().replace(0, np.nan)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_201_sign(df, window_sign=60):
        high_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = high_low.rolling(10).corr(volume)
        sign = np.sign(corr)
        mean_vol = volume.rolling(window_sign).mean()
        raw = sign * mean_vol
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_201_wf(df, p1_quantile=0.3, p2_rolling=20):
        high_low = df['high'] - df['low']
        volume = df['matchingVolume']
        corr = high_low.rolling(10).corr(volume)
        sign = np.sign(corr)
        mean_vol = volume.rolling(5).mean()
        raw = sign * mean_vol
        low = raw.rolling(p2_rolling).quantile(p1_quantile)
        high = raw.rolling(p2_rolling).quantile(1 - p1_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_202_rank(df, window=60, sub_window=40):
        # Tính lợi nhuận log
        log_ret = np.log(df['close'] / df['close'].shift(1))
        # Z-score của lợi nhuận trong 20 kỳ
        mean_log_ret = log_ret.rolling(window, min_periods=1).mean()
        std_log_ret = log_ret.rolling(window, min_periods=1).std().replace(0, np.nan)
        z_ret = (log_ret - mean_log_ret) / std_log_ret
        # Trung bình log của volume trong 10 kỳ
        vol_mean = df['matchingVolume'].rolling(sub_window, min_periods=1).mean()
        log_vol = np.log1p(vol_mean)
        # Nhân hai thành phần
        raw = z_ret * log_vol
        # Rolling Rank đưa về [-1, 1]
        signal = raw.rolling(window, min_periods=1).rank(pct=True) * 2 - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_202_tanh(df, window=20, sub_window=20):
        log_ret = np.log(df['close'] / df['close'].shift(1))
        mean_log_ret = log_ret.rolling(window, min_periods=1).mean()
        std_log_ret = log_ret.rolling(window, min_periods=1).std().replace(0, np.nan)
        z_ret = (log_ret - mean_log_ret) / std_log_ret
        vol_mean = df['matchingVolume'].rolling(sub_window, min_periods=1).mean()
        log_vol = np.log1p(vol_mean)
        raw = z_ret * log_vol
        # Dynamic Tanh
        signal = np.tanh(raw / raw.rolling(window, min_periods=1).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_202_zscore(df, window=60, sub_window=40):
        log_ret = np.log(df['close'] / df['close'].shift(1))
        mean_log_ret = log_ret.rolling(window, min_periods=1).mean()
        std_log_ret = log_ret.rolling(window, min_periods=1).std().replace(0, np.nan)
        z_ret = (log_ret - mean_log_ret) / std_log_ret
        vol_mean = df['matchingVolume'].rolling(sub_window, min_periods=1).mean()
        log_vol = np.log1p(vol_mean)
        raw = z_ret * log_vol
        # Rolling Z-Score/Clip
        mean_raw = raw.rolling(window, min_periods=1).mean()
        std_raw = raw.rolling(window, min_periods=1).std().replace(0, np.nan)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_202_sign(df, window=100, sub_window=10):
        log_ret = np.log(df['close'] / df['close'].shift(1))
        mean_log_ret = log_ret.rolling(window, min_periods=1).mean()
        std_log_ret = log_ret.rolling(window, min_periods=1).std().replace(0, np.nan)
        z_ret = (log_ret - mean_log_ret) / std_log_ret
        vol_mean = df['matchingVolume'].rolling(sub_window, min_periods=1).mean()
        log_vol = np.log1p(vol_mean)
        raw = z_ret * log_vol
        # Sign/Binary Soft
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_202_wf(df, window=40, sub_window=40, p1=0.05, p2=20):
        log_ret = np.log(df['close'] / df['close'].shift(1))
        mean_log_ret = log_ret.rolling(window, min_periods=1).mean()
        std_log_ret = log_ret.rolling(window, min_periods=1).std().replace(0, np.nan)
        z_ret = (log_ret - mean_log_ret) / std_log_ret
        vol_mean = df['matchingVolume'].rolling(sub_window, min_periods=1).mean()
        log_vol = np.log1p(vol_mean)
        raw = z_ret * log_vol
        # Winsorized Fisher Transform
        lo = raw.rolling(p2, min_periods=1).quantile(p1)
        hi = raw.rolling(p2, min_periods=1).quantile(1 - p1)
        winsorized = raw.clip(lower=lo, upper=hi, axis=0)
        normalized = np.arctanh(((winsorized - lo) / (hi - lo + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_203_rank(df, window=20):
        delta = df['close'].diff(1)
        std = df['close'].rolling(window).std()
        raw = (delta / (std + 1e-8)) * (df['close'] / (df['high'].rolling(window).max() + 1e-8))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_203_tanh(df, window=25):
        delta = df['close'].diff(1)
        std = df['close'].rolling(window).std()
        raw = (delta / (std + 1e-8)) * (df['close'] / (df['high'].rolling(window).max() + 1e-8))
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_203_zscore(df, window=20):
        delta = df['close'].diff(1)
        std = df['close'].rolling(window).std()
        raw = (delta / (std + 1e-8)) * (df['close'] / (df['high'].rolling(window).max() + 1e-8))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_203_sign(df, window=75):
        delta = df['close'].diff(1)
        std = df['close'].rolling(window).std()
        raw = (delta / (std + 1e-8)) * (df['close'] / (df['high'].rolling(window).max() + 1e-8))
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_203_wf(df, window=40, quantile=0.1):
        delta = df['close'].diff(1)
        std = df['close'].rolling(window).std()
        raw = (delta / (std + 1e-8)) * (df['close'] / (df['high'].rolling(window).max() + 1e-8))
        p2 = window
        low = raw.rolling(p2).quantile(quantile)
        high = raw.rolling(p2).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_204_rank(df, window=15):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol).fillna(0)
        mean_ret = ret.rolling(window).mean().fillna(0)
        raw = mean_ret * corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_204_tanh(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol).fillna(0)
        mean_ret = ret.rolling(window).mean().fillna(0)
        raw = mean_ret * corr
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1e-9))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_204_zscore(df, window=40):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol).fillna(0)
        mean_ret = ret.rolling(window).mean().fillna(0)
        raw = mean_ret * corr
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_204_sign(df, window=20):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol).fillna(0)
        mean_ret = ret.rolling(window).mean().fillna(0)
        raw = mean_ret * corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_204_wf(df, window=20, quantile=0.7):
        ret = df['close'].pct_change().fillna(0)
        vol = df.get('matchingVolume', df.get('volume', 1))
        corr = ret.rolling(window).corr(vol).fillna(0)
        mean_ret = ret.rolling(window).mean().fillna(0)
        raw = mean_ret * corr
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_205_rank(df, window=100):
        raw = -((df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', 1)))
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_205_tanh(df, window=55):
        raw = -((df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', 1)))
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_205_zscore(df, window=50):
        raw = -((df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', 1)))
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_205_sign(df, window=5):
        raw = -((df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', 1)))
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_205_wf(df, window=7, p2=90):
        raw = -((df['high'] - df['low']).rolling(window).corr(df.get('matchingVolume', 1)))
        p1 = 0.05
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_206_k(df, window=15):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        # Compute raw components
        position = (close - low) / (high - low + 1e-8)
        sign_position = np.sign(position)
        # volume delta using shift (no lookahead)
        vol_delta = volume - volume.shift(1)
        sign_vol_delta = np.sign(vol_delta)
        # return = close.pct_change()
        ret = close.pct_change().fillna(0)
        ts_mean_ret = ret.rolling(window=3).mean().fillna(0)
        raw = sign_position * sign_vol_delta * ts_mean_ret
        # Rolling rank normalization (Case A)
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        norm = norm.fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_206_h(df, window=15):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        position = (close - low) / (high - low + 1e-8)
        sign_position = np.sign(position)
        vol_delta = volume - volume.shift(1)
        sign_vol_delta = np.sign(vol_delta)
        ret = close.pct_change().fillna(0)
        ts_mean_ret = ret.rolling(window=3).mean().fillna(0)
        raw = sign_position * sign_vol_delta * ts_mean_ret
        # Dynamic tanh (Case B)
        std_raw = raw.rolling(window).std().replace(0, np.nan).fillna(1e-8)
        norm = np.tanh(raw / std_raw)
        norm = norm.fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_206_p(df, window=25):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        position = (close - low) / (high - low + 1e-8)
        sign_position = np.sign(position)
        vol_delta = volume - volume.shift(1)
        sign_vol_delta = np.sign(vol_delta)
        ret = close.pct_change().fillna(0)
        ts_mean_ret = ret.rolling(window=3).mean().fillna(0)
        raw = sign_position * sign_vol_delta * ts_mean_ret
        # Rolling Z-score/Clip (Case C)
        rolling_mean = raw.rolling(window).mean().fillna(0)
        rolling_std = raw.rolling(window).std().replace(0, np.nan).fillna(1e-8)
        norm = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        norm = norm.fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_206_y(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        position = (close - low) / (high - low + 1e-8)
        sign_position = np.sign(position)
        vol_delta = volume - volume.shift(1)
        sign_vol_delta = np.sign(vol_delta)
        ret = close.pct_change().fillna(0)
        ts_mean_ret = ret.rolling(window=3).mean().fillna(0)
        raw = sign_position * sign_vol_delta * ts_mean_ret
        # Sign/Binary Soft (Case D) with optional window for smoothing
        smoothed_raw = raw.rolling(window).mean().fillna(0)
        norm = np.sign(smoothed_raw)
        return norm

    @staticmethod
    def alpha_quanta_206_r(df, window=60, quantile_p=0.1):
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['matchingVolume']
        position = (close - low) / (high - low + 1e-8)
        sign_position = np.sign(position)
        vol_delta = volume - volume.shift(1)
        sign_vol_delta = np.sign(vol_delta)
        ret = close.pct_change().fillna(0)
        ts_mean_ret = ret.rolling(window=3).mean().fillna(0)
        raw = sign_position * sign_vol_delta * ts_mean_ret
        # Winsorized Fisher (Case E)
        low_quant = raw.rolling(window).quantile(quantile_p)
        high_quant = raw.rolling(window).quantile(1 - quantile_p)
        winsorized = raw.clip(lower=low_quant, upper=high_quant, axis=0)
        denom = (high_quant - low_quant + 1e-9)
        scaled = ((winsorized - low_quant) / denom) * 1.98 - 0.99
        # Clip scaled to avoid arctanh domain issues
        scaled = scaled.clip(-0.999, 0.999)
        norm = np.arctanh(scaled)
        norm = norm.fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_207_tanh(df, window=15):
        close = df['close']
        ret = close.pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(5).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_207_rank(df, window=5):
        close = df['close']
        ret = close.pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(5).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_207_zscore(df, window=5):
        close = df['close']
        ret = close.pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(5).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_207_sign(df, window=30):
        close = df['close']
        ret = close.pct_change()
        raw = ret.rolling(window).mean() / (ret.rolling(5).std() + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_207_wf(df, window=5, sub_window=10):
        close = df['close']
        ret = close.pct_change()
        raw = ret.rolling(sub_window).mean() / (ret.rolling(5).std() + 1e-8)
        low = raw.rolling(window).quantile(0.1)
        high = raw.rolling(window).quantile(0.9)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_208_rank(df, window=100):
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / (df['open'] + 1e-8))
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_208_tanh(df, window=100):
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / (df['open'] + 1e-8))
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_208_zscore(df, window=100):
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / (df['open'] + 1e-8))
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_208_sign(df, window=80):
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / (df['open'] + 1e-8))
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_208_wf(df, window=100, winsor_quantile=0.1):
        raw = np.sign((df['open'] - df['close'].shift(1)) / (df['close'].rolling(window).std() + 1e-8)) * ((df['high'] - df['low']) / (df['open'] + 1e-8))
        p1 = winsor_quantile
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_209_rank(df, window_rank=35):
        ret = df['close'].pct_change()
        mean_8 = ret.rolling(window_rank).mean()
        corr = ret.rolling(window_rank).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean_4 = ret.rolling(4).mean()
        raw = mean_8 - corr * mean_4
        normalized = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_209_tanh(df, window_tanh=25):
        ret = df['close'].pct_change()
        mean_8 = ret.rolling(window_tanh).mean()
        corr = ret.rolling(window_tanh).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean_4 = ret.rolling(4).mean()
        raw = mean_8 - corr * mean_4
        normalized = np.tanh(raw / raw.rolling(window_tanh).std().replace(0, np.nan))
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_209_zscore(df, window_z=30):
        ret = df['close'].pct_change()
        mean_8 = ret.rolling(window_z).mean()
        corr = ret.rolling(window_z).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean_4 = ret.rolling(4).mean()
        raw = mean_8 - corr * mean_4
        normed = ((raw - raw.rolling(window_z).mean()) / raw.rolling(window_z).std().replace(0, np.nan)).clip(-1, 1)
        return normed.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_209_sign(df, window_sign=30):
        ret = df['close'].pct_change()
        mean_8 = ret.rolling(window_sign).mean()
        corr = ret.rolling(window_sign).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean_4 = ret.rolling(4).mean()
        raw = mean_8 - corr * mean_4
        normalized = np.sign(raw)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_209_wf(df, p1=0.1, p2=10):
        ret = df['close'].pct_change()
        mean_8 = ret.rolling(8).mean()
        corr = ret.rolling(8).corr(df.get('matchingVolume', df.get('volume', 1)))
        mean_4 = ret.rolling(4).mean()
        raw = mean_8 - corr * mean_4
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_210_k(df, window=90):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_vol_3 = volume.rolling(3).mean()
        delay_vol_20 = volume.rolling(20).mean().shift(3)
        raw = mean_vol_3 / (delay_vol_20.replace(0, np.nan).ffill() + 1e-8) - 1
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_210_h(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_vol_3 = volume.rolling(3).mean()
        delay_vol_20 = volume.rolling(20).mean().shift(3)
        raw = mean_vol_3 / (delay_vol_20.replace(0, np.nan).ffill() + 1e-8) - 1
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_210_e(df, window=80):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_vol_3 = volume.rolling(3).mean()
        delay_vol_20 = volume.rolling(20).mean().shift(3)
        raw = mean_vol_3 / (delay_vol_20.replace(0, np.nan).ffill() + 1e-8) - 1
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_210_y(df):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_vol_3 = volume.rolling(3).mean()
        delay_vol_20 = volume.rolling(20).mean().shift(3)
        raw = mean_vol_3 / (delay_vol_20.replace(0, np.nan).ffill() + 1e-8) - 1
        signal = np.sign(raw)
        return -pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_210_r(df, window=100, p1=0.1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        mean_vol_3 = volume.rolling(3).mean()
        delay_vol_20 = volume.rolling(20).mean().shift(3)
        raw = mean_vol_3 / (delay_vol_20.replace(0, np.nan).ffill() + 1e-8) - 1
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_211_k(df, window=100, sub_window=3):
        high_max = df['high'].rolling(window).max()
        close_ratio = df['close'] / (high_max + 1e-8) - 1
        vol_mean = df['matchingVolume'].rolling(sub_window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        corr = close_ratio.rolling(sub_window).corr(vol_ratio).fillna(0)
        raw = np.sign(close_ratio) * corr
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_211_h(df, window=30, sub_window=3):
        high_max = df['high'].rolling(window).max()
        close_ratio = df['close'] / (high_max + 1e-8) - 1
        vol_mean = df['matchingVolume'].rolling(sub_window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        corr = close_ratio.rolling(sub_window).corr(vol_ratio).fillna(0)
        raw = np.sign(close_ratio) * corr
        std = raw.rolling(window).std().replace(0, np.nan).ffill()
        normalized = np.tanh(raw / (std + 1e-8))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_211_e(df, window=30, sub_window=5):
        high_max = df['high'].rolling(window).max()
        close_ratio = df['close'] / (high_max + 1e-8) - 1
        vol_mean = df['matchingVolume'].rolling(sub_window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        corr = close_ratio.rolling(sub_window).corr(vol_ratio).fillna(0)
        raw = np.sign(close_ratio) * corr
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan).ffill()
        normalized = ((raw - mean) / (std + 1e-8)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_211_y(df, window=50, sub_window=3):
        high_max = df['high'].rolling(window).max()
        close_ratio = df['close'] / (high_max + 1e-8) - 1
        vol_mean = df['matchingVolume'].rolling(sub_window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        corr = close_ratio.rolling(sub_window).corr(vol_ratio).fillna(0)
        raw = np.sign(close_ratio) * corr
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_211_r(df, window=100, sub_window=3, p1=0.05):
        high_max = df['high'].rolling(window).max()
        close_ratio = df['close'] / (high_max + 1e-8) - 1
        vol_mean = df['matchingVolume'].rolling(sub_window).mean()
        vol_ratio = df['matchingVolume'] / (vol_mean + 1e-8)
        corr = close_ratio.rolling(sub_window).corr(vol_ratio).fillna(0)
        raw = np.sign(close_ratio) * corr
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_212_k(df, window=40, sub_window=10):
        v = df.get('matchingVolume', df.get('volume', 1))
        mean5 = v.rolling(window).mean()
        delta_mean5 = mean5.diff(1)
        mean10 = v.rolling(window*2).mean()
        part1_raw = delta_mean5 / (mean10 + 1e-8)
        part1 = (part1_raw.rolling(window).rank(pct=True) * 2) - 1
        mean3 = v.rolling(3).mean()
        mean20 = v.rolling(sub_window).mean()
        delay_mean20 = mean20.shift(3)
        part2_raw = mean3 / (delay_mean20 + 1e-8) - 1
        part2 = (part2_raw.rolling(window).rank(pct=True) * 2) - 1
        c = df['close']
        h = df['high']
        max20 = h.rolling(sub_window).max()
        corr_x = c / (max20 + 1e-8)
        corr_y = v / (mean10 + 1e-8)
        corr = corr_x.rolling(10).corr(corr_y)
        corr = corr.fillna(0)
        part3 = corr.clip(-1, 1)
        signal = 0.4 * part1 + 0.35 * part2 + 0.25 * part3
        return -signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_212_h(df, window=40, sub_window=40):
        v = df.get('matchingVolume', df.get('volume', 1))
        mean5 = v.rolling(window).mean()
        delta_mean5 = mean5.diff(1)
        mean10 = v.rolling(window*2).mean()
        part1_raw = delta_mean5 / (mean10 + 1e-8)
        part1 = np.tanh(part1_raw / part1_raw.rolling(window).std().replace(0, np.nan))
        mean3 = v.rolling(3).mean()
        mean20 = v.rolling(sub_window).mean()
        delay_mean20 = mean20.shift(3)
        part2_raw = mean3 / (delay_mean20 + 1e-8) - 1
        part2 = np.tanh(part2_raw / part2_raw.rolling(window).std().replace(0, np.nan))
        c = df['close']
        h = df['high']
        max20 = h.rolling(sub_window).max()
        corr_x = c / (max20 + 1e-8)
        corr_y = v / (mean10 + 1e-8)
        corr = corr_x.rolling(10).corr(corr_y)
        corr = corr.fillna(0).clip(-1, 1)
        part3 = corr
        signal = 0.4 * part1 + 0.35 * part2 + 0.25 * part3
        return -pd.Series(np.tanh(signal), index=df.index)

    @staticmethod
    def alpha_quanta_212_e(df, window=40, sub_window=80):
        v = df.get('matchingVolume', df.get('volume', 1))
        mean5 = v.rolling(window).mean()
        delta_mean5 = mean5.diff(1)
        mean10 = v.rolling(window*2).mean()
        part1_raw = delta_mean5 / (mean10 + 1e-8)
        part1_mean = part1_raw.rolling(window).mean()
        part1_std = part1_raw.rolling(window).std().replace(0, np.nan)
        part1 = ((part1_raw - part1_mean) / part1_std).clip(-1, 1)
        mean3 = v.rolling(3).mean()
        mean20 = v.rolling(sub_window).mean()
        delay_mean20 = mean20.shift(3)
        part2_raw = mean3 / (delay_mean20 + 1e-8) - 1
        part2_mean = part2_raw.rolling(window).mean()
        part2_std = part2_raw.rolling(window).std().replace(0, np.nan)
        part2 = ((part2_raw - part2_mean) / part2_std).clip(-1, 1)
        c = df['close']
        h = df['high']
        max20 = h.rolling(sub_window).max()
        corr_x = c / (max20 + 1e-8)
        corr_y = v / (mean10 + 1e-8)
        corr = corr_x.rolling(10).corr(corr_y)
        corr = corr.fillna(0).clip(-1, 1)
        part3 = corr
        signal = 0.4 * part1 + 0.35 * part2 + 0.25 * part3
        return -signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_212_y(df, window=20, sub_window=10):
        v = df.get('matchingVolume', df.get('volume', 1))
        mean5 = v.rolling(window).mean()
        delta_mean5 = mean5.diff(1)
        mean10 = v.rolling(window*2).mean()
        part1_raw = delta_mean5 / (mean10 + 1e-8)
        part1 = np.sign(part1_raw)
        mean3 = v.rolling(3).mean()
        mean20 = v.rolling(sub_window).mean()
        delay_mean20 = mean20.shift(3)
        part2_raw = mean3 / (delay_mean20 + 1e-8) - 1
        part2 = np.sign(part2_raw)
        c = df['close']
        h = df['high']
        max20 = h.rolling(sub_window).max()
        corr_x = c / (max20 + 1e-8)
        corr_y = v / (mean10 + 1e-8)
        corr = corr_x.rolling(10).corr(corr_y)
        corr = corr.fillna(0).clip(-1, 1)
        part3 = corr
        signal = 0.4 * part1 + 0.35 * part2 + 0.25 * part3
        return -pd.Series(np.sign(signal), index=df.index)

    @staticmethod
    def alpha_quanta_212_r(df, window=40, sub_window=50, clip_quantile=0.05):
        v = df.get('matchingVolume', df.get('volume', 1))
        mean5 = v.rolling(window).mean()
        delta_mean5 = mean5.diff(1)
        mean10 = v.rolling(window*2).mean()
        part1_raw = delta_mean5 / (mean10 + 1e-8)
        p1_low = part1_raw.rolling(window).quantile(clip_quantile)
        p1_high = part1_raw.rolling(window).quantile(1 - clip_quantile)
        winsorized1 = part1_raw.clip(lower=p1_low, upper=p1_high, axis=0)
        norm1 = np.arctanh(((winsorized1 - p1_low) / (p1_high - p1_low + 1e-9)) * 1.98 - 0.99)
        mean3 = v.rolling(3).mean()
        mean20 = v.rolling(sub_window).mean()
        delay_mean20 = mean20.shift(3)
        part2_raw = mean3 / (delay_mean20 + 1e-8) - 1
        p2_low = part2_raw.rolling(window).quantile(clip_quantile)
        p2_high = part2_raw.rolling(window).quantile(1 - clip_quantile)
        winsorized2 = part2_raw.clip(lower=p2_low, upper=p2_high, axis=0)
        norm2 = np.arctanh(((winsorized2 - p2_low) / (p2_high - p2_low + 1e-9)) * 1.98 - 0.99)
        c = df['close']
        h = df['high']
        max20 = h.rolling(sub_window).max()
        corr_x = c / (max20 + 1e-8)
        corr_y = v / (mean10 + 1e-8)
        corr = corr_x.rolling(10).corr(corr_y)
        corr = corr.fillna(0).clip(-1, 1)
        part3 = corr
        signal = 0.4 * norm1.fillna(0) + 0.35 * norm2.fillna(0) + 0.25 * part3
        return -signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_213_rank(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 1))
        vol_ratio_s = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        rank_vol = (vol_ratio_s.rolling(window).rank(pct=True) * 2) - 1
        close_ratio = df['close'] / (df['close'].rolling(10).mean() + 1e-8)
        vol_ratio = volume / (volume.rolling(10).mean() + 1e-8)
        # Compute rolling correlation using vectorized approach
        mean_close_ratio = close_ratio.rolling(10).mean()
        mean_vol_ratio = vol_ratio.rolling(10).mean()
        cov = (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) - (10 * (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) / 10)
        std_close = close_ratio.rolling(10).std()
        std_vol = vol_ratio.rolling(10).std()
        corr = cov / (std_close * std_vol + 1e-8)
        # Ensure corr is between -1 and 1 and clip
        corr = corr.clip(-1, 1)
        signal = rank_vol * corr
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_213_tanh(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 1))
        vol_ratio_s = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        std_vol_ratio = vol_ratio_s.rolling(window).std()
        signal = (vol_ratio_s.rolling(window).mean() / std_vol_ratio) if std_vol_ratio.nunique() > 1 else pd.Series(np.nan, index=df.index)
        # Compute raw signal
        close_ratio = df['close'] / (df['close'].rolling(10).mean() + 1e-8)
        vol_ratio = volume / (volume.rolling(10).mean() + 1e-8)
        mean_close_ratio = close_ratio.rolling(10).mean()
        mean_vol_ratio = vol_ratio.rolling(10).mean()
        cov = (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) - (10 * (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) / 10)
        std_close = close_ratio.rolling(10).std()
        std_vol = vol_ratio.rolling(10).std()
        corr = cov / (std_close * std_vol + 1e-8)
        corr = corr.clip(-1, 1)
        raw = vol_ratio_s * corr
        # Apply dynamic tanh
        std_raw = raw.rolling(window).std()
        signal = np.tanh(raw / (std_raw + 1e-8))
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_213_zscore(df, window=40):
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 1))
        vol_ratio_s = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        close_ratio = df['close'] / (df['close'].rolling(10).mean() + 1e-8)
        vol_ratio = volume / (volume.rolling(10).mean() + 1e-8)
        mean_close_ratio = close_ratio.rolling(10).mean()
        mean_vol_ratio = vol_ratio.rolling(10).mean()
        cov = (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) - (10 * (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) / 10)
        std_close = close_ratio.rolling(10).std()
        std_vol = vol_ratio.rolling(10).std()
        corr = cov / (std_close * std_vol + 1e-8)
        corr = corr.clip(-1, 1)
        raw = vol_ratio_s * corr
        raw_mean = raw.rolling(window).mean()
        raw_std = raw.rolling(window).std()
        signal = ((raw - raw_mean) / (raw_std + 1e-8)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_213_sign(df):
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 1))
        vol_ratio_s = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        close_ratio = df['close'] / (df['close'].rolling(10).mean() + 1e-8)
        vol_ratio = volume / (volume.rolling(10).mean() + 1e-8)
        mean_close_ratio = close_ratio.rolling(10).mean()
        mean_vol_ratio = vol_ratio.rolling(10).mean()
        cov = (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) - (10 * (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) / 10)
        std_close = close_ratio.rolling(10).std()
        std_vol = vol_ratio.rolling(10).std()
        corr = cov / (std_close * std_vol + 1e-8)
        corr = corr.clip(-1, 1)
        raw = vol_ratio_s * corr
        # Use sign of the signal
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_213_wf(df, p1=0.5, p2=20):
        volume = df.get('matchingVolume', df.get('volume', df['close'] * 1))
        vol_ratio_s = volume.rolling(5).mean() / (volume.rolling(20).mean() + 1e-8)
        close_ratio = df['close'] / (df['close'].rolling(10).mean() + 1e-8)
        vol_ratio = volume / (volume.rolling(10).mean() + 1e-8)
        mean_close_ratio = close_ratio.rolling(10).mean()
        mean_vol_ratio = vol_ratio.rolling(10).mean()
        cov = (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) - (10 * (close_ratio.rolling(10).sum() * vol_ratio.rolling(10).sum()) / 10)
        std_close = close_ratio.rolling(10).std()
        std_vol = vol_ratio.rolling(10).std()
        corr = cov / (std_close * std_vol + 1e-8)
        corr = corr.clip(-1, 1)
        raw = vol_ratio_s * corr
        # Winsorized Fisher Transform
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_214_4(df, window=20):
        volume = df.get('matchingVolume', df.get('volume', 1))
        sum_vol = volume.rolling(window).sum()
        max_vol = volume.rolling(window).max()
        raw = sum_vol / (max_vol * window + 1e-8)
        ranked = raw.rolling(window).rank(pct=True) * 2 - 1
        return -ranked.fillna(0)

    @staticmethod
    def alpha_quanta_214_rank(df, window=95):
        volume = df.get('matchingVolume', df.get('volume'))
        raw = volume.rolling(window).sum() / (volume.rolling(window).max() * window + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_214_tanh(df, window=60):
        volume = df.get('matchingVolume', df.get('volume'))
        raw = volume.rolling(window).sum() / (volume.rolling(window).max() * window + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_214_zscore(df, window=20):
        volume = df.get('matchingVolume', df.get('volume'))
        raw = volume.rolling(window).sum() / (volume.rolling(window).max() * window + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_214_sign(df, window=20):
        volume = df.get('matchingVolume', df.get('volume'))
        raw = volume.rolling(window).sum() / (volume.rolling(window).max() * window + 1e-8)
        normalized = np.sign(raw - raw.rolling(window).mean())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_214_wf(df, window=20, p1=0.3):
        p2 = window
        volume = df.get('matchingVolume', df.get('volume'))
        raw = volume.rolling(window).sum() / (volume.rolling(window).max() * window + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        denominator = high - low + 1e-9
        normalized = np.arctanh(((winsorized - low) / denominator) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_215_5(df, window=20):
        # Calculate volume and close volatility ratios
        volume_ratio = df['matchingVolume'].rolling(window).std() / (df['matchingVolume'].rolling(window).mean() + 1e-8)
        close_ratio = df['close'].rolling(window).std() / (df['close'].rolling(window).mean() + 1e-8)
        # Difference of ratios
        raw = volume_ratio - close_ratio
        # Z-score normalization to [-1,1]
        mean_raw = raw.rolling(window).mean()
        std_raw = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean_raw) / std_raw).clip(-1, 1)
        # Handle NaN values
        signal = signal.fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_215_rank(df):
        vol_std = df['close'] * 0  # placeholder
        return vol_std

    @staticmethod
    def alpha_quanta_215_tanh(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_std = volume.rolling(window=window).std()
        vol_mean = volume.rolling(window=window).mean() + 1e-8
        close_std = df['close'].rolling(window=window).std()
        close_mean = df['close'].rolling(window=window).mean() + 1e-8
        raw = vol_std / vol_mean - close_std / close_mean
        normalized = np.tanh(raw / raw.rolling(window=window).std())
        return normalized

    @staticmethod
    def alpha_quanta_215_zscore(df, window=20):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_std = volume.rolling(window=window).std()
        vol_mean = volume.rolling(window=window).mean() + 1e-8
        close_std = df['close'].rolling(window=window).std()
        close_mean = df['close'].rolling(window=window).mean() + 1e-8
        raw = vol_std / vol_mean - close_std / close_mean
        normalized = ((raw - raw.rolling(window=window).mean()) / raw.rolling(window=window).std()).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_215_sign(df, window=35):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_std = volume.rolling(window=window).std()
        vol_mean = volume.rolling(window=window).mean() + 1e-8
        close_std = df['close'].rolling(window=window).std()
        close_mean = df['close'].rolling(window=window).mean() + 1e-8
        raw = vol_std / vol_mean - close_std / close_mean
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_215_wf(df, window=20, p1=0.3, p2=15):
        volume = df.get('matchingVolume', df.get('volume', 1))
        vol_std = volume.rolling(window=window).std()
        vol_mean = volume.rolling(window=window).mean() + 1e-8
        close_std = df['close'].rolling(window=window).std()
        close_mean = df['close'].rolling(window=window).mean() + 1e-8
        raw = vol_std / vol_mean - close_std / close_mean
        low = raw.rolling(window=p2).quantile(p1)
        high = raw.rolling(window=p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_216_rank(df, window=80):
        hi_lo = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        # Rolling covariance / (rolling std of hi_lo * rolling std of vol)
        hi_lo_mean = hi_lo.rolling(window).mean()
        vol_mean = vol.rolling(window).mean()
        cov = (hi_lo * vol).rolling(window).mean() - hi_lo_mean * vol_mean
        std_hi_lo = hi_lo.rolling(window).std(ddof=0)
        std_vol = vol.rolling(window).std(ddof=0)
        corr = cov / (std_hi_lo * std_vol + 1e-9)
        std_sign = (hi_lo.rolling(window).std(ddof=0) > 0).astype(float) * 2 - 1
        raw = corr * std_sign
        # Normalize A: Rolling Rank
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_216_tanh(df, window=5):
        hi_lo = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        hi_lo_mean = hi_lo.rolling(window).mean()
        vol_mean = vol.rolling(window).mean()
        cov = (hi_lo * vol).rolling(window).mean() - hi_lo_mean * vol_mean
        std_hi_lo = hi_lo.rolling(window).std(ddof=0)
        std_vol = vol.rolling(window).std(ddof=0)
        corr = cov / (std_hi_lo * std_vol + 1e-9)
        std_sign = (hi_lo.rolling(window).std(ddof=0) > 0).astype(float) * 2 - 1
        raw = corr * std_sign
        # Normalize B: Dynamic Tanh
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_216_zscore(df, window=80):
        hi_lo = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        hi_lo_mean = hi_lo.rolling(window).mean()
        vol_mean = vol.rolling(window).mean()
        cov = (hi_lo * vol).rolling(window).mean() - hi_lo_mean * vol_mean
        std_hi_lo = hi_lo.rolling(window).std(ddof=0)
        std_vol = vol.rolling(window).std(ddof=0)
        corr = cov / (std_hi_lo * std_vol + 1e-9)
        std_sign = (hi_lo.rolling(window).std(ddof=0) > 0).astype(float) * 2 - 1
        raw = corr * std_sign
        # Normalize C: Rolling Z-Score Clip
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std(ddof=0).replace(0, np.nan)
        result = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_216_sign(df, window=5):
        hi_lo = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        hi_lo_mean = hi_lo.rolling(window).mean()
        vol_mean = vol.rolling(window).mean()
        cov = (hi_lo * vol).rolling(window).mean() - hi_lo_mean * vol_mean
        std_hi_lo = hi_lo.rolling(window).std(ddof=0)
        std_vol = vol.rolling(window).std(ddof=0)
        corr = cov / (std_hi_lo * std_vol + 1e-9)
        std_sign = (hi_lo.rolling(window).std(ddof=0) > 0).astype(float) * 2 - 1
        raw = corr * std_sign
        # Normalize D: Sign Binary Soft
        result = pd.Series(np.sign(raw), index=df.index).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_216_wf(df, window=80):
        hi_lo = df['high'] - df['low']
        vol = np.log1p(df['matchingVolume'])
        hi_lo_mean = hi_lo.rolling(window).mean()
        vol_mean = vol.rolling(window).mean()
        cov = (hi_lo * vol).rolling(window).mean() - hi_lo_mean * vol_mean
        std_hi_lo = hi_lo.rolling(window).std(ddof=0)
        std_vol = vol.rolling(window).std(ddof=0)
        corr = cov / (std_hi_lo * std_vol + 1e-9)
        std_sign = (hi_lo.rolling(window).std(ddof=0) > 0).astype(float) * 2 - 1
        raw = corr * std_sign
        # Normalize E: Winsorized Fisher
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_217_rank(df, window=20):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * df.get('matchingVolume', 1)))
        vol_max = vol.rolling(window).max()
        vol_min = vol.rolling(window).min()
        vol_mean = vol.rolling(window).mean()
        raw = (vol_max - vol_min) / (vol_mean + 1e-8)
        norm = raw.rolling(window).rank(pct=True) * 2 - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_217_tanh(df, window=75):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * df.get('matchingVolume', 1)))
        vol_max = vol.rolling(window).max()
        vol_min = vol.rolling(window).min()
        vol_mean = vol.rolling(window).mean()
        raw = (vol_max - vol_min) / (vol_mean + 1e-8)
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_217_zscore(df, window=20):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * df.get('matchingVolume', 1)))
        vol_max = vol.rolling(window).max()
        vol_min = vol.rolling(window).min()
        vol_mean = vol.rolling(window).mean()
        raw = (vol_max - vol_min) / (vol_mean + 1e-8)
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_217_sign(df, window=60):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * df.get('matchingVolume', 1)))
        vol_max = vol.rolling(window).max()
        vol_min = vol.rolling(window).min()
        vol_mean = vol.rolling(window).mean()
        raw = (vol_max - vol_min) / (vol_mean + 1e-8)
        norm = pd.Series(np.sign(raw), index=df.index)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_217_wf(df, window=90, p1=0.1):
        vol = df.get('matchingVolume', df.get('volume', df['close'] * df.get('matchingVolume', 1)))
        vol_max = vol.rolling(window).max()
        vol_min = vol.rolling(window).min()
        vol_mean = vol.rolling(window).mean()
        raw = (vol_max - vol_min) / (vol_mean + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_218_rank(df, window=20):
        delta = df['close'].diff(1).abs()
        raw = delta.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_218_tanh(df, window=20):
        delta = df['close'].diff(1).abs()
        raw = delta.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-8)
        denom = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / denom)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_218_zscore(df, window=20):
        delta = df['close'].diff(1).abs()
        raw = delta.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_218_sign(df, window=20):
        delta = df['close'].diff(1).abs()
        raw = delta.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-8)
        signal = np.sign(raw - raw.rolling(window).mean())
        signal = pd.Series(signal, index=df.index).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_218_wf(df, window=20, p1=0.05):
        delta = df['close'].diff(1).abs()
        raw = delta.rolling(window).sum() / (df['volume'].rolling(window).sum() + 1e-8)
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_219_rank(df, window=50):
        # A) Rolling Rank
        ret = df['close'].pct_change()
        raw_abs_ret = ret.abs()
        numerator = raw_abs_ret.rolling(window).mean()
        hl_range = df['high'] - df['low']
        denominator = hl_range.rolling(window).std() + 1e-8
        raw = numerator / denominator
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_219_tanh(df, window=95):
        # B) Dynamic Tanh
        ret = df['close'].pct_change()
        raw_abs_ret = ret.abs()
        numerator = raw_abs_ret.rolling(window).mean()
        hl_range = df['high'] - df['low']
        denominator = hl_range.rolling(window).std() + 1e-8
        raw = numerator / denominator
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_219_zscore(df, window=15):
        # C) Rolling Z-Score/Clip
        ret = df['close'].pct_change()
        raw_abs_ret = ret.abs()
        numerator = raw_abs_ret.rolling(window).mean()
        hl_range = df['high'] - df['low']
        denominator = hl_range.rolling(window).std() + 1e-8
        raw = numerator / denominator
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_219_sign(df, window=5):
        # D) Sign/Binary Soft
        ret = df['close'].pct_change()
        raw_abs_ret = ret.abs()
        numerator = raw_abs_ret.rolling(window).mean()
        hl_range = df['high'] - df['low']
        denominator = hl_range.rolling(window).std() + 1e-8
        raw = numerator / denominator
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).ffill().fillna(0)

    @staticmethod
    def alpha_quanta_219_wf(df, window=10, sub_window=40):
        # E) Winsorized Fisher
        ret = df['close'].pct_change()
        raw_abs_ret = ret.abs()
        numerator = raw_abs_ret.rolling(window).mean()
        hl_range = df['high'] - df['low']
        denominator = hl_range.rolling(window).std() + 1e-8
        raw = numerator / denominator
        low = raw.rolling(sub_window).quantile(0.05)
        high = raw.rolling(sub_window).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized
        return signal.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_220_k(df, window=80):
        volume = df.get('matchingVolume', df.get('volume', np.nan))
        close = df['close']
        corr = close.rolling(window).corr(volume)
        std = volume.rolling(window).std()
        raw = corr / (std + 1e-8)
        raw = raw.fillna(method='ffill').fillna(0)
        result = raw.rolling(window).rank(pct=True) * 2 - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_220_h(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', np.nan))
        close = df['close']
        corr = close.rolling(window).corr(volume)
        std = volume.rolling(window).std()
        raw = corr / (std + 1e-8)
        raw = raw.fillna(method='ffill').fillna(0)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan).fillna(method='ffill').fillna(1))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_220_e(df, window=65):
        volume = df.get('matchingVolume', df.get('volume', np.nan))
        close = df['close']
        corr = close.rolling(window).corr(volume)
        std = volume.rolling(window).std()
        raw = corr / (std + 1e-8)
        raw = raw.fillna(method='ffill').fillna(0)
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_220_y(df, window=5):
        volume = df.get('matchingVolume', df.get('volume', np.nan))
        close = df['close']
        corr = close.rolling(window).corr(volume)
        std = volume.rolling(window).std()
        raw = corr / (std + 1e-8)
        raw = raw.fillna(method='ffill').fillna(0)
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_220_r(df, window=80, sub_window=30):
        volume = df.get('matchingVolume', df.get('volume', np.nan))
        close = df['close']
        corr = close.rolling(window).corr(volume)
        std = volume.rolling(window).std()
        raw = corr / (std + 1e-8)
        raw = raw.fillna(method='ffill').fillna(0)
        p1 = 0.1
        p2 = sub_window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.fillna(0)
        result[result == np.inf] = 1
        result[result == -np.inf] = -1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_221_rank(df, window=55):
        ret = df['close'].pct_change()
        delay_close = df['close'].shift(1)
        corr = ret.rolling(window).corr(delay_close)
        raw = (corr.rolling(window).rank(pct=True) * 2) - 1
        result = raw.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_221_tanh(df, window=30):
        ret = df['close'].pct_change()
        delay_close = df['close'].shift(1)
        corr = ret.rolling(window).corr(delay_close)
        raw = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        result = raw.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_221_zscore(df, window=35):
        ret = df['close'].pct_change()
        delay_close = df['close'].shift(1)
        corr = ret.rolling(window).corr(delay_close)
        mean_corr = corr.rolling(window).mean()
        std_corr = corr.rolling(window).std().replace(0, np.nan)
        raw = ((corr - mean_corr) / std_corr).clip(-1, 1)
        result = raw.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_221_sign(df, window=30):
        ret = df['close'].pct_change()
        delay_close = df['close'].shift(1)
        corr = ret.rolling(window).corr(delay_close)
        raw = np.sign(corr)
        result = pd.Series(raw, index=df.index).ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_221_wf(df, window=40, sub_window=80):
        p1 = 0.05
        ret = df['close'].pct_change()
        delay_close = df['close'].shift(1)
        corr = ret.rolling(window).corr(delay_close)
        low = corr.rolling(sub_window).quantile(p1)
        high = corr.rolling(sub_window).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_222_rank(df, window=80):
        ret = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume'].fillna(df.get('volume', 1))
        abs_return_ma = ret.abs().rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        raw = abs_return_ma / (volume_std + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_222_tanh(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume'].fillna(df.get('volume', 1))
        abs_return_ma = ret.abs().rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        corr = volume.rolling(window).corr(df['close']).fillna(0)
        raw = abs_return_ma / (volume_std + 1e-8) + corr
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_222_zscore(df, window=65):
        ret = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume'].fillna(df.get('volume', 1))
        abs_return_ma = ret.abs().rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        corr = volume.rolling(window).corr(df['close']).fillna(0)
        raw = abs_return_ma / (volume_std + 1e-8) + corr
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_222_sign(df, window=5):
        ret = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume'].fillna(df.get('volume', 1))
        abs_return_ma = ret.abs().rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        corr = volume.rolling(window).corr(df['close']).fillna(0)
        raw = abs_return_ma / (volume_std + 1e-8) + corr
        signal = np.sign(raw - raw.rolling(window).mean())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_222_wf(df, window=10, quantile=0.1):
        ret = df['close'].pct_change().fillna(0)
        volume = df['matchingVolume'].fillna(df.get('volume', 1))
        abs_return_ma = ret.abs().rolling(window).mean()
        volume_std = volume.rolling(window).std().replace(0, np.nan)
        corr = volume.rolling(window).corr(df['close']).fillna(0)
        raw = abs_return_ma / (volume_std + 1e-8) + corr
        p2 = window
        p1 = quantile
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_223_k(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        valid = window * 2
        ret_roll = ret.rolling(window)
        vol_roll = volume.rolling(window)
        corr = ret_roll.corr(vol_roll)
        vol_mean = volume.rolling(window).mean()
        raw = corr * vol_mean
        normalized = (raw.rolling(valid).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_223_h(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        valid = window * 2
        ret_roll = ret.rolling(window)
        vol_roll = volume.rolling(window)
        corr = ret_roll.corr(vol_roll)
        vol_mean = volume.rolling(window).mean()
        raw = corr * vol_mean
        normalized = np.tanh(raw / raw.rolling(valid).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_223_e(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        valid = window * 2
        ret_roll = ret.rolling(window)
        vol_roll = volume.rolling(window)
        corr = ret_roll.corr(vol_roll)
        vol_mean = volume.rolling(window).mean()
        raw = corr * vol_mean
        mean_ = raw.rolling(valid).mean()
        std_ = raw.rolling(valid).std().replace(0, np.nan)
        normalized = ((raw - mean_) / std_).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_223_y(df, window=20):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        ret_roll = ret.rolling(window)
        vol_roll = volume.rolling(window)
        corr = ret_roll.corr(vol_roll)
        vol_mean = volume.rolling(window).mean()
        raw = corr * vol_mean
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_223_r(df, window=20, winsor_quantile=0.05):
        close = df['close']
        volume = df['matchingVolume']
        ret = close.pct_change()
        valid = window * 2
        ret_roll = ret.rolling(window)
        vol_roll = volume.rolling(window)
        corr = ret_roll.corr(vol_roll)
        vol_mean = volume.rolling(window).mean()
        raw = corr * vol_mean
        low = raw.rolling(valid).quantile(winsor_quantile)
        high = raw.rolling(valid).quantile(1 - winsor_quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_224_rank(df, window=10):
        close = df['close']
        ret = close.pct_change()
        std5 = ret.rolling(5).std()
        corr = ret.rolling(window).corr(std5)
        rank = corr.rolling(window).rank(pct=True)
        signal = (rank * 2) - 1
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_224_tanh(df, window=5):
        close = df['close']
        ret = close.pct_change()
        std5 = ret.rolling(5).std()
        corr = ret.rolling(window).corr(std5)
        signal = np.tanh(corr / corr.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_224_zscore(df, window=5):
        close = df['close']
        ret = close.pct_change()
        std5 = ret.rolling(5).std()
        corr = ret.rolling(window).corr(std5)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std().replace(0, np.nan)
        signal = ((corr - mean) / std).clip(-1, 1)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_224_sign(df, window=5):
        close = df['close']
        ret = close.pct_change()
        std5 = ret.rolling(5).std()
        corr = ret.rolling(window).corr(std5)
        median = corr.rolling(window).median()
        signal = np.sign(corr - median).fillna(0)
        return signal.clip(-1, 1)

    @staticmethod
    def alpha_quanta_224_wf(df, window=10, winsor_quantile=0.1):
        close = df['close']
        ret = close.pct_change()
        std5 = ret.rolling(5).std()
        corr = ret.rolling(window).corr(std5)
        low = corr.rolling(window).quantile(winsor_quantile)
        high = corr.rolling(window).quantile(1 - winsor_quantile)
        winsorized = corr.clip(lower=low, upper=high)
        numerator = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        numerator = numerator.clip(-0.99, 0.99)
        signal = np.arctanh(numerator)
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_225_g(df, window=15):
        raw = df['close'].rolling(5).mean((df['close'] - df['low']) / (df['low'] + 1e-8)) / (df['close'].rolling(window).std() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_225_h(df, window=15):
        raw = df['close'].rolling(5).mean((df['close'] - df['low']) / (df['low'] + 1e-8)) / (df['close'].rolling(window).std() + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_225_e(df, window=15):
        raw = df['close'].rolling(5).mean((df['close'] - df['low']) / (df['low'] + 1e-8)) / (df['close'].rolling(window).std() + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_225_n(df, window=15):
        raw = df['close'].rolling(5).mean((df['close'] - df['low']) / (df['low'] + 1e-8)) / (df['close'].rolling(window).std() + 1e-8)
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_225_r(df, window=15, p1=0.1, p2=30):
        p1 = 0.1
        p2 = 30
        raw = df['close'].rolling(5).mean((df['close'] - df['low']) / (df['low'] + 1e-8)) / (df['close'].rolling(window).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_226_k(df, window=100):
        ret = df['close'].pct_change()
        vol = np.log1p(df['matchingVolume'])
        corr = ret.rolling(window).corr(vol).fillna(0)
        sign = np.sign(ret.rolling(25).mean().fillna(0))
        raw = corr * sign
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_226_h(df, window=25):
        ret = df['close'].pct_change()
        vol = np.log1p(df['matchingVolume'])
        corr = ret.rolling(window).corr(vol).fillna(0)
        sign = np.sign(ret.rolling(25).mean().fillna(0))
        raw = corr * sign
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).fillna(0)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_226_e(df, window=10):
        ret = df['close'].pct_change()
        vol = np.log1p(df['matchingVolume'])
        corr = ret.rolling(window).corr(vol).fillna(0)
        sign = np.sign(ret.rolling(25).mean().fillna(0))
        raw = corr * sign
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_226_n(df, window=85):
        ret = df['close'].pct_change()
        vol = np.log1p(df['matchingVolume'])
        corr = ret.rolling(window).corr(vol).fillna(0)
        sign = np.sign(ret.rolling(25).mean().fillna(0))
        raw = corr * sign
        normalized = np.sign(raw).fillna(0)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_226_d(df, window=10, p1=0.1):
        ret = df['close'].pct_change()
        vol = np.log1p(df['matchingVolume'])
        corr = ret.rolling(window).corr(vol).fillna(0)
        sign = np.sign(ret.rolling(25).mean().fillna(0))
        raw = corr * sign
        low = raw.rolling(window).quantile(p1)
        high = raw.rolling(window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = (np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)).fillna(0)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_227_rank(df, window=35):
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (high - low) / (close + 1e-8)
        ts_mean = raw.rolling(window).mean()
        ts_std = close.rolling(30).std() + 1e-8
        ratio = ts_mean / ts_std
        ranked = ratio.rolling(window).rank(pct=True) * 2 - 1
        signal = ranked.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_227_tanh(df, window=30):
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (high - low) / (close + 1e-8)
        ts_mean = raw.rolling(window).mean()
        ts_std = close.rolling(30).std() + 1e-8
        ratio = ts_mean / ts_std
        signal = np.tanh(ratio / ratio.rolling(window).std().replace(0, np.nan).fillna(1))
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_227_zscore(df, window=25):
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (high - low) / (close + 1e-8)
        ts_mean = raw.rolling(window).mean()
        ts_std = close.rolling(30).std() + 1e-8
        ratio = ts_mean / ts_std
        signal = ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std().replace(0, np.nan).fillna(1)).clip(-1, 1)
        signal = signal.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_227_sign(df, window=60):
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (high - low) / (close + 1e-8)
        ts_mean = raw.rolling(window).mean()
        ts_std = close.rolling(30).std() + 1e-8
        ratio = ts_mean / ts_std
        signal = np.sign(ratio).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_227_wf(df, window=7, sub_window=90):
        import numpy as np
        high = df['high']
        low = df['low']
        close = df['close']
        raw = (high - low) / (close + 1e-8)
        ts_mean = raw.rolling(window).mean()
        ts_std = close.rolling(30).std() + 1e-8
        ratio = ts_mean / ts_std
        p1 = 0.05
        low_quant = ratio.rolling(sub_window).quantile(p1)
        high_quant = ratio.rolling(sub_window).quantile(1 - p1)
        winsorized = ratio.clip(lower=low_quant, upper=high_quant, axis=0)
        normalized = np.arctanh(((winsorized - low_quant) / (high_quant - low_quant + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_228_k(df, window=30, rank_window=100):
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        corr = ret.rolling(window).corr(delay_ret)
        mean_ret = ret.rolling(40).mean()
        # Dùng mean_ret rolling mean 40 the nhu cau TH A: rolling rank
        raw = corr - mean_ret
        # Fill NaN threshold
        raw = raw.ffill().fillna(0)
        # Chuẩn hóa rolling rank
        result = (raw.rolling(rank_window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_228_h(df, window=40, std_window=70):
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        corr = ret.rolling(window).corr(delay_ret)
        mean_ret = ret.rolling(40).mean()
        raw = corr - mean_ret
        raw = raw.ffill().fillna(0)
        # Chuẩn hóa dynamic tanh
        result = np.tanh(raw / raw.rolling(std_window).std().replace(0, np.nan).ffill())
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_228_e(df, window=20, z_window=80):
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        corr = ret.rolling(window).corr(delay_ret)
        mean_ret = ret.rolling(40).mean()
        raw = corr - mean_ret
        raw = raw.ffill().fillna(0)
        # Chuẩn hóa rolling z-score
        rolling_mean = raw.rolling(z_window).mean()
        rolling_std = raw.rolling(z_window).std().replace(0, np.nan).ffill()
        result = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_228_y(df, window=65):
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        corr = ret.rolling(window).corr(delay_ret)
        mean_ret = ret.rolling(40).mean()
        raw = corr - mean_ret
        raw = raw.ffill().fillna(0)
        # Chuẩn hóa sign binary
        result = np.sign(raw)
        return -pd.Series(result, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_228_r(df, window=20, p1=0.9, p2=60):
        ret = df['close'].pct_change()
        delay_ret = ret.shift(1)
        corr = ret.rolling(window).corr(delay_ret)
        mean_ret = ret.rolling(40).mean()
        raw = corr - mean_ret
        raw = raw.ffill().fillna(0)
        # Winsorized Fisher
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        # Tránh chia cho 0 và tính normalized trong [-1, 1]
        denominator = (high - low).replace(0, np.nan).ffill()
        # Áp dụng Fisher Transform
        transformed = ((winsorized - low) / (denominator + 1e-9)) * 1.98 - 0.99
        # Clip để tránh arctanh infinity
        transformed = transformed.clip(-0.99, 0.99)
        result = np.arctanh(transformed)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_229_rank(df, window1=100, window2=5):
        return_ = df['close'].pct_change()
        mean_60 = return_.rolling(window1).mean()
        std_30 = return_.rolling(window2).std()
        corr = mean_60.rolling(90).corr(std_30)
        raw = corr.rolling(90).rank(pct=True) * 2 - 1
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_229_tanh(df, window1=30, window2=20):
        return_ = df['close'].pct_change()
        mean_60 = return_.rolling(window1).mean()
        std_30 = return_.rolling(window2).std()
        corr = mean_60.rolling(90).corr(std_30)
        raw = np.tanh(corr / corr.rolling(90).std().replace(0, np.nan))
        return raw.fillna(0)

    @staticmethod
    def alpha_quanta_229_zscore(df, window1=100, window2=5):
        return_ = df['close'].pct_change()
        mean_60 = return_.rolling(window1).mean()
        std_30 = return_.rolling(window2).std()
        corr = mean_60.rolling(90).corr(std_30)
        raw = ((corr - corr.rolling(90).mean()) / corr.rolling(90).std().replace(0, np.nan)).clip(-1, 1)
        return -raw.fillna(0)

    @staticmethod
    def alpha_quanta_229_sign(df, window1=30, window2=30):
        return_ = df['close'].pct_change()
        mean_60 = return_.rolling(window1).mean()
        std_30 = return_.rolling(window2).std()
        corr = mean_60.rolling(90).corr(std_30)
        raw = np.sign(corr)
        return pd.Series(raw, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_229_wf(df, window1=100, window2=3, p1=0.05, p2=90):
        return_ = df['close'].pct_change()
        mean_60 = return_.rolling(window1).mean()
        std_30 = return_.rolling(window2).std()
        corr = mean_60.rolling(p2).corr(std_30)
        low = corr.rolling(p2).quantile(p1)
        high = corr.rolling(p2).quantile(1 - p1)
        winsorized = corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_230_rank(df, window=5):
        close = df['close']
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(30).std()
        ts_std = ts_std.replace(0, np.nan)
        # Regression slope: y = ts_mean, x = ts_std, window = 90
        y = ts_mean
        x = ts_std
        cov = y.rolling(90).cov(x)
        var_x = x.rolling(90).var().replace(0, np.nan)
        slope = cov / var_x
        # Rank and normalize
        rank_slope = slope.rolling(180).rank(pct=True) * 2 - 1
        return -rank_slope.fillna(0)

    @staticmethod
    def alpha_quanta_230_tanh(df, window=35):
        close = df['close']
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(30).std()
        ts_std = ts_std.replace(0, np.nan)
        # Regression slope: y = ts_mean, x = ts_std, window = 90
        y = ts_mean
        x = ts_std
        cov = y.rolling(90).cov(x)
        var_x = x.rolling(90).var().replace(0, np.nan)
        slope = cov / var_x
        # Dynamic Tanh
        norm = np.tanh(slope / slope.rolling(180).std().replace(0, np.nan))
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_230_zscore(df, window=5):
        close = df['close']
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(30).std()
        ts_std = ts_std.replace(0, np.nan)
        # Regression slope: y = ts_mean, x = ts_std, window = 90
        y = ts_mean
        x = ts_std
        cov = y.rolling(90).cov(x)
        var_x = x.rolling(90).var().replace(0, np.nan)
        slope = cov / var_x
        # Rolling Z-Score Clip
        z = (slope - slope.rolling(180).mean()) / slope.rolling(180).std().replace(0, np.nan)
        return -z.clip(-1, 1).fillna(0)

    @staticmethod
    def alpha_quanta_230_sign(df, window=35):
        close = df['close']
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(30).std()
        ts_std = ts_std.replace(0, np.nan)
        # Regression slope: y = ts_mean, x = ts_std, window = 90
        y = ts_mean
        x = ts_std
        cov = y.rolling(90).cov(x)
        var_x = x.rolling(90).var().replace(0, np.nan)
        slope = cov / var_x
        # Sign Binary Soft
        return np.sign(slope).fillna(0)

    @staticmethod
    def alpha_quanta_230_wf(df, window=10, p2=60):
        close = df['close']
        ret = close.pct_change()
        ts_mean = ret.rolling(window).mean()
        ts_std = ret.rolling(30).std()
        ts_std = ts_std.replace(0, np.nan)
        # Regression slope: y = ts_mean, x = ts_std, window = 90
        y = ts_mean
        x = ts_std
        cov = y.rolling(90).cov(x)
        var_x = x.rolling(90).var().replace(0, np.nan)
        slope = cov / var_x
        # Winsorized Fisher Transform
        p1 = 0.05
        low = slope.rolling(p2).quantile(p1)
        high = slope.rolling(p2).quantile(1 - p1)
        winsorized = slope.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_231_rank(df, window=35):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        std_30 = ret.rolling(30).std()
        std_90 = ret.rolling(window).std()
        raw = np.sign(mean_ret) * std_30 / (std_90 + 1e-8)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_231_tanh(df, window=20):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        std_30 = ret.rolling(30).std()
        std_90 = ret.rolling(window).std()
        raw = np.sign(mean_ret) * std_30 / (std_90 + 1e-8)
        result = np.tanh(raw / raw.rolling(window).std())
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_231_zscore(df, window=25):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        std_30 = ret.rolling(30).std()
        std_90 = ret.rolling(window).std()
        raw = np.sign(mean_ret) * std_30 / (std_90 + 1e-8)
        result = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_231_sign(df, window=30):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        std_30 = ret.rolling(30).std()
        std_90 = ret.rolling(window).std()
        raw = np.sign(mean_ret) * std_30 / (std_90 + 1e-8)
        result = np.sign(raw)
        result = result.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_231_wf(df, window=20, p1=0.1):
        ret = df['close'].pct_change()
        mean_ret = ret.rolling(window).mean()
        std_30 = ret.rolling(30).std()
        std_90 = ret.rolling(window).std()
        raw = np.sign(mean_ret) * std_30 / (std_90 + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        result = normalized.ffill().fillna(0)
        return result

    @staticmethod
    def alpha_quanta_232_k(df, window=25):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window=window).mean()
        ts_std = ret.rolling(window=window*2).std()
        delta_ts_mean = ts_mean.diff(periods=window)
        raw = delta_ts_mean / (ts_std + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_232_h(df, window=20):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window=window).mean()
        ts_std = ret.rolling(window=window*2).std()
        delta_ts_mean = ts_mean.diff(periods=window)
        raw = delta_ts_mean / (ts_std + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_232_e(df, window=55):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window=window).mean()
        ts_std = ret.rolling(window=window*2).std()
        delta_ts_mean = ts_mean.diff(periods=window)
        raw = delta_ts_mean / (ts_std + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_232_y(df, window=20):
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window=window).mean()
        ts_std = ret.rolling(window=window*2).std()
        delta_ts_mean = ts_mean.diff(periods=window)
        raw = delta_ts_mean / (ts_std + 1e-8)
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_232_r(df, window=40, p1=0.3):
        p2 = window
        ret = df['close'].pct_change()
        ts_mean = ret.rolling(window=window).mean()
        ts_std = ret.rolling(window=window*2).std()
        delta_ts_mean = ts_mean.diff(periods=window)
        raw = delta_ts_mean / (ts_std + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_233_rank(df, window=30):
        ret = df['close'].pct_change()
        ret_lag = ret.shift(5)
        rolling_corr = ret.rolling(window).corr(ret_lag)
        rolling_std = ret.rolling(window).std()
        raw = rolling_corr / (rolling_std + 1e-8)
        rank_raw = raw.rolling(20).rank(pct=True) * 2 - 1
        signal = rank_raw.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_233_tanh(df, window=100):
        ret = df['close'].pct_change()
        ret_lag = ret.shift(5)
        rolling_corr = ret.rolling(window).corr(ret_lag)
        rolling_std = ret.rolling(window).std()
        raw = rolling_corr / (rolling_std + 1e-8)
        signal = np.tanh(raw / raw.rolling(20).std())
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_233_zscore(df, window=30):
        ret = df['close'].pct_change()
        ret_lag = ret.shift(5)
        rolling_corr = ret.rolling(window).corr(ret_lag)
        rolling_std = ret.rolling(window).std()
        raw = rolling_corr / (rolling_std + 1e-8)
        mean_ = raw.rolling(20).mean()
        std_ = raw.rolling(20).std()
        signal = ((raw - mean_) / std_).clip(-1, 1)
        signal = signal.fillna(0).replace([np.inf, -np.inf], 0)
        return -signal

    @staticmethod
    def alpha_quanta_233_sign(df, window=45):
        ret = df['close'].pct_change()
        ret_lag = ret.shift(5)
        rolling_corr = ret.rolling(window).corr(ret_lag)
        rolling_std = ret.rolling(window).std()
        raw = rolling_corr / (rolling_std + 1e-8)
        signal = np.sign(raw)
        signal = pd.Series(signal, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_233_wf(df, window=30, p1=0.1):
        ret = df['close'].pct_change()
        ret_lag = ret.shift(5)
        rolling_corr = ret.rolling(window).corr(ret_lag)
        rolling_std = ret.rolling(window).std()
        raw = rolling_corr / (rolling_std + 1e-8)
        p2 = 20
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return -signal

    @staticmethod
    def alpha_quanta_234_k(df, window=25):
        close = df['close']
        ret = close.pct_change()
        std = ret.rolling(window).std()
        mad = (close - close.rolling(15).mean()).abs().rolling(15).mean()
        raw = std / (mad + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_234_h(df, window=50):
        close = df['close']
        ret = close.pct_change()
        std = ret.rolling(window).std()
        mad = (close - close.rolling(15).mean()).abs().rolling(15).mean()
        raw = std / (mad + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std())
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_234_e(df, window=30):
        close = df['close']
        ret = close.pct_change()
        std = ret.rolling(window).std()
        mad = (close - close.rolling(15).mean()).abs().rolling(15).mean()
        raw = std / (mad + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_234_y(df, window=10):
        close = df['close']
        ret = close.pct_change()
        std = ret.rolling(window).std()
        mad = (close - close.rolling(15).mean()).abs().rolling(15).mean()
        raw = std / (mad + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_234_r(df, window=30, p1=0.1):
        close = df['close']
        ret = close.pct_change()
        std = ret.rolling(window).std()
        mad = (close - close.rolling(15).mean()).abs().rolling(15).mean()
        raw = std / (mad + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_235_rank(df, window=85):
        high = df['high']
        low = df['low']
        raw = (high - low) / (high - low).rolling(10).mean().replace(0, 1e-8) - 1
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_235_tanh(df, window=85):
        high = df['high']
        low = df['low']
        raw = (high - low) / (high - low).rolling(10).mean().replace(0, 1e-8) - 1
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_235_zscore(df, window=80):
        high = df['high']
        low = df['low']
        raw = (high - low) / (high - low).rolling(10).mean().replace(0, 1e-8) - 1
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_235_sign(df):
        high = df['high']
        low = df['low']
        raw = (high - low) / (high - low).rolling(10).mean().replace(0, 1e-8) - 1
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_235_wf(df, window=60):
        high = df['high']
        low = df['low']
        raw = (high - low) / (high - low).rolling(10).mean().replace(0, 1e-8) - 1
        p1 = 0.05
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_236_k(df, window=15):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(5).mean()
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_236_h(df, window=10):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(5).mean()
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_236_p(df, window=5):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(5).mean()
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_236_y(df, window=100):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(window).mean()
        signal = pd.Series(np.sign(raw), index=df.index)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_236_d(df, p1=0.1, p2=20):
        raw = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)).rolling(5).mean()
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_237_rank(df, window=10):
        delta_volume = df['matchingVolume'].diff(1)
        ret = df['close'].pct_change()
        valid = delta_volume.notna() & ret.notna()
        dvol = delta_volume.where(valid, 0)
        ret_clean = ret.where(valid, 0)
        x = dvol.rolling(window)
        y = ret_clean.rolling(window)
        n = x.count()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (dvol * ret_clean).rolling(window).sum()
        sum_x2 = (dvol ** 2).rolling(window).sum()
        sum_y2 = (ret_clean ** 2).rolling(window).sum()
        num = n * sum_xy - sum_x * sum_y
        den = (n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5
        corr = num / den.replace(0, np.nan)
        corr = corr.where((n > window * 0.7) & (den != 0), np.nan).ffill().fillna(0)
        raw = np.sign(corr)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_237_tanh(df, window=10):
        delta_volume = df['matchingVolume'].diff(1)
        ret = df['close'].pct_change()
        valid = delta_volume.notna() & ret.notna()
        dvol = delta_volume.where(valid, 0)
        ret_clean = ret.where(valid, 0)
        x = dvol.rolling(window)
        y = ret_clean.rolling(window)
        n = x.count()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (dvol * ret_clean).rolling(window).sum()
        sum_x2 = (dvol ** 2).rolling(window).sum()
        sum_y2 = (ret_clean ** 2).rolling(window).sum()
        num = n * sum_xy - sum_x * sum_y
        den = (n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5
        corr = num / den.replace(0, np.nan)
        corr = corr.where((n > window * 0.7) & (den != 0), np.nan).ffill().fillna(0)
        raw = np.sign(corr)
        std_val = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        signal = np.tanh(raw / std_val)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_237_zscore(df, window=10):
        delta_volume = df['matchingVolume'].diff(1)
        ret = df['close'].pct_change()
        valid = delta_volume.notna() & ret.notna()
        dvol = delta_volume.where(valid, 0)
        ret_clean = ret.where(valid, 0)
        x = dvol.rolling(window)
        y = ret_clean.rolling(window)
        n = x.count()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (dvol * ret_clean).rolling(window).sum()
        sum_x2 = (dvol ** 2).rolling(window).sum()
        sum_y2 = (ret_clean ** 2).rolling(window).sum()
        num = n * sum_xy - sum_x * sum_y
        den = (n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5
        corr = num / den.replace(0, np.nan)
        corr = corr.where((n > window * 0.7) & (den != 0), np.nan).ffill().fillna(0)
        raw = np.sign(corr)
        mean_val = raw.rolling(window).mean()
        std_val = raw.rolling(window).std().replace(0, np.nan).ffill().fillna(1)
        signal = ((raw - mean_val) / std_val).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_237_sign(df, window=10):
        delta_volume = df['matchingVolume'].diff(1)
        ret = df['close'].pct_change()
        valid = delta_volume.notna() & ret.notna()
        dvol = delta_volume.where(valid, 0)
        ret_clean = ret.where(valid, 0)
        x = dvol.rolling(window)
        y = ret_clean.rolling(window)
        n = x.count()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (dvol * ret_clean).rolling(window).sum()
        sum_x2 = (dvol ** 2).rolling(window).sum()
        sum_y2 = (ret_clean ** 2).rolling(window).sum()
        num = n * sum_xy - sum_x * sum_y
        den = (n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5
        corr = num / den.replace(0, np.nan)
        corr = corr.where((n > window * 0.7) & (den != 0), np.nan).ffill().fillna(0)
        signal = np.sign(corr)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_237_wf(df, p1=0.7, p2=80):
        delta_volume = df['matchingVolume'].diff(1)
        ret = df['close'].pct_change()
        valid = delta_volume.notna() & ret.notna()
        dvol = delta_volume.where(valid, 0)
        ret_clean = ret.where(valid, 0)
        x = dvol.rolling(p2)
        y = ret_clean.rolling(p2)
        n = x.count()
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (dvol * ret_clean).rolling(p2).sum()
        sum_x2 = (dvol ** 2).rolling(p2).sum()
        sum_y2 = (ret_clean ** 2).rolling(p2).sum()
        num = n * sum_xy - sum_x * sum_y
        den = (n * sum_x2 - sum_x ** 2) ** 0.5 * (n * sum_y2 - sum_y ** 2) ** 0.5
        corr = num / den.replace(0, np.nan)
        corr = corr.where((n > p2 * 0.7) & (den != 0), np.nan).ffill().fillna(0)
        raw = np.sign(corr)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_238_8(df, window=100):
        returns = df['close'].pct_change()
        ts_std5 = returns.rolling(5).std()
        ts_mean_std10 = returns.rolling(10).std().rolling(window).mean()
        raw = ts_std5 / (ts_mean_std10 + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_238_rank(df, window=45):
        ret = df['close'].pct_change()
        ts_std_5 = ret.rolling(window).std()
        ts_std_10 = ret.rolling(window * 2).std()
        ts_mean_std = ts_std_10.rolling(window * 2).mean()
        raw = ts_std_5 / (ts_mean_std + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_238_tanh(df, window=75):
        ret = df['close'].pct_change()
        ts_std_5 = ret.rolling(window).std()
        ts_std_10 = ret.rolling(window * 2).std()
        ts_mean_std = ts_std_10.rolling(window * 2).mean()
        raw = ts_std_5 / (ts_mean_std + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_238_zscore(df, window=50):
        ret = df['close'].pct_change()
        ts_std_5 = ret.rolling(window).std()
        ts_std_10 = ret.rolling(window * 2).std()
        ts_mean_std = ts_std_10.rolling(window * 2).mean()
        raw = ts_std_5 / (ts_mean_std + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_238_sign(df, window=5):
        ret = df['close'].pct_change()
        ts_std_5 = ret.rolling(window).std()
        ts_std_10 = ret.rolling(window * 2).std()
        ts_mean_std = ts_std_10.rolling(window * 2).mean()
        raw = ts_std_5 / (ts_mean_std + 1e-8)
        normalized = np.sign(raw)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_238_wf(df, window=30, p1=0.7):
        ret = df['close'].pct_change()
        ts_std_5 = ret.rolling(window).std()
        ts_std_10 = ret.rolling(window * 2).std()
        ts_mean_std = ts_std_10.rolling(window * 2).mean()
        raw = ts_std_5 / (ts_mean_std + 1e-8)
        p2 = window * 4
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_239_rank(df, window=40):
        raw = (df['high'] - df['low']).rolling(window).mean() / ((df['open'] + df['close']) / (df['close'] - df['open']).div(df['open']).rolling(window).std().replace(0, np.nan) + 1e-8)
        denom = ((df['close'] - df['open']) / df['open']).rolling(window).std().replace(0, np.nan)
        raw = ((df['high'] - df['low']).rolling(window).mean() / (df['open'] + df['close'])) / (denom + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_239_tanh(df, window=5):
        raw = (df['high'] - df['low']).rolling(window).mean() / ((df['open'] + df['close']) * 0.5)
        denom = ((df['close'] - df['open']) / df['open']).rolling(window).std().replace(0, np.nan)
        raw_div = raw / (denom + 1e-8)
        normalized = np.tanh(raw_div / raw_div.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_239_zscore(df, window=25):
        raw = (df['high'] - df['low']).rolling(window).mean() / ((df['open'] + df['close']) * 0.5)
        denom = ((df['close'] - df['open']) / df['open']).rolling(window).std().replace(0, np.nan)
        raw_div = raw / (denom + 1e-8)
        zscore = (raw_div - raw_div.rolling(window).mean()) / raw_div.rolling(window).std().replace(0, np.nan)
        normalized = zscore.clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_239_sign(df, window=20):
        raw = (df['high'] - df['low']).rolling(window).mean() / ((df['open'] + df['close']) * 0.5)
        denom = ((df['close'] - df['open']) / df['open']).rolling(window).std().replace(0, np.nan)
        raw_div = raw / (denom + 1e-8)
        normalized = np.sign(raw_div)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_239_wf(df, window=10, sub_window=100):
        raw = (df['high'] - df['low']).rolling(window).mean() / ((df['open'] + df['close']) * 0.5)
        denom = ((df['close'] - df['open']) / df['open']).rolling(window).std().replace(0, np.nan)
        raw = raw / (denom + 1e-8)
        p1 = 0.05
        low = raw.rolling(sub_window).quantile(p1)
        high = raw.rolling(sub_window).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        ratio = ((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99
        ratio = ratio.clip(-0.99, 0.99)  # tránh arctanh vô cực
        normalized = np.arctanh(ratio)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_240_k(df, window=100):
        high_low_range = (df['high'] - df['low']) / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        ret = df['close'].pct_change()
        corr = high_low_range.rolling(window).corr(ret)
        raw = np.sign(corr) * ret.rolling(window).std()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized

    @staticmethod
    def alpha_quanta_240_h(df, window=45):
        high_low_range = (df['high'] - df['low']) / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        ret = df['close'].pct_change()
        corr = high_low_range.rolling(window).corr(ret)
        raw = np.sign(corr) * ret.rolling(window).std()
        std_val = raw.rolling(window).std().replace(0, np.nan)
        normalized = np.tanh(raw / std_val)
        return -normalized

    @staticmethod
    def alpha_quanta_240_e(df, window=10):
        high_low_range = (df['high'] - df['low']) / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        ret = df['close'].pct_change()
        corr = high_low_range.rolling(window).corr(ret)
        raw = np.sign(corr) * ret.rolling(window).std()
        mean_val = raw.rolling(window).mean()
        std_val = raw.rolling(window).std().replace(0, np.nan)
        normalized = ((raw - mean_val) / std_val).clip(-1, 1)
        return normalized

    @staticmethod
    def alpha_quanta_240_y(df, window=50):
        high_low_range = (df['high'] - df['low']) / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        ret = df['close'].pct_change()
        corr = high_low_range.rolling(window).corr(ret)
        raw = np.sign(corr) * ret.rolling(window).std()
        normalized = np.sign(raw)
        return -normalized

    @staticmethod
    def alpha_quanta_240_r(df, window=40, quantile=0.7):
        high_low_range = (df['high'] - df['low']) / (df.get('matchingVolume', df.get('volume', 1)) + 1e-8)
        ret = df['close'].pct_change()
        corr = high_low_range.rolling(window).corr(ret)
        raw = np.sign(corr) * ret.rolling(window).std()
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)).replace(0, np.nan) * 1.98 - 0.99)
        return -normalized

    @staticmethod
    def alpha_quanta_241_rank(df, window_rank=95):
        vol_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        vol_pct = df['matchingVolume'].pct_change(5).fillna(0)
        close_pct = df['close'].pct_change(5).fillna(0)
        corr = vol_pct.rolling(10).corr(close_pct).fillna(0)
        raw = vol_ratio * corr
        rank_val = raw.rolling(window_rank).rank(pct=True) * 2 - 1
        return rank_val.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_241_tanh(df, window_std=100):
        vol_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        vol_pct = df['matchingVolume'].pct_change(5).fillna(0)
        close_pct = df['close'].pct_change(5).fillna(0)
        corr = vol_pct.rolling(10).corr(close_pct).fillna(0)
        raw = vol_ratio * corr
        std = raw.rolling(window_std).std().fillna(1).replace(0, 1)
        normalized = np.tanh(raw / std)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_241_zscore(df, window_z=45):
        vol_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        vol_pct = df['matchingVolume'].pct_change(5).fillna(0)
        close_pct = df['close'].pct_change(5).fillna(0)
        corr = vol_pct.rolling(10).corr(close_pct).fillna(0)
        raw = vol_ratio * corr
        mean = raw.rolling(window_z).mean().fillna(0)
        std = raw.rolling(window_z).std().fillna(1).replace(0, 1)
        normalized = ((raw - mean) / std).clip(-1, 1)
        return normalized.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_241_sign(df, factor=75):
        vol_ratio = df['matchingVolume'].rolling(5).mean() / (df['matchingVolume'].rolling(20).mean() + 1e-8)
        vol_pct = df['matchingVolume'].pct_change(5).fillna(0)
        close_pct = df['close'].pct_change(5).fillna(0)
        corr = vol_pct.rolling(10).corr(close_pct).fillna(0)
        raw = vol_ratio * corr
        normalized = np.sign(raw) * factor
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_241_wf(df, p1=0.1, p2=40):
        vol = df['matchingVolume']
        close = df['close']
        vol_ma5 = vol.rolling(5).mean()
        vol_ma20 = vol.rolling(20).mean()
        vol_ratio = vol_ma5 / (vol_ma20 + 1e-8)
        vol_change = vol.pct_change(5)
        close_change = close.pct_change(5)
        corr = vol_change.rolling(10).corr(close_change)
        raw = vol_ratio * corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        base = (winsorized - low) / (high - low + 1e-9)
        fisher_input = base * 1.98 - 0.99
        fisher_input = fisher_input.clip(-0.99, 0.99)
        sig = np.arctanh(fisher_input)
        return sig.ffill().fillna(0)

    @staticmethod
    def alpha_quanta_242_rank(df, window=50):
        raw = ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).std() / (1e-8 + ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).mean())
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_242_tanh(df, window=95):
        raw = ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).std() / (1e-8 + ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).mean())
        signal = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -signal.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_242_zscore(df, window=35):
        raw = ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).std() / (1e-8 + ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).mean())
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_242_sign(df, window=55):
        raw = ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).std() / (1e-8 + ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).mean())
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_242_wf(df, window=40, quantile=0.7):
        raw = ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).std() / (1e-8 + ((df['high'] - df['low']) / df['close'].replace(0, 1e-8)).rolling(window).mean())
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_243_rank(df, window=90):
        volume = df['matchingVolume']
        raw = volume / volume.rolling(window).mean().replace(0, np.nan)
        raw = raw.ffill().fillna(0)
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_243_tanh(df, window=50):
        volume = df['matchingVolume']
        raw = volume / volume.rolling(window).mean().replace(0, np.nan)
        raw = raw.ffill().fillna(0)
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_243_zscore(df, window=65):
        volume = df['matchingVolume']
        raw = volume / volume.rolling(window).mean().replace(0, np.nan)
        raw = raw.ffill().fillna(0)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - mean) / std).clip(-1, 1)
        return -result.fillna(0)

    @staticmethod
    def alpha_quanta_243_sign(df, window=10):
        volume = df['matchingVolume']
        raw = volume / volume.rolling(window).mean().replace(0, np.nan)
        raw = raw.ffill().fillna(0)
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_243_wf(df, window=70, quantile_param=0.7):
        volume = df['matchingVolume']
        raw = volume / volume.rolling(window).mean().replace(0, np.nan)
        raw = raw.ffill().fillna(0)
        low = raw.rolling(window).quantile(quantile_param)
        high = raw.rolling(window).quantile(1 - quantile_param)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        result = normalized.clip(-1, 1)
        return -result

    @staticmethod
    def alpha_quanta_244_rank(df, window=85):
        log_return -= np.log(df['close'] / df['close'].shift(1))
        high_low_diff = df['high'] - df['low']
        cov = (log_return * high_low_diff).rolling(window).mean() - log_return.rolling(window).mean() * high_low_diff.rolling(window).mean()
        std_log_return = log_return.rolling(window).std()
        std_hl = high_low_diff.rolling(window).std()
        ts_corr = cov / (std_log_return * std_hl + 1e-9)
        ts_corr = ts_corr.fillna(0)
        rank_corr = ts_corr.rolling(window).rank(pct=True) * 2 - 1
        return rank_corr.fillna(0)

    @staticmethod
    def alpha_quanta_244_tanh(df, window=80):
        log_return -= np.log(df['close'] / df['close'].shift(1))
        high_low_diff = df['high'] - df['low']
        cov = (log_return * high_low_diff).rolling(window).mean() - log_return.rolling(window).mean() * high_low_diff.rolling(window).mean()
        std_log_return = log_return.rolling(window).std()
        std_hl = high_low_diff.rolling(window).std()
        ts_corr = cov / (std_log_return * std_hl + 1e-9)
        ts_corr = ts_corr.fillna(0)
        normalized = np.tanh(ts_corr / ts_corr.rolling(window).std().replace(0, np.nan).fillna(1))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_244_zscore(df, window=5):
        log_return = np.log(df['close'] / df['close'].shift(1))
        high_low_diff = df['high'] - df['low']
        cov = (log_return * high_low_diff).rolling(window).mean() - log_return.rolling(window).mean() * high_low_diff.rolling(window).mean()
        std_log_return = log_return.rolling(window).std()
        std_hl = high_low_diff.rolling(window).std()
        ts_corr = cov / (std_log_return * std_hl + 1e-9)
        ts_corr = ts_corr.fillna(0)
        normalized = ((ts_corr - ts_corr.rolling(window).mean()) / ts_corr.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_244_sign(df, window=5):
        log_return = np.log(df['close'] / df['close'].shift(1))
        high_low_diff = df['high'] - df['low']
        cov = (log_return * high_low_diff).rolling(window).mean() - log_return.rolling(window).mean() * high_low_diff.rolling(window).mean()
        std_log_return = log_return.rolling(window).std()
        std_hl = high_low_diff.rolling(window).std()
        ts_corr = cov / (std_log_return * std_hl + 1e-9)
        ts_corr = ts_corr.fillna(0)
        norm = ts_corr.rolling(window).rank(pct=True) * 2 - 1
        signal = np.sign(norm)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_244_wf(df, window=10, p1=0.1):
        p2 = window
        log_return = np.log(df['close'] / df['close'].shift(1))
        high_low_diff = df['high'] - df['low']
        cov = (log_return * high_low_diff).rolling(window).mean() - log_return.rolling(window).mean() * high_low_diff.rolling(window).mean()
        std_log_return = log_return.rolling(window).std()
        std_hl = high_low_diff.rolling(window).std()
        ts_corr = cov / (std_log_return * std_hl + 1e-9)
        ts_corr = ts_corr.fillna(0)
        low = ts_corr.rolling(p2).quantile(p1)
        high = ts_corr.rolling(p2).quantile(1 - p1)
        winsorized = ts_corr.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_245_rank(df, window=60):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_volume = volume.shift(1).fillna(volume)
        vol_ratio = volume / (delay_volume + 1e-8)
        ts_mean = vol_ratio.rolling(window).mean()
        ts_std = vol_ratio.rolling(window).std().replace(0, np.nan)
        zscore = (vol_ratio - ts_mean) / ts_std
        normalized = (zscore.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_245_tanh(df, window=60):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_volume = volume.shift(1).fillna(volume)
        vol_ratio = volume / (delay_volume + 1e-8)
        ts_mean = vol_ratio.rolling(window).mean()
        ts_std = vol_ratio.rolling(window).std().replace(0, np.nan)
        zscore = (vol_ratio - ts_mean) / ts_std
        normalized = np.tanh(zscore / zscore.rolling(window).std().replace(0, np.nan))
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_245_zscore(df, window=25):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_volume = volume.shift(1).fillna(volume)
        vol_ratio = volume / (delay_volume + 1e-8)
        ts_mean = vol_ratio.rolling(window).mean()
        ts_std = vol_ratio.rolling(window).std().replace(0, np.nan)
        zscore = (vol_ratio - ts_mean) / ts_std
        normalized = ((zscore - zscore.rolling(window).mean()) / zscore.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_245_sign(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_volume = volume.shift(1).fillna(volume)
        vol_ratio = volume / (delay_volume + 1e-8)
        ts_mean = vol_ratio.rolling(window).mean()
        ts_std = vol_ratio.rolling(window).std().replace(0, np.nan)
        zscore = (vol_ratio - ts_mean) / ts_std
        normalized = np.sign(zscore)
        normalized = pd.Series(normalized, index=df.index).ffill().fillna(0)
        return -normalized

    @staticmethod
    def alpha_quanta_245_wf(df, window=60, p1=0.7):
        p2 = window
        volume = df.get('matchingVolume', df.get('volume', 1))
        delay_volume = volume.shift(1).fillna(volume)
        vol_ratio = volume / (delay_volume + 1e-8)
        ts_mean = vol_ratio.rolling(window).mean()
        ts_std = vol_ratio.rolling(window).std().replace(0, np.nan)
        zscore = (vol_ratio - ts_mean) / ts_std
        raw = zscore
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.ffill().fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_246_rank(df, window=75):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        vol_zscore = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std().replace(0, np.nan))
        vol_zz = ((vol_zscore - vol_zscore.rolling(window).mean()) / vol_zscore.rolling(window).std().replace(0, np.nan))

        returns = close.pct_change()
        hl_range = high - low
        corr = returns.rolling(window).corr(hl_range)
        rank_corr = corr.rolling(window).rank(pct=True)
        z_corr = ((rank_corr - rank_corr.rolling(window).mean()) / rank_corr.rolling(window).std().replace(0, np.nan))

        oc_ratio = (open_price - close.shift(1)).abs() / (hl_range + 1e-8)
        mean_oc = oc_ratio.rolling(window=5).mean()
        z_oc = ((mean_oc - mean_oc.rolling(window).mean()) / mean_oc.rolling(window).std().replace(0, np.nan))
        z_oc2 = ((z_oc - z_oc.rolling(window).mean()) / z_oc.rolling(window).std().replace(0, np.nan))

        vol_ratio2 = volume / (volume.shift(1) + 1e-8)
        mean_vol2 = vol_ratio2.rolling(window=5).mean()
        z_vol2 = ((mean_vol2 - mean_vol2.rolling(window).mean()) / mean_vol2.rolling(window).std().replace(0, np.nan))
        z_vol22 = ((z_vol2 - z_vol2.rolling(window).mean()) / z_vol2.rolling(window).std().replace(0, np.nan))

        raw = vol_zz + z_corr + z_oc2 + z_vol22
        normalized = raw.rolling(window).rank(pct=True) * 2 - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_246_tanh(df, window=65):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        vol_zscore = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std().replace(0, np.nan))
        vol_zz = ((vol_zscore - vol_zscore.rolling(window).mean()) / vol_zscore.rolling(window).std().replace(0, np.nan))

        returns = close.pct_change()
        hl_range = high - low
        corr = returns.rolling(window).corr(hl_range)
        rank_corr = corr.rolling(window).rank(pct=True)
        z_corr = ((rank_corr - rank_corr.rolling(window).mean()) / rank_corr.rolling(window).std().replace(0, np.nan))

        oc_ratio = (open_price - close.shift(1)).abs() / (hl_range + 1e-8)
        mean_oc = oc_ratio.rolling(window=5).mean()
        z_oc = ((mean_oc - mean_oc.rolling(window).mean()) / mean_oc.rolling(window).std().replace(0, np.nan))
        z_oc2 = ((z_oc - z_oc.rolling(window).mean()) / z_oc.rolling(window).std().replace(0, np.nan))

        vol_ratio2 = volume / (volume.shift(1) + 1e-8)
        mean_vol2 = vol_ratio2.rolling(window=5).mean()
        z_vol2 = ((mean_vol2 - mean_vol2.rolling(window).mean()) / mean_vol2.rolling(window).std().replace(0, np.nan))
        z_vol22 = ((z_vol2 - z_vol2.rolling(window).mean()) / z_vol2.rolling(window).std().replace(0, np.nan))

        raw = vol_zz + z_corr + z_oc2 + z_vol22
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_246_zscore(df, window=10):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        vol_zscore = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std().replace(0, np.nan))
        vol_zz = ((vol_zscore - vol_zscore.rolling(window).mean()) / vol_zscore.rolling(window).std().replace(0, np.nan))

        returns = close.pct_change()
        hl_range = high - low
        corr = returns.rolling(window).corr(hl_range)
        rank_corr = corr.rolling(window).rank(pct=True)
        z_corr = ((rank_corr - rank_corr.rolling(window).mean()) / rank_corr.rolling(window).std().replace(0, np.nan))

        oc_ratio = (open_price - close.shift(1)).abs() / (hl_range + 1e-8)
        mean_oc = oc_ratio.rolling(window=5).mean()
        z_oc = ((mean_oc - mean_oc.rolling(window).mean()) / mean_oc.rolling(window).std().replace(0, np.nan))
        z_oc2 = ((z_oc - z_oc.rolling(window).mean()) / z_oc.rolling(window).std().replace(0, np.nan))

        vol_ratio2 = volume / (volume.shift(1) + 1e-8)
        mean_vol2 = vol_ratio2.rolling(window=5).mean()
        z_vol2 = ((mean_vol2 - mean_vol2.rolling(window).mean()) / mean_vol2.rolling(window).std().replace(0, np.nan))
        z_vol22 = ((z_vol2 - z_vol2.rolling(window).mean()) / z_vol2.rolling(window).std().replace(0, np.nan))

        raw = vol_zz + z_corr + z_oc2 + z_vol22
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_246_sign(df, window=20):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        vol_zscore = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std().replace(0, np.nan))
        vol_zz = ((vol_zscore - vol_zscore.rolling(window).mean()) / vol_zscore.rolling(window).std().replace(0, np.nan))

        returns = close.pct_change()
        hl_range = high - low
        corr = returns.rolling(window).corr(hl_range)
        rank_corr = corr.rolling(window).rank(pct=True)
        z_corr = ((rank_corr - rank_corr.rolling(window).mean()) / rank_corr.rolling(window).std().replace(0, np.nan))

        oc_ratio = (open_price - close.shift(1)).abs() / (hl_range + 1e-8)
        mean_oc = oc_ratio.rolling(window=5).mean()
        z_oc = ((mean_oc - mean_oc.rolling(window).mean()) / mean_oc.rolling(window).std().replace(0, np.nan))
        z_oc2 = ((z_oc - z_oc.rolling(window).mean()) / z_oc.rolling(window).std().replace(0, np.nan))

        vol_ratio2 = volume / (volume.shift(1) + 1e-8)
        mean_vol2 = vol_ratio2.rolling(window=5).mean()
        z_vol2 = ((mean_vol2 - mean_vol2.rolling(window).mean()) / mean_vol2.rolling(window).std().replace(0, np.nan))
        z_vol22 = ((z_vol2 - z_vol2.rolling(window).mean()) / z_vol2.rolling(window).std().replace(0, np.nan))

        raw = vol_zz + z_corr + z_oc2 + z_vol22
        normalized = np.sign(raw)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_246_wf(df, window=10, p1=0.1):
        volume = df.get('matchingVolume', df.get('volume', 1))
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']

        vol_ratio = volume / (volume.rolling(window).mean() + 1e-8)
        vol_zscore = ((vol_ratio - vol_ratio.rolling(window).mean()) / vol_ratio.rolling(window).std().replace(0, np.nan))
        vol_zz = ((vol_zscore - vol_zscore.rolling(window).mean()) / vol_zscore.rolling(window).std().replace(0, np.nan))

        returns = close.pct_change()
        hl_range = high - low
        corr = returns.rolling(window).corr(hl_range)
        rank_corr = corr.rolling(window).rank(pct=True)
        z_corr = ((rank_corr - rank_corr.rolling(window).mean()) / rank_corr.rolling(window).std().replace(0, np.nan))

        oc_ratio = (open_price - close.shift(1)).abs() / (hl_range + 1e-8)
        mean_oc = oc_ratio.rolling(window=5).mean()
        z_oc = ((mean_oc - mean_oc.rolling(window).mean()) / mean_oc.rolling(window).std().replace(0, np.nan))
        z_oc2 = ((z_oc - z_oc.rolling(window).mean()) / z_oc.rolling(window).std().replace(0, np.nan))

        vol_ratio2 = volume / (volume.shift(1) + 1e-8)
        mean_vol2 = vol_ratio2.rolling(window=5).mean()
        z_vol2 = ((mean_vol2 - mean_vol2.rolling(window).mean()) / mean_vol2.rolling(window).std().replace(0, np.nan))
        z_vol22 = ((z_vol2 - z_vol2.rolling(window).mean()) / z_vol2.rolling(window).std().replace(0, np.nan))

        raw = vol_zz + z_corr + z_oc2 + z_vol22
        p2 = window
        low_q = raw.rolling(p2).quantile(p1)
        high_q = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        normalized = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_247_rank(df, window_rank=10):
        delta_vol = df['matchingVolume'].diff().fillna(0) / (df['matchingVolume'] + 1e-8)
        ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = delta_vol.rolling(window_rank).cov(ret) / (delta_vol.rolling(window_rank).std() * ret.rolling(window_rank).std() + 1e-8)
        raw = (corr - corr.rolling(window_rank).mean()) / corr.rolling(window_rank).std()
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_247_tanh(df, window_tanh=10):
        delta_vol = df['matchingVolume'].diff().fillna(0) / (df['matchingVolume'] + 1e-8)
        ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = delta_vol.rolling(window_tanh).cov(ret) / (delta_vol.rolling(window_tanh).std() * ret.rolling(window_tanh).std() + 1e-8)
        raw = (corr - corr.rolling(window_tanh).mean()) / corr.rolling(window_tanh).std()
        signal = np.tanh(raw / (raw.rolling(window_tanh).std() + 1e-8))
        return signal.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_247_zscore(df, window_zscore=10):
        delta_vol = df['matchingVolume'].diff().fillna(0) / (df['matchingVolume'] + 1e-8)
        ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = delta_vol.rolling(window_zscore).cov(ret) / (delta_vol.rolling(window_zscore).std() * ret.rolling(window_zscore).std() + 1e-8)
        raw = (corr - corr.rolling(window_zscore).mean()) / corr.rolling(window_zscore).std()
        signal = ((raw - raw.rolling(window_zscore).mean()) / raw.rolling(window_zscore).std()).clip(-1, 1)
        return signal.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def alpha_quanta_247_sign(df, window_sign=15):
        delta_vol = df['matchingVolume'].diff().fillna(0) / (df['matchingVolume'] + 1e-8)
        ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = delta_vol.rolling(window_sign).cov(ret) / (delta_vol.rolling(window_sign).std() * ret.rolling(window_sign).std() + 1e-8)
        raw = (corr - corr.rolling(window_sign).mean()) / corr.rolling(window_sign).std()
        signal = np.sign(raw)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_247_wf(df, p1_winsor=0.3, p2_roll=20):
        delta_vol = df['matchingVolume'].diff().fillna(0) / (df['matchingVolume'] + 1e-8)
        ret = (df['close'] - df['open']) / (df['open'] + 1e-8)
        corr = delta_vol.rolling(p2_roll).cov(ret) / (delta_vol.rolling(p2_roll).std() * ret.rolling(p2_roll).std() + 1e-8)
        raw = (corr - corr.rolling(p2_roll).mean()) / corr.rolling(p2_roll).std()
        win_low = raw.rolling(p2_roll).quantile(p1_winsor)
        win_high = raw.rolling(p2_roll).quantile(1 - p1_winsor)
        winsorized = raw.clip(lower=win_low, upper=win_high, axis=0)
        normalized = np.arctanh(((winsorized - win_low) / (win_high - win_low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0).replace([np.inf, -np.inf], 0).clip(-1, 1)
        return signal

    @staticmethod
    def alpha_quanta_248_rank(df, window=45):
        close = df['close']
        ret = close.pct_change()
        seq = np.arange(len(df))
        seq_series = pd.Series(seq, index=df.index)
        corr_raw = ret.rolling(window).corr(seq_series)
        corr_mean = corr_raw.rolling(5).mean()
        zscore = (corr_mean - corr_mean.rolling(window).mean()) / corr_mean.rolling(window).std().replace(0, np.nan)
        signal = zscore.ffill().fillna(0)
        norm = ((signal.rolling(window).rank(pct=True) * 2) - 1).ffill().fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_248_tanh(df, window=65):
        close = df['close']
        ret = close.pct_change()
        seq = np.arange(len(df))
        seq_series = pd.Series(seq, index=df.index)
        corr_raw = ret.rolling(window).corr(seq_series)
        corr_mean = corr_raw.rolling(5).mean()
        raw = corr_mean.ffill().fillna(0)
        norm = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan)).ffill().fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_248_zscore(df, window=5):
        close = df['close']
        ret = close.pct_change()
        seq = np.arange(len(df))
        seq_series = pd.Series(seq, index=df.index)
        corr_raw = ret.rolling(window).corr(seq_series)
        corr_mean = corr_raw.rolling(5).mean()
        zscore = (corr_mean - corr_mean.rolling(window).mean()) / corr_mean.rolling(window).std().replace(0, np.nan)
        norm = zscore.ffill().fillna(0).clip(-1, 1)
        return norm

    @staticmethod
    def alpha_quanta_248_sign(df, window=100):
        close = df['close']
        ret = close.pct_change()
        seq = np.arange(len(df))
        seq_series = pd.Series(seq, index=df.index)
        corr_raw = ret.rolling(window).corr(seq_series)
        corr_mean = corr_raw.rolling(5).mean()
        raw = corr_mean.ffill().fillna(0)
        norm = np.sign(raw).astype(float).ffill().fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_248_wf(df, window=80, factor=0.3):
        close = df['close']
        ret = close.pct_change()
        seq = np.arange(len(df))
        seq_series = pd.Series(seq, index=df.index)
        corr_raw = ret.rolling(window).corr(seq_series)
        corr_mean = corr_raw.rolling(5).mean()
        raw = corr_mean.ffill().fillna(0)
        p1 = factor
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        eps = 1e-9
        normalized = np.arctanh(((winsorized - low) / (high - low + eps)) * 1.98 - 0.99)
        norm = normalized.ffill().fillna(0)
        return norm

    @staticmethod
    def alpha_quanta_249_rank(df, window=25):
        ret = df['close'].pct_change()
        raw = (1.0 / (ret.rolling(window).std() + 1e-8)) * ret.rolling(window).mean()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_249_tanh(df, window=15):
        ret = df['close'].pct_change()
        raw = (1.0 / (ret.rolling(window).std() + 1e-8)) * ret.rolling(window).mean()
        normalized = np.tanh(raw / raw.rolling(window).std())
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_249_zscore(df, window=25):
        ret = df['close'].pct_change()
        raw = (1.0 / (ret.rolling(window).std() + 1e-8)) * ret.rolling(window).mean()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_249_sign(df, window=30):
        ret = df['close'].pct_change()
        raw = (1.0 / (ret.rolling(window).std() + 1e-8)) * ret.rolling(window).mean()
        normalized = np.sign(raw)
        return pd.Series(normalized, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_249_wf(df, window=40, quantile=0.3):
        ret = df['close'].pct_change()
        raw = (1.0 / (ret.rolling(window).std() + 1e-8)) * ret.rolling(window).mean()
        low = raw.rolling(window).quantile(quantile)
        high = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_250_k(df, window=25):
        # Tính tổng hợp các thành phần
        frag_5d = df['close'].diff(1).rolling(5).std() / df['close'].rolling(5).mean()
        inst_10d = (df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close'].rolling(10).std()
        fund_20d = df['close'].rolling(20).mean() / df['close'].rolling(20).std()
        vol_20d = df['close'].pct_change().rolling(20).std()

        raw = 0.25 * (frag_5d + inst_10d + fund_20d + vol_20d)

        # Rolling Rank chuẩn hóa
        rank = raw.rolling(window).rank(pct=True)
        signal = rank * 2 - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_250_h(df, window=30):
        # Tính tổng hợp các thành phần
        frag_5d = df['close'].diff(1).rolling(5).std() / df['close'].rolling(5).mean()
        inst_10d = (df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close'].rolling(10).std()
        fund_20d = df['close'].rolling(20).mean() / df['close'].rolling(20).std()
        vol_20d = df['close'].pct_change().rolling(20).std()

        raw = 0.25 * (frag_5d + inst_10d + fund_20d + vol_20d)

        # Dynamic Tanh chuẩn hóa
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        signal = np.tanh(raw / rolling_std)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_250_e(df, window=30):
        # Tính tổng hợp các thành phần
        frag_5d = df['close'].diff(1).rolling(5).std() / df['close'].rolling(5).mean()
        inst_10d = (df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close'].rolling(10).std()
        fund_20d = df['close'].rolling(20).mean() / df['close'].rolling(20).std()
        vol_20d = df['close'].pct_change().rolling(20).std()

        raw = 0.25 * (frag_5d + inst_10d + fund_20d + vol_20d)

        # Rolling Z-Score với clip
        rolling_mean = raw.rolling(window).mean()
        rolling_std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - rolling_mean) / rolling_std).clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_250_y(df):
        # Tính tổng hợp các thành phần
        frag_5d = df['close'].diff(1).rolling(5).std() / df['close'].rolling(5).mean()
        inst_10d = (df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close'].rolling(10).std()
        fund_20d = df['close'].rolling(20).mean() / df['close'].rolling(20).std()
        vol_20d = df['close'].pct_change().rolling(20).std()

        raw = 0.25 * (frag_5d + inst_10d + fund_20d + vol_20d)

        # Sign/Binary Soft
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_250_r(df, window=50):
        # Tính tổng hợp các thành phần
        frag_5d = df['close'].diff(1).rolling(5).std() / df['close'].rolling(5).mean()
        inst_10d = (df['close'].rolling(10).mean() - df['close'].rolling(20).mean()) / df['close'].rolling(10).std()
        fund_20d = df['close'].rolling(20).mean() / df['close'].rolling(20).std()
        vol_20d = df['close'].pct_change().rolling(20).std()

        raw = 0.25 * (frag_5d + inst_10d + fund_20d + vol_20d)

        # Winsorized Fisher Transform
        p1 = 0.05
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_251_rank(df, window_rank=15):
        ret = df['close'].pct_change()
        mean_45 = ret.rolling(window_rank).mean()
        std_90 = ret.rolling(window_rank * 2).std()
        raw = mean_45 / (std_90 + 1e-8)
        signal = (raw.rolling(window_rank).rank(pct=True) * 2) - 1
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_251_tanh(df, window_rank=15):
        ret = df['close'].pct_change()
        mean_45 = ret.rolling(window_rank).mean()
        std_90 = ret.rolling(window_rank * 2).std()
        raw = mean_45 / (std_90 + 1e-8)
        signal = np.tanh(raw / raw.rolling(window_rank).std().replace(0, np.nan))
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_251_zscore(df, window_rank=25):
        ret = df['close'].pct_change()
        mean_45 = ret.rolling(window_rank).mean()
        std_90 = ret.rolling(window_rank * 2).std()
        raw = mean_45 / (std_90 + 1e-8)
        signal = ((raw - raw.rolling(window_rank).mean()) / raw.rolling(window_rank).std().replace(0, np.nan)).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_251_sign(df, window_rank=30):
        ret = df['close'].pct_change()
        mean_45 = ret.rolling(window_rank).mean()
        std_90 = ret.rolling(window_rank * 2).std()
        raw = mean_45 / (std_90 + 1e-8)
        signal = np.sign(raw)
        return pd.Series(signal, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_251_wf(df, window_rank=25):
        ret = df['close'].pct_change()
        mean_45 = ret.rolling(window_rank).mean()
        std_90 = ret.rolling(window_rank * 2).std()
        raw = mean_45 / (std_90 + 1e-8)
        low = raw.rolling(window_rank).quantile(0.05)
        high = raw.rolling(window_rank).quantile(0.95)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        signal = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_252_k(df, window=20, sub_window=10):
        # Tính range normalized
        hilo_range = df['high'] - df['low']
        hilo_std = hilo_range.rolling(sub_window).std().replace(0, np.nan)
        normalized_range = hilo_range / (hilo_std + 1e-8)

        # Tính volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (volume_mean + 1e-8)

        # Tính rolling correlation
        corr = normalized_range.rolling(window).corr(volume_ratio)

        # Chuẩn hóa bằng Rolling Rank (Trường hợp A)
        raw = corr
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_252_h(df, window=3, sub_window=10):
        # Tính range normalized
        hilo_range = df['high'] - df['low']
        hilo_std = hilo_range.rolling(sub_window).std().replace(0, np.nan)
        normalized_range = hilo_range / (hilo_std + 1e-8)

        # Tính volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (volume_mean + 1e-8)

        # Tính rolling correlation
        corr = normalized_range.rolling(window).corr(volume_ratio)

        # Chuẩn hóa bằng Dynamic Tanh (Trường hợp B)
        raw = corr
        signal = np.tanh(raw / (raw.rolling(window).std().replace(0, np.nan) + 1e-8))
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_252_p(df, window=40, sub_window=50):
        # Tính range normalized
        hilo_range = df['high'] - df['low']
        hilo_std = hilo_range.rolling(sub_window).std().replace(0, np.nan)
        normalized_range = hilo_range / (hilo_std + 1e-8)

        # Tính volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (volume_mean + 1e-8)

        # Tính rolling correlation
        corr = normalized_range.rolling(window).corr(volume_ratio)

        # Chuẩn hóa bằng Rolling Z-Score/Clip (Trường hợp C)
        raw = corr
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std().replace(0, np.nan)
        signal = ((raw - mean) / std).clip(-1, 1)
        return signal.fillna(0)

    @staticmethod
    def alpha_quanta_252_y(df, window=3, sub_window=10):
        # Tính range normalized
        hilo_range = df['high'] - df['low']
        hilo_std = hilo_range.rolling(sub_window).std().replace(0, np.nan)
        normalized_range = hilo_range / (hilo_std + 1e-8)

        # Tính volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (volume_mean + 1e-8)

        # Tính rolling correlation
        corr = normalized_range.rolling(window).corr(volume_ratio)

        # Chuẩn hóa bằng Sign/Binary Soft (Trường hợp D)
        raw = corr
        signal = pd.Series(np.sign(raw), index=df.index)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_252_r(df, window=10, sub_window=30, p1=0.05, p2=50):
        # Tính range normalized
        hilo_range = df['high'] - df['low']
        hilo_std = hilo_range.rolling(sub_window).std().replace(0, np.nan)
        normalized_range = hilo_range / (hilo_std + 1e-8)

        # Tính volume ratio
        volume_mean = df.get('matchingVolume', df.get('volume', 1)).rolling(sub_window).mean()
        volume_ratio = df.get('matchingVolume', df.get('volume', 1)) / (volume_mean + 1e-8)

        # Tính rolling correlation
        corr = normalized_range.rolling(window).corr(volume_ratio)

        # Chuẩn hóa bằng Winsorized Fisher (Trường hợp E)
        raw = corr
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.clip(-1, 1)
        return -signal.fillna(0)

    @staticmethod
    def alpha_quanta_253_k(df, window=5):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        ret = (close - open) / (open + 1e-8)
        vol_mean = volume.rolling(window).mean() + 1e-8
        vol_ratio = volume / vol_mean
        corr = ret.rolling(window).corr(vol_ratio).rank(pct=True) * 2 - 1
        return corr.fillna(0)

    @staticmethod
    def alpha_quanta_253_h(df, window=5):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        ret = (close - open) / (open + 1e-8)
        vol_mean = volume.rolling(window).mean() + 1e-8
        vol_ratio = volume / vol_mean
        corr = ret.rolling(window).corr(vol_ratio)
        raw = corr
        std = raw.rolling(window).std()
        return np.tanh(raw / std.replace(0, np.nan)).fillna(0)

    @staticmethod
    def alpha_quanta_253_e(df, window=10):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        ret = (close - open) / (open + 1e-8)
        vol_mean = volume.rolling(window).mean() + 1e-8
        vol_ratio = volume / vol_mean
        corr = ret.rolling(window).corr(vol_ratio)
        mean = corr.rolling(window).mean()
        std = corr.rolling(window).std()
        return ((corr - mean) / std.replace(0, np.nan)).fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_253_y(df, window=75):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        ret = (close - open) / (open + 1e-8)
        vol_mean = volume.rolling(window).mean() + 1e-8
        vol_ratio = volume / vol_mean
        corr = ret.rolling(window).corr(vol_ratio)
        return -np.sign(corr).fillna(0)

    @staticmethod
    def alpha_quanta_253_r(df, window=10, p1=0.1):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        ret = (close - open) / (open + 1e-8)
        vol_mean = volume.rolling(window).mean() + 1e-8
        vol_ratio = volume / vol_mean
        raw = ret.rolling(window).corr(vol_ratio)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        numerator = (winsorized - low) / (high - low + 1e-9)
        normalized = np.arctanh(numerator * 1.98 - 0.99)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_254_rank(df, window_std=50):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = ret.rolling(window_std).std() / (ret.rolling(window_std).std().rolling(window_std * 3).mean() + 1e-8)
        norm = (ratio.rolling(window_std * 2).rank(pct=True) * 2) - 1
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_254_tanh(df, window_std=85):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = ret.rolling(window_std).std() / (ret.rolling(window_std).std().rolling(window_std * 3).mean() + 1e-8)
        norm = np.tanh(ratio / (ratio.rolling(window_std * 2).std() + 1e-8))
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_254_zscore(df, window_std=50):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = ret.rolling(window_std).std() / (ret.rolling(window_std).std().rolling(window_std * 3).mean() + 1e-8)
        norm = ((ratio - ratio.rolling(window_std).mean()) / ratio.rolling(window_std).std()).clip(-1, 1)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_254_sign(df, window_std=5):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = ret.rolling(window_std).std() / (ret.rolling(window_std).std().rolling(window_std * 3).mean() + 1e-8)
        norm = np.sign(ratio - ratio.rolling(window_std).median())
        return pd.Series(norm, index=df.index).fillna(0)

    @staticmethod
    def alpha_quanta_254_wf(df, window_std=40):
        ret = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        ratio = ret.rolling(window_std).std() / (ret.rolling(window_std).std().rolling(window_std * 3).mean() + 1e-8)
        p1 = 0.1
        p2 = window_std * 2
        low = ratio.rolling(p2).quantile(p1)
        high = ratio.rolling(p2).quantile(1 - p1)
        winsorized = ratio.clip(lower=low, upper=high, axis=0)
        norm = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_255_rank(df, window=20):
        # Tính Fundamental Momentum Quality (45,90)
        momentum_45 = df['close'].pct_change(45)
        momentum_90 = df['close'].pct_change(90)
        fundamental_momentum_quality = (momentum_45 + momentum_90) / 2
        # Tính Volatility Regime Weight 20D
        ret = df['close'].pct_change()
        vol = ret.rolling(window).std()
        vol_regime_weight = 1 - (vol / vol.rolling(window).expanding().max())
        vol_regime_weight = vol_regime_weight.fillna(0)
        # Tính Microstructure Fragmentation Composite (5,20)
        spread = (df['high'] - df['low']) / (df['close'] + 1e-9)
        fragmentation_5 = spread.rolling(5).apply(lambda x: np.cov(x, rowvar=False)[0,0], raw=True)  # Sử dụng variance như proxy
        fragmentation_20 = spread.rolling(20).apply(lambda x: np.cov(x, rowvar=False)[0,0], raw=True)
        microstructure_fragmentation = fragmentation_5 + fragmentation_20
        # Tính Session Accumulation Signal
        session_accumulation = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        # Tính TS_MEAN(Volatility_Regime_Weight_20D, 5)
        ts_mean_vol = vol_regime_weight.rolling(5).mean()
        # Tổng hợp
        raw = (fundamental_momentum_quality * (1 - vol_regime_weight) + microstructure_fragmentation * vol_regime_weight + session_accumulation * ts_mean_vol) / 3
        # Chuẩn hóa Rolling Rank (A)
        normalized = (raw.rolling(20).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_255_tanh(df, window=20):
        momentum_45 = df['close'].pct_change(45)
        momentum_90 = df['close'].pct_change(90)
        fundamental_momentum_quality = (momentum_45 + momentum_90) / 2
        ret = df['close'].pct_change()
        vol = ret.rolling(window).std()
        vol_regime_weight = 1 - (vol / vol.rolling(window).expanding().max())
        vol_regime_weight = vol_regime_weight.fillna(0)
        spread = (df['high'] - df['low']) / (df['close'] + 1e-9)
        fragmentation_5 = spread.rolling(5).var()
        fragmentation_20 = spread.rolling(20).var()
        microstructure_fragmentation = fragmentation_5 + fragmentation_20
        session_accumulation = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        ts_mean_vol = vol_regime_weight.rolling(5).mean()
        raw = (fundamental_momentum_quality * (1 - vol_regime_weight) + microstructure_fragmentation * vol_regime_weight + session_accumulation * ts_mean_vol) / 3
        # Chuẩn hóa Dynamic Tanh (B)
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_255_zscore(df, window=20):
        momentum_45 = df['close'].pct_change(45)
        momentum_90 = df['close'].pct_change(90)
        fundamental_momentum_quality = (momentum_45 + momentum_90) / 2
        ret = df['close'].pct_change()
        vol = ret.rolling(window).std()
        vol_regime_weight = 1 - (vol / vol.rolling(window).expanding().max())
        vol_regime_weight = vol_regime_weight.fillna(0)
        spread = (df['high'] - df['low']) / (df['close'] + 1e-9)
        fragmentation_5 = spread.rolling(5).var()
        fragmentation_20 = spread.rolling(20).var()
        microstructure_fragmentation = fragmentation_5 + fragmentation_20
        session_accumulation = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        ts_mean_vol = vol_regime_weight.rolling(5).mean()
        raw = (fundamental_momentum_quality * (1 - vol_regime_weight) + microstructure_fragmentation * vol_regime_weight + session_accumulation * ts_mean_vol) / 3
        # Chuẩn hóa Rolling Z-Score/Clip (C)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_255_sign(df, window=20):
        momentum_45 = df['close'].pct_change(45)
        momentum_90 = df['close'].pct_change(90)
        fundamental_momentum_quality = (momentum_45 + momentum_90) / 2
        ret = df['close'].pct_change()
        vol = ret.rolling(window).std()
        vol_regime_weight = 1 - (vol / vol.rolling(window).expanding().max())
        vol_regime_weight = vol_regime_weight.fillna(0)
        spread = (df['high'] - df['low']) / (df['close'] + 1e-9)
        fragmentation_5 = spread.rolling(5).var()
        fragmentation_20 = spread.rolling(20).var()
        microstructure_fragmentation = fragmentation_5 + fragmentation_20
        session_accumulation = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        ts_mean_vol = vol_regime_weight.rolling(5).mean()
        raw = (fundamental_momentum_quality * (1 - vol_regime_weight) + microstructure_fragmentation * vol_regime_weight + session_accumulation * ts_mean_vol) / 3
        # Chuẩn hóa Sign/Binary Soft (D)
        normalized = np.sign(raw)
        return normalized

    @staticmethod
    def alpha_quanta_255_wf(df, window=20, quantile_percent=0.05):
        momentum_45 = df['close'].pct_change(45)
        momentum_90 = df['close'].pct_change(90)
        fundamental_momentum_quality = (momentum_45 + momentum_90) / 2
        ret = df['close'].pct_change()
        vol = ret.rolling(window).std()
        vol_regime_weight = 1 - (vol / vol.rolling(window).expanding().max())
        vol_regime_weight = vol_regime_weight.fillna(0)
        spread = (df['high'] - df['low']) / (df['close'] + 1e-9)
        fragmentation_5 = spread.rolling(5).var()
        fragmentation_20 = spread.rolling(20).var()
        microstructure_fragmentation = fragmentation_5 + fragmentation_20
        session_accumulation = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-9)
        ts_mean_vol = vol_regime_weight.rolling(5).mean()
        raw = (fundamental_momentum_quality * (1 - vol_regime_weight) + microstructure_fragmentation * vol_regime_weight + session_accumulation * ts_mean_vol) / 3
        # Chuẩn hóa Winsorized Fisher (E)
        p1 = quantile_percent
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return normalized

    @staticmethod
    def alpha_quanta_256_k(df, window=50):
        high_low = df['high'] - df['low']
        ts_std = high_low.rolling(window).std()
        ts_mean = high_low.rolling(window).mean()
        raw = ts_std / (ts_mean + 1e-8)
        signal = ((raw.rolling(window).rank(pct=True)) * 2 - 1).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_256_h(df, window=80):
        high_low = df['high'] - df['low']
        ts_std = high_low.rolling(window).std()
        ts_mean = high_low.rolling(window).mean()
        raw = ts_std / (ts_mean + 1e-8)
        signal = np.tanh(raw / raw.rolling(window).std()).ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_256_p(df, window=35):
        high_low = df['high'] - df['low']
        ts_std = high_low.rolling(window).std()
        ts_mean = high_low.rolling(window).mean()
        raw = ts_std / (ts_mean + 1e-8)
        signal = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_256_y(df, window=85):
        high_low = df['high'] - df['low']
        ts_std = high_low.rolling(window).std()
        ts_mean = high_low.rolling(window).mean()
        raw = ts_std / (ts_mean + 1e-8)
        signal = (pd.Series(np.sign(raw), index=df.index) * 2 - 1).ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_256_r(df, window=100, p1=0.7, p2=20):
        high_low = df['high'] - df['low']
        ts_std = high_low.rolling(window).std()
        ts_mean = high_low.rolling(window).mean()
        raw = ts_std / (ts_mean + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_257_k(df, window=50):
        ret = df['close'].pct_change()
        vol_delta = df.get('matchingVolume', df['close'] * df.get('matchingVolume', 1)).diff()
        abs_ret = ret.abs()
        abs_vol_delta = vol_delta.abs()
        corr = abs_ret.rolling(window).corr(abs_vol_delta)
        raw = np.sign(corr) * corr.abs()
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_257_h(df, window=40):
        ret = df['close'].pct_change()
        vol_delta = df.get('matchingVolume', df['close'] * df.get('matchingVolume', 1)).diff()
        abs_ret = ret.abs()
        abs_vol_delta = vol_delta.abs()
        corr = abs_ret.rolling(window).corr(abs_vol_delta)
        raw = np.sign(corr) * corr.abs()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_257_e(df, window=50):
        ret = df['close'].pct_change()
        vol_delta = df.get('matchingVolume', df['close'] * df.get('matchingVolume', 1)).diff()
        abs_ret = ret.abs()
        abs_vol_delta = vol_delta.abs()
        corr = abs_ret.rolling(window).corr(abs_vol_delta)
        raw = np.sign(corr) * corr.abs()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_257_n(df, window=15):
        ret = df['close'].pct_change()
        vol_delta = df.get('matchingVolume', df['close'] * df.get('matchingVolume', 1)).diff()
        abs_ret = ret.abs()
        abs_vol_delta = vol_delta.abs()
        corr = abs_ret.rolling(window).corr(abs_vol_delta)
        raw = np.sign(corr) * corr.abs()
        normalized = np.sign(raw)
        return normalized.fillna(0)

    @staticmethod
    def alpha_quanta_257_d(df, window=50, p1=0.1):
        ret = df['close'].pct_change()
        vol_delta = df.get('matchingVolume', df['close'] * df.get('matchingVolume', 1)).diff()
        abs_ret = ret.abs()
        abs_vol_delta = vol_delta.abs()
        corr = abs_ret.rolling(window).corr(abs_vol_delta)
        raw = np.sign(corr) * corr.abs()
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return -normalized.fillna(0)

    @staticmethod
    def alpha_quanta_258_rank(df, window=55):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        raw = (close - open) * (close - close.shift(1)) / (volume.rolling(window).mean() + 1e-8)
        signal = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_258_tanh(df, window=100):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        raw = (close - open) * (close - close.shift(1)) / (volume.rolling(window).mean() + 1e-8)
        signal = np.tanh(raw / (raw.rolling(window).std() + 1e-9))
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_258_zscore(df, window=100):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        raw = (close - open) * (close - close.shift(1)) / (volume.rolling(window).mean() + 1e-8)
        mean = raw.rolling(window).mean()
        std = raw.rolling(window).std()
        signal = ((raw - mean) / (std + 1e-9)).clip(-1, 1)
        signal = signal.ffill().fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_258_sign(df, window=5):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        raw = (close - open) * (close - close.shift(1)) / (volume.rolling(window).mean() + 1e-8)
        signal = pd.Series(np.sign(raw), index=df.index)
        signal = signal.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_258_wf(df, window=10, p1=0.1):
        close = df['close']
        open = df['open']
        volume = df['matchingVolume']
        raw = (close - open) * (close - close.shift(1)) / (volume.rolling(window).mean() + 1e-8)
        p2 = window
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.ffill().fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_259_rank(df, window=30):
        raw = (df['high'] - df['low']) / df['close'] / (df['close'].rolling(window).std() + 1e-8)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_259_tanh(df, window=55):
        raw = (df['high'] - df['low']) / df['close'] / (df['close'].rolling(window).std() + 1e-8)
        normalized = np.tanh(raw / raw.rolling(window).std())
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_259_zscore(df, window=40):
        raw = (df['high'] - df['low']) / df['close'] / (df['close'].rolling(window).std() + 1e-8)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_259_sign(df, window=45):
        raw = (df['high'] - df['low']) / df['close'] / (df['close'].rolling(window).std() + 1e-8)
        normalized = np.sign(raw)
        signal = pd.Series(normalized, index=df.index).fillna(0)
        return signal

    @staticmethod
    def alpha_quanta_259_wf(df, window=50, p1=0.3, p2=50):
        raw = (df['high'] - df['low']) / df['close'] / (df['close'].rolling(window).std() + 1e-8)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        signal = normalized.fillna(0)
        return -signal

    @staticmethod
    def alpha_quanta_260_rank(df, window=80):
        delta_close = df['close'].diff()
        abs_delta = delta_close.abs()
        corr = abs_delta.rolling(window).corr(df['matchingVolume'])
        raw = np.sign(delta_close) * (1 - corr)
        # Rolling Rank normalization
        result = (raw.rolling(window).rank(pct=True) * 2) - 1
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_260_tanh(df, window=40):
        delta_close = df['close'].diff()
        abs_delta = delta_close.abs()
        corr = abs_delta.rolling(window).corr(df['matchingVolume'])
        raw = np.sign(delta_close) * (1 - corr)
        # Dynamic Tanh normalization
        result = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_260_zscore(df, window=35):
        delta_close = df['close'].diff()
        abs_delta = delta_close.abs()
        corr = abs_delta.rolling(window).corr(df['matchingVolume'])
        raw = np.sign(delta_close) * (1 - corr)
        # Rolling Z-Score normalization
        roll_mean = raw.rolling(window).mean()
        roll_std = raw.rolling(window).std().replace(0, np.nan)
        result = ((raw - roll_mean) / roll_std).clip(-1, 1)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_260_sign(df, window=95):
        delta_close = df['close'].diff()
        abs_delta = delta_close.abs()
        corr = abs_delta.rolling(window).corr(df['matchingVolume'])
        raw = np.sign(delta_close) * (1 - corr)
        # Sign/Binary Soft normalization
        result = np.sign(raw)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_260_wf(df, p1=0.1, p2=40):
        delta_close = df['close'].diff()
        abs_delta = delta_close.abs()
        corr = abs_delta.rolling(p2).corr(df['matchingVolume'])
        raw = np.sign(delta_close) * (1 - corr)
        # Winsorized Fisher normalization
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        result = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        return result.fillna(0)

    @staticmethod
    def alpha_quanta_261_rank(df, window=45):
        close = df['close']
        high = df['high']
        low = df['low']
        spread = (high - low).rolling(window).mean() + 1e-8
        raw_close = close / spread
        zscore_close = (close - close.rolling(window).mean()) / close.rolling(window).std()
        raw = raw_close - zscore_close
        norm = (raw.rolling(window).rank(pct=True) * 2) - 1
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_261_tanh(df, window=50):
        close = df['close']
        high = df['high']
        low = df['low']
        spread = (high - low).rolling(window).mean() + 1e-8
        raw_close = close / spread
        zscore_close = (close - close.rolling(window).mean()) / close.rolling(window).std()
        raw = raw_close - zscore_close
        norm = np.tanh(raw / raw.rolling(window).std())
        return -norm.fillna(0)

    @staticmethod
    def alpha_quanta_261_zscore(df, window=50):
        close = df['close']
        high = df['high']
        low = df['low']
        spread = (high - low).rolling(window).mean() + 1e-8
        raw_close = close / spread
        zscore_close = (close - close.rolling(window).mean()) / close.rolling(window).std()
        raw = raw_close - zscore_close
        norm = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std()).clip(-1, 1)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_261_sign(df, window=85):
        close = df['close']
        high = df['high']
        low = df['low']
        spread = (high - low).rolling(window).mean() + 1e-8
        raw_close = close / spread
        zscore_close = (close - close.rolling(window).mean()) / close.rolling(window).std()
        raw = raw_close - zscore_close
        norm = np.sign(raw)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_261_wf(df, window=100, quantile=0.7):
        close = df['close']
        high = df['high']
        low = df['low']
        spread = (high - low).rolling(window).mean() + 1e-8
        raw_close = close / spread
        zscore_close = (close - close.rolling(window).mean()) / close.rolling(window).std()
        raw = raw_close - zscore_close
        low_q = raw.rolling(window).quantile(quantile)
        high_q = raw.rolling(window).quantile(1 - quantile)
        winsorized = raw.clip(lower=low_q, upper=high_q, axis=0)
        norm = np.arctanh(((winsorized - low_q) / (high_q - low_q + 1e-9)) * 1.98 - 0.99)
        return norm.fillna(0)

    @staticmethod
    def alpha_quanta_262_rank(df, window=70):
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_close = close.rolling(window).std().add(1e-8)
        raw = np.sign(mean_ret) * (mean_ret / std_close)
        normalized = (raw.rolling(window).rank(pct=True) * 2) - 1
        normalized = normalized.fillna(0)
        return -normalized - normalized.mean() if normalized.isna().all() else normalized

    @staticmethod
    def alpha_quanta_262_tanh(df, window=75):
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_close = close.rolling(window).std().add(1e-8)
        raw = np.sign(mean_ret) * (mean_ret / std_close)
        normalized = np.tanh(raw / raw.rolling(window).std().add(1e-8))
        normalized = normalized.fillna(0)
        return -normalized - normalized.mean() if normalized.isna().all() else normalized

    @staticmethod
    def alpha_quanta_262_zscore(df, window=65):
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_close = close.rolling(window).std().add(1e-8)
        raw = np.sign(mean_ret) * (mean_ret / std_close)
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().add(1e-8)).clip(-1, 1)
        normalized = normalized.fillna(0)
        return -normalized - normalized.mean() if normalized.isna().all() else normalized

    @staticmethod
    def alpha_quanta_262_sign(df, window=55):
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window).mean()
        std_close = close.rolling(window).std().add(1e-8)
        raw = np.sign(mean_ret) * (mean_ret / std_close)
        normalized = np.sign(raw)
        normalized = normalized.fillna(0)
        return normalized - normalized.mean() if normalized.isna().all() else normalized

    @staticmethod
    def alpha_quanta_262_wf(df, window_rank=60, winsor_quantile=0.9):
        p1 = winsor_quantile
        p2 = window_rank * 5
        close = df['close']
        ret = close.pct_change()
        mean_ret = ret.rolling(window_rank).mean()
        std_close = close.rolling(window_rank).std().add(1e-8)
        raw = np.sign(mean_ret) * (mean_ret / std_close)
        low = raw.rolling(p2).quantile(p1)
        high = raw.rolling(p2).quantile(1 - p1)
        winsorized = raw.clip(lower=low, upper=high, axis=0)
        normalized = np.arctanh(((winsorized - low) / (high - low + 1e-9)) * 1.98 - 0.99)
        normalized = normalized.fillna(0)
        return -normalized - normalized.mean() if normalized.isna().all() else normalized

    @staticmethod
    def alpha_quanta_263_rank(df, window=30):
        median = df['close'].rolling(window).median()
        std = df['close'].rolling(window).std()
        raw = (df['close'] - median) / (std + 1e-8) * df['close'].diff()
        zscore = (raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)
        normalized = (zscore.rolling(window).rank(pct=True) * 2) - 1
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_263_tanh(df, window=55):
        median = df['close'].rolling(window).median()
        std = df['close'].rolling(window).std()
        raw = (df['close'] - median) / (std + 1e-8) * df['close'].diff()
        normalized = np.tanh(raw / raw.rolling(window).std().replace(0, np.nan))
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_263_zscore(df, window=20):
        median = df['close'].rolling(window).median()
        std = df['close'].rolling(window).std()
        raw = (df['close'] - median) / (std + 1e-8) * df['close'].diff()
        normalized = ((raw - raw.rolling(window).mean()) / raw.rolling(window).std().replace(0, np.nan)).clip(-1, 1)
        return -normalized.fillna(0).clip(-1, 1)

    @staticmethod
    def alpha_quanta_263_sign(df, window=40):
        median = df['close'].rolling(window).median()
        std = df['close'].rolling(window).std()
        raw = (df['close'] - median) / (std + 1e-8) * df['close'].diff()
        normalized = np.sign(raw)
        return -pd.Series(normalized, index=df.index).fillna(0).clip(-1, 1)