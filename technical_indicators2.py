
from ta.trend import *
from ta.volatility import *
from ta.trend import MACD
from ta.momentum import ROCIndicator
from ta.momentum import RSIIndicator
from ta.momentum import WilliamsRIndicator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator
from ta.volume import ChaikinMoneyFlowIndicator
from ta.trend import WMAIndicator
from ta.trend import TRIXIndicator
from ta.trend import DPOIndicator
from ta.trend import KSTIndicator
from ta.trend import ADXIndicator
from ta.volume import ForceIndexIndicator
from ta.volume import EaseOfMovementIndicator
from ta.volatility import AverageTrueRange
import time
from stockstats import StockDataFrame as sdf
from tqdm.auto import tqdm
import numpy as np

# Class setup indicators with ta library:
class TechnicalIndicators():
    def __init__(self, dataframe):
        self.data = dataframe
        # TODO initialize df here


    def get_macd(self, col_name: str, w_fast: int, w_slow: int, w_sign: int):
        indicator_macd = MACD(col_name, w_fast, w_slow, w_sign)
        self.data['macd_{}_{}'.format(w_fast, col_name)] = indicator_macd.macd()

    def get_roc(self, col_name: str, window: int):
        indicator_roc = ROCIndicator(col_name, window)
        self.data['roc_{}_{}'.format(window, col_name)] = indicator_roc.roc()

    def get_rsi(self, col_name: str, window: int):
        indicator_rsi = RSIIndicator(col_name, window)
        self.data['rsi_{}_{}'.format(window, col_name)] = indicator_rsi.rsi()

    def get_mfi(self, high: str, low: str, close: str, volume: str, window: int):
        indicator_mfi = MFIIndicator(high, low, close, volume, window)
        self.data['mfi_{}'.format(window)] = indicator_mfi.money_flow_index()

    def get_cmf(self, high: str, low: str, close: str, volume: str, window: int):
        indicator_cmf = ChaikinMoneyFlowIndicator(high, low, close, volume, window)
        self.data['cmf_{}'.format(window)] = indicator_cmf.chaikin_money_flow()

    def get_wma(self, col_name: str, window: int):
        indicator_wma = WMAIndicator(col_name, window)
        self.data['wma_{}_{}'.format(window, col_name)] = indicator_wma.wma()

    def get_trix(self, close: str, window: int):
        indicator_trix = TRIXIndicator(close, window)
        self.data['trix_{}'.format(window)] = indicator_trix.trix()

    def get_dpo(self, close: str, window: int):
        indicator_dpo = DPOIndicator(close, window)
        self.data['dpo_{}'.format(window)] = indicator_dpo.dpo()

    def get_kst(self, close: str, roc1: int, roc2: int, roc3: int, roc4: int, window1: int, window2: int, window3: int,
                window4: int, nsig: int):
        indicator_kst = KSTIndicator(close, roc1, roc2, roc3, roc4, window1, window2, window3, window4, nsig)
        self.data['kst'] = indicator_kst.kst()

    def get_adx(self, high: str, low: str, close: str, window: int):
        indicator_adx = ADXIndicator(high, low, close, window)
        self.data['adx_{}'.format(window)] = indicator_adx.adx()

    def get_fi(self, close: str, volume: str, window: int):
        indicator_fi = ForceIndexIndicator(close, volume, window)
        self.data['fi_{}'.format(window)] = indicator_fi.force_index()

    def get_emv(self, high: str, low: str, volume: str, window: int):
        indicator_emv = EaseOfMovementIndicator(high, low, volume, window)
        self.data['emv_{}'.format(window)] = indicator_emv.ease_of_movement()

    def get_bb(self, close: str, window: int):
        indicator_bb = BollingerBands(close, window)
        self.data['bb_bbm'] = indicator_bb.bollinger_mavg()
        self.data['bb_bbh'] = indicator_bb.bollinger_hband()
        self.data['bb_bbl'] = indicator_bb.bollinger_lband()
        self.data['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
        self.data['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
        self.data['bb_bbhi'] = indicator_bb.bollinger_hband()
        self.data['bb_bbw'] = indicator_bb.bollinger_wband()
        self.data['bb_bbp'] = indicator_bb.bollinger_pband()

    def get_atr(self, high: str, low: str, close: str, window: int):
        indicator_atr = AverageTrueRange(high, low, close, window)
        self.data['atr_{}'.format(window)] = indicator_atr.average_true_range()

    def get_williamR(self, col_name: str, intervals: int):
        """
        both libs gave same result
        Momentum indicator
        """
        stime = time.time()
        print("Calculating WilliamR")
        # df_ss = sdf.retype(df)
        for i in tqdm(intervals):
            # df['wr_'+str(i)] = df_ss['wr_'+str(i)]
            self.data["wr_" + str(i)] = WilliamsRIndicator(self.data['high'], self.data['low'], self.data['close'], i, fillna=True).williams_r()


    def get_SMA(self, col_name: str, intervals: int):
        """
        Momentum indicator
        """
        stime = time.time()
        print("Calculating SMA")
        df_ss = sdf.retype(self.data)
        for i in tqdm(intervals):
            self.data[col_name + '_sma_' + str(i)] = df_ss[col_name + '_' + str(i) + '_sma']
            del self.data[col_name + '_' + str(i) + '_sma']

    def get_EMA(self, col_name: str, intervals: int):  # not working?
        """
        Needs validation
        Momentum indicator
        """
        stime = time.time()
        print("Calculating EMA")
        df_ss = sdf.retype(self.data)
        for i in tqdm(intervals):
            self.data['ema_' + str(i)] = df_ss[col_name + '_' + str(i) + '_ema']
            del self.data[col_name + '_' + str(i) + '_ema']
            # df["ema_"+str(intervals[0])+'_1'] = ema_indicator(df['close'], i, fillna=True)

    def get_CMO(self, col_name: str, intervals: int):
        """
        Chande Momentum Oscillator
        As per https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/cmo

        CMO = 100 * ((Sum(ups) - Sum(downs))/ ( (Sum(ups) + Sum(downs) ) )
        range = +100 to -100

        params: df -> dataframe with financial instrument history
                col_name -> column name for which CMO is to be calculated
                intervals -> list of periods for which to calculated

        return: None (adds the result in a column)
        """

        print("Calculating CMO")
        stime = time.time()

        def calculate_CMO(series, period):
            # num_gains = (series >= 0).sum()
            # num_losses = (series < 0).sum()
            sum_gains = series[series >= 0].sum()
            sum_losses = np.abs(series[series < 0].sum())
            cmo = 100 * ((sum_gains - sum_losses) / (sum_gains + sum_losses))
            return np.round(cmo, 3)

        diff = self.data[col_name].diff()[1:]  # skip na
        for period in tqdm(intervals):
            self.data['cmo_' + str(period)] = np.nan
            res = diff.rolling(period).apply(calculate_CMO, args=(period,), raw=False)
            self.data['cmo_' + str(period)][1:] = res

    def get_WMA(self, col_name, intervals, hma_step=0):
        """
        Momentum indicator
        """
        stime = time.time()
        if (hma_step == 0):
            # don't show progress for internal WMA calculation for HMA
            print("Calculating WMA")

        def wavg(rolling_prices, period):
            weights = pd.Series(range(1, period + 1))
            return np.multiply(rolling_prices.values, weights.values).sum() / weights.sum()

        temp_col_count_dict = {}
        for i in tqdm(intervals, disable=(hma_step != 0)):
            res = self.data[col_name].rolling(i).apply(wavg, args=(i,), raw=False)
            # print("interval {} has unique values {}".format(i, res.unique()))
            if hma_step == 0:
                self.data['wma_' + str(i)] = res
            elif hma_step == 1:
                if 'hma_wma_' + str(i) in temp_col_count_dict.keys():
                    temp_col_count_dict['hma_wma_' + str(i)] = temp_col_count_dict['hma_wma_' + str(i)] + 1
                else:
                    temp_col_count_dict['hma_wma_' + str(i)] = 0
                # after halving the periods and rounding, there may be two intervals with same value e.g.
                # 2.6 & 2.8 both would lead to same value (3) after rounding. So save as diff columns
                self.data['hma_wma_' + str(i) + '_' + str(temp_col_count_dict['hma_wma_' + str(i)])] = 2 * res
            elif hma_step == 3:
                import re
                expr = r"^hma_[0-9]{1}"
                columns = list(self.data.columns)
                # print("searching", expr, "in", columns, "res=", list(filter(re.compile(expr).search, columns)))
                self.data['hma_' + str(len(list(filter(re.compile(expr).search, columns))))] = res

    def get_HMA(self, col_name: str, intervals: int):
        import re
        stime = time.time()
        print("Calculating HMA")
        expr = r"^wma_.*"

        if len(list(filter(re.compile(expr).search, list(self.data.columns)))) > 0:
            print("WMA calculated already. Proceed with HMA")
        else:
            print("Need WMA first...")
            self.get_WMA(col_name, intervals)

        intervals_half = np.round([i / 2 for i in intervals]).astype(int)

        # step 1 = WMA for interval/2
        # this creates cols with prefix 'hma_wma_*'
        self.get_WMA(col_name, intervals_half, 1)
        # print("step 1 done", list(df.columns))

        # step 2 = step 1 - WMA
        columns = list(self.data.columns)
        expr = r"^hma_wma.*"
        hma_wma_cols = list(filter(re.compile(expr).search, columns))
        rest_cols = [x for x in columns if x not in hma_wma_cols]
        expr = r"^wma.*"
        wma_cols = list(filter(re.compile(expr).search, rest_cols))

        self.data[hma_wma_cols] = self.data[hma_wma_cols].sub(self.data[wma_cols].values,
                                                fill_value=0)  # .rename(index=str, columns={"close": "col1", "rsi_6": "col2"})
        # df[0:10].copy().reset_index(drop=True).merge(temp.reset_index(drop=True), left_index=True, right_index=True)

        # step 3 = WMA(step 2, interval = sqrt(n))
        intervals_sqrt = np.round([np.sqrt(i) for i in intervals]).astype(int)
        for i, col in tqdm(enumerate(hma_wma_cols)):
            # print("step 3", col, intervals_sqrt[i])
            self.get_WMA(col, [intervals_sqrt[i]], 3)
        self.data.drop(columns=hma_wma_cols, inplace=True)

    def get_CCI(self, col_name: str, intervals: int):
        print("Calculating CCI")
        for i in tqdm(intervals):
            self.data['cci_' + str(i)] = cci(self.data['high'], self.data['low'], self.data['close'], i, fillna=True)
