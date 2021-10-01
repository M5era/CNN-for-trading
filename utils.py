"""
Utility functions
"""

import shutil
import os
from PIL import Image
from ta.volatility import *
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import yfinance as yf
from technical_indicators import get_SMA

def seconds_to_minutes(seconds):
    return str(seconds // 60) + " minutes " + str(np.round(seconds % 60)) + " seconds"


def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds_to_minutes(seconds))


def get_readable_ctime():
    return time.strftime("%d-%m-%Y %H_%M_%S")


def download_financial_data(symbol, path_to_save_file):
    print("Starting download " + symbol)
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="max")
    cols_to_delete = ['Dividends', 'Stock Splits']
    for c in cols_to_delete:
        if c in df.columns:
            df.drop(axis=1, columns=c, inplace=True)

    df.reset_index(inplace=True)
    df = df.rename(columns={'Date': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close',
                            'Volume': 'volume'})
    df.to_csv(path_to_save_file)


def save_array_as_images(x, img_width, img_height, path, file_names):
    if os.path.exists(path):
        shutil.rmtree(path)
        print("deleted old files")

    os.makedirs(path)
    print("Image Directory created", path)
    x_temp = np.zeros((len(x), img_height, img_width))
    print("saving images...")
    stime = time.time()
    for i in tqdm(range(x.shape[0])):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))
        img = Image.fromarray(x_temp[i], 'RGB')
        img.save(os.path.join(path, str(file_names[i]) + '.png'))

    print_time("Images saved at " + path, stime)
    return x_temp


def reshape_as_image(x, img_width, img_height):
    x_temp = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_temp[i] = np.reshape(x[i], (img_height, img_width))

    return x_temp


def show_images(rows, columns, path):
    w = 15
    h = 15
    fig = plt.figure(figsize=(15, 15))
    files = os.listdir(path)
    for i in range(1, columns * rows + 1):
        index = np.random.randint(len(files))
        img = np.asarray(Image.open(os.path.join(path, files[index])))
        fig.add_subplot(rows, columns, i)
        plt.title(files[i], fontsize=10)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.imshow(img)
    plt.show()


def dict_to_str(d):
    return str(d).replace("{", '').replace("}", '').replace("'", "").replace(' ', '')


def cleanup_file_path(path):
    return path.replace('\\', '/').replace(" ", "_").replace(':', '_')


def white_noise_check(tags_list, logger=None, *pd_series_args):
    if len(tags_list) != len(pd_series_args):
        raise Exception("Length of tags_list and series params different. Should be same.")
    for idx, s in enumerate(pd_series_args):
        # logger.append_log("1st, 2nd element {}, {}".format(s.iloc[0], s.iloc[1]))
        m = s.mean()
        std = s.std()
        logger.append_log("mean & std for {} is {}, {}".format(tags_list[idx], m, std))


def plot(y, title, output_path, x=None):
    fig = plt.figure(figsize=(10, 10))
    # x = x if x is not None else np.arange(len(y))
    plt.title(title)
    if x is not None:
        plt.plot(x, y, 'o-')
    else:
        plt.plot(y, 'o-')
        plt.savefig(output_path)


def col1_gt_col2(col1, col2, df):
    compare_series = df[col1] > df[col2]
    print(df.iloc[compare_series[compare_series == True].index])


def console_pretty_print_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)

def create_labels(df, col_name, window_size=11):
    """
    Data is labeled as per the logic in research paper
    Label code : BUY => 1, SELL => 0, HOLD => 2

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels with
              size = total-(window_size)+1
    """

    print("creating label with bazel's strategy")
    row_counter = 0
    total_rows = len(df)
    labels = np.zeros(total_rows)
    labels[:] = np.nan
    print("Calculating labels")
    pbar = tqdm(total=total_rows)

    while row_counter < total_rows:
        if row_counter >= window_size - 1:
            window_begin = row_counter - (window_size - 1)
            window_end = row_counter
            window_middle = int((window_begin + window_end) / 2)

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin, window_end + 1):
                price = df.iloc[i][col_name]
                if price < min_:
                    min_ = price
                    min_index = i
                if price > max_:
                    max_ = price
                    max_index = i

            if max_index == window_middle:
                labels[window_middle] = 0
            elif min_index == window_middle:
                labels[window_middle] = 1
            else:
                labels[window_middle] = 2

        row_counter = row_counter + 1
        pbar.update(1)

    pbar.close()
    return labels


def create_label_short_long_ma_crossover(df, short, long):
    """
    if short = 30 and long = 90,
    Buy when 30 day MA < 90 day MA
    Sell when 30 day MA > 90 day MA

    Label code : BUY => 1, SELL => 0, HOLD => 2

    params :
        df => Dataframe with data
        col_name => name of column which should be used to determine strategy

    returns : numpy array with integer codes for labels
    """

    print("creating label with {}_{}_ma".format(short, long))

    def detect_crossover(diff_prev, diff):
        if diff_prev >= 0 > diff:
            # buy
            return 1
        elif diff_prev <= 0 < diff:
            return 0
        else:
            return 2

    get_SMA(df, 'close', [short, long])
    labels = np.zeros((len(df)))
    labels[:] = np.nan
    diff = df['close_sma_' + str(short)] - df['close_sma_' + str(long)]
    diff_prev = diff.shift()
    df['diff_prev'] = diff_prev
    df['diff'] = diff

    res = df.apply(lambda row: detect_crossover(row['diff_prev'], row['diff']), axis=1)
    print("labels count", np.unique(res, return_counts=True))
    df.drop(columns=['diff_prev', 'diff'], inplace=True)
    return res
