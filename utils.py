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


def print_with_timestamp(text, stime):
    delta_secs = (time.time() - stime)
    new_time = str(delta_secs // 60) + " mins, " + str(np.round(delta_secs % 60)) + " secs"
    print(text, new_time)


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


def save_array_as_images(x, img_width, img_height, path, file_names):  # TODO rewrite
    if os.path.exists(path):
        shutil.rmtree(path) #remove folder
    os.makedirs(path)  # make new folder
    x_as_images = np.zeros((len(x), img_height, img_width))
    for i in tqdm(range(x.shape[0])):
        x_as_images[i] = np.reshape(x[i], (img_height, img_width))
        img = Image.fromarray(x_as_images[i], 'RGB')
        img.save(os.path.join(path, str(file_names[i]) + '.png'))

    print_with_timestamp("Images saved at " + path, time.time())
    return x_as_images


def reshape_array_as_image(x, img_width, img_height, save_to_disk=False, path="", file_names=None):
    x_as_images = np.zeros((len(x), img_height, img_width))
    for i in range(x.shape[0]):
        x_as_images[i] = np.reshape(x[i], (img_height, img_width))

    if save_to_disk:
        if os.path.exists(path):
            shutil.rmtree(path)  # remove folder including old files
        os.makedirs(path)  # add new folder

        for i in tqdm(range(x.shape[0])):
            img = Image.fromarray(x_as_images[i], 'RGB')
            img.save(os.path.join(path, str(file_names[i]) + '.png'))
        print_with_timestamp("Images saved at " + path, time.time())

    return x_as_images


def show_images(rows, columns, arr=None, path=None):
    """
    show images that were generated for the CNN.

    rows: int
    columns: int
    arr: np.array
    path: String
    """

    if path is None and arr is None:
        raise AttributeError('Both attributes "arr" and "path" are None. One of them has to be defined.')

    w = h = 15
    fig = plt.figure(figsize=(w, h))

    for i in range(1, columns * rows + 1):
        if not path:
            index = np.random.randint(len(arr))
            img = arr[index]
            title = 'image_'+str(index)
        else:
            files = os.listdir(path)
            index = np.random.randint(len(files))
            img = np.asarray(Image.open(os.path.join(path, files[index])))
            title = files[i]
        fig.add_subplot(rows, columns, i)
        plt.title(title, fontsize=10)
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.imshow(img)
    plt.show()
    return

def plot(y, title, output_path, x=None):
    fig = plt.figure(figsize=(10, 10))
    # x = x if x is not None else np.arange(len(y))
    plt.title(title)
    if x is not None:
        plt.plot(x, y, 'o-')
    else:
        plt.plot(y, 'o-')
        plt.savefig(output_path)


def create_labels(df, col_name, window_size=11):  # according to BAZEL
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
    counter_row = 0
    number_of_days_in_File = len(df)
    labels = np.zeros(number_of_days_in_File)
    labels[:] = np.nan
    print("Calculating labels")
    pbar = tqdm(total=number_of_days_in_File)

    while counter_row < number_of_days_in_File:
        counter_row += 1
        if counter_row > window_size:
            window_begin_index = counter_row - window_size
            window_end_index = window_begin_index + window_size - 1
            window_middle_index = int((window_begin_index + window_end_index) / 2)

            min_ = np.inf
            min_index = -1
            max_ = -np.inf
            max_index = -1
            for i in range(window_begin_index, window_end_index + 1):
                number = df.iloc[i][col_name]  # number is the price
                if number < min_:
                    min_ = number
                    min_index = i
                if number > max_:
                    max_ = number
                    max_index = i

            if max_index == window_middle_index:
                labels[window_middle_index] = 0  # SELL
            elif min_index == window_middle_index:
                labels[window_middle_index] = 1  # BUY
            else:
                labels[window_middle_index] = 2  # HOLD

        pbar.update(1)

    pbar.close()
    return labels

# TRIPLE BARRIER METHOD LABELLING


# Label creation if only using daily data
def get_daily_volatility(df,span0=20):
    # simple percentage returns
    df0 = df.close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0=df0.ewm(span=span0).std().to_frame("volatility")
    df_clean = df0.dropna()
    return df0, df_clean

def adjust_data(df, volatilities_raw):
    df_clean = df[volatilities_raw.isna()['volatility'] == False]
    return df_clean

def get_barriers(df, volatilities, upper_lower_multipliers):
    barriers = df[['close','high','low']].copy()
    barriers['volatility'] = volatilities['volatility']
    top_barrier = [0]
    bottom_barrier = [0]
    for i in range(len(barriers)-1):
        vol = volatilities.volatility.iloc[i]
        if upper_lower_multipliers[0] > 0:
            top_barrier.append(barriers.close.iloc[i] + barriers.close.iloc[i] * upper_lower_multipliers[0] * vol)
        else:
            #set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        #set the bottom barrier

        if upper_lower_multipliers[1] > 0:
            bottom_barrier.append(barriers.close.iloc[i] - barriers.close.iloc[i] * upper_lower_multipliers[1] * vol)
        else:
            #set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
    barriers['top_barrier'] = top_barrier
    barriers['bottom_barrier'] = bottom_barrier
    return barriers

def get_labels_daily_triple_barrier(df, upper_lower_multipliers=(2, 2)):
    """
    top_barrier: profit taking limit
    bottom_barrier:stop loss limit
    daily_volatiliy: average daily volatility based on 20-day moving average
    barriers_df: DataFrame containing top and bottom barriers on a per-day base
    """
    daily_volatility_raw, daily_volatility_clean = get_daily_volatility(df)
    df = adjust_data(df, daily_volatility_raw)
    barriers_df = get_barriers(df = df, volatilities = daily_volatility_clean, upper_lower_multipliers = upper_lower_multipliers)
    labels = [0,0]
    nr_double_labels = 0
    for i in range(len(barriers_df.index)-1):
        if barriers_df.high.iloc[i+1] >= barriers_df.top_barrier.iloc[i+1]:
            labels.append(1)
        elif barriers_df.low.iloc[i+1] <= barriers_df.bottom_barrier.iloc[i+1]:
            labels.append(-1)
        else:
            labels.append(0)

        if barriers_df.high.iloc[i+1] >= barriers_df.top_barrier.iloc[i+1] and barriers_df.low.iloc[i+1] <= barriers_df.bottom_barrier.iloc[i+1]:
            nr_double_labels += 1

    labels.append(0)
    perc_double_labels = round(nr_double_labels / len(df),4)
    #barriers_df['label'] = labels
    #return barriers_df, barriers_df.label, perc_double_labels
    return labels
