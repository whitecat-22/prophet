import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

data = pd.DataFrame()
args = sys.argv
file_name = args[1] #ここでデータファイルを読み込む
data2 = pd.read_csv(file_name, skiprows=1, header=None, names=['ds','Open','High','Low','Close','Adj_Close','Volume'])
data3 = data2.dropna(how='any')
data = data.append(data3)

plt.style.use('ggplot')
fig = plt.figure(figsize=(18,8))
ax_Nikkei = fig.add_subplot(224)
ax_Nikkei.plot(data.loc["1965-04-21": ,['Adj_Close']], label='Nikkei', color='r')
ax_Nikkei.set_title('NIKKEI')

data['Adj_Close_log'] = np.log(data.Adj_Close).diff()
data.head()
data.tail()

fig = plt.figure(figsize=(18,8))
ax_Nikkei_log = fig.add_subplot(224)
ax_Nikkei_log.plot(data.loc["1965-04-21": ,['Adj_Close_log']], label='log diff', color='b')
ax_Nikkei_log.set_title('log earning rate')

model = Prophet()
model.fit(data.rename(columns={'Adj_Close':'y'}))
future_data = model.make_future_dataframe(periods=365, freq= 'd')
forecast_data = model.predict(future_data)
fig = model.plot(forecast_data)

df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
df_cv.head()

df_p = performance_metrics(df_cv)
df_p.head()

#MSEをプロットする関数をデバッグのために再定義
def plot_cross_validation_metric(
    df_cv, metric, rolling_window=0.1, ax=None, figsize=(10, 6)
):
    """Plot a performance metric vs. forecast horizon from cross validation.
    Cross validation produces a collection of out-of-sample model predictions
    that can be compared to actual values, at a range of different horizons
    (distance from the cutoff). This computes a specified performance metric
    for each prediction, and aggregated over a rolling window with horizon.
    This uses fbprophet.diagnostics.performance_metrics to compute the metrics.
    Valid values of metric are 'mse', 'rmse', 'mae', 'mape', and 'coverage'.
    rolling_window is the proportion of data included in the rolling window of
    aggregation. The default value of 0.1 means 10% of data are included in the
    aggregation for computing the metric.
    As a concrete example, if metric='mse', then this plot will show the
    squared error for each cross validation prediction, along with the MSE
    averaged over rolling windows of 10% of the data.
    Parameters
    ----------
    df_cv: The output from fbprophet.diagnostics.cross_validation.
    metric: Metric name, one of ['mse', 'rmse', 'mae', 'mape', 'coverage'].
    rolling_window: Proportion of data to use for rolling average of metric.
        In [0, 1]. Defaults to 0.1.
    ax: Optional matplotlib axis on which to plot. If not given, a new figure
        will be created.
    Returns
    -------
    a matplotlib figure.
    """
    if ax is None:
        fig = plt.figure(facecolor='w', figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        fig = ax.get_figure()
    # Get the metric at the level of individual predictions, and with the rolling window.
    df_none = performance_metrics(df_cv, metrics=[metric], rolling_window=0)
    df_h = performance_metrics(df_cv, metrics=[metric], rolling_window=rolling_window)

    # Some work because matplotlib does not handle timedelta
    # Target ~10 ticks.
    tick_w = max(df_none['horizon'].astype('timedelta64[ns]')) / 10.
    # Find the largest time resolution that has <1 unit per bin.
    dts = ['D', 'h', 'm', 's', 'ms', 'us', 'ns']
    dt_names = [
        'days', 'hours', 'minutes', 'seconds', 'milliseconds', 'microseconds',
        'nanoseconds'
    ]
    dt_conversions = [
        24 * 60 * 60 * 10 ** 9,
        60 * 60 * 10 ** 9,
        60 * 10 ** 9,
        10 ** 9,
        10 ** 6,
        10 ** 3,
        1.,
    ]
    for i, dt in enumerate(dts):
        if np.timedelta64(1, dt) < np.timedelta64(tick_w, 'ns'):
            break

    x_plt = df_none['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])
    x_plt_h = df_h['horizon'].astype('timedelta64[ns]').astype(np.int64) / float(dt_conversions[i])

    ax.plot(x_plt, df_none[metric], '.', alpha=0.5, c='gray')
    ax.plot(x_plt_h, df_h[metric], '-', c='b')
    ax.grid(True)

    ax.set_xlabel('Horizon ({})'.format(dt_names[i]))
    ax.set_ylabel(metric)
    return fig
# 日経平均株価をそのままモデリング(MSEが高くなり当てはまりが悪い)
fig = plot_cross_validation_metric(df_cv, metric='mse')#MSEをプロット

model2 = Prophet()
model2.fit(data.rename(columns={'Adj_Close_log':'y'}))
future_data2 = model2.make_future_dataframe(periods=365, freq= 'd')
forecast_data2 = model2.predict(future_data2)
fig2 = model2.plot(forecast_data2)

df_cv2 = cross_validation(model2, initial='730 days', period='180 days', horizon = '365 days')
df_cv2.head()

df_p2 = performance_metrics(df_cv2)
df_p2.head()
# 株価ではなく対数収益率をモデリング（当てはまりが高い)
fig = plot_cross_validation_metric(df_cv2, metric='mse')#MSEをプロット
plt.show()
