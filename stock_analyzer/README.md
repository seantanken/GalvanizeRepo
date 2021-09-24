# Stock Analyzer and Predictor

[TOC]

Prediction of future Amazon stock trends and prices using a recursive neural network and historical stock data related to Amazon performance.

**Overview:** A project using stock data scraped with Selenium and downloaded from NASDAQ and Yahoo Finance websites then analyzed using time series analysis and a Keras LSTM neural network model to predict future trends and prices for the Amazon stock (AMZN). The results are then demonstrated with a Flask app. This model could be used as a guide to help inform investment and trading choices in the stock market.

**Results:** I found that the best features for predicting future prices began with smoothing the jaggedness of the stock market by using a rolling average and diversifying the features with other related stock data. I then ran the results through a custom made scaler method which checks if the trend is increasing or decreasing and multiplies by a scaler to adjust the predictions and get much closer to the actual recorded prices. In the end, the adjusted predictions were still quite a bit off recorded prices, but the scaler will be investigated more and could be used to predict trends.

[Google slide presentation](https://docs.google.com/presentation/d/11U0RmnSMoXtRPwjT8FSqfppO0JOVBeGqeniUY8rMYaQ/edit?usp=sharing)

[Youtube video presentation](https://www.youtube.com/watch?v=7UKiBRLuKeM&list=PLs8Sv4AQpgXpVj9fIoIUVD_6SMG8BQIL2&index=6)

## Background and Motivation

Many people, both young and old, are jumping into investing in the stock market and trying their hand at trading. With this rise in popularity, I also began to take an interest in invested in some big-name companies like Amazon. I want to make a tool using time series analysis and a neural network to predict the volatility of the market and help investors increase their gains.

## Method

**Selenium web scraping**

A custom made script used to scrape the stock price and trade volume for AMZN by the minute along with the price and volume for each trade done which is run after every trading day (1 pm PST). This is used to build my own high definition database for AMZN.

**Time series analysis with Pandas, Numpy**

The scraped and downloaded datasets are then cleaned and organized into csv files, then into a dictionary using time series analysis for input into the neural network.

**Keras LSTM recursive neural network prediction:**

The neural network is made up of five bidirectional LSTM layers with 0.2 dropout layers in between and a dense layer at the end. Units = 256, loss = mean squared error, optimizer = Adamax, metric = mean squared logarithmic error. These were decided on after repeated testing. The neural network uses 40 days of data (approximately 2 months of trading days) to predict the next day.

**Scaler method:**

Looks at the first derivative of the predicted results to see if it is positive or negative and multiplies by 1.01 or 0.99 respectively. This causes the predicted results to change direction much quicker and have larger extremes, which is much closer to how the actual stock market acts.

## Results

### **First Iteration:**

**Features used:** AMZN price, AMZN volume

![](C:\Users\seant\stock_analyzer\actual vs prediction graphs\bi, shuf.png)

Follows the targets almost exactly with a delay, essentially saying that the price for tomorrow is the same as the price for today.

High variance (overfitting), needs more diversity.

### Second Iteration:

**Initial features:** 5 day rolling average close price, volume, and mean percent change from previous day for AMZN, AAPL, GOOG

**Features used:** 5 day rolling average AMZN close price, AMZN volume, 5 day rolling average GOOG close price

**Initial predictions:**



![](C:\Users\seant\stock_analyzer\capstone_3\models\final model\pred vs act.png)

Still has a delay, but predictions do not exactly follow targets.

**After scaling:**

![](C:\Users\seant\stock_analyzer\capstone_3\models\final model\scaled pred vs act.png)

Follows large changes much better. 30-35% increase in r^2 score compared to unscaled predictions across all models tested.

## Current and Future Work

While the price of the stock is difficult to predict with a high degree of accuracy, the scaler could be used to predict large changes in price and can potentially be used to inform about these jumps rather than predict the price after the jump.

## Acknowledgements

I would like to thank Lisa Beatty for taking an interest in this project when I first started it as well as Noah Shreve and Ryan Kasichainula for helping me with the tougher parts during my second iteration. I would also like to thank all my classmates and instructors in the Remote Part Time 2 cohort at Galvanize for creating a great environment to learn and explore.

## References

https://github.com/cshawnkeech/rainforest_audio/blob/main/notebooks/CURRENT_BEST-1-4_mel_shape_FREQ_ADJUST.ipynb

https://www.youtube.com/watch?v=PuZY9q-aKLw

https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

https://www.thepythoncode.com/article/stock-price-prediction-in-python-using-tensorflow-2-and-keras

https://www.youtube.com/watch?v=PuZY9q-aKLw&t=426s

https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233