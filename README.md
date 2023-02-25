# Predicting Bitcoin price movements with Machine Learning
This repository contains the code and data used in the paper "Sparse trading strategies for
high-frequency bitcoin trading". The aim of this study is to explore the potential for machine learning algorithms to predict the price of Bitcoin and develop a profitable trading strategy. 

### Abstract
Bitcoin prices are notoriously volatile, but the fluctuations may exhibit patterns that can be detected using machine learning. In this study, a Long Short-Term Memory (LSTM) network and a random forest were trained on recent, high-frequency price data to predict future Bitcoin prices. Combining the predictions of these two models in an ensemble led to more accurate predictions and more profitable trading strategies.

### Getting Started
###### Note: the code for this thesis is currently being refactored, therefore the procedure below is subject to change.
To reproduce our results, please follow these steps:

1. Clone this repository to your local machine.
2. Install the required packages listed in requirements.txt.
3. Download the data files from the link provided in the paper.
4. Run the code in `full_notebook.ipynb` to construct predictions and results.

<? 4. Run the `data_preparation.ipynb` notebook to preprocess the data.
5. Run the `model_training.ipynb` notebook to train the LSTM and random forest models.
6. Run the `ensemble.ipynb` notebook to combine the predictions of the two models and evaluate their performance. ?>

### Data
The study used historical Bitcoin price data from the Kraken exchange, obtained [here](http://api.bitcoincharts.com/v1/csv/). A subset of the data that covers the period from January 2019 to January 2021 is used.

### Model
Two models were trained to predict future Bitcoin prices: a Long Short-Term Memory (LSTM) network and a random forest. The LSTM network is a type of recurrent neural network that is well-suited for time series prediction, while the random forest is an ensemble of decision trees that is known for its robustness and efficiency on large datasets. 

### Results
Experiments showed that the LSTM network and random forest were both able to predict future Bitcoin price movements with an accuracy exceeding 50%. Combining the predictions of the two models in an ensemble led to more accurate predictions and more profitable trading strategies. By introducing a threshold for trading, the profitability of the ensembles improved in backtesting.

### Conclusion
The study suggests that machine learning algorithms might predict the price movements of Bitcoin with reasonable accuracy and that ensembling different models can lead to more profitable trading strategies. However, it's important to note that transaction fees and slippage can have a significant impact on the profitability of these strategies, and that the buy and hold strategy likely still outperformed the ensemble trading strategies in many cases.
