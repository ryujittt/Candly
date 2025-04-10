import sys
import ccxt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QTabWidget,QApplication, QMainWindow, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QCompleter,QWidget, QLabel, QCheckBox, QLineEdit
from PyQt5.QtGui import QIcon
import pandas as pd
import numpy as np
import time
from scipy.signal import correlate
from sklearn.metrics import r2_score
import os
import pandas as pd
import ccxt
from scipy.stats import ttest_ind
import traceback
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from sklearn.neighbors import NearestNeighbors


import mplfinance as mpf



class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=120):
        self.fig, self.axs = plt.subplots(nrows=1 , ncols=1, figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the ccxt exchange object
        self.exchange = ccxt.binance()

        self.init_ui()

    def init_ui(self):
        self.setGeometry(100, 100, 1000, 600)
        self.setWindowTitle('Candly')

        # Create ComboBoxes for symbol, timeframe, and limit
        self.symbol_combobox = QComboBox(self)
        self.symbol_combobox.addItems(['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'AUD/JPY', 'EUR/AUD', 'GBP/EUR', 'USD/SGD', 'USD/HKD','XRPUSDT'])
        self.symbol_combobox.setCurrentIndex(0)

        self.timeframe_combobox = QComboBox(self)
        self.timeframe_combobox.addItems(['1m', '5m', '15m', '1h', '4h','1d'])
        self.timeframe_combobox.setCurrentIndex(2)

        self.limit_combobox = QComboBox(self)
        self.limit_combobox.addItems(['5', '10', '15', '20', '50', '100', '200','400','600','1000'])
        self.limit_combobox.setCurrentIndex(9)



        self.future_steps_combobox = QComboBox(self)
        self.future_steps_combobox.addItems(['2', '3', '4', '5', '10', '15', '20', '25', '30', '35', '40', '45'])
        self.future_steps_combobox.setCurrentIndex(4)
        
        self.past_steps_combobox = QComboBox(self)
        self.past_steps_combobox.addItems(['2', '3', '4', '5', '8', '10', '15', '25', '30', '35', '40', '45','100','200'])
        self.past_steps_combobox.setCurrentIndex(4)
        
        self.to_predict_combobox = QComboBox(self)
        self.to_predict_combobox.addItems(['Body','Close', 'Open', 'High', 'Low', 'Volume'])
        self.to_predict_combobox.setCurrentIndex(2)
        






        # Create buttons for fetching and plotting data
    
        self.next = QPushButton('Next', self)
        icon = QIcon.fromTheme("go-forward")  
        self.next.setIcon(icon)
        self.next.clicked.connect(self.select_next_value)
        
        
        self.last = QPushButton('Last', self)
        icon = QIcon.fromTheme("go-back")  
        self.last.setIcon(icon)
        self.last.clicked.connect(self.select_last_value)
        
        

        
#         self.fetch_button.clicked.connect(self.calculate_and_plot)
        
        self.fetch_button = QPushButton('Process', self)
        self.fetch_button.clicked.connect(self.calculate_and_plot)
        
        self.correlation_button = QPushButton('Corr', self)
        self.correlation_button.clicked.connect(self.sort_by_correlation)
        
        
        
        self.ml_button = QPushButton('ML', self)
        self.ml_button.clicked.connect(self.apply_machine_learning)

        self.historic_button = QPushButton('Historic OHLCV', self)
        self.historic_button.clicked.connect(self.load_historic_ohlcv_data)

        self.test_button = QPushButton('Current OHLCV', self)
        self.test_button.clicked.connect(self.load_test_ohlcv_data)

        # Create labels for the ComboBoxes and plot
        self.symbol_label = QLabel('Symbol:')
        self.timeframe_label = QLabel('Timeframe:')
        self.limit_label = QLabel('Limit:')

        self.future_steps_label = QLabel('Future Steps:')
        self.past_steps_label = QLabel('Past Steps:')
        self.start_candle_label = QLabel('Candle:')
        self.to_predict_label = QLabel('To Predict:')
        self.round_to_label = QLabel('Round To:')

        self.start_candle_edit = QLineEdit(self)
        self.start_candle_edit.setText('2')

        
        

        self.file_dir_label = QLabel('Dir Key:', self)
        self.file_dir_edit = QLineEdit(self)
        self.file_dir_edit.setText('BTC')

        
        holder = ''
        suggestions = []
        
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # Set the full path for the file
        fdir = os.path.join(desktop_path, "ohlcv_data/")
        
        for filename in os.listdir(fdir):
            holder += f"{filename[:-4]},"
            suggestions.append(filename)
            
        self.file_dir_edit.setText('BTC')
        self.file_dir_edit.setPlaceholderText(holder)
        
        # Create a QCompleter and set it for the QLineEdit
        completer = QCompleter(suggestions, self)
        self.file_dir_edit.setCompleter(completer)
        

        # Create HBoxes for ComboBoxes, labels, and the button
        hbox_comboboxes = QHBoxLayout()
        hbox_comboboxes.addWidget(self.symbol_label)
        hbox_comboboxes.addWidget(self.symbol_combobox)
        hbox_comboboxes.addWidget(self.timeframe_label)
        hbox_comboboxes.addWidget(self.timeframe_combobox)
        hbox_comboboxes.addWidget(self.limit_label)
        hbox_comboboxes.addWidget(self.limit_combobox)

        hbox_comboboxes.addWidget(self.future_steps_label)
        hbox_comboboxes.addWidget(self.future_steps_combobox)
        hbox_comboboxes.addWidget(self.past_steps_label)
        hbox_comboboxes.addWidget(self.past_steps_combobox)





        

        hbox_button = QHBoxLayout()
        
        hbox_button.addWidget(self.ml_button)
        hbox_button.addWidget(self.correlation_button)
        hbox_button.addWidget(self.fetch_button)
        hbox_button.addWidget(self.test_button)
        hbox_button.addWidget(self.historic_button)
        
        hbox_button.addWidget(self.last)
        hbox_button.addWidget(self.next)
        

        hbox_button.addWidget(self.to_predict_label)
        hbox_button.addWidget(self.to_predict_combobox)

        hbox_button.addWidget(self.start_candle_label)
        hbox_button.addWidget(self.start_candle_edit)





        hbox_button.addWidget(self.file_dir_label)
        hbox_button.addWidget(self.file_dir_edit)

        


        # Create a central widget to hold layout
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.addLayout(hbox_comboboxes)
        central_layout.addLayout(hbox_button)

        self.max_price_label = QLabel(' ', self)
        central_layout.addWidget(self.max_price_label)
        
        self.predicted_canvas = MplCanvas(self, width=12, height=8, dpi=100)  # Increase width to 12 for larger subplots

        self.original_canvas = MplCanvas(self, width=12, height=8, dpi=100)  # Increase width to 12 for larger subplots

                # Create a tab widget
        self.tabs = QTabWidget(self)

        # Create three tab pages
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(self.tab1, "Predicted")
        self.tabs.addTab(self.tab2, "Original")
        
        
                # Set self.predicted_canvas as the central widget for tab1
        self.tab1_layout = QVBoxLayout()
        self.tab1_layout.addWidget(self.predicted_canvas)
        self.tab1.setLayout(self.tab1_layout)
        
        self.tab2_layout = QVBoxLayout()
        self.tab2_layout.addWidget(self.original_canvas)
        self.tab2.setLayout(self.tab2_layout)
        
        # Set the central widget to the tab widget
        central_layout.addWidget(self.tabs)
        central_widget.setLayout(central_layout)
        self.setCentralWidget(central_widget)
        
        
        

        # Initialize DataFrame and closing_prices_b
        self.df = None



        
        self.messages = ''
    def load_historic_ohlcv_data(self):
        self.df_list = []
        substring_list_text = self.file_dir_edit.text()
        substring_list = [substring.strip() for substring in substring_list_text.split(',')]

        # Define the directory where your CSV files are located
        
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # Set the full path for the file
        directory = os.path.join(desktop_path, "ohlcv_data/")
        
        for filename in os.listdir(directory):
            if filename.endswith("csv") and any(substring in filename for substring in substring_list):                
                file_path = os.path.join(directory, filename)
                df_ = pd.read_csv(file_path)
                df_.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


                self.df_list.append(df_)   
        self.max_price_label.setText('Historical OHLCV loaded')

    def load_test_ohlcv_data(self):
        symbol = self.symbol_combobox.currentText()
        timeframe = self.timeframe_combobox.currentText()
        limit = int(self.limit_combobox.currentText())
        

        future_steps = int(self.future_steps_combobox.currentText())
        to_predict = self.to_predict_combobox.currentText()

        
        exchange = ccxt.binance()
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe,  limit=limit) 
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        df_list_raw = self.df_list.copy()


        df_list_raw.append(df)

        self.combined_df = pd.concat(df_list_raw, ignore_index=True)

        self.combined_df['Date'] = pd.to_datetime(self.combined_df['Date'], unit='ms')
        self.max_price_label.setText('Current OHLCV Loaded')
            
    def calculate_and_plot(self):
        df = None
        df = self.combined_df
        steps = int(self.past_steps_combobox.currentText())
        future_steps = int(self.future_steps_combobox.currentText())
        to_predict = self.to_predict_combobox.currentText()


        start_point= len(df) - int(self.start_candle_edit.text())

        
        df['Body Length'] = abs(df['Close'] - df['Open'])
        df['Balance'] = df['Volume']*(df['Close'] - df['Open'])/(df['High'] - df['Low'])
        df['MaxOC'] = df[['Open', 'Close']].max(axis=1)  # Get the maximum value between Open and Close
        df['Upper Shadow'] = df['High'] - df['MaxOC']

        df['MinOC'] = df[['Open', 'Close']].min(axis=1)  # Get the maximum value between Open and Close
        df['Lower Shadow'] = df['MinOC'] - df['Low']

        df['Color'] = df.apply(lambda row: 1 if row['Close'] > row['Open'] else -1, axis=1)

        df['Body'] = df['Close'] - df['Open']
        df['Candle Size'] = abs(df['High'] - df['Low'])
        df['CloseOpen'] = (df['Close'] + df['Open'])/2
        df['HighLow'] = (df['High'] + df['Low'])/2

        # df['ID1'] = df[(df['Upper Shadow'] > df['Body Length']) and (df['Body Length'] > df['Lower Shadow'])]

        df['ID1'] = ((df['Upper Shadow'] > df['Body Length']) & (df['Body Length'] > df['Lower Shadow']))
        df['ID2'] = ((df['Lower Shadow'] + df['Body Length']) < df['Upper Shadow'])
        df['ID3'] = (df['Lower Shadow'] < df['Upper Shadow'])
        df['ID4'] = ((df['Upper Shadow'] + df['Lower Shadow']) < df['Body Length'])

        df['ID5'] = ((df['Body Length'] / df['Lower Shadow']) < (df['Body Length'] / df['Upper Shadow']))

        df['ID6'] = (df['Body Length'] > (df['Lower Shadow'] + df['Upper Shadow']))

        df['ID7'] = (df['Volume'] > (df['Volume'].mean()))

        # Formula to identify Hammer pattern
        df['ID8'] = ((df['Lower Shadow'] + df['Body Length']) < df['Upper Shadow'])

        # Formula to identify Doji pattern
        df['ID9'] = (abs(df['Open'] - df['Close']) <= (0.1 * df['Body Length']))

        # Formula to identify Bullish Engulfing pattern
        df['ID10'] = ((df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open']) &
                     (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)))
        # Formula to identify Shooting Star pattern
        df['ID11'] = ((df['Upper Shadow'] >= 2 * df['Body Length']) & (df['Upper Shadow'] > df['Lower Shadow']))

        # Formula to identify Bearish Engulfing pattern
        df['ID12'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close']) &
                     (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1)))

        # Formula to identify Three White Soldiers pattern
        df['ID13'] = ((df['Open'] < df['Close']) & (df['Open'].shift(1) < df['Close'].shift(1)) &
                     (df['Close'].shift(1) < df['Open']) & (df['Close'] > df['Open'].shift(1)))

        # Note: In the Bearish Engulfing and Three White Soldiers patterns, we compare the current candle with the previous candle.

        # Formula to identify Bullish Harami pattern
        df['ID14'] = ((df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open']) &
                     (df['Close'] < df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)))

        # Note: In the Bullish Harami pattern, we compare the current candle with the previous candle.

        # Formula to identify Bearish Harami pattern
        df['ID15'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close']) &
                     (df['Open'] < df['Open'].shift(1)) & (df['Close'] > df['Close'].shift(1)))

        df['ID16'] = ((df['Volume'] > df['Volume'].shift(1)) & (df['Body Length'] > df['Body Length'].shift(1)))

        df['ID17'] = ((df['Volume'] /df['Body Length'])  >  df['Volume'].shift(1) / df['Body Length'].shift(1)) &  ((df['Volume'] /df['Body Length'])  >  df['Volume'].shift(2) / df['Body Length'].shift(3)) &  ((df['Volume'] /df['Body Length'])  >  df['Volume'].shift(2) / df['Body Length'].shift(3)) 


        # Formula to identify Inverted Hammer pattern
        df['ID18'] = ((df['Upper Shadow'] < 0.1 * df['Body Length']) & (df['Lower Shadow'] > 2 * df['Body Length']))

        # Formula to identify Piercing pattern
        df['ID19'] = ((df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open']) &
                      (df['Close'] > df['Body Length'].shift(1) * 0.5 + df['Open'].shift(1)) &
                      (df['Open'] < df['Close'].shift(1)))

        # Formula to identify Dark Cloud Cover pattern
        df['ID20'] = ((df['Open'].shift(1) < df['Close'].shift(1)) & (df['Open'] > df['Close']) &
                      (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1)))

        # Formula to identify Morning Star pattern
        df['ID21'] = ((df['Open'].shift(2) > df['Close'].shift(2)) & (df['Close'].shift(2) > df['Open'].shift(1)) &
                      (df['Close'].shift(1) < df['Open']) & (df['Close'] > df['Body Length'].shift(1) * 0.5 + df['Open'].shift(1)))

        # Formula to identify Evening Star pattern
        df['ID22'] = ((df['Open'].shift(2) < df['Close'].shift(2)) & (df['Close'].shift(2) < df['Open'].shift(1)) &
                      (df['Close'].shift(1) > df['Open']) & (df['Close'] < df['Close'].shift(1) - df['Body Length'].shift(1) * 0.5))


        # Formula to identify Bullish Hammer pattern with High Volume
        df['ID23'] = ((df['Lower Shadow'] + df['Body Length']) < df['Upper Shadow']) & (df['Volume'] > df['Volume'].shift(1))

        # Formula to identify Bearish Shooting Star pattern with High Volume
        df['ID24'] = ((df['Upper Shadow'] >= 2 * df['Body Length']) & (df['Upper Shadow'] > df['Lower Shadow']) &
                      (df['Volume'] > df['Volume'].shift(1)))

        # Formula to identify Bullish Harami Cross pattern with High Volume
        df['ID25'] = ((df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open']) &
                      (df['Close'] < df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)) &
                      (df['Volume'] > df['Volume'].shift(1)))

        # Formula to identify Bearish Harami Cross pattern with High Volume
        df['ID26'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close']) &
                      (df['Open'] < df['Open'].shift(1)) & (df['Close'] > df['Close'].shift(1)) &
                      (df['Volume'] > df['Volume'].shift(1)))

        # Formula to identify Bullish Engulfing pattern with High Volume
        df['ID27'] = ((df['Open'].shift(1) > df['Close'].shift(1)) & (df['Close'] > df['Open']) &
                      (df['Close'] > df['Open'].shift(1)) & (df['Open'] < df['Close'].shift(1)) &
                      (df['Volume'] > df['Volume'].shift(1)))

        # Formula to identify Bearish Engulfing pattern with High Volume
        df['ID28'] = ((df['Close'].shift(1) > df['Open'].shift(1)) & (df['Open'] > df['Close']) &
                      (df['Open'] > df['Close'].shift(1)) & (df['Close'] < df['Open'].shift(1)) &
                      (df['Volume'] > df['Volume'].shift(1)))

        # Formula to identify Bullish Abandoned Baby pattern
        df['ID29'] = ((df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2)) &
                      (df['Low'].shift(1) > df['High'].shift(2)) & (df['Low'] < df['Open']) &
                      (df['Close'] > df['High'].shift(1)))

        # Formula to identify Bearish Abandoned Baby pattern
        df['ID30'] = ((df['Close'].shift(2) > df['Open'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(2)) &
                      (df['High'].shift(1) < df['Low'].shift(2)) & (df['High'] > df['Open']) &
                      (df['Close'] < df['Low'].shift(1)))



        # Formula to identify Bullish Three Inside Up pattern
        df['ID31'] = ((df['Close'].shift(2) > df['Open'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2)) &
                      (df['Close'].shift(1) < df['Open']) & (df['Close'] > df['Open']))

        # Formula to identify Bearish Three Inside Down pattern
        df['ID32'] = ((df['Open'].shift(2) > df['Close'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(2)) &
                      (df['Open'].shift(1) > df['Close']) & (df['Close'] < df['Open']))

        # Formula to identify Bullish Three Outside Up pattern
        df['ID33'] = ((df['Close'].shift(2) > df['Open'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2)) &
                      (df['Close'].shift(1) > df['Open']) & (df['Close'] > df['Open']))

        # Formula to identify Bearish Three Outside Down pattern
        df['ID34'] = ((df['Open'].shift(2) > df['Close'].shift(2)) & (df['Close'].shift(1) < df['Open'].shift(2)) &
                      (df['Open'].shift(1) < df['Close']) & (df['Close'] < df['Open']))

        # Formula to identify Bullish Three Stars in the South pattern
        df['ID35'] = ((df['Open'].shift(2) > df['Close'].shift(2)) & (df['Open'].shift(1) > df['Close'].shift(2)) &
                      (df['Close'].shift(1) > df['Open']) & (df['Close'] > df['Open']) &
                      (df['Close'] > df['Open'].shift(1)))

        # Formula to identify Bearish Three Stars in the North pattern
        df['ID36'] = ((df['Close'].shift(2) > df['Open'].shift(2)) & (df['Close'].shift(1) > df['Open'].shift(2)) &
                      (df['Open'].shift(1) > df['Close']) & (df['Close'] < df['Open']) &
                      (df['Close'] < df['Open'].shift(1)))


        df['ID37'] = ((df['Color'] == 1))

        specific_rows = df[(df['ID1'] == df['ID1'].iloc[start_point]) & 
                            (df['ID2'] == df['ID2'].iloc[start_point ]) &
                               (df['ID3'] == df['ID3'].iloc[start_point]) &
                               (df['ID4'] == df['ID4'].iloc[start_point])&
                           (df['ID5'] == df['ID5'].iloc[start_point]) & 
                           (df['ID6'] == df['ID6'].iloc[start_point]) & 
                                  (df['ID7'] == df['ID7'].iloc[start_point]) & 
                                  (df['ID8'] == df['ID8'].iloc[start_point]) & 
                                  (df['ID9'] == df['ID9'].iloc[start_point]) & 
                           (df['ID11'] == df['ID11'].iloc[start_point]) & 
                           (df['ID12'] == df['ID12'].iloc[start_point]) & 
                           (df['ID13'] == df['ID13'].iloc[start_point]) & 
                           (df['ID14'] == df['ID14'].iloc[start_point]) & 
                           (df['ID15'] == df['ID15'].iloc[start_point]) & 
                            (df['ID16'] == df['ID16'].iloc[start_point]) & 
                           (df['ID17'] == df['ID17'].iloc[start_point]) & 
                           (df['ID18'] == df['ID18'].iloc[start_point]) & 
                           (df['ID19'] == df['ID19'].iloc[start_point]) & 
                           (df['ID20'] == df['ID20'].iloc[start_point]) & 
                           (df['ID21'] == df['ID21'].iloc[start_point]) & 
                           (df['ID22'] == df['ID22'].loc[start_point]) & 
                           (df['ID23'] == df['ID23'].loc[start_point]) & 
                           (df['ID24'] == df['ID24'].loc[start_point]) & 
                           (df['ID25'] == df['ID25'].loc[start_point]) & 
                           (df['ID26'] == df['ID26'].loc[start_point]) & 
                           (df['ID27'] == df['ID27'].iloc[start_point]) &
                           (df['ID28'] == df['ID28'].iloc[start_point])&
                          (df['ID29'] == df['ID29'].iloc[start_point])&
                          (df['ID30'] == df['ID30'].iloc[start_point])&
                          (df['ID31'] == df['ID31'].iloc[start_point])&
                          (df['ID32'] == df['ID32'].iloc[start_point])&
                          (df['ID33'] == df['ID33'].iloc[start_point])&
                          (df['ID34'] == df['ID34'].iloc[start_point])&
                          (df['ID35'] == df['ID35'].iloc[start_point])&
                          (df['ID36'] == df['ID36'].iloc[start_point] )& 
                          (df['ID10'] == df['ID10'].iloc[start_point] )& 


                        (df['ID26'] == df['ID26'].loc[start_point]) &  
                           (df['ID27'] == df['ID27'].iloc[start_point]) & 
                           (df['ID28'] == df['ID28'].iloc[start_point])&
                          (df['ID29'] == df['ID29'].iloc[start_point])&
                          (df['ID30'] == df['ID30'].iloc[start_point])&
                          (df['ID31'] == df['ID31'].iloc[start_point])&
                          (df['ID32'] == df['ID32'].iloc[start_point])&
                          (df['ID33'] == df['ID33'].iloc[start_point])&
                          (df['ID34'] == df['ID34'].iloc[start_point])&
                          (df['ID35'] == df['ID35'].iloc[start_point])&
                          (df['ID36'] == df['ID36'].iloc[start_point]  ) &
                          (df['ID10'] == df['ID10'].iloc[start_point]  ) &
                          (df['ID37'] == df['ID37'].iloc[start_point])]





#         accumulated_price = []

        self.original_ohlc = df[['Date','Body', 'Open', 'High', 'Low', 'Close','Volume']].iloc[start_point-steps:start_point+future_steps].reset_index() 


        self.original_ohlc.set_index('Date', inplace=True)
        
        self.original_arrays = {}
        self.norm_original_arrays = {}
        # Loop through the columns in df
        for column_name in [ 'Open', 'High', 'Low', 'Close','Body' , 'Color','Balance']:
        # Extract the desired range of values from the column and convert it to a list
            selected_values = df[column_name].iloc[start_point - steps:start_point + future_steps].to_numpy()
            norm_values = selected_values
            self.original_arrays[column_name]      = selected_values
            self.norm_original_arrays[column_name] = norm_values
    

        
        if 'level_0' in df.columns:
            df = df.drop('level_0', axis=1)
            df = df.reset_index(inplace = True)
            

        to_test_price = df[to_predict].iloc[start_point]



        
        # original_price = new_close

        count = 1
        similiarity_pass = 0
        std_pass = 0 
        Collecting = False
        difference_stage = False
        std_stage = False


        
        results = []
        # Define the target diff and std values

        arrays_close = []
        arrays_open = []
        indexs = []
        coefficients = []
        coefficients_open = []
        list_of_coefficients = []
        self.i = 0
        

        
        original_factors = {}
        original_factors['Derivative'] = np.diff(self.original_arrays[to_predict][:steps+1])
        original_factors['Derivative_Body'] = np.diff(self.original_arrays['Body'][:steps+1])
        original_factors['Std'] = np.std(self.original_arrays[to_predict][:steps+1]) 
        original_factors['Std_Body'] = np.std(self.original_arrays['Body'][:steps+1])
        

        list_correlations_dict = []
        list_norm_predicted_arrays_dict = []
        list_indexs = []
        
        for i, j in zip(specific_rows.index, specific_rows[to_predict]):
            predicted_arrays = {}
            norm_predicted_arrays = {}
        
            predicted_factors = {}
            correlations_dict = {}
        
            if j != to_test_price:
                if not Collecting:
                    
                    self.max_price_label.setText(self.messages)
                    Collecting = True
                try:
                    
                    rows = df.iloc[i - steps : (i + future_steps)]

                    if 'level_0' in rows.columns:
                        rows = rows.drop('level_0', axis=1)
                        rows = rows.reset_index()

                    for column_name in [ 'Open', 'High', 'Low', 'Close','Body' , 'Color','Balance']:
                        # Extract the desired range of values from the column and convert it to a list
                        selected_values = rows[column_name][:steps+1]

                        norm_values = selected_values
                        predicted_arrays[column_name]      = selected_values
                        norm_predicted_arrays[column_name] = norm_values

                    predicted_factors['Derivative'] = np.diff(norm_predicted_arrays[to_predict])
                    predicted_factors['Derivative_Body'] = np.diff(norm_predicted_arrays['Body'])
                    # Calculate standard deviation
                    predicted_factors['Std'] = np.std(norm_predicted_arrays[to_predict])
                    predicted_factors['Std_Body'] = np.std(norm_predicted_arrays['Body'])

                    correlations_dict['Similarity'] = 1/ (1 - cosine(norm_predicted_arrays[to_predict], self.norm_original_arrays[to_predict][:steps+1]))
                    dtw_distance, _  = fastdtw(norm_predicted_arrays[to_predict], self.norm_original_arrays[to_predict][:steps+1])
                    correlations_dict['Dtw_Distance'] = dtw_distance
                    pearsonr_coefficient, _ = pearsonr(norm_predicted_arrays[to_predict], self.norm_original_arrays[to_predict][:steps+1])
                    correlations_dict['Pearson']  = 1 / abs(pearsonr_coefficient) 
                    correlations_dict['Cross_Correlation_Close']  = 1/np.max(correlate(norm_predicted_arrays['Close'], self.norm_original_arrays['Close'][:steps+1], mode='full'))
                    correlations_dict['Cross_Correlation_Open']  = 1/np.max(correlate(norm_predicted_arrays['Open'], self.norm_original_arrays['Open'][:steps+1], mode='full'))
                    correlations_dict['Cross_Correlation_Balance']  = 1/np.max(correlate(norm_predicted_arrays['Balance'], self.norm_original_arrays['Balance'][:steps+1], mode='full'))
                    correlations_dict['Cross_Correlation_Body']  = 1/np.max(correlate(norm_predicted_arrays['Body'], self.norm_original_arrays['Body'][:steps+1], mode='full'))

                    correlations_dict['Derivative']  = np.linalg.norm(original_factors['Derivative'] - predicted_factors['Derivative'])
                    correlations_dict['Derivative_Body']  = np.linalg.norm(original_factors['Derivative_Body'] - predicted_factors['Derivative_Body'])
                    correlations_dict['Std']  = original_factors['Std'] - predicted_factors['Std']
                    correlations_dict['Std_Body']  = original_factors['Std_Body'] - predicted_factors['Std_Body']



                    list_correlations_dict.append(correlations_dict)
                    list_norm_predicted_arrays_dict.append(norm_predicted_arrays)

                    list_indexs.append(i)



        

                except Exception as e: 
                    print(f'{str(e)}')
                

            
            
        # Initialize an empty dictionary to store the result
        correlations_dict = {}
        predicted_arrays_dict = {}
        norm_correlations_dict = {}
        norm_predicted_arrays_dict = {}

        # Iterate through the list of dictionaries

        for dictionary in list_correlations_dict:
            for key, value in dictionary.items():
                # Check if the key is already in the result_dict
                if key not in correlations_dict:
                    correlations_dict[key] = []  # Initialize an empty list for the key if it doesn't exist
                correlations_dict[key].append(value)  # Append the value to the list associated with the key

                

        # Normalize the values
        for key, value in correlations_dict.items():
            value_array = np.array(value)

            normalized_values = (value_array - np.min(value_array))/(np.max(value_array) - np.min(value_array))
            norm_correlations_dict[key] = normalized_values
#         norm_correlations_dict['Pearson'] = 1 - np.array(norm_correlations_dict['Pearson'])
    

        for dictionary in list_norm_predicted_arrays_dict:
            for key, value in dictionary.items():
                # Check if the key is already in the result_dict
                if key not in predicted_arrays_dict:
                    predicted_arrays_dict[key] = []  # Initialize an empty list for the key if it doesn't exist
                predicted_arrays_dict[key].append(value)  # Append the value to the list associated with the key
        norm_predicted_arrays_dict       = predicted_arrays_dict 

        sum_of_correlations =  norm_correlations_dict['Cross_Correlation_Close'] 

#         sum_of_correlations =  norm_correlations_dict['Cross_Correlation_Balance'] + norm_correlations_dict['Cross_Correlation_Body'] + norm_correlations_dict['Cross_Correlation_Close'] + norm_correlations_dict['Cross_Correlation_Open'] + norm_correlations_dict['Similarity'] + norm_correlations_dict['Dtw_Distance'] #+ norm_correlations_dict['Pearson']
        predicted_value = norm_predicted_arrays_dict[to_predict]
        index = list_indexs
        data = [np.array(sum_of_correlations),np.array(index),np.array(predicted_value),np.array(norm_correlations_dict['Cross_Correlation_Close']),np.array(norm_correlations_dict['Similarity']),np.array(norm_correlations_dict['Pearson']),np.array(norm_correlations_dict['Dtw_Distance'])]

        sorted_coefficients  = np.argsort(data[0])
        sorted_indexs = data[1][sorted_coefficients][:]
        sorted_values = data[2][sorted_coefficients][:]
        Cross_Correlation_Close = data[3][sorted_coefficients][:3]
        Similarity = data[4][sorted_coefficients][:3]
        Dtw_Distance = data[5][sorted_coefficients][:3]
        Pearson = data[6][sorted_coefficients][:3]



        
        
        self.final_dataFrame = df.copy()
        

        series_data = self.original_arrays[to_predict][:steps+1]
        # Convert the Series to a NumPy array
        original_price_array = series_data

        curves_2d = np.array(sorted_values).reshape(-1, len(original_price_array))



        # Create a KNN model
        knn = NearestNeighbors(n_neighbors=1)  # Find the nearest neighbor

        # Fit the model on the set of curves
        knn.fit(curves_2d)


        reference_plot_2d = original_price_array.reshape(1, -1)

        # Find the most similar curve using the KNN model
        most_similar_index = knn.kneighbors(reference_plot_2d, n_neighbors=1, return_distance=False)[0][0]
        self.best_index = sorted_indexs[most_similar_index] 

        self.sorted_correlations = sorted_indexs
        self.messages = f'OHLCV Data Processed , Total Match: {len(self.final_dataFrame)}\n'
        self.max_price_label.setText(self.messages)
    def sort_by_correlation(self):
        self.finla_result = None
        steps = int(self.past_steps_combobox.currentText())
        future_steps = int(self.future_steps_combobox.currentText())
        self.closest_index = self.sorted_correlations[self.i]



        start_point= - int(self.start_candle_edit.text())

        self.finla_result = self.final_dataFrame.iloc[self.closest_index-steps:(self.closest_index+future_steps)].reset_index()

        if 'level_0' in self.finla_result.columns:
            self.finla_result = self.finla_result.drop('level_0', axis=1)
            self.finla_result = self.finla_result.reset_index()
        self.Plotting()
        self.messages = f'Total Match: {len(self.sorted_correlations)}/{len(self.final_dataFrame)} , Pattern Index: {self.i} Cross Correlation\n'
        self.max_price_label.setText(self.messages)
        
    def apply_machine_learning(self):
        self.finla_result = None
        steps = int(self.past_steps_combobox.currentText())
        future_steps = int(self.future_steps_combobox.currentText())

        start_point= - int(self.start_candle_edit.text())
        self.finla_result = self.final_dataFrame.iloc[self.best_index-steps:(self.best_index+future_steps)]
        if 'level_0' in self.finla_result.columns:
            self.finla_result = self.finla_result.drop('level_0', axis=1)
            self.finla_result = self.finla_result.reset_index()
        self.Plotting()
        self.messages = f'Total Match: {len(self.sorted_correlations)}/{len(self.final_dataFrame)} , Pattern Index: {self.i} KNN\n'
        self.max_price_label.setText(self.messages)
    def Plotting(self):

        steps = int(self.past_steps_combobox.currentText())
        future_steps = int(self.future_steps_combobox.currentText())



        try:


            predicted_ohlc = self.finla_result[['Date', 'Open', 'High', 'Low', 'Close','Volume']]

            predicted_ohlc.set_index('Date', inplace=True)


            


            self.original_canvas.axs.clear()
            self.predicted_canvas.axs.clear()
            

            middle_original = (self.original_ohlc['Close'].iloc[steps+1]  + self.original_ohlc['Open'].iloc[steps+1] )/2
            middle_predicted = (predicted_ohlc['Close'].iloc[steps+1]  + predicted_ohlc['Open'].iloc[steps+1] )/2
            referance = middle_original - middle_predicted
            predicted_ohlc['High']  = predicted_ohlc['High'] +  referance
            predicted_ohlc['Low']  = predicted_ohlc['Low'] +   referance
            predicted_ohlc['Close']  = predicted_ohlc['Close'] +   referance
            predicted_ohlc['Open']  = predicted_ohlc['Open'] +  referance
            

    
            mpf.plot(self.original_ohlc, type='candle', style='yahoo', ax=self.original_canvas.axs)
            self.original_canvas.axs.axvline(x=steps, color='red', linestyle='--')

            self.original_canvas.axs.grid(True)
            self.original_canvas.axs.legend(loc='best')

            mpf.plot(predicted_ohlc, type='candle', style='yahoo', ax=self.predicted_canvas.axs)
            self.predicted_canvas.axs.axvline(x=steps, color='red', linestyle='--')
            self.predicted_canvas.axs.grid(True)
            self.predicted_canvas.axs.legend(loc='best')

            # Customize subplot titles, labels, etc. as needed
            self.original_canvas.axs.set_title('Original Price')
            self.predicted_canvas.axs.set_title('Predicted Price')

            plt.tight_layout()  # Ensure subplots don't overlap

            self.predicted_canvas.draw()
            self.original_canvas.draw()
        except Exception as e :
            self.messages += f"Operation failed,  Total Predicted Values: {len(self.original_ohlc)}\n {str(e)}"
            self.max_price_label.setText(self.messages)
            traceback.print_exc()
            


    def select_next_value(self):
        if self.i < len(self.sorted_correlations)-1 :
            self.i += 1
            self.sort_by_correlation()
            
    def select_last_value(self):
        if self.i > 0 :
            self.i -= 1
            self.sort_by_correlation()
        




app = None
if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
