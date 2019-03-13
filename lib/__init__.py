

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Flatten

import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import keras



class Controller:
    data=None

    def __init__(self, path='data3.csv', train_model=True, scaler_path='data3.scale.pkl'):
        if train_model:
            self.data = pd.read_csv(filepath_or_buffer='time_series/data/'+path)
            self.sample = self.select()
            self.target = self.extract_target()
            self.xscaler = MinMaxScaler(feature_range=(0, 1))
            self.yscaler = MinMaxScaler(feature_range=(0, 1))
            self.train, self.test ,self.train_length = self.set_train()
            self.x_train,self.y_train= self.split_data()
            self.x_test, self.y_test = self.split_data(False)
            self.model = self.set_model()
        else:
            self.model = load_model(path)
            with open(scaler_path, 'rb') as f:
                self.xscaler, self.yscaler = pickle.load(f)
        self.graph = tf.get_default_graph()

    def select(self):
        cols = [col for col in self.data.iloc[:, :].columns if 'trade' in col]
        sample = self.data.loc[:, cols]
        sample.dropna(inplace=True)
        return sample

    def extract_target(self):
        target = self.sample.trade_price.values
        self.sample.drop(['trade_price'],axis=1,inplace=True)
        return target

    def scale(self):
        return self.xscaler.fit_transform(self.x_train), self.yscaler.fit_transform(self.y_train.reshape(-1, 1))

    def set_train(self):
        total = self.sample.shape[0]
        train_length = int(total * 0.75)
        train = range(train_length)
        test = range(train_length, total)
        return train,test,train_length

    def split_data(self, train_set=True):
        import numpy as np
        index=self.test
        if train_set:
            index=self.train
        x = self.sample.iloc[index].values
        #x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        y = np.array([self.target[v] for v in index])
        return x,y

    def set_model(self):
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def fit_model(self,save=False):
        x, scaled_target = self.scale()
        scaled_data = np.reshape(x, (x.shape[0], x.shape[1], 1))
        self.model.fit(scaled_data, scaled_target, epochs=101, batch_size=90, verbose=2)
        if save:
            self.model.save('data3.model.h5py')
            with open('data3.scale.pkl', 'wb') as f:
                pickle.dump((self.xscaler, self.yscaler), f)

    def predict(self, x):
        x=self.xscaler.transform(x)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))
        with self.graph.as_default():
            y_predict = self.model.predict(x)
        return self.yscaler.inverse_transform(y_predict)

    def plot(self,y_base, y_predict):
        final = {}
        final['valid'] = y_base
        final['prediction'] = [None for i in range(self.train_length)] + list(y_predict)
        final = pd.DataFrame(final)
        import matplotlib.pyplot as plt
        from matplotlib.pylab import rcParams
        rcParams['figure.figsize'] = 20, 10
        plt.figure(figsize=(16, 8))
        #plt.ylim(0, max(y_predict))
        plt.plot(final[['valid', 'prediction']])
