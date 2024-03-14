import pandas as pd
import numpy as np
from logic.DataAnalizer import DataAnalizer
from log import log
import os


import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import load_model
import joblib
from sklearn.svm import SVC


class ApacheAnalizer(DataAnalizer):
    
    def __init__(self) -> None:
        self.params_tokenizer = Tokenizer(char_level=False) # минхэши тут должны быть
        self.max_sec_params_len = 0 
        self.min_for_train = 0
        self.classes_count = 0

        self.samples_dir = './samples/Apache/'
        self.model_dir = './models/Apache/'

        self.train_samples = pd.DataFrame()
        self.new_samples = pd.DataFrame()

        self.y_enc = LabelEncoder()
        self.model = Sequential()

        self.load()

    def train_prepare(self, df):
        # пол. колво классов
        df['y'] = self.y_enc.fit_transform(df['y'])
        self.classes_count = len(self.y_enc.classes_)

        self.params_tokenizer.fit_on_texts(df['params'])
        params_sec = self.params_tokenizer.texts_to_sequences(df['params'])

        self.max_sec_params_len = max(len(seq) for seq in params_sec)
        df['params'] = pad_sequences(params_sec, maxlen=self.max_sec_params_len, padding='post')

        return df

    def train(self, input, y):
        log('Started train model process...', lvl='debug')

        report = {}

        df = pd.DataFrame(input)
        df['y'] = y

        self.new_samples = pd.concat([self.new_samples, df], ignore_index=True)

        if self.new_samples.shape[0] >= self.min_for_train:
            # набрали достаточное количество для тренировки
            # новые образцы сохр в файлы
            # прочитьать все образцы из файлов
            # если классов < 2, ждем новые образцы другого класса

            self.store()
            self.train_samples = pd.DataFrame()
            self.new_samples = pd.DataFrame()
            self.classes_count = 0

            self.load() 

            if self.classes_count < 2:
                log('Classes count < 2. No train', lvl='error')
                report['error'] = 'Classes count < 2'
                return report

            # образцов достаточно, классов >= 2
            # готовим данные для тренировки ембеддинга
            df = self.train_prepare(self.train_samples)

            self.model = Sequential()
            self.model.add(Embedding(input_dim=len(self.params_tokenizer.word_index) + 1, # размер словаря
                                        output_dim=10,                                     # размерность выхода
                                        input_length=self.max_sec_params_len))               # колво входов
            
            self.model.compile(optimizer='adam',          # выберите оптимизатор
                                  loss='sparse_categorical_crossentropy',  # выберите функцию потерь
                                  metrics=['accuracy'])     # выберите метрики, которые вам интересны

            
            X = self.model.predict(df['params'])
            
            X = np.array(X).reshape(len(X), -1)
            y = df['y'].values

            self.svm = SVC(kernel='rbf', decision_function_shape='ovo')
            self.svm.fit(X, y)


            predicted_labels = self.svm.predict(X)
            accuracy = accuracy_score(y, predicted_labels)
            report['accuracy'] = accuracy

        else:
            log('Мало данных:', self.new_samples.shape[0], lvl='error')
            report['error'] = 'Too less data for train'
            report['more'] = 'Train samples: ' + str(self.new_samples.shape[0])


        return report

    def load(self):
        files = []
        log('Reading samples from', self.samples_dir, lvl='info')
        if os.path.isdir(self.samples_dir):
            contents = os.listdir(self.samples_dir)
            log('Files in dir: ', contents, lvl='info')
            for item in contents:
                full_params = os.path.join(self.samples_dir, item)
                if os.path.isfile(full_params):
                    files.append(full_params)
        
        for file in files:
            log('Reading file: ', file, '...', lvl='info')
            df = pd.read_csv(file, sep='\t')
            self.train_samples = pd.concat([self.train_samples, df], ignore_index=True)

        self.classes_count = len(files)
        log('Reading samples completed, classes count:', self.classes_count, lvl='info')


        if len(os.listdir(self.model_dir)) == 4:
            self.model = load_model(self.model_dir + 'model.h5')
            self.params_tokenizer = joblib.load(self.model_dir + 'params_tokenizer.pkl')
            self.y_enc = joblib.load(self.model_dir + 'y_enc.pkl')
            self.svm = joblib.load(self.model_dir  + 'svm_model.joblib')
            self.classes_count = len(self.y_enc.classes_)

    def analize(self, input):
        report = {}
        report['results'] = []

        try:
            df = pd.DataFrame(input)

            self.params_tokenizer.fit_on_texts(df['params'])
            params_sec = self.params_tokenizer.texts_to_sequences(df['params'])
            df['params'] = pad_sequences(params_sec, maxlen=self.max_sec_params_len, padding='post')
    
    
            X = self.model.predict(df['params'])
            X = np.array(X).reshape(len(X), -1)

            for x, i in zip(X, input):
                report['results'].append({'input' : i, 
                               'predicted': str(self.y_enc.classes_[self.svm.predict([x])]) })
        except:
            log('Fatal error in analize method', lvl='fatal')
            report['error'] = 'Fatal error in analize method'

        return report

    def store(self):
        log('Saving samples and model...', lvl='info')
        print(self.new_samples)
            
        # сохраняем образцы в файлы 
        grouped = self.new_samples.groupby('y')
        for group_name, group_df in grouped:
            filename = f"{group_name}.csv"  # Имя файла на основе значения y
            if os.path.exists(self.samples_dir + filename):
                group_df.to_csv(self.samples_dir + filename, sep='\t', index=False, header=None, mode='a')
            else: group_df.to_csv(self.samples_dir + filename, sep='\t', index=False, mode='w')


        try:
            self.model.save(self.model_dir + 'model.h5')

            joblib.dump(self.params_tokenizer, self.model_dir + 'params_tokenizer.pkl')
            joblib.dump(self.y_enc, self.model_dir + 'y_enc.pkl')
            joblib.dump(self.svm, self.model_dir  + 'svm_model.joblib')
        except:
            log('Unable save not compiled model to files', lvl='error')