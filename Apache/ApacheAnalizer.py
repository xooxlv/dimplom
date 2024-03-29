import pandas as pd
import numpy as np
from logic.DataAnalizer import DataAnalizer
from log import log
import os


import pandas as pd
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Flatten, Reshape
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, Concatenate
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
        self.params_tokenizer = Tokenizer()
        self.url_tokenizer = Tokenizer()
        self.ua_tokenizer = Tokenizer()

        self.status_code_encoder = LabelEncoder()
        self.method_encoder = LabelEncoder()

        self.max_seq_url_len = 0
        self.max_seq_ua_len = 0
        self.max_seq_params_len = 0


        self.min_for_train = 0
        self.classes_count = 0

        self.samples_dir = './samples/Apache/'
        self.model_dir = './models/Apache/'

        self.train_samples = pd.DataFrame()
        self.new_samples = pd.DataFrame()

        self.y_enc = LabelEncoder()
        self.model = Sequential()

        self.load()

    def train_prepare(self, df: pd.DataFrame):
        # Преобразование целевой переменной в числовой формат
        df['y'] = self.y_enc.fit_transform(df['y'])
        self.classes_count = len(self.y_enc.classes_)

        # Обработка столбца "url"
        self.url_tokenizer.fit_on_texts(df['url'])
        url_sec = self.url_tokenizer.texts_to_sequences(df['url'])
        self.max_seq_url_len = max(len(seq) for seq in url_sec)
        url_input = pad_sequences(url_sec, maxlen=self.max_seq_url_len, padding='post')

        # Обработка столбца "params"
        self.params_tokenizer.fit_on_texts(df['params'])
        params_sec = self.params_tokenizer.texts_to_sequences(df['params'])
        self.max_seq_params_len = max(len(seq) for seq in params_sec)
        params_input = pad_sequences(params_sec, maxlen=self.max_seq_params_len, padding='post')

        # Обработка столбца "ua" (user_agent)
        self.ua_tokenizer.fit_on_texts(df['ua'])
        ua_sec = self.ua_tokenizer.texts_to_sequences(df['ua'])
        self.max_seq_ua_len = max(len(seq) for seq in ua_sec)
        ua_input = pad_sequences(ua_sec, maxlen=self.max_seq_ua_len, padding='post')

        # Преобразование столбца "status_code" в числовой формат
        df['code'] = self.status_code_encoder.fit_transform(df['code'])
        status_code_input = np.array(df['code']).reshape(-1, 1)

        # Преобразование столбца "method" в числовой формат
        df['method'] = self.method_encoder.fit_transform(df['method'])
        method_input = np.array(df['method']).reshape(-1, 1)

        # Объединение всех входных данных в один массив
        X = [url_input, params_input, ua_input, status_code_input, method_input]

        return X, df['y']

    def create_model(self):
        # Входные данные для каждого из столбцов
        input_url = Input(shape=(self.max_seq_url_len,))
        input_params = Input(shape=(self.max_seq_params_len,))
        input_ua = Input(shape=(self.max_seq_ua_len,))
        input_status_code = Input(shape=(1,))
        input_method = Input(shape=(1,))

        # Эмбеддинг для каждого из столбцов
        emb_url = Embedding(input_dim=len(self.url_tokenizer.word_index) + 1, output_dim=10, input_length=self.max_seq_url_len)(input_url)
        emb_params = Embedding(input_dim=len(self.params_tokenizer.word_index) + 1, output_dim=10, input_length=self.max_seq_params_len)(input_params)
        emb_ua = Embedding(input_dim=len(self.ua_tokenizer.word_index) + 1, output_dim=10, input_length=self.max_seq_ua_len)(input_ua)

        # Сверточные слои для каждого из эмбеддингов
        conv_url = Conv1D(64, 3, activation='relu')(emb_url)
        pool_url = MaxPooling1D(2)(conv_url)
        flat_url = GlobalMaxPooling1D()(pool_url)

        conv_params = Conv1D(64, 3, activation='relu')(emb_params)
        pool_params = MaxPooling1D(2)(conv_params)
        flat_params = GlobalMaxPooling1D()(pool_params)

        conv_ua = Conv1D(64, 3, activation='relu')(emb_ua)
        pool_ua = MaxPooling1D(2)(conv_ua)
        flat_ua = GlobalMaxPooling1D()(pool_ua)

        # Объединение эмбеддингов и других входных данных
        merged = Concatenate()([flat_url, flat_params, flat_ua, input_status_code, input_method])

        # Выходной слой с функцией активации softmax (по количеству классов)
        output = Dense(self.classes_count, activation='softmax')(merged)

        # Создание модели
        model = Model(inputs=[input_url, input_params, input_ua, input_status_code, input_method], outputs=output)

        return model

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
            try:
                X, y = self.train_prepare(self.train_samples)
                self.model = self.create_model()
                self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                self.model.fit(X, y, epochs=1, batch_size=32, validation_split=0.2)
                train_loss, train_accuracy = self.model.evaluate(X, y, verbose=0)
                report['accuracy'] = train_accuracy
            except Exception as ex:
                log('Fatal error in train method:', ex.with_traceback(),  lvl='error')
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

        try:
            self.model = load_model(self.model_dir + 'model.h5')
            self.params_tokenizer = joblib.load(self.model_dir + 'params_tokenizer.pkl')
            self.url_tokenizer = joblib.load(self.model_dir + 'url_tokenizer.pkl')
            self.ua_tokenizer = joblib.load(self.model_dir + 'ua_tokenizer.pkl')
            
            self.y_enc = joblib.load(self.model_dir + 'y_enc.pkl')
            self.status_code_encoder = joblib.load(self.model_dir + 'status_code_encoder.pkl')
            self.method_encoder = joblib.load(self.model_dir + 'method_encoder.pkl')

            self.classes_count = len(self.y_enc.classes_)
        except:
            log('Error while read model from', self.model_dir, lvl='error')

    def analize(self, input):
        report = {}
        report['results'] = []

        try:
            df = pd.DataFrame(input)

            url_sec = self.url_tokenizer.texts_to_sequences(df['url'])
            params_sec = self.params_tokenizer.texts_to_sequences(df['params'])
            ua_sec = self.ua_tokenizer.texts_to_sequences(df['ua'])

            ua_input = pad_sequences(ua_sec, maxlen=self.max_seq_ua_len, padding='post')
            params_input = pad_sequences(params_sec, maxlen=self.max_seq_params_len, padding='post')
            url_input = pad_sequences(url_sec, maxlen=self.max_seq_url_len, padding='post')

            tr_code = self.status_code_encoder.fit_transform(df['code'])
            code = np.array(tr_code).reshape(-1, 1)    

            tr_method = self.method_encoder.fit_transform(df['method'])
            method = np.array(tr_method).reshape(-1, 1)
    
            X = [url_input, params_input, ua_input, code, method]

            # Предсказание классов
            pred_probs = self.model.predict(X)
            pred_classes = np.argmax(pred_probs, axis=1)

            # Добавление результатов в отчет
            for url, params, ua, code, method, pred_class in zip(df['url'], df['params'], df['ua'], df['code'], df['method'], self.y_enc.inverse_transform(pred_classes)):
                result = {'url': url,
                          'params': params,
                          'ua': ua,
                          'code' : code,
                          'method': method,
                          'predicted_class': pred_class}
                report['results'].append(result)
            
        except Exception as e:
            log('Fatal error in analize method: {}'.format(str(e)), lvl='fatal')
            report['error'] = 'Fatal error in analize method: {}'.format(str(e))

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
            joblib.dump(self.url_tokenizer, self.model_dir + 'url_tokenizer.pkl')
            joblib.dump(self.ua_tokenizer, self.model_dir + 'ua_tokenizer.pkl')

            joblib.dump(self.status_code_encoder, self.model_dir + 'status_code_encoder.pkl')
            joblib.dump(self.method_encoder, self.model_dir + 'method_encoder.pkl')
            
            joblib.dump(self.y_enc, self.model_dir + 'y_enc.pkl')
        except:
            log('Unable save not compiled model to files', lvl='error')