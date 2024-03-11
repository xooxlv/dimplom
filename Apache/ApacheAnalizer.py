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



class ApacheAnalizer(DataAnalizer):

    ############ CONFIG SECTION #############
    min_records_for_train = 1000 # минимальное количество записей, необходимых для тренировки нс
    samples_dir = './samples/Apache/'
    model_dir = './models/Apache/'
    known_classes_count = 0
    max_urlparams_len = 100
    ############ CONFIG SECTION #############


    ############ MODEL SECTION #############
    model = None
    encoder_methods = LabelEncoder()
    encoder_y = LabelEncoder()

    scaler_count_codes = MinMaxScaler()
    scaler_coutn_requests = MinMaxScaler()
    scaler_trafic_volume = MinMaxScaler()

    scaler_request_len = MinMaxScaler()
    scaler_request_code = MinMaxScaler()

    url_param_tokenizer = Tokenizer(num_words=max_urlparams_len)
    ############ MODEL SECTION #############

    samples = pd.DataFrame()

    def __init__(self, min_for_train) -> None:
        self.min_records_for_train = min_for_train
        self.load()
        self.samples = pd.DataFrame()



    def save_all_samples(self):

        log('Сохраняем образцы', lvl='info')
        print(self.samples)


        # сохраняем образцы в файлы 
        grouped = self.samples.groupby('y')
        for group_name, group_df in grouped:
            filename = f"{group_name}.csv"  # Имя файла на основе значения y
            group_df.to_csv(self.samples_dir + filename, sep='\t', index=False, mode='w')
        log('Образцы сохранены')

    def save_all_models(self):
        # сохраняем модель в файлы
        self.model.save(self.model_dir + 'model.h5')

        joblib.dump(self.encoder_methods, self.model_dir + 'encoder_methods.pkl')
        joblib.dump(self.encoder_y, self.model_dir + 'encoder_y.pkl')

        joblib.dump(self.scaler_count_codes, self.model_dir + 'scaler_count_codes.pkl')
        joblib.dump(self.scaler_coutn_requests, self.model_dir + 'scaler_coutn_requests.pkl')
        joblib.dump(self.scaler_trafic_volume, self.model_dir + 'scaler_trafic_volume.pkl')
        joblib.dump(self.scaler_request_len, self.model_dir + 'scaler_request_len.pkl')
        joblib.dump(self.scaler_request_code, self.model_dir + 'scaler_request_code.pkl')

        joblib.dump(self.url_param_tokenizer, self.model_dir + 'url_param_tokenizer.pkl')


    # при завершении работы
    def store(self):
        self.save_all_samples()
        self.save_all_models()
        
     
    def load_all_samples(self):

        files = []
        log('Reading samples from', self.samples_dir, lvl='info')
        if os.path.isdir(self.samples_dir):
            contents = os.listdir(self.samples_dir)
            log('Files in dir: ', contents, lvl='info')
            for item in contents:
                full_path = os.path.join(self.samples_dir, item)
                if os.path.isfile(full_path):
                    files.append(full_path)
        
        for file in files:
            log('Reading file: ', file, '...', lvl='info')
            df = pd.read_csv(file, sep='\t')
            self.samples = pd.concat([self.samples, df], ignore_index=True)

        log('Reading samples completed', lvl='info')

    def load_all_models(self):
        log('Loading model from', self.model_dir, lvl='info')
        files = os.listdir(self.model_dir)
        log('Files in dir: ', files, lvl='info')

        if len(files) != 9:
            log('Unable load model from', self.model_dir, 'found', len(files), 'files, expected 9', lvl='error')
            return

        self.encoder_method = joblib.load(self.model_dir + 'encoder_methods.pkl')
        self.encoder_y = joblib.load(self.model_dir + 'encoder_y.pkl')

        self.scaler_count_codes = joblib.load(self.model_dir + 'scaler_count_codes.pkl')
        self.scaler_coutn_requests = joblib.load(self.model_dir + 'scaler_coutn_requests.pkl')
        self.scaler_trafic_volume = joblib.load(self.model_dir + 'scaler_trafic_volume.pkl')
        self.scaler_request_len = joblib.load(self.model_dir + 'scaler_request_len.pkl')
        self.scaler_request_code = joblib.load(self.model_dir + 'scaler_request_code.pkl')
    
        self.url_param_tokenizer = joblib.load(self.model_dir + 'url_param_tokenizer.pkl')

        self.model = load_model(self.model_dir + 'model.h5')

        log('Model loaded successfully', lvl='info')
    
    # загружаем модель и образцы с диска если доступны
    def load(self):
        self.load_all_models()
        self.load_all_samples()


    # готовим выборку для загрузки в нс
    def prepare_train_data(self, df):
        df['y'] = self.encoder_y.fit_transform(df['y'])
        self.known_classes_count = len(self.encoder_y.classes_)

        df['method'] = self.encoder_methods.fit_transform(df['method'])

        code_column = 'code'
        len_column = 'len'

        for column in ['rcm1', 'rcm5', 'rcm30', 'rch3', 'rcd']:
            df[column] = self.scaler_coutn_requests.fit_transform(df[[column]])

        for column in ['c200m1', 'c200m5', 'c200m30', 'c200h3', 'c200d', 
                        'c404m1', 'c404m5', 'c404m30', 'c404h3', 'c404d', 
                        'c403m1', 'c403m5', 'c403m30', 'c403h3', 'c403d']:
            df[column] = self.scaler_count_codes.fit_transform(df[[column]])

        for column in ['volm1', 'volm5', 'volm30', 'volh3', 'vold']:
            df[column] = self.scaler_trafic_volume.fit_transform(df[[column]])

        df['len'] = self.scaler_request_len.fit_transform(df[[len_column]])
        df['code'] = self.scaler_request_code.fit_transform(df[[code_column]])
        
        urls = df['url'].tolist()
        params = df['params'].fillna('0').tolist() 

        combined_text = [str(url) + ' ' + str(param) for url, param in zip(urls, params)]
        self.url_param_tokenizer.fit_on_texts(combined_text)
        sequences = self.url_param_tokenizer.texts_to_sequences(combined_text)

        padded_sequences = pad_sequences(sequences, maxlen=self.max_urlparams_len, padding='post')

        df.drop(columns=['url', 'ua', 'ip', 'params'], inplace=True)

        y = df['y'].values        
        df.drop(columns=['y'], inplace=True)
        X = np.column_stack((padded_sequences, df.values))


        return train_test_split(X, y, test_size=0.2, random_state=42)

    # готовую выборку прогоняем через нс, 
    # при каждом вызове модель перекомпилируется
    def train_model(self):

        samples_copy = self.samples.copy()
        X_train, X_test, y_train, y_test = self.prepare_train_data(samples_copy)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        self.model = Sequential()
        self.model.add(Embedding(input_dim=self.max_urlparams_len+1+len(self.url_param_tokenizer.word_index)+2,
                    output_dim=self.max_urlparams_len,
                    input_length=X_train.shape[1]))
        
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

        self.model.add(Dense(self.known_classes_count, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        res = self.model.fit(X_train, y_train, epochs=1, batch_size=32)

        y_pred = self.model.predict(X_test)

        y_pred_classes = np.argmax(y_pred, axis=1)

        y_true_classes = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        return accuracy



    # набираем данные, запускаем тренировку модели
    # если набрали достаточное количество
    def train(self, records, label):

        report = {}
        
        df = pd.DataFrame(records)
        df['y'] = label


        # загрузить все с диска
        self.load()

        # если данных не было на диске то добавляем только что полученные данные в samples
        if self.samples.empty:
            self.samples = df

        # а если были, то добавляем новые к существующим
        else:   
            self.samples = pd.concat([self.samples, df], ignore_index=True)

            # если количество классов больше 1
            classes = self.samples['y'].unique()
            if len(classes) > 1:
                # готовим отчет с количеством данных каждого из класссов
                grouped = self.samples.groupby('y')
                report['classes_count'] = {}
                for group_name, group_df in grouped:
                    report['classes_count'][group_name] = group_df.shape[0]
                    report['accuracy'] = self.train_model()

            else:
                report['error'] = 'Too less classes for train model'
                report['more'] = 'Existing classes: ' + ', '.join(classes)


        # сохраняем все образцы на диск, чистим samples
        self.store()
        self.samples = pd.DataFrame()

        return report


    def prepare_analize_data(self, recs):
        df = pd.DataFrame(recs)

        # проверить, что все encoders и тому подобные != none

        df['method'] = self.encoder_methods.fit_transform(df['method'])

        code_column = 'code'
        len_column = 'len'

        for column in ['rcm1', 'rcm5', 'rcm30', 'rch3', 'rcd']:
            df[column] = self.scaler_coutn_requests.fit_transform(df[[column]])

        for column in ['c200m1', 'c200m5', 'c200m30', 'c200h3', 'c200d', 
                        'c404m1', 'c404m5', 'c404m30', 'c404h3', 'c404d', 
                        'c403m1', 'c403m5', 'c403m30', 'c403h3', 'c403d']:
            df[column] = self.scaler_count_codes.fit_transform(df[[column]])

        for column in ['volm1', 'volm5', 'volm30', 'volh3', 'vold']:
            df[column] = self.scaler_trafic_volume.fit_transform(df[[column]])

        df['len'] = self.scaler_request_len.fit_transform(df[[len_column]])
        df['code'] = self.scaler_request_code.fit_transform(df[[code_column]])
        
        urls = df['url'].tolist()
        params = df['params'].fillna('0').tolist() 

        combined_text = [url + ' ' + param for url, param in zip(urls, params)]
        sequences = self.url_param_tokenizer.texts_to_sequences(combined_text)

        padded_sequences = pad_sequences(sequences, maxlen=self.max_urlparams_len, padding='post')

        df.drop(columns=['url', 'ua', 'ip', 'params'], inplace=True)

        X = np.column_stack((padded_sequences, df.values))
        return X

    
    # используя записи по конкретному ip пробуем анализировать
    def analize(self, records):
        X = self.prepare_analize_data(records)
        # проверить, что модель норм
        result = self.model.predict(X)
        print(result)
        return 'ok'
