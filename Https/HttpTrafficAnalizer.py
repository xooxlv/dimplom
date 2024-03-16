from logic.DataAnalizer import DataAnalizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from keras.models import load_model
import joblib
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from log import log

from sklearn.svm import SVC
import os, json


class HttpTrafficAnalizer(DataAnalizer):

    def __init__(self) -> None:
        self.path_tokenizer = Tokenizer(char_level=False) # минхэши тут должны быть
        self.max_sec_path_len = 0 
        self.new_samples = pd.DataFrame()
        self.min_for_train = 0
        self.classes_count = 0

        self.samples_dir = './samples/Https/'
        self.model_dir = './models/Https/'

        self.train_samples = pd.DataFrame()
        self.y_enc = LabelEncoder()
        self.emb_path = Sequential()

        self.params_vectorizer = CountVectorizer()

        self.load()

    def fill_params(self, df):
        for index, row in df.iterrows():
            if row['formdata'] == '{}':  # Проверяем, равно ли значение столбца 'formdata' '{}'
                df.at[index, 'params'] = row['query']  # Если равно, то копируем данные из 'query' в 'params'
            elif row['query'] == '{}':  # Проверяем, равно ли значение столбца 'query' '{}'
                df.at[index, 'params'] = row['formdata']  # Если равно, то копируем данные из 'formdata' в 'params'
            else:
                df['params'] = '{\'key\': \'no\'}'

        return df[['params', 'path', 'y']]

    def train_prepare(self, df):
        print(df)
        df['y'] = self.y_enc.fit_transform(df['y'])
        self.classes_count = len(self.y_enc.classes_)

        self.path_tokenizer.fit_on_texts(df['path'])
        path_sec = self.path_tokenizer.texts_to_sequences(df['path'])

        max_sec_path_len = max(len(seq) for seq in path_sec)
        padded_path_seq = pad_sequences(path_sec, maxlen=max_sec_path_len, padding='post')
        
        df['params'] = df['params'].str.replace("'", "\"")
        values_list = []
        for index, row in df.iterrows():
            try:
                data_dict = json.loads(row['params'])
                values_list.append(data_dict.values())
            except:
                log(f'Unable parse from json:\n{row}\n',  lvl='error')
                values_list.append(['no'])

        storage = []
        for values in values_list:
            storage.append(''.join(values))

        transformed_params = self.params_vectorizer.fit_transform(storage)
        combined_features = hstack((padded_path_seq, transformed_params))
        combined_features_array = combined_features.toarray() 

        return combined_features_array, df['y']

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
            df = self.fill_params(self.train_samples.copy())
            X, y = self.train_prepare(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Пример: 20% тестовых данных

            svm = SVC(kernel='rbf', decision_function_shape='ovo')
            svm.fit(X_train, y_train)
            y_pred = svm.predict(X_test)
            print(y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            print("Accuracy:", accuracy)

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
                full_path = os.path.join(self.samples_dir, item)
                if os.path.isfile(full_path):
                    files.append(full_path)
        
        for file in files:
            log('Reading file: ', file, '...', lvl='info')
            df = pd.read_csv(file, sep='\t')
            self.train_samples = pd.concat([self.train_samples, df], ignore_index=True)

        self.classes_count = len(files)
        log('Reading samples completed, classes count:', self.classes_count, lvl='info')


        if len(os.listdir(self.model_dir)) == 4:
            self.emb_path = load_model(self.model_dir + 'model.h5')
            self.path_tokenizer = joblib.load(self.model_dir + 'path_tokenizer.pkl')
            self.y_enc = joblib.load(self.model_dir + 'y_enc.pkl')
            self.svm = joblib.load(self.model_dir  + 'svm_model.joblib')
            self.classes_count = len(self.y_enc.classes_)

    def analize(self, input):
        report = {}
        report['results'] = []

        try:
            df = pd.DataFrame(input)
            df['y'] = 'no label'

            df = self.fill_params(df)[['path', 'params']]

            path_sec = self.path_tokenizer.texts_to_sequences(df['path'])
            padded_path_seq = pad_sequences(path_sec, maxlen=self.max_sec_path_len, padding='post')

            values_list = []
            for index, row in df.iterrows():
                try:
                    data_dict = json.loads(row['params'])
                    values_list.append(data_dict.values())
                except:
                    log(f'Unable parse from json:\n{row}\n',  lvl='error')
                    values_list.append(['no'])

            storage = []
            for values in values_list:
                storage.append(''.join(values))
        
            transformed_params = self.params_vectorizer.transform(storage)
            combined_features = hstack((padded_path_seq, transformed_params))
            X = combined_features.toarray() 

            predictions = self.svm.predict(X)
            pred_labels = self.y_enc.inverse_transform(predictions)
            
            # Создаем отчет
            for idx, pred_label in enumerate(pred_labels):
                result = {
                    'path': df.iloc[idx]['path'],
                    'params': df.iloc[idx]['params']
                }
                report['results'].append({'input': result, 'prediction': pred_label})

        except Exception as ex:
            log(f'Fatal error in analize method: {ex}', lvl='fatal')
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
            self.emb_path.save(self.model_dir + 'model.h5')

            joblib.dump(self.path_tokenizer, self.model_dir + 'path_tokenizer.pkl')
            joblib.dump(self.y_enc, self.model_dir + 'y_enc.pkl')
            joblib.dump(self.svm, self.model_dir  + 'svm_model.joblib')
        except:
            log('Unable save not compiled model to files', lvl='error')