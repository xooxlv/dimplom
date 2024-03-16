
from flask import Flask, request
import tkinter as tk
from tkinter import ttk
from threading import Thread
import json
from log import log

app = Flask(__name__)

ids_modules = {'test': {  'accuracy' : 0,
                           'active' : False, 
                           'mode': 'Обучение',
                           'y' : 'Тестовый класс', 
                           'results' : [{'url' : '/url', 'predicted': 'false'}]}}


class ConnectionAccepter:
    def __init__(self, cons, client, cl_accept):
        self.root = tk.Tk()
        self.connections = cons
        self.client = client
        self.cl_acpt = cl_accept
        self.root.title("Новое подключение")

        # Создаем рамку для элементов интерфейса
        self.frame = ttk.Frame(self.root)       
        self.frame.pack(padx=10, pady=10)

        # Название клиента
        self.client_name_label = ttk.Label(self.frame, text="Запрос на подключение модулем " + self.client)
        self.client_name_label.grid(row=0, column=0, padx=5, pady=5)

        # Кнопка для принятия нового подключения
        self.accept_button = ttk.Button(self.frame, text="Принять подключение", command=self.accept_connection)
        self.accept_button.grid(row=1, column=0, padx=5, pady=5)

        # Кнопка для отказа в соединении
        self.reject_button = ttk.Button(self.frame, text="Отклонить подключение", command=self.reject_connection)
        self.reject_button.grid(row=1, column=1, padx=5, pady=5)

    def decide(self):
        self.root.mainloop()

    def accept_connection(self):    
        log('Accepted connection:', self.client, lvl='info')
        self.connections : dict
        self.connections[self.client] = {'active': True,'accuracy' : 0,'mode': '','y' : '','results' : []}
        self.root.destroy()
        self.cl_acpt = True
        main_window.update_combobox_values()

    def reject_connection(self):
        log('Denied connection:', self.client, lvl='info')
        self.root.destroy()
        self.cl_acpt = False



class MainWindow:
    def __init__(self, ids_modules: dict):
        self.root = tk.Tk()
        self.root.title("Модуль контроля и управления")

        # Создание рамки для элементов интерфейса
        self.frame_controls = ttk.Frame(self.root)
        self.frame_controls.pack(pady=15)

        self.selected_sensor = 'test'

        self.mode_label = ttk.Label(self.frame_controls, text="Режим работы:")
        self.mode_label.grid(row=0, column=0, padx=5, pady=5)
        self.mode_var = tk.StringVar()
        self.mode_var.set(ids_modules[self.selected_sensor]['mode'])
        self.mode_button = ttk.Button(self.frame_controls, textvariable=self.mode_var, command=self.toggle_mode)
        self.mode_button.grid(row=0, column=1, padx=5, pady=5)

        self.input_label = ttk.Label(self.frame_controls, text="Метка класса")
        self.input_label.grid(row=0, column=2, padx=5, pady=5)
        self.input_entry = ttk.Entry(self.frame_controls, width=30)
        self.input_entry.grid(row=0, column=3, padx=5, pady=5)
        self.input_entry.insert(0, ids_modules[self.selected_sensor]['y'])

        self.sensor_label = ttk.Label(self.frame_controls, text="Выберите сенсор:")
        self.sensor_label.grid(row=0, column=4, padx=5, pady=5)
        self.sensor_combobox = ttk.Combobox(self.frame_controls, values=[x for x in ids_modules.keys()])
        self.sensor_combobox.grid(row=0, column=5, padx=5, pady=5)
        index = self.sensor_combobox['values'].index(self.selected_sensor)
        self.sensor_combobox.current(index)
        self.sensor_combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)

        self.accuracy_label = ttk.Label(self.frame_controls, text='Точность модели: ' + str(ids_modules[self.selected_sensor]['accuracy']) + '%')
        self.accuracy_label.grid(row=0, column=6, padx=5, pady=5)

        self.frame_table = ttk.Frame(self.root)
        self.frame_table.pack(pady=100)

        self.table = ttk.Treeview(self.frame_table, 
                                  columns=[x for x in ids_modules[self.selected_sensor]['results'][0].keys()],
                                  show="headings",
                                  height=30)

    # вызывать после подключения нового модуля для добавления его в комбобокс 
    def update_combobox_values(self):
        new_values = list(ids_modules.keys())
        self.sensor_combobox['values'] = new_values
        self.sensor_combobox.set(new_values[0])  # Устанавливаем первый элемент по умолчанию
    
    # переключение режимов обучение - анализ (если обучение, заполни "y" поле!!)
    def toggle_mode(self):
        current_mode = self.mode_var.get()
        new_mode = "Обучение" if current_mode == "Анализ" else "Анализ"
        self.mode_var.set(new_mode)
        ids_modules[self.selected_sensor]['mode'] = new_mode
        ids_modules[self.selected_sensor]['y'] = self.input_entry.get()

    # выбран другой сенсор почистить старые данные в полях, отобразить новые
    def on_combobox_select(self, event):
        # заменяем режимы работы и текст y выборки на те, которрые у этого модуля
        self.selected_sensor = self.sensor_combobox.get()
        self.input_entry.delete(0, tk.END)
        self.mode_var.set(ids_modules[self.selected_sensor]['mode'])
        self.input_entry.insert(0, ids_modules[self.selected_sensor]['y'])

        log('Selected sensor - ', self.selected_sensor, lvl='debug')

        # заменяем точность предыдущей модели на точность текущей
        self.accuracy_label.config(text='Точность модели: ' + str(ids_modules[self.selected_sensor]['accuracy']) + '%')

        self.table.delete(*self.table.get_children())

        self.table['columns'] = [x for x in ids_modules[self.selected_sensor]['results'][0].keys()]
        for col in self.table['columns']:
            self.table.heading(col, text=col)

        for result in ids_modules[self.selected_sensor]['results']:
            self.table.insert('', 'end', values=tuple(result.values()))

        self.table.pack()

    # вызывать после отчета об окончании обучения модели (проверь, что выбранный модуль == тому от которого пришел отчет)
    def update_accuracy(self):    
        self.accuracy_label.config(text='Точность модели: ' + str(ids_modules[self.selected_sensor]['accuracy']) + '%')

    # чтобы не отображать уже отображенные строки, содержащиеся в таблице
    def is_record_displayed(self, record_values):
        # Получаем идентификаторы всех строк в таблице
        all_records = self.table.get_children()

        # Проходимся по всем строкам в таблице
        for record_id in all_records:
            # Получаем значения текущей строки
            current_record_values = self.table.item(record_id, 'values')

            # Сравниваем значения текущей строки с данными записи, которую мы хотим проверить
            if current_record_values == record_values:
                # Если значения совпадают, значит, запись уже отображена в таблице
                return True

        # Если не найдено совпадений, значит, запись еще не отображена в таблице
        return False

    # вызывать после получения отчета с модуля безопасности (проверь, что выбранный модуль == тому от которого пришел отчет)
    def update_table(self):

        for record in ids_modules[self.selected_sensor]['results']:
            record_value = tuple(record.values())
            if self.is_record_displayed(record_value):
                pass
                #log('Record already displayed in table\n', record_value, lvl='warn')
            else:
                #log('Display new record\n', record_value, lvl='warn')
                self.table.insert('', 'end', values=record_value)

    def mainloop(self):
        self.root.mainloop()



@app.route('/connect', methods=['GET'])
def connect():
    client_name = request.args.get('name')
    is_client_connected = False
    cur_client_cnt = len(ids_modules.keys())

    ConnectionAccepter(ids_modules, client_name, is_client_connected).decide()

    if len(ids_modules.keys()) > cur_client_cnt:
        return 'ACCEPT'
    else: return 'ACCEPT'

@app.route('/is_work', methods=['GET'])
def is_work():
    client_name = request.args.get('name')
    if ids_modules[client_name].get('active') == True:
        return 'True'
    else: return 'False'

@app.route('/mode', methods=['GET'])
def get_mode():
    client_name = request.args.get('name')
    mode = ids_modules[client_name].get('mode')
    if mode == 'Обучение':
        return 'train'
    elif mode == 'Анализ':
        return 'analize'
    else: return ''
     
@app.route('/y', methods=['GET'])
def get_y():
    client_name = request.args.get('name')
    return ids_modules[client_name].get('y')

@app.route('/report', methods=['POST'])
def report_recv():
    j = request.data.decode('utf-8')
    data_list = eval(j)
    client = data_list['name']
    report = data_list['report']


    log('New report from', client, lvl='info')
    print(report, '\n')

    if 'accuracy' in report.keys():
        ids_modules[client]['accuracy'] = report['accuracy']
        main_window.update_accuracy()
        # вызвать на окне обновление accuracy

    if 'results' in report.keys():
        for result in report['results']:
            ids_modules[client]['results'].append(result)
            main_window.update_table()
        # вызывать на окне добавление в таблицу новой записи

    return 'OK'
    



def run_flask():
    app.run(host='127.0.0.1', port=5669)

main_window = None

if __name__ == '__main__':
    flask_thread = Thread(target=run_flask)
    flask_thread.start()

    main_window = MainWindow(ids_modules)
    main_window.mainloop()

 