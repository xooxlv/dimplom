from requests import *

class ControlCenterClient():

    def set_module(self, module_name):
        self.name = {'name' : module_name}
    
    def connect(self):
        return request('GET', 'http://127.0.0.1:5669/connect', params=self.name).content.decode()

    def is_work(self):
        return request('GET', 'http://127.0.0.1:5669/is_work', params=self.name).content.decode()

    def get_mode(self):
        return request('GET', 'http://127.0.0.1:5669/mode', params=self.name).content.decode()
    
    def get_y(self):
        return request('GET', 'http://127.0.0.1:5669/y', params=self.name).content.decode()
    
    def send_report(self, report):
        request('POST', 'http://127.0.0.1:5669/report', data=str({'name': self.name['name'],
                                                                  'report': report}).encode('utf-8'))