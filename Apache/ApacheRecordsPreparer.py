from datetime import datetime, timedelta
from collections import defaultdict
from log import log

from logic.DataPreparer import DataPreparer

class ApacheRecordsPreparer(DataPreparer):

    def __init__(self):
        self.requests_list = []
        self.users_stat = defaultdict(lambda: {'requests_count': {interval: 0 for interval in ['min1', 'min5', 'min30', 'hour3', 'day']},
                                               'status_code_count': {interval: {'200': 0, '404': 0, '403': 0} for interval in ['min1', 'min5', 'min30', 'hour3', 'day']},
                                               'traffic_volume': {interval: 0 for interval in ['min1', 'min5', 'min30', 'hour3', 'day']}})
        self.intervals = {
            'min1': timedelta(minutes=1),
            'min5': timedelta(minutes=5),
            'min30': timedelta(minutes=30),
            'hour3': timedelta(hours=3),
            'day': timedelta(days=1)
        }

    def update_request_list(self, req):
        self.requests_list.append(req)
        return self

    def update_users_stat(self):
        current_time = datetime.now()
        for r in self.requests_list:
            user = r['client_ip']
            request_time = r['request_datetime']
            content_length = r.get('content_length')
            status_code = r.get('status_code')
            for interval_name, interval in self.intervals.items():
                try:
                    if current_time - request_time <= interval:
                        self.users_stat[user]['requests_count'][interval_name] += 1
                        if status_code is not None:
                            self.users_stat[user]['status_code_count'][interval_name][status_code] += 1
                        if content_length is not None:
                            self.users_stat[user]['traffic_volume'][interval_name] += int(content_length)
                except:
                    pass

    def get_user_stat(self, user):
        return self.users_stat[user]


    temp_buffer = []

    def prepare(self, request):
        self.update_request_list(request).update_users_stat()
        user_stat = self.get_user_stat((request['client_ip']))
        record = {
            'url' : request['url'],
            'params':  request['params'] if request['params'] != '' else 'no', 
            'code':   request['status_code'], 
            'len':   request['content_length'], 
            'ip' :  request['client_ip'], 
            'method' :  request['method'],
            'ua' : request['user_agent'],

            # 'rcm1' : user_stat['requests_count']['min1'], 
            # 'rcm5' : user_stat['requests_count']['min5'],
            # 'rcm30' :  user_stat['requests_count']['min30'],
            # 'rch3' :  user_stat['requests_count']['hour3'],
            # 'rcd' :   user_stat['requests_count']['day'],

                # 200 статусы для анализа брута логинов
            # 'c200m1' : user_stat['status_code_count']['min1']['200'], 
            # 'c200m5' : user_stat['status_code_count']['min5']['200'], 
            # 'c200m30' :  user_stat['status_code_count']['min30']['200'], 
            # 'c200h3' : user_stat['status_code_count']['hour3']['200'], 
            # 'c200d' : user_stat['status_code_count']['day']['200'], 
                            
                # 404 статусы для анализа брута директорий
            # 'c404m1' :  user_stat['status_code_count']['min1']['404'], 
            # 'c404m5' :   user_stat['status_code_count']['min5']['404'], 
            # 'c404m30' :  user_stat['status_code_count']['min30']['404'], 
            # 'c404h3' :   user_stat['status_code_count']['hour3']['404'], 
            # 'c404d' :   user_stat['status_code_count']['day']['404'], 

                # 403 статусы для анализа брута директорий
            # 'c403m1' :   user_stat['status_code_count']['min1']['403'], 
            # 'c403m5' :   user_stat['status_code_count']['min5']['403'], 
            # 'c403m30' :   user_stat['status_code_count']['min30']['403'], 
            # 'c403h3' :   user_stat['status_code_count']['hour3']['403'], 
            # 'c403d' :   user_stat['status_code_count']['day']['403'], 

                # объем трафика пользователя по таймфреймам
            # 'volm1' :  user_stat['traffic_volume']['min1'], 
            # 'volm5' :   user_stat['traffic_volume']['min5'], 
            # 'volm30' :   user_stat['traffic_volume']['min30'], 
            # 'volh3' :   user_stat['traffic_volume']['hour3'], 
            # 'vold' :   user_stat['traffic_volume']['day'] 
        }

        self.temp_buffer.append(record)


    def flush(self):
        self.temp_buffer.clear()

    def get_all(self):
        return self.temp_buffer.copy()