from logic.DataParser import DataParser
from collections import deque
import re, log, datetime
from urllib.parse import urlparse, parse_qs

class ApacheLogParser(DataParser):

    def __init__(self) -> None:
        self.temp_storage = deque()

    # сохранить новые данные во временное хранилище
    def save_to_temp(self, data: str):
        lines_data = data.split('\n')
        for line in lines_data:
            self.temp_storage.append(line)
    
    # извлечь данные, которые находятся между A и Z метками
    def pop_a2z_records(self):
        a_mark_passed = False
        z_mark_passed = False
        to_return = []
        record = []

        while len(self.temp_storage) > 0:
            line = self.temp_storage.popleft()
            found_a = re.search(r'--[a-z0-9]{8}-A--', line)
            found_z = re.search(r'--[a-z0-9]{8}-Z--', line)

            if found_a:
                #log.log('Found A', line, lvl='debug')
                a_mark_passed = True
                record.append(line)
                continue
            
            if a_mark_passed:
                if found_z:
                    #log.log('Found Z while A passed', line, lvl='debug')
                    z_mark_passed = True
                record.append(line)

            if z_mark_passed:
                a_mark_passed = False
                z_mark_passed = False
                to_return.append(record)
                record = []

        return to_return


    # получив эти данные, собрать из них массив объектов
    # { ip, timestamp, url, 
    # }
    def parse(self, records):
        parsed_objs = []
        for record in records:
            ip = None
            dt = None
            id = None
            code = None
            ua = None
            content_length = None
            url = None
            method = None
            params = None

            post_params_line_started = False

            parsed_record = {}
            for line in record:
                if id is None:
                    id_found = re.search(r'\s.{27}\s', line)
                    id = None if id_found is None else id_found.group(0).strip()

                if ip is None:
                    ip_found = re.search(r'([0-9]{1,3}[\.]){3}[0-9]{1,3}', line)
                    ip = None if ip_found is None else ip_found.group(0).strip()

                if dt is None:
                    time_found = re.search(r'\[.*\]', line)
                    time = None if time_found is None else time_found.group(0).replace(']', '').replace('[', '')[:-6]
                    if time is not None:
                        time = time.replace(' ', '')
                        dt = datetime.datetime.strptime(time, "%d/%b/%Y:%H:%M:%S.%f")

                if code is None:
                    code_found = re.search(r'HTTP/\d+\.\d+\s(\d+)', line)
                    code = None if code_found is None else code_found.group(1).strip()

                if ua is None:
                    ua_found = re.search(r'User-Agent:\s(.+)', line)
                    ua = None if ua_found is None else ua_found.group(1).strip()

                if content_length is None:
                    content_length_found = re.search(r'Content-Length:\s(\d+)', line)
                    content_length = None if content_length_found is None else content_length_found.group(1).strip()

                if url is None and method is None:
                    if re.search(r'GET|POST|PUT|DELETE', line):
                        rline = line.split(' ')
                        method = rline[0]
                        url = urlparse(rline[1]).path


                if method == 'GET':
                    if re.search(r'GET|POST|PUT|DELETE', line):
                        rline = line.split(' ')
                        params = urlparse(rline[1]).query
                elif method == 'POST':
                    if not post_params_line_started:
                        if re.findall(r'--[0-z]{8}-C--', line):
                            post_params_line_started = True
                            continue
                    else:
                        params = line
                        post_params_line_started = False
                        

                parsed_record['client_ip'] = ip
                parsed_record['request_datetime'] = dt
                parsed_record['request_id'] = id
                parsed_record['status_code'] = code
                parsed_record['user_agent'] = ua
                parsed_record['content_length'] = content_length
                parsed_record['url'] = url
                parsed_record['method'] = method
                parsed_record['params'] = params


            parsed_objs.append(parsed_record)

        return parsed_objs
    
    def parse_data(self, data):
        self.save_to_temp(data)
        raw_records = self.pop_a2z_records()
        if raw_records is not None:
            return self.parse(raw_records)
      
