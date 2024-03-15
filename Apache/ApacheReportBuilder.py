from logic.ReportBuilder import *

class ApacheReportBuilder(ReportBuilder):
    def make_report(self, net_results):
        good_report = {}

        rus = {'url': 'URL',
               'ua' : 'User-Agent',
               'params': 'POST параметры',
               'method' : 'Метод',
               'code': 'Статус код',
               'predicted_class': 'Класс атаки'}
        
        attack_ignore = ['test', 'normal']

        if 'results' in net_results:
            good_report['results'] = []
            records = net_results['results']
            for record in records:
                rus_record = {}
                
                for key in record:
                    if key in rus.keys():
                        rus_record[rus[key]] = record[key]

                if rus_record['Класс атаки'] not in attack_ignore:
                    good_report['results'].append(rus_record)

        elif 'accuracy' in net_results:
            good_report['accuracy'] = net_results['accuracy']

        print(good_report)
        return good_report
    
    