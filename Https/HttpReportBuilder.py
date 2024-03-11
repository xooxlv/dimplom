from logic.ReportBuilder import ReportBuilder

class HttpReportBuilder(ReportBuilder):
    def make_report(self, net_results):
        
        good_report = {}

        rus = {'path': 'URL', 
               'predicted': 'Класс атаки'}
        
        attack_ignore = ['test', 'normal']

        if 'results' in net_results:
            good_report['results'] = []
            res = net_results['results']
            for item in res:
                good_record = {}
                if 'input' in item:
                    for col in item['input']:
                        good_record[col] = item['input'][col]
                if 'predicted' in item:
                    good_record['predicted'] = item['predicted']
                
                # перевод на русский
                
                best_record = {}
                for key in good_record:
                    if key in rus:
                        best_record[rus[key]] = good_record.get(key)

                best_record['Класс атаки'] = best_record['Класс атаки'].replace('\'', '').replace('[', '').replace(']', '')

                # если это не атака, то не отправляем
                if best_record['Класс атаки'] not in attack_ignore:
                    good_report['results'].append(best_record)


        if 'accuracy' in net_results:
            good_report['accuracy'] = net_results['accuracy']

        return good_report
    
    