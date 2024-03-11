from logic.SecurityModule import init_module, run_module
from Apache.ApacheLogFile import ApacheLogFile
from Apache.ApacheLogParser import ApacheLogParser
from Apache.ApacheRecordsPreparer import ApacheRecordsPreparer
from Apache.ApacheAnalizer import ApacheAnalizer
from Apache.ApacheReportBuilder import ApacheReportBuilder
from Apache.ApacheReportHandler import ApacheReportHandler
from control.Client import ControlCenterClient

ccc = ControlCenterClient()
ccc.set_module('ApacheIdsModule')

arh = ApacheReportHandler() 
arh.set_handle_instrument(ccc) ## отчеты кидаем в control center


init_module(ApacheLogFile('/var/log/apache2/ids.log'), 
            ApacheLogParser(),
            ApacheRecordsPreparer(),
            ApacheAnalizer(1),
            ApacheReportBuilder(),
            arh, 
            ccc)

run_module()