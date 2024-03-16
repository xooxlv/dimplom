
from logic.SecurityModule import init_module, run_module

from Https.HttpsProxyLog import HttpsProxyLog
from Https.HttpTrafficParser import HttpTrafficParser
from Https.HttpRecordsPreparer import HttpRecordsPreparer
from Https.HttpTrafficAnalizer import HttpTrafficAnalizer
from Https.HttpReportBuilder import HttpReportBuilder
from Https.HttpReportHandler import HttpReportHandler
from control.Client import ControlCenterClient

ccc = ControlCenterClient()
ccc.set_module('HttpsIdsModule')

rh = HttpReportHandler()
rh.set_client(ccc)


init_module(HttpsProxyLog('./proxy.log'),
            HttpTrafficParser(),
            HttpRecordsPreparer(),
            HttpTrafficAnalizer(),
            HttpReportBuilder(),
            rh,
            ccc)

run_module()