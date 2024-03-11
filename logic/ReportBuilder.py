from abc import ABC, abstractmethod

class ReportBuilder():
    @abstractmethod
    def make_report(self, net_results):
        return None