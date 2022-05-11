from abc import abstractmethod


class Watcher(object):

    @abstractmethod
    def action(self,res):
        pass