from ..watcher.watcher import Watcher

class DefaultWatcher(Watcher):
    def action(self, res):
        print(res)