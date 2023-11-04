from copy import deepcopy
from plugins import BasePlugin

class PluginChain:
    def __init__(self):
        self.plugins = []

    def add(self, plugin):
        if not isinstance(plugin, BasePlugin):
            raise TypeError("Plugin must be a subclass of BasePlugin")
        self.plugins.append(plugin)

    def remove(self, plugin):
        self.plugins.remove(plugin)

    def run(self, data):
        for plugin in self.plugins:
            data = plugin.run(data)
        return data

    def run_yield(self, data, modify_in_place=True):
        for plugin in self.plugins:
            processed_data = plugin.run(deepcopy(data) if not modify_in_place else data)
            yield (plugin, processed_data)