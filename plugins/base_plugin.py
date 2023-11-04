class BasePlugin:
    def __init__(self, model):
        self.model = model

    def pre(self, data):
        raise NotImplementedError("This method should be overridden by subclass")

    def apply(self, preprocessed_data):
        raise NotImplementedError("This method should be overridden by subclass")

    def post(self, model_output):
        raise NotImplementedError("This method should be overridden by subclass")

    def cleanup(self):
        raise NotImplementedError("This method should be overridden by subclass")

    def run(self, data):
        preprocessed_data = self.pre(data)
        model_output = self.apply(preprocessed_data)
        postprocessed_output = self.post(model_output)
        self.cleanup()
        return postprocessed_output
