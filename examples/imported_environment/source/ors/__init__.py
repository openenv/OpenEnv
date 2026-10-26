class Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self):
        return dict(self.__dict__)


class Split(Model):
    pass


class ToolSpec(Model):
    pass


class ListToolsOutput(Model):
    pass


class TextBlock(Model):
    def __init__(self, text, detail=None, type="text"):
        super().__init__(text=text, detail=detail, type=type)


class ToolOutput(Model):
    pass


class RunToolSuccess:
    ok = True

    def __init__(self, output):
        self.output = output


class RunToolOutput:
    def __init__(self, output):
        self.root = RunToolSuccess(output)


class Environment:
    def __init__(self, task_spec=None, secrets=None):
        self.task_spec = task_spec or {}
        self.secrets = secrets or {}

    def setup(self):
        pass

    def teardown(self):
        pass
