from .context import Context

class Step:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, context: Context):
        pass

    def pre_run(self, context: Context):
        context.logger.info(f"="*60)
        context.logger.info(f"[{self.name}] {self.description}")

    def post_run(self, context: Context):
        pass  
