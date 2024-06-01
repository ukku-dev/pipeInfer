class Strategy:
    def execute(self, data):
        pass

class ConcreteStrategyA(Strategy):
    def execute(self, data):
        print(f"Strategy A executed with data: {data}")

class ConcreteStrategyB(Strategy):
    def execute(self, data):
        print(f"Strategy B executed with data: {data}")

class Context:
    def __init__(self, strategy: Strategy):
        self._strategy = strategy

    def set_strategy(self, strategy: Strategy):
        self._strategy = strategy

    def execute_strategy(self, data):
        self._strategy.execute(data)

if __name__ == "__main__":
    data = "some data"
    
    context = Context(ConcreteStrategyA())
    context.execute_strategy(data)
    
    context.set_strategy(ConcreteStrategyB())
    context.execute_strategy(data)
