
class Counter:

    def __init__(self, starting_value, step_value, stop_value):
        self.value = starting_value
        self.step = step_value
        self.stop_value = stop_value

    def increase_counter(self):
        self.value += self.step

    def is_running(self):
        return self.value < self.stop_value

    def add(self, value):
        self.value += value

    def decrease_counter(self):
        self.value -= self.step

    def sub(self, value):
        self.value -= value
