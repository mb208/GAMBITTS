from abc import ABC, abstractmethod

class AgentBase(ABC):
    @abstractmethod
    def action(self, context):
        pass

    @abstractmethod
    def update(self, context, action, text_representation, reward):
        pass