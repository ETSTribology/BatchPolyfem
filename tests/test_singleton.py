import pytest
from singleton import SingletonMeta, Config

class TestSingletonClass(metaclass=SingletonMeta):
    def __init__(self, value):
        self.value = value

def test_singleton_instance_creation():
    instance1 = TestSingletonClass("First")
    instance2 = TestSingletonClass("Second")

    assert instance1 is instance2, "Singleton instances are not identical."
    assert instance1.value == "Second", "Singleton instance does not update value as expected."

    print("Test Passed: Singleton behavior is correct.")
