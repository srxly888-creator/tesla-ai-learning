"""
Python 高级教程 - 第2章

涵盖内容：
- 元类
- 描述符
- 装饰器进阶
- 生成器进阶
- 协程
- 上下文管理器进阶
- 反射
- 动态属性
- 槽位
- 弱引用
"""

# 元类
class MetaClass(type):
    """自定义元类"""
    def __new__(cls, name, bases, namespace):
        print(f"Creating class {name}")
        # 添加新属性
        namespace['meta_info'] = f"Created by MetaClass"
        return super().__new__(cls, name, bases, namespace)

class MyClass(metaclass=MetaClass):
    def __init__(self):
        self.data = "Hello"

obj = MyClass()
print(obj.meta_info)

# 描述符
class Descriptor:
    """属性描述符"""
    def __init__(self, name=None):
        self.name = name
        self.value = None
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return f"Getting {self.name}: {self.value}"
    
    def __set__(self, instance, value):
        self.value = value
        print(f"Setting {self.name} to {value}")

class Person:
    name = Descriptor("name")
    age = Descriptor("age")

person = Person()
person.name = "Alice"
person.age = 30
print(person.name)
print(person.age)

# 装饰器类
class CountDecorator:
    """计数装饰器类"""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        print(f"Function called {self.count} times")
        return self.func(*args, **kwargs)

@CountDecorator
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
print(greet("Bob"))
print(greet("Charlie"))

# 带参数的装饰器
def repeat(times):
    """重复调用装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def say_hello():
    print("Hello!")

say_hello()

# 生成器表达式
squares_gen = (x**2 for x in range(10))
print(f"Generator: {list(squares_gen)}")

# 协程
async def producer():
    """生产者协程"""
    for i in range(5):
        print(f"Producing {i}")
        await asyncio.sleep(0.1)

async def consumer():
    """消费者协程"""
    for i in range(5):
        print(f"Consuming {i}")
        await asyncio.sleep(0.1)

async def main_coroutine():
    await asyncio.gather(producer(), consumer())

# asyncio.run(main_coroutine())

# 反射
class ReflectiveClass:
    def __init__(self):
        self.public_var = "public"
        self._protected_var = "protected"
        self.__private_var = "private"

obj = ReflectiveClass()

# 获取所有属性
print(f"Attributes: {dir(obj)}")

# 动态获取属性
print(f"public_var: {getattr(obj, 'public_var')}")

# 动态设置属性
setattr(obj, 'new_var', 'new value')
print(f"new_var: {obj.new_var}")

# 检查属性是否存在
print(f"Has public_var: {hasattr(obj, 'public_var')}")

# 删除属性
delattr(obj, 'new_var')
print(f"Has new_var: {hasattr(obj, 'new_var')}")

# 调用方法
if hasattr(obj, '__init__'):
    print("Has __init__ method")

# 动态属性
class DynamicClass:
    def __getattr__(self, name):
        return f"Dynamic attribute: {name}"
    
    def __setattr__(self, name, value):
        print(f"Setting {name} to {value}")
        object.__setattr__(self, name, value)
    
    def __delattr__(self, name):
        print(f"Deleting {name}")
        object.__delattr__(self, name)

dynamic_obj = DynamicClass()
print(dynamic_obj.nonexistent)
dynamic_obj.new_attr = "value"
del dynamic_obj.new_attr

# 槽位
class SlotClass:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

slot_obj = SlotClass("Alice", 30)
print(f"Name: {slot_obj.name}")

# 弱引用
import weakref

class BigObject:
    def __del__(self):
        print("BigObject deleted")

big_obj = BigObject()
weak_ref = weakref.ref(big_obj)

print(f"Object exists: {weak_ref() is not None}")
del big_obj
print(f"Object exists after delete: {weak_ref() is not None}")

# 上下文管理器进阶
class DatabaseConnection:
    def __init__(self, dbname):
        self.dbname = dbname
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.dbname}")
        self.connection = f"Connection to {self.dbname}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing connection to {self.dbname}")
        if exc_type:
            print(f"Error occurred: {exc_val}")
        return True  # 抑制异常

with DatabaseConnection("mydb") as conn:
    print(f"Connection: {conn}")
    # 模拟错误
    raise Exception("Database error!")

# 魔术方法
class MagicMethods:
    def __init__(self, value):
        self.value = value
    
    def __str__(self):
        return f"MagicMethods({self.value})"
    
    def __repr__(self):
        return f"<MagicMethods: {self.value}>"
    
    def __add__(self, other):
        return MagicMethods(self.value + other.value)
    
    def __eq__(self, other):
        return self.value == other.value
    
    def __len__(self):
        return len(str(self.value))
    
    def __getitem__(self, key):
        return str(self.value)[key]
    
    def __setitem__(self, key, value):
        # 演示用法
        pass

magic1 = MagicMethods(10)
magic2 = MagicMethods(20)
print(f"String: {magic1}")
print(f"Repr: {repr(magic1)}")
print(f"Addition: {magic1 + magic2}")
print(f"Equal: {magic1 == magic2}")
print(f"Length: {len(magic1)}")
print(f"Item: {magic1[0]}")

# 迭代器进阶
class PrimeIterator:
    """素数迭代器"""
    def __init__(self):
        self.current = 2
    
    def __iter__(self):
        return self
    
    def __next__(self):
        prime = self.current
        self.current = self._next_prime()
        return prime
    
    def _next_prime(self):
        """查找下一个素数"""
        candidate = self.current + 1
        while not self._is_prime(candidate):
            candidate += 1
        return candidate
    
    def _is_prime(self, n):
        """检查是否为素数"""
        if n < 2:
            return False
        for i in range(2, int(n ** 0.5) + 1):
            if n % i == 0:
                return False
        return True

prime_iter = PrimeIterator()
primes = [next(prime_iter) for _ in range(10)]
print(f"First 10 primes: {primes}")

# 可迭代对象
class FibonacciIterable:
    """斐波那契数列可迭代对象"""
    def __init__(self, limit):
        self.limit = limit
        self.a, self.b = 0, 1
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.a > self.limit:
            raise StopIteration
        current = self.a
        self.a, self.b = self.b, self.a + self.b
        return current

fib_iter = FibonacciIterable(100)
print(f"Fibonacci numbers <= 100: {list(fib_iter)}")

# 上下文管理器嵌套
from contextlib import ExitStack

class Resource1:
    def __enter__(self):
        print("Resource1 acquired")
        return "Resource1"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Resource1 released")

class Resource2:
    def __enter__(self):
        print("Resource2 acquired")
        return "Resource2"
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Resource2 released")

with ExitStack() as stack:
    r1 = stack.enter_context(Resource1())
    r2 = stack.enter_context(Resource2())
    print(f"Using {r1} and {r2}")

# 类型提示进阶
from typing import List, Dict, Optional, Union, Callable, TypeVar, Generic

T = TypeVar('T')

class Container(Generic[T]):
    def __init__(self, value: T):
        self.value = value
    
    def get(self) -> T:
        return self.value

def process_data(
    numbers: List[int],
    mapping: Dict[str, int],
    callback: Optional[Callable[[int], int]] = None
) -> Union[int, None]:
    """处理数据的函数"""
    result = 0
    for num in numbers:
        if callback:
            num = callback(num)
        result += num
    return result

numbers = [1, 2, 3, 4, 5]
mapping = {"a": 1, "b": 2}
result = process_data(numbers, mapping, lambda x: x * 2)
print(f"Processed result: {result}")

# 抽象基类进阶
from abc import ABC, abstractmethod, ABCMeta

class AbstractShape(ABC, metaclass=ABCMeta):
    @abstractmethod
    def area(self) -> float:
        pass
    
    @abstractmethod
    def perimeter(self) -> float:
        pass
    
    def describe(self):
        return f"Shape with area {self.area()} and perimeter {self.perimeter()}"

class Square(AbstractShape):
    def __init__(self, side: float):
        self.side = side
    
    def area(self) -> float:
        return self.side ** 2
    
    def perimeter(self) -> float:
        return 4 * self.side

square = Square(5)
print(square.describe())

# 协议（Python 3.8+）
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> None:
        ...

class Circle:
    def draw(self) -> None:
        print("Drawing a circle")

class Rectangle:
    def draw(self) -> None:
        print("Drawing a rectangle")

def draw_shape(shape: Drawable) -> None:
    shape.draw()

circle = Circle()
rectangle = Rectangle()
draw_shape(circle)
draw_shape(rectangle)

# 数据类进阶
from dataclasses import dataclass, field
from typing import List

@dataclass
class Student:
    name: str
    age: int
    grades: List[int] = field(default_factory=list)
    
    def average(self) -> float:
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)

student = Student("Alice", 20, [90, 85, 95])
print(f"Average grade: {student.average():.2f}")

# 单例模式
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(f"Same instance: {s1 is s2}")

# 工厂模式
class AnimalFactory:
    @staticmethod
    def create_animal(animal_type: str):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        else:
            raise ValueError(f"Unknown animal type: {animal_type}")

class Dog:
    def speak(self):
        print("Woof!")

class Cat:
    def speak(self):
        print("Meow!")

dog = AnimalFactory.create_animal("dog")
cat = AnimalFactory.create_animal("cat")
dog.speak()
cat.speak()

# 观察者模式
from typing import List, Callable

class Observable:
    def __init__(self):
        self._observers: List[Callable] = []
    
    def attach(self, observer: Callable):
        self._observers.append(observer)
    
    def detach(self, observer: Callable):
        self._observers.remove(observer)
    
    def notify(self, data):
        for observer in self._observers:
            observer(data)

class Observer:
    def update(self, data):
        print(f"Received data: {data}")

observable = Observable()
observer1 = Observer()
observer2 = Observer()

observable.attach(observer1.update)
observable.attach(observer2.update)
observable.notify("Hello, observers!")

# 策略模式
class SortingStrategy:
    def sort(self, data: List) -> List:
        pass

class BubbleSort(SortingStrategy):
    def sort(self, data: List) -> List:
        n = len(data)
        for i in range(n):
            for j in range(0, n-i-1):
                if data[j] > data[j+1]:
                    data[j], data[j+1] = data[j+1], data[j]
        return data

class QuickSort(SortingStrategy):
    def sort(self, data: List) -> List:
        if len(data) <= 1:
            return data
        pivot = data[len(data) // 2]
        left = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)

class Sorter:
    def __init__(self, strategy: SortingStrategy):
        self.strategy = strategy
    
    def set_strategy(self, strategy: SortingStrategy):
        self.strategy = strategy
    
    def sort(self, data: List) -> List:
        return self.strategy.sort(data)

data = [64, 34, 25, 12, 22, 11, 90]
sorter = Sorter(BubbleSort())
print(f"Bubble sort: {sorter.sort(data.copy())}")
sorter.set_strategy(QuickSort())
print(f"Quick sort: {sorter.sort(data.copy())}")

# 责任链模式
class Handler:
    def __init__(self):
        self.next_handler = None
    
    def set_next(self, handler: 'Handler') -> 'Handler':
        self.next_handler = handler
        return handler
    
    def handle(self, request):
        if self.next_handler:
            return self.next_handler.handle(request)
        return None

class ConcreteHandler1(Handler):
    def handle(self, request):
        if request == "request1":
            return f"Handled by Handler1: {request}"
        else:
            return super().handle(request)

class ConcreteHandler2(Handler):
    def handle(self, request):
        if request == "request2":
            return f"Handled by Handler2: {request}"
        else:
            return super().handle(request)

handler1 = ConcreteHandler1()
handler2 = ConcreteHandler2()
handler1.set_next(handler2)

print(handler1.handle("request1"))
print(handler1.handle("request2"))
print(handler1.handle("request3"))

print("=== Python 高级教程完成 ===")
