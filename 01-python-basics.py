"""
Python 基础教程 - 第1章

涵盖内容：
- 变量
- 数据类型
- 运算符
- 控制流
"""

# 变量定义
x = 10
y = 20
name = "Python"
version = 3.11

# 数据类型
number = 42
string = "Hello, World!"
boolean = True
list_data = [1, 2, 3, 4, 5]
dict_data = {"name": "Python", "version": 3.11}

# 运算符
addition = x + y  # 30
subtraction = x - y  # -10
multiplication = x * y  # 200
division = y / x  # 2.0

# 控制流 - if 语句
if x > 0:
    print(f"{x} is positive")
elif x < 0:
    print(f"{x} is negative")
else:
    print(f"{x} is zero")

# 控制流 - for 循环
for i in range(5):
    print(f"Iteration {i}")

# 控制流 - while 循环
count = 0
while count < 5:
    print(f"Count: {count}")
    count += 1

# 函数定义
def greet(name):
    """向用户打招呼"""
    return f"Hello, {name}!"

# 调用函数
message = greet("World")
print(message)

# 列表推导式
squares = [x**2 for x in range(10)]
print(f"Squares: {squares}")

# 字典推导式
square_dict = {x: x**2 for x in range(5)}
print(f"Square dictionary: {square_dict}")

# 异常处理
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")
finally:
    print("Execution completed")

# 类定义
class Person:
    """人类"""
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def introduce(self):
        return f"My name is {self.name} and I am {self.age} years old"

# 使用类
person = Person("Alice", 30)
print(person.introduce())

# 文件操作
with open("output.txt", "w") as f:
    f.write("Hello, File!\n")

# 模块导入
import math
import random
import datetime

# 使用数学函数
print(f"Square root of 16: {math.sqrt(16)}")
print(f"Random number: {random.randint(1, 100)}")
print(f"Current time: {datetime.datetime.now()}")

# 字符串操作
text = "Hello, Python"
print(f"Uppercase: {text.upper()}")
print(f"Lowercase: {text.lower()}")
print(f"Replace: {text.replace('Python', 'World')}")

# 列表操作
numbers = [1, 2, 3, 4, 5]
print(f"First element: {numbers[0]}")
print(f"Last element: {numbers[-1]}")
print(f"Slice: {numbers[1:4]}")
print(f"Length: {len(numbers)}")

# 字典操作
person_dict = {"name": "Bob", "age": 25}
print(f"Keys: {list(person_dict.keys())}")
print(f"Values: {list(person_dict.values())}")
print(f"Items: {list(person_dict.items())}")

# 集合操作
set_a = {1, 2, 3, 4, 5}
set_b = {4, 5, 6, 7, 8}
print(f"Union: {set_a | set_b}")
print(f"Intersection: {set_a & set_b}")
print(f"Difference: {set_a - set_b}")

# 元组操作
point = (10, 20)
print(f"X coordinate: {point[0]}")
print(f"Y coordinate: {point[1]}")

# 格式化字符串
name = "Charlie"
age = 35
print(f"{name} is {age} years old")
print("{} is {} years old".format(name, age))
print("%s is %d years old" % (name, age))

# Lambda 函数
multiply = lambda x, y: x * y
print(f"5 * 6 = {multiply(5, 6)}")

# Map, Filter, Reduce
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, numbers))
even = list(filter(lambda x: x % 2 == 0, numbers))
from functools import reduce
summed = reduce(lambda x, y: x + y, numbers)

print(f"Squared: {squared}")
print(f"Even numbers: {even}")
print(f"Sum: {summed}")

# 装饰器
def decorator(func):
    def wrapper():
        print("Before function call")
        func()
        print("After function call")
    return wrapper

@decorator
def say_hello():
    print("Hello!")

say_hello()

# 生成器
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

print(f"Fibonacci sequence: {list(fibonacci(10))}")

# 上下文管理器
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

# 使用自定义上下文管理器
with FileManager("test.txt", "w") as f:
    f.write("Hello, Context Manager!")

# 类型提示
def add_numbers(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

result = add_numbers(10, 20)
print(f"Result: {result}")

# 枚举
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

print(f"Red color: {Color.RED}")

# 数据类
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

point = Point(10, 20)
print(f"Point: {point}")

# 序列化和反序列化
import json
import pickle

data = {"name": "Dave", "age": 40}

# JSON 序列化
json_str = json.dumps(data)
print(f"JSON: {json_str}")

# 反序列化
parsed_data = json.loads(json_str)
print(f"Parsed: {parsed_data}")

# Pickle 序列化
with open("data.pkl", "wb") as f:
    pickle.dump(data, f)

# 反序列化
with open("data.pkl", "rb") as f:
    loaded_data = pickle.load(f)
print(f"Loaded: {loaded_data}")

# 多线程
import threading
import time

def worker(name, delay):
    print(f"Worker {name} started")
    time.sleep(delay)
    print(f"Worker {name} completed")

threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(f"Thread-{i}", 2))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# 多进程
import multiprocessing

def square(x):
    return x * x

if __name__ == "__main__":
    numbers = [1, 2, 3, 4, 5]
    with multiprocessing.Pool(processes=2) as pool:
        results = pool.map(square, numbers)
    print(f"Squared numbers: {results}")

# 异步编程
import asyncio

async def hello(name):
    print(f"Hello, {name}!")
    await asyncio.sleep(1)
    print(f"Goodbye, {name}!")

async def main():
    await asyncio.gather(
        hello("Alice"),
        hello("Bob"),
        hello("Charlie")
    )

# asyncio.run(main())

# 装饰器进阶
import functools

def count_calls(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"Call {wrapper.calls}")
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper

@count_calls
def say_hi():
    print("Hi!")

say_hi()
say_hi()
say_hi()

# 属性
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14 * self._radius ** 2

circle = Circle(5)
print(f"Area: {circle.area}")

# 静态方法和类方法
class Calculator:
    @staticmethod
    def add(a, b):
        return a + b
    
    @classmethod
    def multiply(cls, a, b):
        return a * b

print(f"Add: {Calculator.add(10, 20)}")
print(f"Multiply: {Calculator.multiply(10, 20)}")

# 抽象基类
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

rect = Rectangle(10, 5)
print(f"Rectangle area: {rect.area()}")

# 迭代器协议
class CountDown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start

for num in CountDown(5):
    print(num)

# 上下文管理器装饰器
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")

with timer():
    sum(range(1000000))

# 类型检查
if isinstance(42, int):
    print("42 is an integer")

if isinstance("Hello", str):
    print('"Hello" is a string')

# 深拷贝和浅拷贝
import copy
original = [[1, 2], [3, 4]]
shallow = copy.copy(original)
deep = copy.deepcopy(original)

shallow[0][0] = 99
deep[0][0] = 88

print(f"Original: {original}")
print(f"Shallow copy: {shallow}")
print(f"Deep copy: {deep}")

print("=== Python 基础教程完成 ===")
