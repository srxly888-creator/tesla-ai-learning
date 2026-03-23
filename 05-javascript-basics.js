/**
 * JavaScript 基础教程 - 第5章
 * 
 * 涵盖内容：
 * - 变量和数据类型
 * - 运算符
 * - 控制流
 * - 函数
 * - 对象和数组
 * - DOM 操作
 * - 事件处理
 * - 异步编程
 * - 模块化
 */

// ============================================
// 第1节：变量和数据类型
// ============================================

// 变量声明
var oldWay = "旧的声明方式";
let modernWay = "现代的声明方式";
const constant = "常量";

// 数据类型
let string = "Hello, World!";
let number = 42;
let float = 3.14;
let boolean = true;
let nullValue = null;
let undefinedValue = undefined;
let bigInt = 9007199254740991n;

// 类型检查
console.log(typeof string);      // "string"
console.log(typeof number);       // "number"
console.log(typeof boolean);      // "boolean"
console.log(typeof undefined);    // "undefined"
console.log(typeof null);         // "object" (历史遗留问题)

// 类型转换
let numStr = "123";
let num = Number(numStr);         // 123
let str = String(456);            // "456"
let bool = Boolean(1);            // true

console.log(num, str, bool);

// ============================================
// 第2节：运算符
// ============================================

// 算术运算符
let a = 10;
let b = 3;
console.log(a + b);   // 13 加法
console.log(a - b);   // 7  减法
console.log(a * b);   // 30 乘法
console.log(a / b);   // 3.333... 除法
console.log(a % b);   // 1  取模
console.log(a ** b);  // 1000 幂运算

// 比较运算符
console.log(5 == 5);    // true
console.log(5 != 3);    // true
console.log(5 > 3);     // true
console.log(5 < 3);     // false
console.log(5 >= 5);    // true
console.log(5 <= 3);    // false

// 逻辑运算符
console.log(true && false);  // false
console.log(true || false);  // true
console.log(!true);          // false

// 字符串运算符
let firstName = "John";
let lastName = "Doe";
let fullName = firstName + " " + lastName;
console.log(fullName);  // "John Doe"

// 三元运算符
let age = 20;
let canVote = age >= 18 ? "Yes" : "No";
console.log(canVote);  // "Yes"

// ============================================
// 第3节：控制流
// ============================================

// if-else 语句
let score = 85;
if (score >= 90) {
    console.log("A");
} else if (score >= 80) {
    console.log("B");  // 输出这个
} else if (score >= 70) {
    console.log("C");
} else {
    console.log("F");
}

// switch 语句
let day = 3;
switch (day) {
    case 1:
        console.log("Monday");
        break;
    case 2:
        console.log("Tuesday");
        break;
    case 3:
        console.log("Wednesday");  // 输出这个
        break;
    default:
        console.log("Other day");
}

// for 循环
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// while 循环
let count = 0;
while (count < 3) {
    console.log(count);
    count++;
}

// do-while 循环
let num2 = 0;
do {
    console.log(num2);
    num2++;
} while (num2 < 3);

// for...in 循环（对象）
let person = {name: "Alice", age: 30, city: "NYC"};
for (let key in person) {
    console.log(key + ": " + person[key]);
}

// for...of 循环（数组）
let colors = ["red", "green", "blue"];
for (let color of colors) {
    console.log(color);
}

// break 和 continue
for (let i = 0; i < 10; i++) {
    if (i === 3) continue;  // 跳过 3
    if (i === 7) break;     // 在 7 时停止
    console.log(i);
}

// ============================================
// 第4节：函数
// ============================================

// 函数声明
function greet(name) {
    return "Hello, " + name + "!";
}
console.log(greet("World"));  // "Hello, World!"

// 函数表达式
let add = function(a, b) {
    return a + b;
};
console.log(add(5, 3));  // 8

// 箭头函数
let multiply = (a, b) => a * b;
console.log(multiply(4, 5));  // 20

// 箭头函数（多行）
let calculate = (a, b, operation) => {
    if (operation === "add") return a + b;
    if (operation === "subtract") return a - b;
    if (operation === "multiply") return a * b;
    if (operation === "divide") return a / b;
};
console.log(calculate(10, 5, "add"));       // 15
console.log(calculate(10, 5, "subtract")); // 5

// 默认参数
let greetWithDefault = (name = "Guest") => {
    return "Hello, " + name + "!";
};
console.log(greetWithDefault());       // "Hello, Guest!"
console.log(greetWithDefault("Alice")); // "Hello, Alice!"

// Rest 参数
let sumAll = (...numbers) => {
    return numbers.reduce((sum, num) => sum + num, 0);
};
console.log(sumAll(1, 2, 3, 4, 5));  // 15

// 解构参数
let displayPerson = ({name, age}) => {
    console.log(name + " is " + age + " years old");
};
displayPerson({name: "Bob", age: 25});

// 高阶函数
let applyOperation = (a, b, operation) => {
    return operation(a, b);
};
let addition = (x, y) => x + y;
let subtraction = (x, y) => x - y;
console.log(applyOperation(10, 5, addition));     // 15
console.log(applyOperation(10, 5, subtraction));  // 5

// 闭包
let createCounter = () => {
    let count = 0;
    return {
        increment: () => ++count,
        decrement: () => --count,
        getCount: () => count
    };
};
let counter = createCounter();
console.log(counter.increment());  // 1
console.log(counter.increment());  // 2
console.log(counter.getCount());   // 2

// 递归
let factorial = (n) => {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
};
console.log(factorial(5));  // 120

// 立即执行函数 (IIFE)
let result = (() => {
    let x = 10;
    let y = 20;
    return x + y;
})();
console.log(result);  // 30

// ============================================
// 第5节：对象和数组
// ============================================

// 对象创建
let user = {
    name: "Alice",
    age: 30,
    email: "alice@example.com",
    greet: function() {
        return "Hi, I'm " + this.name;
    }
};
console.log(user.name);        // "Alice"
console.log(user.greet());     // "Hi, I'm Alice"

// 对象属性访问
console.log(user["name"]);     // "Alice"
let key = "age";
console.log(user[key]);        // 30

// 添加/删除属性
user.phone = "123-456-7890";
console.log(user.phone);
delete user.phone;

// 对象方法
let person2 = {
    firstName: "John",
    lastName: "Doe",
    getFullName: function() {
        return this.firstName + " " + this.lastName;
    },
    setFirstName: function(name) {
        this.firstName = name;
    }
};
console.log(person2.getFullName());  // "John Doe"
person2.setFirstName("Jane");
console.log(person2.getFullName());  // "Jane Doe"

// Object 方法
let keys = Object.keys(user);
let values = Object.values(user);
let entries = Object.entries(user);
console.log(keys);     // ["name", "age", "email", "greet"]
console.log(values);   // ["Alice", 30, "alice@example.com", ...]
console.log(entries);  // [["name", "Alice"], ...]

// 数组创建
let arr1 = [1, 2, 3, 4, 5];
let arr2 = new Array(1, 2, 3, 4, 5);
let arr3 = Array.from("Hello");  // ["H", "e", "l", "l", "o"]

// 数组方法
let numbers2 = [1, 2, 3, 4, 5];

// map
let doubled = numbers2.map(x => x * 2);
console.log(doubled);  // [2, 4, 6, 8, 10]

// filter
let evens = numbers2.filter(x => x % 2 === 0);
console.log(evens);  // [2, 4]

// reduce
let sum = numbers2.reduce((acc, val) => acc + val, 0);
console.log(sum);  // 15

// find
let found = numbers2.find(x => x > 3);
console.log(found);  // 4

// findIndex
let index = numbers2.findIndex(x => x > 3);
console.log(index);  // 3

// some
let hasEven = numbers2.some(x => x % 2 === 0);
console.log(hasEven);  // true

// every
let allPositive = numbers2.every(x => x > 0);
console.log(allPositive);  // true

// sort
let unsorted = [3, 1, 4, 1, 5, 9, 2, 6];
unsorted.sort((a, b) => a - b);
console.log(unsorted);  // [1, 1, 2, 3, 4, 5, 6, 9]

// reverse
numbers2.reverse();
console.log(numbers2);  // [5, 4, 3, 2, 1]

// push, pop, shift, unshift
let arr4 = [1, 2, 3];
arr4.push(4);        // [1, 2, 3, 4]
arr4.pop();          // [1, 2, 3]
arr4.unshift(0);     // [0, 1, 2, 3]
arr4.shift();        // [1, 2, 3]

// slice, splice
let arr5 = [1, 2, 3, 4, 5];
let sliced = arr5.slice(1, 4);  // [2, 3, 4]
let spliced = arr5.splice(2, 1, "a", "b");  // [3], arr5 = [1, 2, "a", "b", 4, 5]

// concat, join
let arr6 = [1, 2];
let arr7 = [3, 4];
let combined = arr6.concat(arr7);  // [1, 2, 3, 4]
let joined = combined.join("-");  // "1-2-3-4"

// includes, indexOf
let arr8 = [1, 2, 3, 4, 5];
console.log(arr8.includes(3));     // true
console.log(arr8.indexOf(4));      // 3
console.log(arr8.lastIndexOf(5));  // 4

// forEach
arr8.forEach((value, index) => {
    console.log(`Index ${index}: ${value}`);
});

// ============================================
// 第6节：DOM 操作
// ============================================

// 获取元素
/*
let element = document.getElementById("myId");
let elements = document.getElementsByClassName("myClass");
let tags = document.getElementsByTagName("div");
let query = document.querySelector(".myClass");
let queryAll = document.querySelectorAll(".myClass");
*/

// 创建元素
/*
let div = document.createElement("div");
div.id = "newDiv";
div.className = "container";
div.innerHTML = "<p>Hello, DOM!</p>";
document.body.appendChild(div);
*/

// 修改元素
/*
let elem = document.getElementById("myElement");
elem.textContent = "New text content";
elem.innerHTML = "<strong>Bold text</strong>";
elem.setAttribute("data-custom", "value");
elem.style.color = "red";
elem.style.backgroundColor = "blue";
elem.classList.add("active");
elem.classList.remove("inactive");
elem.classList.toggle("highlight");
*/

// 删除元素
/*
let toRemove = document.getElementById("removeMe");
toRemove.remove();
// 或者
toRemove.parentNode.removeChild(toRemove);
*/

// 事件监听
/*
let button = document.getElementById("myButton");
button.addEventListener("click", function(event) {
    console.log("Button clicked!");
    console.log("Event type:", event.type);
    console.log("Target:", event.target);
});

// 事件委托
document.addEventListener("click", function(event) {
    if (event.target.matches(".dynamic-button")) {
        console.log("Dynamic button clicked!");
    }
});

// 阻止默认行为
let form = document.getElementById("myForm");
form.addEventListener("submit", function(event) {
    event.preventDefault();
    console.log("Form submission prevented");
});

// 事件冒泡和捕获
let parent = document.getElementById("parent");
let child = document.getElementById("child");

parent.addEventListener("click", function() {
    console.log("Parent clicked (bubbling)");
}, false);

parent.addEventListener("click", function() {
    console.log("Parent clicked (capturing)");
}, true);

child.addEventListener("click", function() {
    console.log("Child clicked");
});
*/

// ============================================
// 第7节：事件处理
// ============================================

// 事件类型示例
/*
// 鼠标事件
element.addEventListener("click", handler);
element.addEventListener("dblclick", handler);
element.addEventListener("mouseenter", handler);
element.addEventListener("mouseleave", handler);
element.addEventListener("mousemove", handler);

// 键盘事件
document.addEventListener("keydown", function(event) {
    console.log("Key pressed:", event.key);
    console.log("Key code:", event.code);
    console.log("Ctrl key:", event.ctrlKey);
    console.log("Shift key:", event.shiftKey);
});

document.addEventListener("keyup", handler);

// 表单事件
let formElement = document.getElementById("myForm");
formElement.addEventListener("submit", handler);
formElement.addEventListener("reset", handler);

let input = document.getElementById("myInput");
input.addEventListener("input", handler);
input.addEventListener("change", handler);
input.addEventListener("focus", handler);
input.addEventListener("blur", handler);

// 窗口事件
window.addEventListener("load", handler);
window.addEventListener("resize", handler);
window.addEventListener("scroll", handler);
window.addEventListener("beforeunload", handler);

// 触摸事件
element.addEventListener("touchstart", handler);
element.addEventListener("touchmove", handler);
element.addEventListener("touchend", handler);
element.addEventListener("touchcancel", handler);

// 拖放事件
element.addEventListener("drag", handler);
element.addEventListener("dragstart", handler);
element.addEventListener("dragend", handler);
element.addEventListener("dragover", handler);
element.addEventListener("dragenter", handler);
element.addEventListener("dragleave", handler);
element.addEventListener("drop", handler);
*/

// ============================================
// 第8节：异步编程
// ============================================

// 回调函数
function fetchData(callback) {
    setTimeout(() => {
        callback(null, "Data loaded");
    }, 1000);
}

fetchData((error, data) => {
    if (error) {
        console.error(error);
    } else {
        console.log(data);  // "Data loaded"
    }
});

// Promise
let promise = new Promise((resolve, reject) => {
    setTimeout(() => {
        let success = true;
        if (success) {
            resolve("Operation succeeded");
        } else {
            reject("Operation failed");
        }
    }, 1000);
});

promise
    .then(result => {
        console.log(result);  // "Operation succeeded"
        return "Next step";
    })
    .then(nextResult => {
        console.log(nextResult);  // "Next step"
    })
    .catch(error => {
        console.error(error);
    })
    .finally(() => {
        console.log("Cleanup");
    });

// Promise 静态方法
let p1 = Promise.resolve("Immediate value");
let p2 = Promise.reject("Immediate error");
let p3 = new Promise(resolve => setTimeout(() => resolve("One"), 1000));
let p4 = new Promise(resolve => setTimeout(() => resolve("Two"), 2000));
let p5 = new Promise(resolve => setTimeout(() => resolve("Three"), 1500));

Promise.all([p3, p4, p5])
    .then(results => console.log("All done:", results));

Promise.race([p3, p4, p5])
    .then(result => console.log("First done:", result));

Promise.allSettled([p3, p2, p5])
    .then(results => console.log("All settled:", results));

// async/await
async function asyncFunction() {
    try {
        let result1 = await new Promise(resolve => 
            setTimeout(() => resolve("Step 1"), 1000)
        );
        console.log(result1);
        
        let result2 = await new Promise(resolve => 
            setTimeout(() => resolve("Step 2"), 1000)
        );
        console.log(result2);
        
        return "All steps completed";
    } catch (error) {
        console.error("Error:", error);
        throw error;
    }
}

asyncFunction().then(result => console.log(result));

// 并行执行
async function parallelExecution() {
    let [r1, r2, r3] = await Promise.all([
        new Promise(resolve => setTimeout(() => resolve("A"), 1000)),
        new Promise(resolve => setTimeout(() => resolve("B"), 1000)),
        new Promise(resolve => setTimeout(() => resolve("C"), 1000))
    ]);
    console.log(r1, r2, r3);
}

parallelExecution();

// ============================================
// 第9节：模块化
// ============================================

// ES6 模块
/*
// math.js
export const PI = 3.14159;

export function add(a, b) {
    return a + b;
}

export function subtract(a, b) {
    return a - b;
}

export default class Calculator {
    add(a, b) {
        return a + b;
    }
}

// main.js
import { PI, add, subtract } from './math.js';
import Calculator from './math.js';
import * as MathModule from './math.js';

console.log(PI);
console.log(add(5, 3));
let calc = new Calculator();
console.log(calc.add(10, 20));
*/

// CommonJS 模块
/*
// utils.js
const PI = 3.14159;

function add(a, b) {
    return a + b;
}

module.exports = {
    PI: PI,
    add: add
};

// main.js
const utils = require('./utils.js');
console.log(utils.PI);
console.log(utils.add(5, 3));
*/

// 动态导入
/*
async function loadModule() {
    const module = await import('./math.js');
    console.log(module.add(5, 3));
}

loadModule();
*/

// ============================================
// 第10节：错误处理
// ============================================

// try-catch
try {
    let result = someUndefinedVariable;
} catch (error) {
    console.error("Error caught:", error.message);
}

// throw 语句
function divide(a, b) {
    if (b === 0) {
        throw new Error("Division by zero");
    }
    return a / b;
}

try {
    divide(10, 0);
} catch (error) {
    console.error(error.message);  // "Division by zero"
}

// 自定义错误
class CustomError extends Error {
    constructor(message) {
        super(message);
        this.name = "CustomError";
    }
}

try {
    throw new CustomError("Something went wrong");
} catch (error) {
    if (error instanceof CustomError) {
        console.error("Custom error:", error.message);
    } else {
        console.error("Unknown error:", error);
    }
}

// finally
try {
    console.log("Trying...");
} catch (error) {
    console.error(error);
} finally {
    console.log("Cleanup code runs regardless");
}

console.log("=== JavaScript 基础教程完成 ===");
