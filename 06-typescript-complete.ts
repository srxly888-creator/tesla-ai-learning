/**
 * TypeScript 完整教程 - 第6章
 * 
 * 涵盖内容：
 * - 基础类型
 * - 接口
 * - 类
 * - 泛型
 * - 装饰器
 * - 高级类型
 * - 模块
 * - 命名空间
 */

// ============================================
// 第1节：基础类型
// ============================================

// 布尔值
let isDone: boolean = false;

// 数字
let decimal: number = 6;
let hex: number = 0xf00d;
let binary: number = 0b1010;
let octal: number = 0o744;

// 字符串
let color: string = "blue";
color = 'red';
let fullName: string = `Bob Bobbington`;
let age: number = 37;
let sentence: string = `Hello, my name is ${fullName}.

I'll be ${age + 1} years old next month.`;

// 数组
let list: number[] = [1, 2, 3];
let list2: Array<number> = [1, 2, 3];

// 元组
let x: [string, number];
x = ["hello", 10]; // OK
// x = [10, "hello"]; // Error

// 枚举
enum Color {Red, Green, Blue}
let c: Color = Color.Green;

enum Color2 {Red = 1, Green = 2, Blue = 4}
let c2: Color2 = Color2.Green;

enum Color3 {Red = "RED", Green = "GREEN", Blue = "BLUE"}
let c3: Color3 = Color3.Green;

// Any
let notSure: any = 4;
notSure = "maybe a string instead";
notSure = false;

let listAny: any[] = [1, true, "free"];
listAny[1] = 100;

// Void
function warnUser(): void {
    console.log("This is my warning message");
}

let unusable: void = undefined;

// Null 和 Undefined
let u: undefined = undefined;
let n: null = null;

// Never
function error(message: string): never {
    throw new Error(message);
}

function fail() {
    return error("Something failed");
}

function infiniteLoop(): never {
    while (true) {
    }
}

// Object
declare function create(o: object | null): void;
create({ prop: 0 });

// 类型断言
let someValue: any = "this is a string";
let strLength: number = (<string>someValue).length;
let strLength2: number = (someValue as string).length;

// ============================================
// 第2节：接口
// ============================================

// 基本接口
interface LabelledValue {
    label: string;
}

function printLabel(labelledObj: LabelledValue) {
    console.log(labelledObj.label);
}

let myObj = {size: 10, label: "Size 10 Object"};
printLabel(myObj);

// 可选属性
interface SquareConfig {
    color?: string;
    width?: number;
}

function createSquare(config: SquareConfig): { color: string; area: number } {
    let newSquare = {color: "white", area: 100};
    if (config.color) {
        newSquare.color = config.color;
    }
    if (config.width) {
        newSquare.area = config.width * config.width;
    }
    return newSquare;
}

let mySquare = createSquare({color: "black"});

// 只读属性
interface Point {
    readonly x: number;
    readonly y: number;
}

let p1: Point = { x: 10, y: 20 };
// p1.x = 5; // Error!

// 函数类型
interface SearchFunc {
    (source: string, subString: string): boolean;
}

let mySearch: SearchFunc;
mySearch = function(source: string, subString: string): boolean {
    let result = source.search(subString);
    return result > -1;
}

// 可索引类型
interface StringArray {
    [index: number]: string;
}

let myArray: StringArray;
myArray = ["Bob", "Fred"];

interface NumberDictionary {
    [index: string]: number;
    length: number;
    width: number;
}

// 类类型
interface ClockInterface {
    currentTime: Date;
    setTime(d: Date): void;
}

class Clock implements ClockInterface {
    currentTime: Date = new Date();
    setTime(d: Date) {
        this.currentTime = d;
    }
    constructor(h: number, m: number) {}
}

// 接口继承
interface Shape {
    color: string;
}

interface PenStroke {
    penWidth: number;
}

interface Square extends Shape, PenStroke {
    sideLength: number;
}

let square = <Square>{};
square.color = "blue";
square.sideLength = 10;
square.penWidth = 5.0;

// 混合类型
interface Counter {
    (start: number): string;
    interval: number;
    reset(): void;
}

function getCounter(): Counter {
    let counter = <Counter>function (start: number) {
        return "Started at " + start;
    };
    counter.interval = 5.0;
    counter.reset = function() { };
    return counter;
}

let c4 = getCounter();
c4(10);
c4.reset();
c4.interval = 5.0;

// ============================================
// 第3节：类
// ============================================

// 基本类
class Greeter {
    greeting: string;
    
    constructor(message: string) {
        this.greeting = message;
    }
    
    greet() {
        return "Hello, " + this.greeting;
    }
}

let greeter = new Greeter("world");

// 继承
class Animal {
    move(distanceInMeters: number = 0) {
        console.log(`Animal moved ${distanceInMeters}m.`);
    }
}

class Dog extends Animal {
    bark() {
        console.log('Woof! Woof!');
    }
}

const dog = new Dog();
dog.bark();
dog.move(10);
dog.bark();

// 公共，私有与受保护的修饰符
class Animal2 {
    public name: string;
    private privateName: string;
    protected protectedName: string;
    
    constructor(theName: string) {
        this.name = theName;
        this.privateName = theName;
        this.protectedName = theName;
    }
}

class Rhino extends Animal2 {
    constructor() {
        super("Rhino");
    }
    
    public getProtectedName(): string {
        return this.protectedName;
    }
}

let animal = new Animal2("Goat");
console.log(animal.name);
// console.log(animal.privateName); // Error
// console.log(animal.protectedName); // Error

// 存取器
let passcode = "secret passcode";

class Employee {
    private _fullName: string = "";

    get fullName(): string {
        return this._fullName;
    }

    set fullName(newName: string) {
        if (passcode && passcode == "secret passcode") {
            this._fullName = newName;
        }
        else {
            console.log("Error: Unauthorized update of employee!");
        }
    }
}

let employee = new Employee();
employee.fullName = "Bob Smith";
if (employee.fullName) {
    console.log(employee.fullName);
}

// 静态属性
class Grid {
    static origin = {x: 0, y: 0};
    
    calculateDistanceFromOrigin(point: {x: number; y: number;}) {
        let xDist = (point.x - Grid.origin.x);
        let yDist = (point.y - Grid.origin.y);
        return Math.sqrt(xDist * xDist + yDist * yDist) / this.scale;
    }
    
    constructor(public scale: number) {}
}

let grid = new Grid(1.0);

// 抽象类
abstract class Department {
    constructor(public name: string) {
    }
    
    printName(): void {
        console.log("Department name: " + this.name);
    }
    
    abstract printMeeting(): void;
}

class AccountingDepartment extends Department {
    constructor() {
        super("Accounting and Auditing");
    }
    
    printMeeting(): void {
        console.log("The Accounting Department meets each Monday at 10am.");
    }
    
    generateReports(): void {
        console.log("Generating accounting reports...");
    }
}

let department: Department;
department = new AccountingDepartment();
department.printName();
department.printMeeting();
// department.generateReports(); // Error

// 构造函数签名
interface ClockConstructor {
    new (hour: number, minute: number): ClockInterface2;
}

interface ClockInterface2 {
    tick(): void;
}

function createClock(ctor: ClockConstructor, hour: number, minute: number): ClockInterface2 {
    return new ctor(hour, minute);
}

class DigitalClock implements ClockInterface2 {
    constructor(h: number, m: number) {}
    
    tick() {
        console.log("beep beep");
    }
}

class AnalogClock implements ClockInterface2 {
    constructor(h: number, m: number) {}
    
    tick() {
        console.log("tick tock");
    }
}

let digital = createClock(DigitalClock, 12, 17);
let analog = createClock(AnalogClock, 7, 32);

// ============================================
// 第4节：函数
// ============================================

// 函数类型
let myAdd: (x: number, y: number) => number = function(x: number, y: number): number {
    return x + y;
};

// 可选参数和默认参数
function buildName(firstName: string, lastName?: string) {
    if (lastName)
        return firstName + " " + lastName;
    else
        return firstName;
}

let result1 = buildName("Bob");
let result2 = buildName("Bob", "Adams");

function buildName2(firstName: string, lastName = "Smith") {
    return firstName + " " + lastName;
}

let result3 = buildName2("Bob");
let result4 = buildName2("Bob", undefined);
let result5 = buildName2("Bob", "Adams");

// 剩余参数
function buildName3(firstName: string, ...restOfName: string[]) {
    return firstName + " " + restOfName.join(" ");
}

let employeeName = buildName3("Joseph", "Samuel", "Lucas", "MacKinzie");

// this 参数
interface Card {
    suit: string;
    card: number;
}

interface Deck {
    suits: string[];
    cards: number[];
    createCardPicker(this: Deck): () => Card;
}

let deck: Deck = {
    suits: ["hearts", "spades", "clubs", "diamonds"],
    cards: Array(52),
    createCardPicker: function(this: Deck) {
        return () => {
            let pickedCard = Math.floor(Math.random() * 52);
            let pickedSuit = Math.floor(pickedCard / 13);
            
            return {suit: this.suits[pickedSuit], card: pickedCard % 13};
        };
    }
};

let cardPicker = deck.createCardPicker();
let pickedCard = cardPicker();

// 重载
let suits = ["hearts", "spades", "clubs", "diamonds"];

function pickCard(x: {suit: string; card: number; }[]): number;
function pickCard(x: number): {suit: string; card: number; };
function pickCard(x: any): any {
    if (typeof x == "object") {
        let pickedCard = Math.floor(Math.random() * x.length);
        return pickedCard;
    }
    else if (typeof x == "number") {
        let pickedSuit = Math.floor(x / 13);
        return { suit: suits[pickedSuit], card: x % 13 };
    }
}

let myDeck = [{ suit: "diamonds", card: 2 }, { suit: "spades", card: 10 }, { suit: "hearts", card: 4 }];
let pickedCard1 = myDeck[pickCard(myDeck)];
let pickedCard2 = pickCard(15);

console.log("card: " + pickedCard1.card + " of " + pickedCard1.suit);
console.log("card: " + pickedCard2.card + " of " + pickedCard2.suit);

// ============================================
// 第5节：泛型
// ============================================

// 基本泛型
function identity<T>(arg: T): T {
    return arg;
}

let output1 = identity<string>("myString");
let output2 = identity("myString");

// 泛型变量
function loggingIdentity<T>(arg: T[]): T[] {
    console.log(arg.length);
    return arg;
}

// 泛型类型
let myIdentity: <T>(arg: T) => T = identity;

let myIdentity2: {<T>(arg: T): T} = identity;

interface GenericIdentityFn {
    <T>(arg: T): T;
}

let myIdentity3: GenericIdentityFn = identity;

// 泛型类
class GenericNumber<T> {
    zeroValue: T;
    add: (x: T, y: T) => T;
}

let myGenericNumber = new GenericNumber<number>();
myGenericNumber.zeroValue = 0;
myGenericNumber.add = function(x, y) { return x + y; };

// 泛型约束
interface Lengthwise {
    length: number;
}

function loggingIdentity2<T extends Lengthwise>(arg: T): T {
    console.log(arg.length);
    return arg;
}

// loggingIdentity2(3);  // Error
loggingIdentity2({length: 10, value: 3});

// 在泛型约束中使用类型参数
function getProperty<T, K extends keyof T>(obj: T, key: K) {
    return obj[key];
}

let x2 = { a: 1, b: 2, c: 3, d: 4 };

getProperty(x2, "a");
// getProperty(x, "m"); // Error

// 泛型类约束
class BeeKeeper {
    hasMask: boolean = true;
}

class ZooKeeper {
    nametag: string = "Mikle";
}

class Animal3 {
    numLegs: number = 4;
}

class Bee extends Animal3 {
    keeper: BeeKeeper = new BeeKeeper();
}

class Lion extends Animal3 {
    keeper: ZooKeeper = new ZooKeeper();
}

function createInstance<A extends Animal3>(c: new () => A): A {
    return new c();
}

createInstance(Lion).keeper.nametag;
createInstance(Bee).keeper.hasMask;

console.log("=== TypeScript 完整教程完成 ===");
