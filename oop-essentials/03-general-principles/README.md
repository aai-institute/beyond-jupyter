# General Design and Development Principles

## SOLID Design

SOLID is a collection of five principles introduced by Robert C. Martin:

 1. **Single Responsibility Principle**

    “Every class in a computer program should have responsibility over a single part of the program's functionality, which it should encapsulate.”
    
    Adhering to this principle critically improves *clarity* and *maintainability*:
    * The components are relatively small pieces of code that are easy to scan and understand.
    * The component where a change/fix would need to be implemented is easy to identify and modifications can be made more quickly.

 2. **Open-Closed Principle**

    “Software entities (classes, modules, functions, etc.) should be open for extension, but closed for modification,” i.e. entities should allow their behaviour to be extended without modifying the source code.
    
    This critically improves *extensibility*:
    * It is easier to add new functionality, as it can be done without interfering with the development of existing components.
    * Extensions can be made by users who do not even have access to the original source code.

 3. **Liskov Substitution Principle**

    “If S is a subtype of T, then objects of type T may be replaced with objects of type S without altering any of the desirable properties of the program.”
    
    This enforces well-defined interface contracts in abstractions, rendering concrete implementations interchangeable.

 4. **Interface Segregation Principle**

    “No client should be forced to depend on methods it does not use. Large interfaces are split into smaller and more specific ones so that clients will only have to know about the methods that are of interest to them.”
    
    This critically improves clarity and simplifies the client implementation process:
    * Clients are not confused by functionality they do not need and quickly find the functions they are looking for.
    * Deciding on the interfaces to depend on is straightforward.

 5. **Dependency Inversion Principle**

    “High-level modules should not depend on low-level modules. Both should depend on abstractions (e.g., interfaces).  
    Abstractions should not depend on details. Details (concrete implementations) should depend on abstractions.”
    
    This critically decouples modules from one another and focuses dependencies between modules on interfaces and abstract interactions:
    * Dependencies are weaker, allowing implementations to be exchanged.
    * Interfaces are clear and reduced to the necessary minimum.
   

*Examples.* Even though our case study was a very simple example, it essentially exhibits all of the above principles:

 * Single Responsibility Principle: All of the classes `Metric`, `ModelEvaluation` and `Results` serve well-defined purposes and do not mix different levels of abstraction.
 * Open-Closed Principle: `ModelEvaluation` provides some extensibility by depending on one or more `Metric` instances, which the user can specify. Users are free to implement their own metrics and do not need to modify the implementation of the evaluation at all in order to get them applied.
 * Liskov Substitution Principle: The `ModelEvaluation` does not care which `Metric` evaluation it receives and is unaware of its specifics. As long as the `Metric` implementation correctly implements the interface, any implementation can be provided.
 * Interface Segregation Principle: The interfaces we defined are all small to begin with, so this is already satisfied because our abstractions have a single responsibility. In larger applications, however, it can make sense for an abstraction to simultaneously implement more than one interface (abstract base class) if the respective abstraction is a complex concept which can be viewed as a specialisation of several, typically more primitive ones.
 * Dependency Inversion Principle: `ModelEvaluation` depending on the abstract `Metric` rather than using a (perhaps hard-coded) instance of a particular metric is an application of this principle.

# DRY: Don't Repeat Yourself

The DRY principle calls for a high degree of *factorisation* in order to avoid code duplication: 
If a piece of code repeats, factor it out into a reusable function/class which is then applied multiple times; this is akin to factorisation in mathematics: `ab+ac = a(b+c)`.

Note, however, that factorisation can be overdone. While two pieces of code may use the same sub-routine at present, it may be the case that the implementations have different needs in the future, requiring different extensions. 
Instead of applying factorisation and extracting a function that can be parametrised in a myriad of ways only to support both cases, it can, therefore,  sometimes be better - for the sake of clarity - to have two copies of the code, which are then free to develop independently.

# YAGNI: You Aren't Gonna Need It

The YAGNI principle seeks to find the simplest solution that will work, without overdoing it or preparing for future needs that may never arise.
Overengineering a solution to prepare for future needs is virtually *always* a bad thing, and unexperienced programmers typically do not realise how rare it is for these needs to actually materialise.
Instead, keep your design clean, such that it will be easy to refactor it once new requirements arise.

# KISS (Keep It Simple and Stupid) and the Principle of Least Surprise

The KISS principle calls for simple solutions rather than complex ones. 
If there is no (good) reason for the design to be complex, prefer simple design, as it will facilitate understandability.

This is closely tied to the *principle of least surprise*, which calls for design choices that won't elicit surprise on the user's part. 
Understand people as part of the system, and choose a design that matches people's expectations, experience, and mental models. 
If possible, the design should prefer idiomatic language constructs over exotic, non-standard ones that aren't easily understood - even if they may solve a problem slightly more elegantly.

# Extreme Programming (XP)

YAGNI is one of the core principles of extreme programming (aka XP), which addresses not only principles pertaining to code but also collaboration.

The core values of XP are:
  * **Communication**: XP stresses the need for communication between developers, for knowledge transfer and alignment.
  * **Simplicity** (= YAGNI)
  * **Feedback**: Periodically reflecting on past performance can help to identify areas for improvement, both in terms of code and the development process being applied.
  * **Courage**: Courage is required in order to raise issues that impede the development process, e.g organisational issues or even issues pertaining to the general direction the product is headed in.
  * **Respect**: Mutual respect is required in order to foster communication and to provide and accept feedback.

XP furthermore defines a set of practices, including
  * **Simple Design**: Build software to a simple but always adequate design.
  * **Pair Programming**: Two developers directly collaborate to produce a single piece of code. This immediately creates knowledge transfer (as more than one person is familiar with every piece of code) and eliminates the need for additional reviews.
  * **Refactoring**: Constantly refactoring the code to retain the quality of simple, adequate code that has no technical debt.
  * **Test-Driven Development**: Thinking about how to functionality can be tested from the very beginng.
  * **Continuous Integration**: Constantly integrating changes that improve the software product and applying automated tests helps to keep quality standards high.
  * **Collective Code Ownership**: Every piece of code can be immediately maintained by at least two developers. All code gets the benefit of many people’s attention, which increases code quality and reduces defects.

For the full set of principles and ideas, please refer to ["What is Extreme Programming?" by Ron Jeffries](https://ronjeffries.com/xprog/what-is-extreme-programming/).