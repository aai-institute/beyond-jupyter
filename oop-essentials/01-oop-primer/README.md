# A Primer on Object-Oriented Programming

## Classes, Objects and Encapsulation

**Classes** act as blueprints that define the data **attributes** and behaviors (**methods**) of objects; 
**objects** are instances of these classes created at runtime with their own state and identity. 
*Encapsulation* binds the data and methods of a class together into a single unit and prevents external code from 
directly accessing the internal representation. Instead, interaction with the object is managed through a well-defined public interface. 
This encapsulation provides *abstraction* and *information hiding*, which reduces complexity and promotes *loose coupling*. 
Multiple objects can be instantiated from the same class and contain their own copies of the data attributes defined by that class. 
The object's methods operate on the its internal data.

In object-oriented design, classes can be designed to align with concrete or abstract entities in the application domain,
enabling *domain-driven design*.
A key design aspect is the determination of the right set of classes and abstractions/interfaces, which will result in sensible and intuitive computational and representational units.

## Inheritance

Inheritance allows new classes to be defined that reuse, extend, and modify the behavior of existing classes. 
The new class is called the subclass or child class, and the one it inherits from is the parent class or superclass. 
The subclass inherits the attributes and methods of the parent, so code does not need to be rewritten. 
The child class can also add new attributes and behaviors, extending the functionality of the parent.  
Inheritance establishes hierarchical relationships between classes and promotes code reuse. 

Parent classes which do not implement all the methods they declare are called *abstract* and 
cannot be instantiated at runtime. 
Such classes thus define abstractions which serve as interfaces in object-oriented programming.


## Subtype Polymorphism: Abstractions as Interfaces

Inheritance supports subtype polymorphism, where subclass objects can be used in place of parent objects, 
i.e. if a function expects an instance of a parent class as a parameter, an instance of any of its subclasses can be passed instead.
This enables certain generalisations:
  * We can treat (collections of) different types of objects in the same way as long as they are instances of a common superclass.
  * We can modify behaviour by exchanging the implementations that are being passed to algorithms at runtime.

Abstract classes are particularly convenient interface specifications, which are preferable to purely function-based interfaces (i.e. higher-order functions) in most cases, because:

 - The base class provides a type bound which straightforwardly enables discovery of potential implementations
 - Objects have representations which can more readily be logged and stored
 - Objects can straightforwardly parametrise their behavior through attributes,   
   which alleviates the need for cumbersome currying (e.g. through `lambda` functions or `functools.partial`)
 - Type relationships are explicit (in contrast to duck typing concepts such as `Protocol`).

 Many of the above points will become clear if we look at concrete examples,
 which we shall do in the subsequent [case study](../02a-case-study-0-unstructured-script/).