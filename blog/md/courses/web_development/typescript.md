# Typescript Training

## Origin

TypeScript (abbreviated as TS) is a free and open-source high-level programming language developed by Microsoft that adds static typing with optional type annotations to JavaScript. It is designed for the development of large applications and transpiles to JavaScript.

TypeScript may be used to develop JavaScript applications for both client-side and server-side execution (as with Node.js, Deno or Bun). Multiple options are available for transpilation. The default TypeScript Compiler can be used, or the Babel compiler can be invoked to convert TypeScript to JavaScript.

TypeScript supports definition files that can contain type information of existing JavaScript libraries, much like C++ header files can describe the structure of existing object files. This enables other programs to use the values defined in the files as if they were statically typed TypeScript entities. There are third-party header files for popular libraries such as jQuery, MongoDB, and D3.js. TypeScript headers for the Node.js library modules are also available, allowing development of Node.js programs within TypeScript.

## Release History (Skipped)

Please refer to wikipedia or its own website for more details.

## Features

TypeScript adds the following syntax extensions to JavaScript:

* Static Typing

Type annotation, inference and type checking.

* Interfaces

Interfaces define the structure of objects, ensuring that objects adhere to a specific contract.

* Generics, Tuples and other language supports

* Namespaces and modules

Modules help organize code into logical units, making it easier to manage and reuse code.

* Explicit Resource Management

* Transpilation

TypeScript code is transpiled into plain JavaScript, which can run on any browser or JavaScript engine.

## Linting tools

TSLint scans TypeScript code for conformance to a set of standards and guidelines. ESLint, a standard JavaScript linter, also provided some support for TypeScript via community plugins. However, ESLint's inability to leverage TypeScript's language services precluded certain forms of semantic linting and program-wide analysis. In early 2019, the TSLint team announced the linter's deprecation in favor of typescript-eslint, a joint effort of the TSLint, ESLint and TypeScript teams to consolidate linting under the ESLint umbrella for improved performance, community unity and developer accessibility.

## Quick Guide

* **Installation**

`npm install -g typescript`

* **Create a TypeScript file (e.g., app.ts):**

```javascript
 function greet(name: string): string {
       return `Hello, ${name}!`;
   }

   const message = greet("John"); 
   console.log(message); 
```

