# Java Enterprise Development with Springboot - Topic 6 - Lombok

## What is Lombok

IntelliJ Lombok plugin for code generation.

```java
A plugin that adds first-class support for Project Lombok

@Getter and @Setter
@FieldNameConstants
@ToString
@EqualsAndHashCode
@AllArgsConstructor, @RequiredArgsConstructor and @NoArgsConstructor
@Log,@Log4j, @Log4j2, @Slf4j, @XSlf4j, @CommonsLog, @JBossLog, @Flogger, @CustomLog
@Data
@Builder
```

## How to get Lombok in IntelliJ

* Go to the site of following for more info,

https://plugins.jetbrains.com/plugin/6317-lombok

* Install the plugin by downloading the zip.

![lombok_installation.jpg](https://kevinli-webbertech.github.io/blog/images/springboot/lombok_installation.png)

or

* Install online (Recommended)

![lombok_installation.jpg](https://kevinli-webbertech.github.io/blog/images/springboot/lombok_install_online.png)

You need to Enable Annotation Processing on IntelliJ IDEA,

`> Settings > Build, Execution, Deployment > Compiler > Annotation Processors`

## Update our code

Now let us add these annotations and imports,

![lombok usage1](https://kevinli-webbertech.github.io/blog/images/springboot/lombok1.jpg)

Another one,

![lombok usage2](https://kevinli-webbertech.github.io/blog/images/springboot/lombok2.jpg)

Now, there are no getter and setters!!

![no setter and getter](https://kevinli-webbertech.github.io/blog/images/springboot/No_Getter_Setter.jpg)

**Does this violate our rules of Java programming?**

The answer is: No, because this is not the final Java file. It would have the preprocessor, similar to C's macro, so the final expanded Java file would have everything.
This is again another nice feature to allow you to ease your rapid development.

## Test our code

You should start Springboot project and re check that everything should work as usual. For the conciseness, I will not show the screenshots here.