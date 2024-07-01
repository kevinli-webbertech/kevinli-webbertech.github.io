# Java Enterprise Development with Springboot - Topic 9 - Spring Security

## Spring Security Architecture

Spring Security is a framework that provides,

* authentication

* authorization

* protection against common attacks

https://docs.spring.io/spring-security/reference/features/exploits/index.html

such as CSRF (Cross Site Request Forgery (CSRF) attacks)

More can be found here,

https://docs.spring.io/spring-security/reference/features/exploits/csrf.html


On top of the current Springboot developments, it supports two mainstream applications,

* imperative

https://docs.spring.io/spring-security/reference/servlet/index.html

* reactive

https://docs.spring.io/spring-security/reference/reactive/index.html


Spring Security’s high-level architecture within Servlet based applications is based on Servlet Filters, so it is helpful to look at the role of Filters generally first. The following image shows the typical layering of the handlers for a single HTTP request.

**DelegatingFilterProxy**

Spring provides a Filter implementation named DelegatingFilterProxy that allows bridging between the Servlet container’s lifecycle and Spring’s ApplicationContext. The Servlet container allows registering Filter instances by using its own standards, but it is not aware of Spring-defined Beans. You can register DelegatingFilterProxy through the standard Servlet container mechanisms but delegate all the work to a Spring Bean that implements Filter.

![alt text](https://kevinli-webbertech.github.io/blog/images/springboot/DelegatingFilterProxy.png=200*200)

**FilterChainProxy**

Spring Security’s Servlet support is contained within FilterChainProxy. FilterChainProxy is a special Filter provided by Spring Security that allows delegating to many Filter instances through SecurityFilterChain. Since FilterChainProxy is a Bean, it is typically wrapped in a DelegatingFilterProxy.


![alt text](https://kevinli-webbertech.github.io/blog/images/springboot/FilterChainProxy.png=200*200)

**SecurityFilterChain**

![alt text](https://kevinli-webbertech.github.io/blog/images/springboot/SecurityFilterChain.png=200*200)

SecurityFilterChain is used by FilterChainProxy to determine which Spring Security Filter instances should be invoked for the current request.


The Security Filters in SecurityFilterChain are typically Beans, but they are registered with FilterChainProxy instead of DelegatingFilterProxy. FilterChainProxy provides a number of advantages to registering directly with the Servlet container or DelegatingFilterProxy. First, it provides a starting point for all of Spring Security’s Servlet support. For that reason, if you try to troubleshoot Spring Security’s Servlet support, adding a debug point in FilterChainProxy is a great place to start.

Second, since FilterChainProxy is central to Spring Security usage, it can perform tasks that are not viewed as optional. For example, it clears out the SecurityContext to avoid memory leaks. It also applies Spring Security’s HttpFirewall to protect applications against certain types of attacks.

In addition, it provides more flexibility in determining when a SecurityFilterChain should be invoked. In a Servlet container, Filter instances are invoked based upon the URL alone. However, FilterChainProxy can determine invocation based upon anything in the HttpServletRequest by using the RequestMatcher interface.

![alt text](https://kevinli-webbertech.github.io/blog/images/springboot/MultipleSecurityChain.png=200*200)

```java
@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
            .csrf(Customizer.withDefaults())
            .authorizeHttpRequests(authorize -> authorize
                .anyRequest().authenticated()
            )
            .httpBasic(Customizer.withDefaults())
            .formLogin(Customizer.withDefaults());
        return http.build();
    }

}
```

![alt text](https://kevinli-webbertech.github.io/blog/images/springboot/filterOrder.png=200*200)

## Securing a Web Application

## Securing a SPA(single app application)


### Ref

- https://spring.io/guides/topicals/spring-security-architecture
- https://spring.io/guides/gs/securing-web/
- https://spring.io/guides/tutorials/spring-security-and-angular-js