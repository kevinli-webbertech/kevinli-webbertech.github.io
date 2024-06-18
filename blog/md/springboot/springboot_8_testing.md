# Java Enterprise Development with Springboot - Topic 8 - Springboot Testing

## Testing web layer

### Step 1

* git clone https://github.com/spring-guides/gs-testing-web.git
* cd into gs-testing-web/complete

### Step 2

check we have the following code in place,

```java
src/main/java/com/example/testingweb/HomeController.java shows how to do so:

package com.example.testingweb;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HomeController {

	@RequestMapping("/")
	public @ResponseBody String greeting() {
		return "Hello, World";
	}

}
```

* Get test file

The `@SpringBootTest` annotation tells Spring Boot to look for a main configuration class (one with `@SpringBootApplication`, for instance) and use that to start a Spring application context. You can run this test in your IDE or on the command line (by running `./mvnw test` or `./gradlew test`), and it should pass. To convince yourself that the context is creating your controller, you could add an assertion, as the following example (from src/test/java/com/example/testingweb/SmokeTest.java) shows:

```java
package com.example.testingweb;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
class SmokeTest {

	@Autowired
	private HomeController controller;

	@Test
	void contextLoads() throws Exception {
		assertThat(controller).isNotNull();
	}
}
```

* Then we need to do some api testing besides the sanity test,

The following listing (from src/test/java/com/example/testingweb/HttpRequestTest.java) shows how to do so:

```java
package com.example.testingweb;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.context.SpringBootTest.WebEnvironment;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.test.web.server.LocalServerPort;
import static org.assertj.core.api.Assertions.assertThat;

```java
@SpringBootTest(webEnvironment = WebEnvironment.RANDOM_PORT)
class HttpRequestTest {

	@LocalServerPort
	private int port;

	@Autowired
	private TestRestTemplate restTemplate;

	@Test
	void greetingShouldReturnDefaultMessage() throws Exception {
		assertThat(this.restTemplate.getForObject("http://localhost:" + port + "/",
				String.class)).contains("Hello, World");
	}
}
```

### ref

https://spring.io/guides/gs/testing-web/
