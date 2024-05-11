# Java Enterprise Development with Springboot - Data Access

## Outline of today's Lab

* Mysql Configuration
* GemFire cache
* MongoDB
* Neo4j

## Lab 1 Accessing data with MySQL

## Lab 2 Accessing Data with JPA

## Lab 3 Accessing Data with MongoDB

## Lab 4 Accessing Data in Pivotal GemFire

### What You Will build

You will use Spring Data for Apache Geode to store and retrieve POJOs.

 - `git clone https://github.com/spring-guides/gs-accessing-data-gemfire.git`

 - `cd into gs-accessing-data-gemfire/initial`

### Define a Simple Entity

Apache Geode is an `In-Memory Data Grid (IMDG)` that maps data to regions. You can configure distributed regions that partition and replicate data across multiple nodes in a cluster. However, in this guide, we use a LOCAL region so that you need not set up anything extra, such as an entire cluster of servers.

`Apache Geode` is a `key/value store`, and a region implements the java.util.concurrent.ConcurrentMap interface. Though you can treat a region as a java.util.Map, it is quite a bit more sophisticated than just a simple Java Map, given that data is distributed, replicated, and generally managed inside the region.

In this example, you store Person objects in Apache Geode (a region) by using only a few annotations.

`src/main/java/hello/Person.java`

```
package hello;

import java.io.Serializable;

import org.springframework.data.annotation.Id;
import org.springframework.data.annotation.PersistenceConstructor;
import org.springframework.data.gemfire.mapping.annotation.Region;

import lombok.Getter;

@Region(value = "People")
public class Person implements Serializable {

  @Id
  @Getter
  private final String name;

  @Getter
  private final int age;

  @PersistenceConstructor
  public Person(String name, int age) {
    this.name = name;
    this.age = age;
  }

  @Override
  public String toString() {
    return String.format("%s is %d years old", getName(), getAge());
  }
}
```

### Create Simple Queries

Spring Data for Apache Geode focuses on storing and accessing data in Apache Geode using Spring. It also inherits powerful functionality from the Spring Data Commons project, such as the ability to derive queries. Essentially, you need not learn the query language of Apache Geode (OQL). You can write a handful of methods, and the framework writes the queries for you.

To see how this works, create an interface that queries Person objects stored in Apache Geode:

`src/main/java/hello/PersonRepository.java`

```
package hello;

import org.springframework.data.gemfire.repository.query.annotation.Trace;
import org.springframework.data.repository.CrudRepository;

public interface PersonRepository extends CrudRepository<Person, String> {

  @Trace
  Person findByName(String name);

  @Trace
  Iterable<Person> findByAgeGreaterThan(int age);

  @Trace
  Iterable<Person> findByAgeLessThan(int age);

  @Trace
  Iterable<Person> findByAgeGreaterThanAndAgeLessThan(int greaterThanAge, int lessThanAge);

}

```

PersonRepository extends the CrudRepository interface from Spring Data Commons and specifies types for the generic type parameters for both the value and the ID (key) with which the Repository works (Person and String, respectively). This interface comes with many operations, including basic CRUD (create, read, update, delete) and simple query data access operations (such a findById(..)).

You can define other queries as needed by declaring their method signature. In this case, we add findByName, which essentially searches for objects of type Person and finds one that matches on name.

You also have:

- findByAgeGreaterThan: To find people above a certain age

- findByAgeLessThan: To find people below a certain age

- findByAgeGreaterThanAndAgeLessThan: To find people in a certain age range

### Create an Application Class

The following example creates an application class with all the components:

`src/main/java/hello/Application.java`

```
package hello;

import static java.util.Arrays.asList;
import static java.util.stream.StreamSupport.stream;

import org.apache.geode.cache.client.ClientRegionShortcut;

import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.data.gemfire.config.annotation.ClientCacheApplication;
import org.springframework.data.gemfire.config.annotation.EnableEntityDefinedRegions;
import org.springframework.data.gemfire.repository.config.EnableGemfireRepositories;

@SpringBootApplication
@ClientCacheApplication(name = "AccessingDataGemFireApplication")
@EnableEntityDefinedRegions(
  basePackageClasses = Person.class,
  clientRegionShortcut = ClientRegionShortcut.LOCAL
)
@EnableGemfireRepositories
public class Application {

  public static void main(String[] args) {
    SpringApplication.run(Application.class, args);
  }

  @Bean
  ApplicationRunner run(PersonRepository personRepository) {

    return args -> {

      Person alice = new Person("Adult Alice", 40);
      Person bob = new Person("Baby Bob", 1);
      Person carol = new Person("Teen Carol", 13);

      System.out.println("Before accessing data in Apache Geode...");

      asList(alice, bob, carol).forEach(person -> System.out.println("\t" + person));

      System.out.println("Saving Alice, Bob and Carol to Pivotal GemFire...");

      personRepository.save(alice);
      personRepository.save(bob);
      personRepository.save(carol);

      System.out.println("Lookup each person by name...");

      asList(alice.getName(), bob.getName(), carol.getName())
        .forEach(name -> System.out.println("\t" + personRepository.findByName(name)));

      System.out.println("Query adults (over 18):");

      stream(personRepository.findByAgeGreaterThan(18).spliterator(), false)
        .forEach(person -> System.out.println("\t" + person));

      System.out.println("Query babies (less than 5):");

      stream(personRepository.findByAgeLessThan(5).spliterator(), false)
        .forEach(person -> System.out.println("\t" + person));

      System.out.println("Query teens (between 12 and 20):");

      stream(personRepository.findByAgeGreaterThanAndAgeLessThan(12, 20).spliterator(), false)
        .forEach(person -> System.out.println("\t" + person));
    };
  }
}
```

In the configuration, you need to add the `@EnableGemfireRepositories` annotation.

By default, `@EnableGemfireRepositories` scans the current package for any interfaces that extend one of Spring Data’s repository interfaces. You can use its basePackageClasses = MyRepository.class to safely tell Spring Data for Apache Geode to scan a different root package by type for application-specific Repository extensions.

A Apache Geode cache containing one or more regions is required to store all the data. For that, you use one of Spring Data for Apache Geode’s convenient configuration-based annotations: `@ClientCacheApplication`, `@PeerCacheApplication`, or `@CacheServerApplication`.

Apache Geode supports different cache topologies, such as client/server, peer-to-peer (p2p), and even WAN arrangements. In p2p, a peer cache instance is embedded in the application, and your application would have the ability to participate in a cluster as a peer cache member. However, your application is subject to all the constraints of being a peer member in the cluster, so this is not as commonly used as, say, the client/server topology.

In our case, we use `@ClientCacheApplication` to create a “client” cache instance, which has the ability to connect to and communicate with a cluster of servers. However, to keep things simple, the client stores data locally by using a LOCAL client region, without the need to setup or run any servers.

Now, remember how you tagged Person to be stored in a region called People by using the SDG mapping annotation, `@Region("People")?` You define that region here by using the ClientRegionFactoryBean<String, Person> bean definition. You need to inject an instance of the cache you just defined while also naming it People.

### Build an executable JAR

You can run the application from the command line with Gradle or Maven. You can also build a single executable JAR file that contains all the necessary dependencies, classes, and resources and run that. Building an executable jar makes it easy to ship, version, and deploy the service as an application throughout the development lifecycle, across different environments, and so forth.


### Run executable JAR

If you use Gradle, you can run the application by using `./gradlew bootRun`. Alternatively, you can build the JAR file by using `./gradlew build` and then run the JAR file, as follows:

`java -jar build/libs/gs-accessing-data-gemfire-0.1.0.jar`

If you use Maven, you can run the application by using ./mvnw spring-boot:run. Alternatively, you can build the JAR file with ./mvnw clean package and then run the JAR file, as follows:

`java -jar target/gs-accessing-data-gemfire-0.1.0.jar`

You should see something like this (with other content, such as queries):



## Accessing Data with Neo4j

### Prepare the lab

- `git clone https://github.com/spring-guides/gs-accessing-data-neo4j.git`
- `cd into gs-accessing-data-neo4j/initial`
- install neo4j
  1.  ` brew install neo4j`
  2. For other options, visit https://neo4j.com/download/community-edition/.

Once installed, launch it with its default settings by running the following command:
`$ neo4j start`

You should see output similar to the following:

```
Starting Neo4j.
Started neo4j (pid 96416). By default, it is available at http://localhost:7474/
There may be a short delay until the server is ready.
See /usr/local/Cellar/neo4j/3.0.6/libexec/logs/neo4j.log for current status.
```

## Ref

- https://spring.io/guides/gs/accessing-data-mysql/
- https://spring.io/guides/gs/accessing-data-jpa/
- https://spring.io/guides/gs/accessing-data-gemfire
- https://spring.io/guides/gs/accessing-data-mongodb
- https://spring.io/guides/gs/accessing-data-neo4j