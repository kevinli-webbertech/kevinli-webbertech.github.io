package com.example.todoapp;

import com.example.todoapp.entity.Priority;
import com.example.todoapp.entity.Task;
import com.example.todoapp.repository.TaskRepository;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import java.time.LocalDateTime;

//https://www.baeldung.com/spring-data-jpa-query

@SpringBootApplication
public class TodoApplication {

	public static void main(String[] args) {
		SpringApplication.run(TodoApplication.class, args);
	}

	@Bean
	public CommandLineRunner demo(TaskRepository repository) {
		return (args) -> {
			// save a few customers
			repository.save(new Task().setCompleted(1).setTitle("testme").setNote("check...").setDueDay(LocalDateTime.now()).setPriority(Priority.LOW));
			repository.save(new Task().setCompleted(1).setTitle("testme2").setNote("check...").setDueDay(LocalDateTime.now()).setPriority(Priority.LOW));
		};
	}
}
