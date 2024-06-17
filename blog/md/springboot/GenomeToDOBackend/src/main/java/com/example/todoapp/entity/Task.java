package com.example.todoapp.entity;

import jakarta.persistence.*;
import lombok.Builder;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import java.io.Serializable;
import java.time.LocalDateTime;

@Entity
@Getter
@Setter
@Builder
@EqualsAndHashCode
public class Task implements Serializable {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Integer tId;

    @Column(nullable = false)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String note;

    private LocalDateTime dueDay;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Priority priority;

    @Column(nullable = false)
    private Integer completed = 0;

    @ManyToOne
    @JoinColumn(name = "list_id")
    private TaskList taskList;
}
