# Homework 1

## Submission

* zip your code in upload. It has to be .zip format, no windows WinRAR or tar ball.
* write a word document with your name, info, and screenshots of your code how it should show to prove you have done it right for each of the questions. The word document can be sitting in parallel with the zip file, no need to put it inside of the zip.

## Requirements

`Total points: 100`

1/ configure your springboot project to deploy using WAR file. 
Run your project with apache tomcat server in your computer. (25 pts)

2/ Configure your springboot project to connect to mysql database. Create a student table with the following data, (25 pts)

{
  {
    "student_id": 1,
    "student_name": "John"
    "age": 21
  },
  {
    "student_id": 2,
    "student_name": "Jack"
    "age": 22
  },
  {
    "student_id": 3,
    "student_name": "Tom"
    "age": 23
  }
}

Endpoint: `/students/all`

To test it with http get, try to go to your browser and hit `localhost:8080/students/all`, and it should show the above json in the web page.

3/ Write a Http post method endpoint `/students/add` to add a record to your database, and prove the record is added using `http get` endpoint from step 2. (25 pts)

4/ Write a summary of the following annotation. (25 pts)

@Autowired
@SpringBootApplication
@Component
@Service
@Repository
@Configuration
@RequestMapping vs @GetMapping
@Bean
@Qualifier
@RequestBody

