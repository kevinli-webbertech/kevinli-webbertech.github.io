## Maven Pipeline Configuration & GitHub Integration

In this tutorial, we use a Maven-based Java project from GitHub to go from freestyle jobs to scripted pipelines.

# Step 1 Create a Jenkin Job Pipeline

First screen,

![jenkin_pipeline.png](../../../../images/dev_ops/jenkin/jenkin_pipeline.png)

Next,

![jenkin_pipeline1.png](../../../../images/dev_ops/jenkin/jenkin_pipeline1.png)

Next,

![jenkin_pipeline2.png](../../../../images/dev_ops/jenkin/jenkin_pipeline2.png)

Next,

![jenkin_pipeline3.png](../../../../images/dev_ops/jenkin/jenkin_pipeline3.png)

Next,

![jenkin_pipeline4.png](../../../../images/dev_ops/jenkin/jenkin_pipeline4.png)

Next,

![jenkin_pipeline5.png](../../../../images/dev_ops/jenkin/jenkin_pipeline5.png)

>Note: In Jenkins, a `Spring Boot Hello World` project was used as the base for creating a Pipeline job.
>GitHub Repo: <https://github.com/spring-guides/gs-spring-boot>

# Step 2 Create a Jenkinsfile

* Create a Github Project (register a github account)

1/ Click on the green `new` button,

![create_git_repo.png](../../../../images/dev_ops/jenkin/create_git_repo.png)

2/ Create a new repo,

![create_git_repo1.png](create_git_repo1.png)

* Fork the above springboot project from https://github.com/spring-guides/gs-spring-boot

1/ First you need to checkout your own repo,

* Create a Jenkinsfile below in your project layout.

```
pipeline {
agent any

options {
    skipStagesAfterUnstable()
}

tools {
    maven ‘Maven 3.9.6’
}

stages {
    stage(‘Build’) {
        steps {
            sh ‘mvn clean compile’
        }
    }

    stage(‘Test’) {
        steps {
            sh ‘mvn test’
        }
        post {
            always {
                junit ‘target/surefire-reports/*.xml’
            }
        }
    }
    stage(‘Package’) {
        steps {
            sh ‘mvn package’
        }
    }
    stage(‘Deliver’) {
        steps {
            sh ‘./jenkins/scripts/deliver.sh’
        }
    }
}
}
```

![My three failed attempts...](https://miro.medium.com/v2/resize:fit:786/format:webp/1*h2sBSI_hZms6VdAukapTfg.png)

It will successfully build the project and produced a .jar file.

![Build #4 status](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*enVxCjmZfr0agSH0YqS5-g.png)

## Testing, Debugging & Final Pipeline

Concentrated on incorporating automated testing into the pipeline on day three. There was already a JUnit-written test class in my Maven project. I modified the Jenkinsfile to execute `mvn test` and use the JUnit plugin to report the outcomes.

![Jenkinsfile](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*WA-EZCYSJfyeN0YDNFNcpg.png)

![Jenkinsfile2](https://miro.medium.com/v2/resize:fit:1100/format:webp/1*q7PnUlyYrLEPGRtGvXR91g.png)

Here is the test stage I added to my Jenkinsfile:

```commandline
stage(‘Test’) {
    steps {
        sh ‘mvn test’
    }
    post {
        always {
            junit ‘target/surefire-reports/TEST-*.xml’
        }
    }
}
```

After building the pipeline, the test results were collected and displayed under my Jenkins job.

![Test results](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*qUZ4fNd2kH_puksI)

Explored test result locations, checked the contents of `target/surefire-reports`, and validated my XML and TXT outputs.

Confirmed the JUnit plugin was properly installed:

![Installed plugins](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*jMIwJDM5GkxZlhQ1)

Both test cases passed, and Jenkins displayed a green success badge.

Also double-checked the status and artifacts for each pipeline build.

![Build #11 status](https://miro.medium.com/v2/resize:fit:1100/format:webp/0*nqzS9s1xpn7KuQGu)

We are now aware of the connections between builds, tests, artifacts, and outputs. Jenkins has a lot of capability, but learning pipelines and test reporting takes effort. It’s important to consider console output rather than visual tabs for everything. Have patience; most issues are the result of small naming or syntax mistakes.

## Reflections

This three-day trip was both difficult and rewarding, as I went from being a complete newbie to using a Maven project in Jenkins to execute builds and tests.
I asked questions, made mistakes, and gained knowledge at every turn. Jenkins no longer seems so menacing.

I didn’t just learn Jenkins; I gained knowledge about DevOps workflows! Starting with code, testing automatically, and seeing results all in one tool has been eye-opening.