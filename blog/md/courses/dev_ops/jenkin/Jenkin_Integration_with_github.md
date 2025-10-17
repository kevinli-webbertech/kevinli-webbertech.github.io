# Maven Pipeline Configuration & GitHub Integration

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

## Step 2 Create a Jenkinsfile

* Create a Github Project (register a github account)

1/ Click on the green `new` button,

![create_git_repo.png](../../../../images/dev_ops/jenkin/create_git_repo.png)

2/ Create a new repo,

![create_git_repo1.png](../../../../images/dev_ops/jenkin/create_git_repo1.png)

Next screen provides the instructions to init and checkout your new repo to your local,

![create_git_repo2.png](../../../../images/dev_ops/jenkin/create_git_repo2.png)

Open a terminal and checkout the repo to your local with the instruction on the left hand side,

![create_git_repo3.png](../../../../images/dev_ops/jenkin/create_git_repo3.png)

```commandline
kevin@kevin-li:~/git$ git clone https://github.com/spring-guides/gs-spring-boot.git
Cloning into 'gs-spring-boot'...
remote: Enumerating objects: 1745, done.
remote: Counting objects: 100% (41/41), done.
remote: Compressing objects: 100% (24/24), done.
remote: Total 1745 (delta 29), reused 17 (delta 17), pack-reused 1704 (from 2)
Receiving objects: 100% (1745/1745), 1.07 MiB | 4.54 MiB/s, done.
Resolving deltas: 100% (1104/1104), done.
```

* Fork the above Springboot project from https://github.com/spring-guides/gs-spring-boot

`kevin@kevin-li:~/git/gs-spring-boot/complete$ cp -rf * ~/git/my-gs-spring-boot/`

And make sure you have everything,

```commandline
kevin@kevin-li:~/git/my-gs-spring-boot$ ls
build.gradle  gradle  gradlew  gradlew.bat  mvnw  mvnw.cmd  pom.xml  settings.gradle  src
```

* Create a Jenkinsfile below in your project layout.

```
pipeline {
    agent any

    options {
        skipStagesAfterUnstable()
    }

    tools {
        maven '3.9.11'
    }

    stages {
        stage('Checkout Source Code') {
            steps {
                git branch: 'main', url: 'https://github.com/kevinli-webbertech/gs-spring-boot.git'
            }
        }

        stage('Test') {
            steps {
                sh 'git --version'
                sh 'mvn --version'
                sh 'mvn clean test' // Example for a Maven project
            }
        }

        stage('Build and Package') {
            steps {
                sh 'mvn clean package -DskipTests'
            }
        }

        stage('Archive Artifacts') {
            steps {
                archiveArtifacts artifacts: 'target/*.jar', fingerprint: true
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
        success {
            echo 'Build successful!'
        }
        failure {
            echo 'Build failed!'
        }
    }

}
```

Now use the following command to see what files needs to be committed to the git repo,

`git status`

![Jenkinsfile.png](../../../../images/dev_ops/jenkin/Jenkinsfile.png)

Then, we run the following commands to make sure we add all the files, 

`git add *`

Next, run `git status` again, you will see these files turned green.

And then do the following to commit these files,

`git commit  -m "initial commit"`

Then, push them to your remote repo by running the following commands,

`git push`

> For the first time, you would need to enter username and password, please use the login credential you use to login to the github.io. 

## Create a pipeline job

![pipeline_job.png](../../../../images/dev_ops/jenkin/pipeline_job.png)

Put your github url there,

Image 1,

![pipeline_job1.png](../../../../images/dev_ops/jenkin/pipeline_job1.png)

Image 2,

![pipeline_job2.png](../../../../images/dev_ops/jenkin/pipeline_job2.png)

Image 3,

![pipeline_job3.png](../../../../images/dev_ops/jenkin/pipeline_job3.png)

Image 4,

![pipeline_job4.png](../../../../images/dev_ops/jenkin/pipeline_job4.png)

Image 5,

![pipeline_job5.png](../../../../images/dev_ops/jenkin/pipeline_job5.png)

Image 6,

![pipeline_job6.png](../../../../images/dev_ops/jenkin/pipeline_job6.png)

Image 7,

![pipeline_job7.png](../../../../images/dev_ops/jenkin/pipeline_job7.png)

### Configure the github repo in the Jenkin plugin

Click the `Manage` button on the right top corner,

![Jenkins_management.png](../../../../images/dev_ops/jenkin/Jenkins_management.png)

Then you are going to see the following screen,

![Jenkins_management1.png](../../../../images/dev_ops/jenkin/Jenkins_management1.png)

Click on `Tools`, and then scroll down to the bottom, 

![Jenkins_tools.png](../../../../images/dev_ops/jenkin/Jenkins_tools.png)

Next to click on `Maven Installations`, and you will the UI looks like the following,

![Jenkins_maven_plugin.png](../../../../images/dev_ops/jenkin/Jenkins_maven_plugin.png)

In this pre-built image, we already have this maven version `3.9.11` installed. But here is a caveat,
you need to make sure that 3.9.11 is filled into the `Name`, so you see above image, there are two 3.9.11.

### Double-check the maven plugin and versions in your `Jenkinsfile`

And also make sure that, this version is coded in your `Jenkinsfile` in your springboot project root layout.
For example, it should look like this,

![Jenkinsfile_maven_version.png](../../../../images/dev_ops/jenkin/Jenkinsfile_maven_version.png)

### Jenkins_pipeline_configuration - Configure the github repo

After we created the `pipeline_job`, we will go to the `configure`, it is found on the left panel in the following image,

![Jenkins_configure.png](../../../../images/dev_ops/jenkin/Jenkins_configure.png)

On your left side, make sure you click on `Pipeline` button,

![Jenkins_pipeline_configuration.png](../../../../images/dev_ops/jenkin/Jenkins_pipeline_configuration.png)

In the `Definition` dropdown menu, we select `Pipeline script from SCM`,

And in the `SCM` dropdown menu, we select `Git` (see in the image above),

Then copy&paste your new repo where we put the `hello world` example of springboot project, and PLEASE MAKE SURE
your `Jenkinsfile` is in there,

Once you are done, click `Save`.

### Execute a Jenkin job

Click on `Pipeline_job` in the breadcrum,

![pipeline_job.png](../../../../images/dev_ops/jenkin/pipeline_job.png)

Then click on `Build Now` on the left panel,  and it will successfully build the project and produced a .jar file.

The following `green` shows the successful jobs, and the `red` shows the failed jobs.

![jenkin_jobs.png](../../../../images/dev_ops/jenkin/jenkin_jobs.png)

For instance, the Job `#11` was the one I just run, click into it, and check the `Console output`.
Normally the failed job would have some errors in it.

![jenkin_build_jar.png](../../../../images/dev_ops/jenkin/jenkin_build_jar.png)

If your run job is `red` and failed, please click into it, and view the 

## Default Admin Password

![admin_login.png](../../../../images/dev_ops/jenkin/admin_login.png)

Retrieve the default admin password, if you lost or forgot the default hash string of password, you could retrieve it like the following,

![admin_login_pwd.png](../../../../images/dev_ops/jenkin/admin_login_pwd.png)

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
