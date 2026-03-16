# Setup Jenkins for private GitHub repository

## Goal

This lab deals with both Jenkin configuration and private Github repository.
The goal of this lab is to learn how to configure jenkin environmental variable and create
ssh private/public key and authenticate with Github to checkout its private repo.

### Knowledge

* Private/Public key

You can find some introduction here,

- https://kevinli-webbertech.github.io/blog/html/courses/cybersecurity/public_private_key.html
- https://kevinli-webbertech.github.io/blog/html/courses/cybersecurity/mathematics.html

* Jenkin Environmental variables

* Github Configurations

## Labs

### Generate SSH private&public key + Add the public key to the github server

See https://kevinli-webbertech.github.io/blog/md/courses/dev_ops/github/labs/Setup_SSH_KEY_GITHUB_lab2.pdf

### Create a Jenkin Job

![jenkin_repo.png](../../../../images/dev_ops/github/jenkin_repo.png)

Next step is job configuration. Let’s set only General part. So, we need to set Jenkins to Discard old builds(experiment with your own values) and to Execute concurrent builds if necessary.

![jenkin_configuration.png](../../../../images/dev_ops/github/jenkin_configuration.png)


### Continue job configuration

>Note: remember the following,

```commandline
 kevin@kevin-li:~$ docker exec e78a71ef7079 cat /var/jenkins_home/secrets/initialAdminPassword
61bace988c00479faead1029783c3780
```

With the password you can login as "admin" to Jenkin dashboard.

Let’s connect Jenkins job to our repository. In job configuration go to Source Code Management section and choose Git.


![git_credential.png](../../../../images/dev_ops/github/git_credential.png)

In the picture you can see I have some credentials already, it’s because I created them in the previous picture. Click Add and you’ll see the same screen. Pick Kind: SSH Username with private key and type username and again, let’s go to terminal.

```commandline
more /Users/Shared/Jenkins/.ssh/id_rsa # This time private key
(space, space, space until you see the entire private key)
```

Copy entire private key as in the picture (including BEGIN and END stuff). Save and choose just created credentials (as in the last screenshot).

Go to Build Triggers and checkoff Poll SCM. Schedule it for every 15mins.

`H/15 * * * *`

![build_trigger.png](../../../../images/dev_ops/github/build_trigger.png)

When you save here and run the build, it will pass. It will connect via SSH to your repository (and that’s basically all it will do).

## Ref

- https://medium.com/facademy/setup-jenkins-for-private-repository-9060f54eeac9
