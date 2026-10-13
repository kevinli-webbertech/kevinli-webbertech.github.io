# Git Private Repo Integration with Jenkins (Docker)

## Goal

To check out code from a **private** GitHub repository using the official `jenkins/jenkins:lts` Docker image, you need to spin up the container and provide Jenkins with authenticating credentials. The image already comes with `git` pre-installed, so you do not need to modify the image itself. ([ref1](https://piotrminkowski.com/2018/06/12/local-continuous-delivery-environment-with-docker-and-jenkins/), [ref2](https://www.youtube.com/watch?v=vsEI0HIv8b4))

You can accomplish this using either **HTTPS (Personal Access Token)** or **SSH (Deploy Keys)**. ([ref1](https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27), [ref2](https://www.youtube.com/watch?v=vsEI0HIv8b4))

## Labs

### Step 1: Run the `jenkins/jenkins:lts` Container

Start by running the official LTS container: ([ref1](https://piotrminkowski.com/2018/06/12/local-continuous-delivery-environment-with-docker-and-jenkins/))

```bash
docker run -d \
  --name jenkins \
  -p 8080:8080 \
  -p 50000:50000 \
  -v jenkins_home:/var/jenkins_home \
  jenkins/jenkins:lts
```

Retrieve your initial administrator password by running `docker logs jenkins`, navigate to `http://localhost:8080`, and finish the initial setup wizard by installing the recommended plugins (which includes the **Git plugin**). ([ref1](https://piotrminkowski.com/2018/06/12/local-continuous-delivery-environment-with-docker-and-jenkins/), [ref2](https://tesh.digital/set-up-a-cicd-server-and-configure-for-use-with-private-github-repositories))

### Step 2: Choose Your Authentication Method

#### Option A: Authenticate via HTTPS (Personal Access Token) - Easiest

1. **Generate a PAT on GitHub**: Go to **Settings** > **Developer Settings** > **Personal Access Tokens** > **Tokens (classic)**. Click **Generate new token** and check the `repo` scope box. Copy the generated token. ([ref1](https://www.youtube.com/watch?v=vsEI0HIv8b4), [ref2](https://tesh.digital/set-up-a-cicd-server-and-configure-for-use-with-private-github-repositories))

2. **Add Credentials to Jenkins**: Go to **Manage Jenkins** > **Credentials** > **System** > **Global credentials** > **Add Credentials**.

   * **Kind:** Username with password
   * **Username:** Your GitHub username
   * **Password:** Paste your GitHub Personal Access Token
   * **ID:** `github-token` (or a descriptive name) ([ref1](https://www.youtube.com/watch?v=Xs6bXpJsGu4&t=150), [ref2](https://www.youtube.com/watch?v=t-UAzkZTKZI&t=362))

3. **Configure your Pipeline/Job**: Set the Repository URL to the HTTPS variant (e.g., `https://github.com/<username>/<repo>.git`) and select your newly created credential from the dropdown menu. ([ref1](https://www.youtube.com/watch?v=t-UAzkZTKZI&t=362))

#### Option B: Authenticate via SSH (Deploy Keys) - More Secure

1. **Generate SSH Keypair**: Open your terminal and generate an SSH key pair: ([ref1](https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27))

   ```bash
   ssh-keygen -t ed25519 -C "jenkins-deploy-key" -f ./jenkins_key
   ```

2. **Add Public Key to GitHub**: Go to your GitHub Private Repository > **Settings** > **Deploy keys** > **Add deploy key**. Paste the contents of `jenkins_key.pub`. ([ref1](https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27))

3. **Add Private Key to Jenkins**: Go to **Manage Jenkins** > **Credentials** > **System** > **Global credentials** > **Add Credentials**.

   * **Kind:** SSH Username with private key
   * **Username:** `git`
   * **Private Key:** Select **Enter directly** and paste the entire contents of your private key file (`jenkins_key`).
   * **ID:** `github-ssh` ([ref1](https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27), [ref2](https://www.youtube.com/watch?v=Xs6bXpJsGu4&t=150))

4. **Configure your Pipeline/Job**: Use the SSH repository URL (e.g., `git@github.com:your-username/your-repo.git`) and link your SSH credentials. ([ref1](https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27))

### Step 3: Implement Checkout in your Jenkinsfile

Once your credentials are configured, you can call them seamlessly inside a Jenkins Pipeline script using the `checkout` or `git` syntax: ([ref1](https://stackoverflow.com/questions/9767919/in-jenkins-how-to-checkout-a-project-into-a-specific-directory-using-git))

```groovy
pipeline {
    agent any
    stages {
        stage('Checkout Code') {
            steps {
                // If using Option A (HTTPS)
                git url: 'https://github.com/<username>/<repo>.git',
                    branch: 'main',
                    credentialsId: 'github-token'

                // OR If using Option B (SSH)
                // git url: 'git@github.com:your-username/your-repo.git',
                //    branch: 'main',
                //    credentialsId: 'github-ssh'
            }
        }
    }
}
```

## Troubleshooting

If you run into issues, check the following:

* Which **authentication type** (HTTPS token or SSH key) you are using.
* Any **error codes** or console output you see during checkout.

## Ref

- https://piotrminkowski.com/2018/06/12/local-continuous-delivery-environment-with-docker-and-jenkins/
- https://www.youtube.com/watch?v=vsEI0HIv8b4
- https://medium.com/@mai.vly/setup-jenkins-for-your-application-from-the-private-github-repo-with-docker-bac79335ff27
- https://tesh.digital/set-up-a-cicd-server-and-configure-for-use-with-private-github-repositories
- https://www.youtube.com/watch?v=Xs6bXpJsGu4&t=150
- https://www.youtube.com/watch?v=t-UAzkZTKZI&t=362
- https://stackoverflow.com/questions/9767919/in-jenkins-how-to-checkout-a-project-into-a-specific-directory-using-git

## Lab Document

- https://kevinli-webbertech.github.io/blog/md/courses/dev_ops/github/labs/GitPrivateRepoIntegrationWithJenkin.pdf
