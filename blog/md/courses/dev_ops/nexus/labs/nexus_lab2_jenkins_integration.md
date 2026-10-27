# Nexus Lab2: Jenkins Integration — Build a Jar and Push It to Nexus

Written by: Kevin Li

## Goal

Lab1 uploaded a jar to Nexus by hand through the web UI. This lab automates that step: a
Jenkins pipeline checks out the `gs-spring-boot` project, builds the jar with Maven, and
publishes it to a local Nexus Maven repository with `mvn deploy`.

Reference project: [https://github.com/kevinli-webbertech/gs-spring-boot](https://github.com/kevinli-webbertech/gs-spring-boot)
(`groupId: com.example`, `artifactId: spring-boot-complete`, `version: 0.0.1-SNAPSHOT`).

### Knowledge

* Maven's `distributionManagement` section tells `mvn deploy` which remote repository to
  publish to, for release versions and for `-SNAPSHOT` versions respectively.
* Maven does **not** accept `-Dusername`/`-Dpassword` on the command line to authenticate a
  deploy. Credentials must come from a `<server>` entry in `settings.xml` whose `id` matches
  the repository id used in `distributionManagement` (or on `-DrepositoryId` for
  `deploy:deploy-file`). A Jenkins pipeline can generate that `settings.xml` on the fly from a
  Jenkins credential, so nothing secret is stored in the repo.
* Because `com.example:spring-boot-complete` is currently `0.0.1-SNAPSHOT`, Maven routes the
  deploy to the **snapshot** repository, not the release one.

## Labs

### Step 1: Confirm Nexus Is Running

Reuse the Nexus container from Lab1 (`docker run -d -p 8081:8081 ... sonatype/nexus3`). Confirm
it is reachable at `http://localhost:8081` and that a `maven-snapshots` hosted repository exists
(Nexus ships this repository by default under **Repository → Repositories**).

### Step 2: Add `distributionManagement` to `pom.xml`

```xml
<distributionManagement>
    <repository>
        <id>nexus-releases</id>
        <url>http://localhost:8081/repository/maven-releases/</url>
    </repository>
    <snapshotRepository>
        <id>nexus-snapshots</id>
        <url>http://localhost:8081/repository/maven-snapshots/</url>
    </snapshotRepository>
</distributionManagement>
```

The `id` values (`nexus-releases`, `nexus-snapshots`) are arbitrary, but whatever you choose
here must match the `<server>` ids in the `settings.xml` used at deploy time (Step 4).

### Step 3: Create a Jenkins Credential for the Nexus Deploy User

**Manage Jenkins → Credentials → System → Global credentials → Add Credentials**

* **Kind:** Username with password
* **Username:** a Nexus user allowed to deploy to `maven-snapshots` (e.g. `admin`, or a
  dedicated deploy account)
* **Password:** that user's Nexus password
* **ID:** `nexus-credentials`

### Step 4: Add a "Deploy to Nexus" Stage to the Jenkinsfile

```groovy
environment {
    NEXUS_RELEASE_REPO_ID  = 'nexus-releases'
    NEXUS_SNAPSHOT_REPO_ID = 'nexus-snapshots'
}

stages {
    // ... Checkout, Test, Build and Package, Archive Artifacts stages ...

    stage('Deploy to Nexus') {
        steps {
            withCredentials([usernamePassword(
                credentialsId: 'nexus-credentials',
                usernameVariable: 'NEXUS_USER',
                passwordVariable: 'NEXUS_PASSWORD'
            )]) {
                writeFile file: 'nexus-settings.xml', text: """<settings>
  <servers>
    <server>
      <id>${NEXUS_RELEASE_REPO_ID}</id>
      <username>${NEXUS_USER}</username>
      <password>${NEXUS_PASSWORD}</password>
    </server>
    <server>
      <id>${NEXUS_SNAPSHOT_REPO_ID}</id>
      <username>${NEXUS_USER}</username>
      <password>${NEXUS_PASSWORD}</password>
    </server>
  </servers>
</settings>"""
                sh 'mvn deploy -DskipTests -s nexus-settings.xml'
            }
        }
    }
}

post {
    always {
        sh 'rm -f nexus-settings.xml'
    }
}
```

`withCredentials` injects `NEXUS_USER`/`NEXUS_PASSWORD` only for the duration of that block and
masks them in the console log. The generated `nexus-settings.xml` is written to the job's
workspace and removed in `post { always { ... } }` so the credentials never persist on disk.

The full, working Jenkinsfile for this lab is in the project repo:
[https://github.com/kevinli-webbertech/gs-spring-boot/blob/main/Jenkinsfile](https://github.com/kevinli-webbertech/gs-spring-boot/blob/main/Jenkinsfile)

### Step 5: Run the Pipeline and Verify in Nexus

Run the Jenkins job. On success, open Nexus → **Browse** → `maven-snapshots` and confirm the
artifact was published under:

```text
com/example/spring-boot-complete/0.0.1-SNAPSHOT/
```

Each snapshot deploy adds a new timestamped build to that folder; Nexus keeps the version
directory the same (`0.0.1-SNAPSHOT`) while the underlying jar/pom filenames get a unique
timestamp and build number.

## Troubleshooting

* **401/403 on deploy** — the Nexus user in the `nexus-credentials` Jenkins credential does not
  have deploy permission on `maven-snapshots`, or the password is wrong.
* **"Failed to deploy artifacts... Return code is: 400"** — you tried to deploy a non-SNAPSHOT
  version to the `maven-snapshots` repository (or vice versa); the repository's **Version
  Policy** must match the artifact's version.
* **`mvn deploy` can't find the server entry** — the `id` used in `pom.xml`'s
  `distributionManagement` does not exactly match a `<server><id>` in the `settings.xml` passed
  with `-s`.
* **Jenkins can't reach `localhost:8081`** — if Jenkins itself runs in a container, `localhost`
  inside that container is the container, not the Docker host. Run Jenkins and Nexus on the same
  Docker network, or use the host's reachable address instead of `localhost`.

## Ref

- [https://maven.apache.org/plugins/maven-deploy-plugin/](https://maven.apache.org/plugins/maven-deploy-plugin/)
- [https://help.sonatype.com/en/maven-repositories.html](https://help.sonatype.com/en/maven-repositories.html)
- Lab1: [nexus_lab1_webui_usage.md](./nexus_lab1_webui_usage.md)
