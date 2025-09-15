# Python App Deployment

## Dockerfile, syntax and more

Docker is a simple tool and its syntax is very easy. However, if you following a tutorial of the following site will show you more examples you can try after the class,

https://docker-curriculum.com/#webapps-with-docker

In this class, we would learn to deploy a simple Python flask webserver app into a docker, so you get a feel of what
the container is and what a containerized image is.

Here is a practical example of writing a `Dockerfile` for a simple Python application, with a detailed explanation of each step.
The reason we choose `python` instead of `java` is to avoid complex installation of JDK in your operating system, especially for Mac.

### Example Project Structure

Imagine you have a simple Python web application using the Flask framework. Your project folder looks like this:

```
my-python-app/
├── app.py
├── requirements.txt
└── Dockerfile  <-- We will create this
```

**`app.py`** (Your application code):
```python
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, Docker World!'

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
```
*Note: `host='0.0.0.0'` is crucial. It makes the server available outside the container.*

**`requirements.txt`** (Your Python dependencies):
```
Flask==2.3.3
```

---

### The Dockerfile

Create a new file named **`Dockerfile`** (no file extension) in the same directory as `app.py`.

```Dockerfile
# Step 1: Choose the base image. This provides the foundation.
# We use an official Python runtime with a specific version and a slim OS (Debian).
FROM python:3.9-slim-buster

# Step 2: Set the working directory inside the container.
# All subsequent commands (COPY, RUN, CMD) will be run from this directory.
WORKDIR /app

# Step 3: Copy the requirements file first.
# We do this separately from the app code to leverage Docker's cache layer.
# If requirements.txt doesn't change, Docker won't reinstall dependencies on subsequent builds.
COPY requirements.txt requirements.txt

# Step 4: Install the Python dependencies defined in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of the application code from the host to the container's working directory.
COPY . .

# Step 6: Inform Docker that the container listens on port 5000 at runtime.
# This is mostly documentation; the actual mapping is done at `docker run`.
EXPOSE 5000

# Step 7: Define the command to run the application when the container starts.
# This is the process that will keep the container alive.
# We use the "list" form (CMD ["executable", "param1", "param2"]) for best results.
CMD ["python", "app.py"]
```

---

### Building the Image and Running the Container

Now that you have your `Dockerfile`, follow these steps:

1.  **Open a terminal** and navigate to your `my-python-app` directory.

2.  **Build the Docker Image.** The `-t` flag tags your image with a name (`my-python-app`) and the `.` tells Docker to look for the `Dockerfile` in the current directory.
    ```bash
    docker build -t my-python-app .
    ```

3.  **Run the Container from your Image.** The `-p` flag maps your local machine's port 5000 to the container's internal port 5000 (which we exposed with `EXPOSE`). The `-d` flag runs the container in "detached" mode (in the background).
    ```bash
    docker run -d -p 5000:5000 --name my-running-app my-python-app
    ```

4.  **Test it!** Open your web browser and go to `http://localhost:5000`. You should see the message "Hello, Docker World!".

5.  **See your running container:**
    ```bash
    docker ps
    ```

6.  **Stop the container when you're done:**
    ```bash
    docker stop my-running-app
    ```

---

### Key Dockerfile Instructions Explained

| Instruction | Purpose | Example |
| :--- | :--- | :--- |
| **`FROM`** | **Mandatory first instruction.** Sets the base image to build upon. | `FROM python:3.9-slim-buster` |
| **`WORKDIR`** | Sets the working directory for any `RUN`, `CMD`, `ENTRYPOINT`, `COPY`, and `ADD` instructions that follow. | `WORKDIR /app` |
| **`COPY`** | Copies files or directories from the host machine (your computer) to the container's filesystem. | `COPY . .` |
| **`RUN`** | Executes a command in a new layer *on top of the current image* and commits the result. Used for installing packages. | `RUN pip install -r requirements.txt` |
| **`EXPOSE`** | Documents which port the container application listens on. This is informational. | `EXPOSE 5000` |
| **`CMD`** | Provides the default command to run when the container starts. There can only be one `CMD` in a Dockerfile. | `CMD ["python", "app.py"]` |

### Best Practices Demonstrated

1.  **Use a `.dockerignore` file:** Create a file called `.dockerignore` in the same directory to prevent copying unnecessary files (like local virtual environments, `__pycache__`, etc.), which makes your build faster and your image smaller.
    ```
    __pycache__
    *.pyc
    .env
    venv/
    Dockerfile
    .git
    .gitignore
    ```

2.  **Leverage Cache:** Copying `requirements.txt` and installing dependencies *before* copying the application code allows Docker to use its build cache. If your app code changes but your dependencies don't, it won't waste time re-running the `pip install` step.

3.  **Choose Small Base Images:** Using `python:3.9-slim-buster` instead of the default `python:3.9` results in a much smaller and more secure image, as it contains only the minimal packages needed to run Python.

This is the fundamental workflow for containerizing an application with Docker. You define your environment in a `Dockerfile`, build it into an image, and then run instances of that image as containers.