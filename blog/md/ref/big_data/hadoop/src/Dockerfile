FROM ubuntu:latest
LABEL authors="thoophoms"

ENTRYPOINT ["top", "-b"]

# Dockerfile
FROM ubuntu:22.04 AS builder

# Install Java Development Kit
RUN apt-get update && apt-get install -y openjdk-8-jdk

# Set the working directory
WORKDIR /app

# Add HelloWorld.java to the working directory
ADD HelloWorld.java .

# Compile the Java program
RUN javac -source 8 -target 8 HelloWorld.java

# Use a smaller base image for the final image
FROM ubuntu:22.04

# Set the working directory
WORKDIR /

# Copy the compiled Java class from the builder stage
COPY --from=builder /app/HelloWorld.class .

# Keep the container running
CMD ["tail", "-f", "/dev/null"]
