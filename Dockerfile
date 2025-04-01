# Use a public Python slim image as the base
FROM python:3.10-slim

# Update and install system packages including ffmpeg, Node.js dependencies, and procps for the ps command
RUN apt-get update && \
    apt-get install -y curl gnupg build-essential procps ffmpeg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the Node.js package files and install frontend dependencies
COPY package*.json ./
RUN npm install

# Copy the Python requirements (assuming they're in src/requirements.txt) and install backend dependencies
COPY src/requirements.txt ./src/
RUN pip install --upgrade pip && pip install -r src/requirements.txt

# Install lightning-app if needed
RUN pip install lightning-app

# Copy the rest of your project files into the container
COPY . .

# Expose the port used by the frontend dev server (assuming 5173)
EXPOSE 5173

# Command to run your app (adjust if necessary)
CMD ["python", "src/server.py"]