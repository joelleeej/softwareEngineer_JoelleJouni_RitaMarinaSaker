# Use the official Python image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container
COPY . .

# Install Python dependencies
RUN pip install poetry
RUN poetry install

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the Flask app
CMD ["poetry", "run", "python", "app.py"]
