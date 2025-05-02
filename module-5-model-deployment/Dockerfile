# official Python runtime image
FROM python:3.9-slim

# working directory in the container
WORKDIR /app



COPY pyproject.toml .
COPY README.md .


# Install the package and its dependencies defined in pyproject.toml
RUN pip install --no-cache-dir --trusted-host pypi.python.org .

#  Application Code and Artifacts 
COPY ./src /app/src

# Copy artifacts 
COPY ./artifacts /app/artifacts


# Points to the Flask app instance 'app' within the 'api.py' module
# inside the 'iris_predictor' package.
ENV FLASK_APP=src.iris_predictor.api:app
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Expose the port 
EXPOSE 5000

# Command to run the Flask development server 
# 'flask run' will use the FLASK_APP, FLASK_RUN_HOST, FLASK_RUN_PORT variables
CMD ["flask", "run"]


