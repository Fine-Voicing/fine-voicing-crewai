#!/bin/bash

# Create a temporary combined env file
ENV_FILE=".env.docker"

# Always start with .env
cat .env > $ENV_FILE

# If .env.local exists, append it (it will override .env values)
if [ -f .env.local ]; then
    echo "Found .env.local, applying overrides..."
    cat .env.local >> $ENV_FILE
fi

# Run docker-compose with the combined env file
docker compose --env-file $ENV_FILE "$@"

# Clean up
rm $ENV_FILE
