#!/bin/bash
# scripts/init_db.sh

# Ensure we're in the project root
cd "$(dirname "$0")/.."

# Create necessary directories if they don't exist
mkdir -p data
mkdir -p logs

# Run the database initialization script
echo "Running database initialization..."
python src/scripts/init_database.py

if [ $? -eq 0 ]; then
    echo "Database initialization completed successfully!"
else
    echo "Error: Database initialization failed!"
    exit 1
fi