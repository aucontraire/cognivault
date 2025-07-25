-- PostgreSQL initialization script for CogniVault
-- This script sets up the database with required extensions

-- Enable the vector extension for pgvector (similarity search)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create a basic schema structure for future use
-- (Tables will be created by Alembic migrations later)

-- Set timezone
SET timezone = 'UTC';

-- Basic configuration
SELECT 'Database initialized successfully for CogniVault' AS status;