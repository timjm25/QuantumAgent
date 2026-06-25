"""
Central configuration for the QuantumAgent platform.
All settings are environment-variable overridable via pydantic-settings.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "QuantumAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    environment: str = "production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False

    # Security
    secret_key: str = Field(default="change-me-in-production-use-256-bit-key")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24  # 24h
    refresh_token_expire_days: int = 30

    # Database
    database_url: str = "sqlite:///./quantumagent.db"
    redis_url: str = "redis://localhost:6379/0"

    # Vector store (Qdrant)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "quantum_research"

    # Knowledge Graph (Neo4j)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "quantumagent"

    # Message broker (RabbitMQ / Redis)
    broker_url: str = "redis://localhost:6379/1"
    result_backend: str = "redis://localhost:6379/2"

    # Claude / LLM backend (Claude 4.x via Anthropic API)
    anthropic_api_key: Optional[str] = None
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.1

    # Quantum hardware APIs
    ibm_quantum_token: Optional[str] = None
    ionq_api_key: Optional[str] = None
    quantinuum_api_key: Optional[str] = None
    rigetti_api_key: Optional[str] = None
    aws_braket_region: str = "us-east-1"
    azure_quantum_subscription: Optional[str] = None

    # Scientific APIs
    semantic_scholar_key: Optional[str] = None
    pubmed_email: Optional[str] = None
    arxiv_max_results: int = 50

    # Storage
    storage_path: Path = Path("./data")
    experiment_store_path: Path = Path("./data/experiments")
    model_checkpoint_path: Path = Path("./data/checkpoints")

    # Neural models
    embedding_dim: int = 768
    gnn_hidden_dim: int = 256
    gnn_num_layers: int = 4
    transformer_heads: int = 8
    transformer_layers: int = 6

    # Workflow engine
    max_concurrent_agents: int = 12
    agent_timeout_seconds: int = 3600
    workflow_checkpoint_interval: int = 60

    # CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "https://quantumagent.ai",
    ]

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
