# Contributing to LiteLLM Local

## Development Setup

1. Fork and clone the repository
2. Copy `.env.example` to `.env` and configure
3. Start services with `./start_vllm.sh`

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run specific tests:
```bash
pytest tests/test_models.py::TestEmbeddings -v
```

## Code Style

- Follow PEP 8 for Python code
- Use descriptive variable names
- Add docstrings to functions
- Keep functions focused and small

## Submitting Changes

1. Create a feature branch: `git checkout -b feature-name`
2. Make your changes and test thoroughly
3. Commit with clear messages: `git commit -m "Add feature X"`
4. Push and create a pull request

## Project Structure

```
litellm_local/
├── docker-compose.vllmMin.yml    # 3 services (embed, chat, OCR)
├── docker-compose.vllmFull.yml   # 4 services (+ Whisper audio)
├── docker-compose.litellm.yml    # API gateway
├── litellm_config.yaml           # Model routing config
├── start_vllm.sh                 # Startup script
├── start_ollama.sh               # Alternative: use Ollama
├── tests/                        # Test suite
│   └── test_models.py
└── .env.example                  # Environment template
```

## Adding New Models

1. Update `.env.example` with new model name
2. Add service to appropriate docker-compose file
3. Add model configuration to `litellm_config.yaml`
4. Add tests to `tests/test_models.py`
5. Update README.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
