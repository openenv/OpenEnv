<!-- openenv-source: websearch_env -->
# Web Search Environment

A web search environment that searches the web with Google Search API (via Serper.dev).

## Prerequisites

### API Key Setup

This environment requires a Serper.dev API key to function. 

1. **Get your API Key:**
   - Visit [Serper.dev](https://serper.dev/) and sign up for an account
   - Navigate to your dashboard to get your API key
   - Free tier includes 2,500 free searches

2. **Configure the API Key:**

   **For Local Development:**
   ```bash
   export SERPER_API_KEY="your-api-key-here"
   ```

   **For Docker:**
   ```bash
   docker run -e SERPER_API_KEY="your-api-key-here" web_search-env:latest
   ```

   **For Hugging Face Spaces:** pass it as a secret when you push (see [Deploying to Hugging Face Spaces](#deploying-to-hugging-face-spaces)), or add a `SERPER_API_KEY` secret in the Space settings.

   > **Important:** Never commit your API key to code. Always use environment variables or secrets management.

## Quick Start

The simplest way to use the Web Search environment is through the `WebSearchEnv` client:

```python
from envs.websearch_env import WebSearchAction, WebSearchEnv

try:
    # Create environment from Docker image
    web_search_env = WebSearchEnv.from_docker_image("web_search-env:latest").sync()

    # Reset
    result = web_search_env.reset()
    print(f"Reset: {result.observation.content}")

    # Send a search query
    query = "What is the capital of China?"

    result = web_search_env.step(WebSearchAction(query=query))
    print(f"Formatted search result:", result.observation.content)
    print(f"Individual web contents:", result.observation.web_contents)

finally:
    # Always clean up
    web_search_env.close()
```

That's it! The `WebSearchEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From the websearch_env directory
cd envs/websearch_env
docker build -t web_search-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

From `envs/websearch_env/`:

```bash
openenv push --repo-id my-org/websearch-env --secret SERPER_API_KEY=your-api-key-here
```

The environment doesn't work without `SERPER_API_KEY`. `--secret` stores it as a Space secret and never logs it. See the [`openenv push` reference](https://huggingface.co/docs/openenv/reference/cli#openenv-push) for all options. The Space serves the web UI at `/web`, the API docs at `/docs` and a health check at `/health`.

## Environment Details

### Action
**WebSearchAction**: Contains a single field
- `query` (str) - The query to search for
- `temp_api_key` (str) - Temporary Serper.dev API key if not set in environment variables.

### Observation
**WebSearchObservation**: Contains the echo response and metadata
- `content` (str) - The formatted prompt that aggregates both query and web contents
- `web_contents` (list) - List of web contents for top ranked web pages
- `reward` (float) - Not computed (see [Reward](#reward))
- `done` (bool) - Always False for search environment
- `metadata` (dict) - Additional info like step count

### Reward
The environment doesn't compute a reward. `reward` is `0.0` after `reset()` and `None` after each step.

## Advanced Usage

### Connecting to an Existing Server

If you already have a Web Search environment server running, you can connect directly:

```python
from envs.websearch_env import WebSearchAction, WebSearchEnv

# Connect to existing server
web_search_env = WebSearchEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = web_search_env.reset()
result = web_search_env.step(WebSearchAction(query="What is the capital of China?"))
```

Note: When connecting to an existing server, `web_search_env.close()` will NOT stop the server.

## Running Locally

```bash
# Make sure to set your API key first
export SERPER_API_KEY="your-api-key-here"

# Then run the server
uvicorn server.app:app --reload
```
