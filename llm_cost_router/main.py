"""Main entry point for running the FastAPI server."""

import uvicorn

from .api import create_app


def main():
    """Run the FastAPI server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
