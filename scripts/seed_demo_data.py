#!/usr/bin/env python3
"""Seed demo data by running varied prompts through the router."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx


# Sample prompts across complexity tiers
SAMPLE_PROMPTS = [
    # Simple
    "What is 2+2?",
    "Tell me a joke",
    "What's the capital of France?",
    "Define 'algorithm'",
    "What is Python?",
    "Name three colors",
    "What is JSON?",
    "Who wrote Harry Potter?",
    "What is the speed of light?",
    "Define 'function'",
    "What is CSS?",
    "Name a web browser",
    "What is HTML?",
    "What is a database?",
    "What is an API?",
    "Who is Albert Einstein?",
    "What is Linux?",
    "What is a URL?",
    "What is Docker?",
    "What is REST?",
    # Moderate
    "Explain how photosynthesis works",
    "Compare Python and JavaScript for web development",
    "Write a short poem about coding",
    "How do I create a simple todo app?",
    "Explain the difference between HTTP and HTTPS",
    "What are the benefits of using TypeScript?",
    "Write a function to reverse a string in Python",
    "Explain the concept of recursion with an example",
    "How does OAuth 2.0 work?",
    "What are the SOLID principles?",
    "Explain the difference between REST and GraphQL",
    "How do I implement pagination in an API?",
    "Write a script to fetch data from an API",
    "Explain how DNS resolution works",
    "What are the differences between TCP and UDP?",
    "How do I deploy a web app to AWS?",
    "Explain the concept of microservices",
    "Write a function to find duplicates in an array",
    "How does JWT authentication work?",
    "Explain the event loop in JavaScript",
    "How do I implement rate limiting?",
    "Write a function to implement binary search",
    "Explain how CDNs improve website performance",
    "What is the difference between cookies and sessions?",
    "Explain the MVC architecture pattern",
    "Write a function to validate a JSON schema",
    "How does HTTPS encryption work?",
    "What are the benefits of using Redis?",
    "Explain how message queues work",
    "Write a function to flatten a nested array",
    # Complex
    "Design and implement a distributed rate limiting system that works across multiple servers with accurate enforcement and minimal latency",
    "Build a production-ready recommendation engine using collaborative and content-based filtering with cold-start handling",
    "Create a comprehensive CI/CD platform supporting parallel builds, complex workflows, and deployment strategies",
    "Design a sophisticated fraud detection system using machine learning with real-time analysis and explainable predictions",
    "Implement a high-performance order matching engine for trading with microsecond latency",
    "Build a scalable video processing pipeline with transcoding, CDN distribution, and adaptive bitrate streaming",
    "Create a production-grade API gateway with routing, authentication, rate limiting, and circuit breaking",
    "Design a comprehensive data governance platform with lineage tracking and policy enforcement",
    "Build a sophisticated chatbot platform with NLU, context management, and multi-channel support",
    "Implement a distributed tracing system with intelligent sampling and dependency mapping",
]


async def seed_data():
    """Seed demo data."""
    print("=" * 80)
    print("LLM Cost Router - Demo Data Seeding")
    print("=" * 80)
    print()

    api_url = "http://localhost:8000/v1/route"
    
    print(f"API URL: {api_url}")
    print(f"Total prompts to process: {len(SAMPLE_PROMPTS)}")
    print()
    print("Starting data seeding...")
    print("-" * 80)

    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, prompt in enumerate(SAMPLE_PROMPTS, 1):
            print(f"\n[{i}/{len(SAMPLE_PROMPTS)}] Processing: {prompt[:70]}...")
            
            try:
                response = await client.post(
                    api_url,
                    json={
                        "prompt": prompt,
                        "run_eval": True,
                    },
                )
                response.raise_for_status()
                data = response.json()

                routing = data['routing_decision']
                print(f"  ✓ Routed to: {routing['selected_model']} ({routing['selected_tier']} tier)")
                print(f"    Cost: ${data['cost_usd']:.6f}, Savings: ${data['savings_usd']:.6f} ({data['savings_percent']:.1f}%)")
                
                if data.get('eval_scores'):
                    eval_scores = data['eval_scores']
                    status = "✓ Passed" if eval_scores['passed'] else "✗ Failed"
                    print(f"    Quality: {eval_scores['quality_score']:.3f} {status}")

            except httpx.HTTPError as e:
                print(f"  ✗ Request failed: {e}")
            except Exception as e:
                print(f"  ✗ Error: {e}")

            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.1)

    print()
    print("-" * 80)
    print("Demo data seeding complete!")
    print("=" * 80)


def main():
    """Main entry point."""
    asyncio.run(seed_data())


if __name__ == "__main__":
    main()
