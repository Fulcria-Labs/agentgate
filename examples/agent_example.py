#!/usr/bin/env python3
"""Example: How an AI agent uses AgentGate to securely access third-party services.

This script demonstrates the complete agent authentication flow:
  1. Authenticate with an API key (Bearer token)
  2. Request a scoped OAuth token for a specific service
  3. Handle step-up authentication (CIBA) when required
  4. Use the token to call the third-party API

Usage:
    export AGENTGATE_URL=http://localhost:8000
    export AGENTGATE_API_KEY=ag_xxxxxx
    python examples/agent_example.py
"""

import os
import sys
import time

import requests

BASE_URL = os.environ.get("AGENTGATE_URL", "http://localhost:8000")
API_KEY = os.environ.get("AGENTGATE_API_KEY", "")


def headers():
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def request_token(agent_id: str, service: str, scopes: list[str]) -> dict:
    """Request a scoped OAuth token through AgentGate.

    The policy engine enforces:
      - Agent must have an active policy
      - Service must be in the agent's allowed services
      - Requested scopes must be a subset of allowed scopes
      - Rate limits, time windows, and IP allowlist are checked
    """
    resp = requests.post(
        f"{BASE_URL}/api/v1/token",
        headers=headers(),
        json={"agent_id": agent_id, "service": service, "scopes": scopes},
    )

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code == 202:
        # Step-up authentication required (CIBA flow)
        return handle_step_up(resp.json())

    print(f"Token request denied ({resp.status_code}): {resp.json().get('detail', resp.text)}")
    sys.exit(1)


def handle_step_up(step_up_response: dict) -> dict:
    """Handle CIBA step-up authentication.

    When a service requires human approval, AgentGate initiates a push
    notification to the user's device. The agent polls until approved.
    """
    auth_req_id = step_up_response["auth_req_id"]
    print(f"Step-up auth required. Waiting for user approval (ID: {auth_req_id})...")

    for attempt in range(30):
        time.sleep(2)
        resp = requests.get(
            f"{BASE_URL}/api/v1/step-up/status/{auth_req_id}",
            headers=headers(),
        )
        data = resp.json()

        if data.get("status") == "approved":
            print("User approved! Token received.")
            return data

        if data.get("status") == "denied":
            print("User denied the request.")
            sys.exit(1)

        print(f"  Polling... ({attempt + 1}/30)")

    print("Step-up auth timed out.")
    sys.exit(1)


def list_available_services(agent_id: str) -> list[str]:
    """List which services this agent is authorized to access."""
    resp = requests.get(
        f"{BASE_URL}/api/v1/services",
        headers=headers(),
        params={"agent_id": agent_id},
    )
    if resp.status_code == 200:
        return resp.json().get("services", [])
    return []


def main():
    if not API_KEY:
        print("Error: Set AGENTGATE_API_KEY environment variable")
        print("  1. Log into the AgentGate dashboard")
        print("  2. Go to API Keys section")
        print("  3. Generate a key and bind it to your agent ID")
        sys.exit(1)

    agent_id = "my-github-bot"

    # Step 1: Check what services are available
    services = list_available_services(agent_id)
    print(f"Available services for {agent_id}: {services}")

    # Step 2: Request a scoped GitHub token
    print("\nRequesting GitHub token with 'repo' scope...")
    token_data = request_token(agent_id, "github", ["repo"])

    access_token = token_data.get("access_token", "")
    print(f"Token received (expires: {token_data.get('expires_in', '?')}s)")

    # Step 3: Use the token to call GitHub API
    print("\nFetching repos with scoped token...")
    gh_resp = requests.get(
        "https://api.github.com/user/repos?per_page=5",
        headers={"Authorization": f"token {access_token}"},
    )

    if gh_resp.status_code == 200:
        repos = gh_resp.json()
        for repo in repos:
            print(f"  - {repo['full_name']} ({'private' if repo['private'] else 'public'})")
    else:
        print(f"GitHub API error: {gh_resp.status_code}")


if __name__ == "__main__":
    main()
