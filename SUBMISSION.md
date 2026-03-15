# AgentGate - Hackathon Submission

## Tagline

A security-first gateway that gives users full control over how AI agents access their third-party services, powered by Auth0 Token Vault.

## What It Does

AgentGate sits between AI agents and the services they access on behalf of users. Instead of handing agents raw API keys with full access, AgentGate enforces per-agent policies, scoped tokens, rate limits, and a complete audit trail -- all powered by Auth0's Token Vault, CIBA, and Connected Accounts.

## The Problem

AI agents increasingly need to access services like GitHub, Slack, and Google on behalf of users. Current approaches are dangerously insecure:

- Agents store raw API keys with full, unscoped access
- No visibility into what actions agents perform with credentials
- No way to scope, rate-limit, or revoke access per agent
- No audit trail linking token usage back to specific agents
- No mechanism for users to approve sensitive operations in real time

## How Auth0 Powers the Solution

AgentGate is built entirely on Auth0 primitives:

1. **Token Vault + Connected Accounts**: Users authorize services once via OAuth. Auth0 manages the full token lifecycle (storage, refresh, rotation). Agents never see raw credentials.

2. **Token Exchange (RFC 8693)**: When an agent needs to access a service, AgentGate requests a scoped token from Token Vault. The policy engine ensures the agent only gets the scopes it is authorized to use.

3. **CIBA (Client-Initiated Backchannel Authentication)**: Sensitive operations trigger step-up auth. AgentGate pushes a consent request to the user, and the agent waits for approval before proceeding.

4. **My Account API**: Users manage their connected accounts, see active sessions, and revoke access from the AgentGate dashboard.

## Security Model (Judging Criteria)

| Control | Implementation |
|---------|---------------|
| Per-agent ACLs | Each agent has a policy defining exactly which services and scopes it can access |
| Scope enforcement | Token requests are validated against the policy -- excess scopes are rejected |
| Time-based access | Restrict agents to specific hours/days (UTC) |
| IP allowlist | CIDR-aware IP filtering (supports IPv4, IPv6, and ranges like 10.0.0.0/24) |
| Policy expiration | Auto-disable agents after a configurable date |
| Rate limiting | Per-agent, per-service rate limits (configurable, defaults to 60/min) |
| Step-up auth | Services can be marked to require CIBA approval before token issuance |
| Emergency revoke | Kill switch to instantly disable all agent policies and revoke all keys |
| Agent delegation | Agents delegate narrowed scopes to sub-agents with depth limits and cascade revocation |
| Policy simulation | Dry-run authorization checks return detailed pass/fail report without issuing tokens |
| Webhook notifications | HMAC-signed real-time alerts for 17 security event types with auto-disable on failures |
| Policy templates | 8 pre-built security profiles (read-only, dev, CI/CD, admin, paranoid) with risk assessment |
| Usage quotas | Daily/monthly/total token budgets with configurable actions (deny, warn, step-up) |
| API key hashing | Keys are SHA-256 hashed at rest. Raw keys shown only once at creation |
| Token rotation | Token Vault handles automatic refresh and rotation |
| Audit logging | Every token request, policy change, and connection event is logged |
| Instant revocation | Disable an agent or revoke an API key with a single click |

## User Control (Judging Criteria)

- **Real-time dashboard**: See all connected services, agent policies, API keys, and recent activity in one view
- **Granular policy creation**: Choose exactly which services, scopes, and rate limits each agent gets via a modal form
- **Toggle agents on/off**: Disable an agent without deleting its policy
- **API key lifecycle**: Create, view, and revoke API keys from the dashboard. Keys show prefix, agent binding, creation date, expiration, and last-used timestamp
- **Service management**: Connect and disconnect services with one click
- **Full audit trail**: Every action is timestamped, tagged with agent ID, and filterable

## Technical Execution (Judging Criteria)

- **FastAPI + async throughout**: All database operations and HTTP calls use `async/await`
- **Clean architecture**: Separate modules for auth, policy engine, database, and configuration
- **Policy engine**: Eight-stage enforcement (existence, active status, ownership, expiration, time windows, IP allowlist, service authorization, scope validation, rate limiting)
- **Dual authentication**: Session-based auth for dashboard users, API key auth (Bearer tokens) for agents
- **Agent-key binding**: API keys are bound to specific agents -- an agent cannot use another agent's key to request tokens
- **Test suite**: 2,333 tests across 61 test files covering policy enforcement, CIDR IP validation, time windows, rate limiting, scope intersection, multi-tenant isolation, security injection, API key lifecycle, audit trails, agent delegation chains, policy simulation, webhook notifications, policy templates, usage quotas, and edge cases
- **Type safety**: Pydantic models for all API requests, dataclasses for domain objects
- **Starlette TemplateResponse**: Updated to current API format (no deprecation warnings)

## Design (Judging Criteria)

- **Dark theme**: Clean, professional interface using CSS custom properties
- **Responsive layout**: Grid-based service cards and tables adapt to screen size
- **Visual status indicators**: Color-coded badges for active/disabled/denied/warning states
- **Modal forms**: Policy and API key creation happen in-page without navigation
- **One-click interactions**: Toggle agent status, revoke keys, disconnect services
- **Timestamp formatting**: Unix timestamps are converted to human-readable locale strings client-side

## Potential Impact (Judging Criteria)

Every AI agent deployment faces the same unsolved problem: how do you let agents access user services without giving them uncontrolled access? AgentGate provides a reusable, auditable security layer that:

- Works with any AI agent framework (the agent just calls a REST API)
- Scales from a single personal agent to an organization managing dozens
- Provides the compliance trail enterprises need (who accessed what, when, with which scopes)
- Leverages Auth0's existing infrastructure rather than reinventing token management

As AI agents become standard infrastructure, a gateway like AgentGate becomes a mandatory security control -- not an optional add-on.

## Insight Value (Judging Criteria)

The core insight is that **agent authorization is a fundamentally different problem from user authorization**. Users authenticate once and get broad access. Agents should authenticate per-action with minimal scopes, auditable trails, and human-in-the-loop approval for sensitive operations.

Auth0's Token Vault, CIBA, and Connected Accounts already provide the building blocks for this. AgentGate demonstrates how these primitives compose into a complete agent security platform:

- Token Vault = secure credential lifecycle without agents touching raw tokens
- CIBA = human-in-the-loop consent for agent actions
- Connected Accounts = user-managed service authorization
- Policy engine = the missing layer that maps agent identity to permitted actions
- Delegation chains = safe agent-to-agent permission sharing with scope narrowing at every hop
- Policy simulation = debuggable authorization without side effects
- Webhook notifications = real-time incident response for security teams
- Policy templates = security best practices encoded as reusable profiles
- Usage quotas = budget-based access control for cost and risk management

## Architecture

```
User --> Auth0 Login --> AgentGate Dashboard
                              |
                    +---------+---------+
                    |         |         |
                    v         v         v
              +---------+ +--------+ +--------+
              | GitHub  | | Slack  | | Google |  ... Connected Services
              | Token   | | Token  | | Token  |      via Token Vault
              +----+----+ +---+----+ +---+----+
                   |          |          |
                   +----------+----------+
                              |
                    +---------v---------+
                    |   Token Vault     |
                    |   (Auth0)         |
                    |                   |
                    |  - Token storage  |
                    |  - Auto-refresh   |
                    |  - Scope mgmt     |
                    +---------+---------+
                              |
                    +---------v---------+
                    |   AgentGate       |
                    |   Policy Engine   |
                    |                   |
                    |  - Per-agent ACLs |
                    |  - Rate limits    |
                    |  - Audit logging  |
                    |  - Step-up auth   |
                    +---------+---------+
                              |
              +---------------+---------------+
              |               |               |
              v               v               v
        +----------+   +----------+   +----------+
        | Agent A  |   | Agent B  |   | Agent C  |
        | (Code    |   | (Slack   |   | (Email   |
        |  Review) |   |  Bot)    |   |  Assist) |
        +----------+   +----------+   +----------+
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (async) |
| Auth | Auth0 (OAuth 2.0 + CIBA + Token Vault) |
| Templates | Jinja2 |
| Database | SQLite (aiosqlite) |
| HTTP Client | httpx (async) |
| OAuth Library | Authlib |
| Validation | Pydantic |
| Testing | pytest + pytest-asyncio |

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Configure Auth0 (see AUTH0_SETUP.md)
cp .env.example .env
# Edit .env with your Auth0 credentials

# Start the server
uvicorn src.app:app --reload

# Run tests
pytest tests/ -v
```

## API Endpoints

### User Endpoints
- `GET /` -- Dashboard with services, policies, keys, and audit
- `GET /login` -- Auth0 login flow
- `GET /callback` -- OAuth callback handler
- `POST /connect/{service}` -- Connect a service via Token Vault
- `DELETE /connect/{service}` -- Disconnect a service
- `GET /audit` -- Full audit trail

### Agent Endpoints
- `POST /api/v1/token` -- Request a scoped token (session or API key auth)
- `GET /api/v1/services?agent_id=xxx` -- List available services for an agent
- `POST /api/v1/step-up` -- Trigger CIBA step-up authentication
- `GET /api/v1/step-up/status/{auth_req_id}` -- Poll step-up status

### Management Endpoints
- `POST /api/v1/policies` -- Create an agent policy (with time windows, IP allowlist, expiration)
- `GET /api/v1/policies` -- List all policies
- `POST /api/v1/policies/{agent_id}/toggle` -- Enable/disable an agent
- `DELETE /api/v1/policies/{agent_id}` -- Delete an agent policy
- `POST /api/v1/emergency-revoke` -- Kill switch: disable all agents and revoke all keys
- `POST /api/v1/keys` -- Create an API key
- `GET /api/v1/keys` -- List API keys
- `DELETE /api/v1/keys/{key_id}` -- Revoke an API key

## Example Integration

See `examples/agent_example.py` for a complete working example of how an AI agent authenticates through AgentGate to access third-party services.
