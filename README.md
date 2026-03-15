# AgentGate - Secure AI Agent Gateway with Auth0 Token Vault

A security-first gateway that manages how AI agents access third-party services on behalf of users, powered by Auth0 Token Vault.

**Every token request is scoped, logged, and revocable.**

## The Problem

AI agents need to access services (GitHub, Slack, Google, etc.) on behalf of users. Current approaches are dangerous:
- Agents store raw API keys with full access
- No visibility into what agents do with credentials
- No way to scope or revoke access per-agent
- No audit trail of token usage

## The Solution

AgentGate uses Auth0 Token Vault to provide a secure, observable layer between AI agents and user services:

| Feature | Description |
|---------|-------------|
| **Scoped Access** | Define exactly which APIs and scopes each agent can use |
| **Audit Trail** | Every token request logged with agent ID, action, timestamp |
| **Step-Up Auth** | Sensitive operations require additional user confirmation |
| **Auto-Revocation** | Tokens expire, rotate, and revoke automatically |
| **Real-Time Dashboard** | See what agents are doing with your credentials right now |
| **Anomaly Detection** | AI-powered behavioral analysis flags suspicious agent patterns |
| **Compliance Reports** | SOC2/GDPR-ready audit exports with risk scoring |

## Architecture

```
User ──> Auth0 Login ──> AgentGate Dashboard
                              │
                    ┌─────────┼─────────┐
                    │         │         │
                    v         v         v
              ┌─────────┐ ┌────────┐ ┌────────┐
              │ GitHub  │ │ Slack  │ │ Google │  ... Connected Services
              │ Token   │ │ Token  │ │ Token  │      via Token Vault
              └────┬────┘ └───┬────┘ └───┬────┘
                   │          │          │
                   └──────────┼──────────┘
                              │
                    ┌─────────v─────────┐
                    │   Token Vault     │
                    │   (Auth0)         │
                    │                   │
                    │  • Token storage  │
                    │  • Auto-refresh   │
                    │  • Scope mgmt     │
                    └─────────┬─────────┘
                              │
                    ┌─────────v─────────┐
                    │   AgentGate       │
                    │   Policy Engine   │
                    │                   │
                    │  • Per-agent ACLs │
                    │  • Rate limits    │
                    │  • Audit logging  │
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              v               v               v
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Agent A  │   │ Agent B  │   │ Agent C  │
        │ (Code    │   │ (Slack   │   │ (Email   │
        │  Review) │   │  Bot)    │   │  Assist) │
        └──────────┘   └──────────┘   └──────────┘
```

## How Auth0 Token Vault Powers AgentGate

1. **Connected Accounts** - Users authorize services once via OAuth; Token Vault manages the full token lifecycle
2. **Token Exchange (RFC 8693)** - Agents request scoped tokens through AgentGate, which exchanges them via Token Vault
3. **CIBA Flow** - Step-up authentication pushes consent requests to users for sensitive operations
4. **My Account API** - Users manage their connected accounts and see active sessions

## Quick Start

```bash
# Clone and install
git clone <repo-url>
cd auth0-agentgate
pip install -r requirements.txt

# Configure Auth0
cp .env.example .env
# Edit .env with your Auth0 credentials

# Run
uvicorn src.app:app --reload
```

## Configuration

Required Auth0 setup:
1. Create an Auth0 application (Regular Web Application)
2. Enable Token Vault and CIBA grant types
3. Activate My Account API with Connected Accounts scopes
4. Configure social connections (GitHub, Slack, Google)

See [AUTH0_SETUP.md](AUTH0_SETUP.md) for detailed instructions.

## API Endpoints

### User Endpoints
- `GET /` - Dashboard with connected services and agent activity
- `GET /login` - Auth0 login flow
- `GET /callback` - OAuth callback handler
- `POST /connect/{service}` - Connect a new service via Token Vault
- `DELETE /connect/{service}` - Disconnect a service
- `GET /audit` - View audit trail

### Agent Endpoints
- `POST /api/v1/token` - Request a scoped token for a service
- `GET /api/v1/services` - List available services for this agent
- `POST /api/v1/step-up` - Trigger step-up auth for sensitive ops

### Analytics & Compliance
- `GET /api/v1/analytics?hours=24` - Token usage analytics with anomaly detection
- `GET /api/v1/compliance?days=30` - SOC2/GDPR compliance audit report

## Security Features

### Anomaly Detection
AgentGate continuously monitors agent behavior and flags suspicious patterns:
- **Burst Detection** - Alerts when an agent makes unusually many requests in a short window
- **Off-Hours Access** - Flags access attempts outside configured business hours
- **High Denial Rate** - Identifies agents whose requests are frequently denied (possible compromise)
- **New IP Detection** - Alerts when agents start connecting from new IP addresses
- **Scope Escalation** - Detects attempts to access scopes beyond the agent's policy

### Risk Scoring
Each agent receives a 0.0-1.0 risk score based on:
- Denial rate, anomaly severity, IP diversity, and request volume
- High-risk agents are highlighted in the analytics dashboard

### Compliance Reporting
Generate audit-ready reports with:
- Access event summaries, emergency revocations, policy changes
- Step-up authentication challenge logs
- Anomaly analysis across all agents

## Testing

```bash
pytest tests/ -v
```

## Prize Categories

- **Security Model** - Per-agent ACLs, step-up auth, automatic token rotation
- **User Control** - Real-time dashboard, granular permissions, instant revocation
- **Technical Execution** - Full Token Vault + CIBA + My Account API integration
- **Potential Impact** - Every AI agent deployment needs this security layer

## License

MIT
