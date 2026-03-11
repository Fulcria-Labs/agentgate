# Auth0 Setup Guide for AgentGate

This guide walks through configuring Auth0 to power AgentGate's secure agent gateway. You will set up authentication, Token Vault for connected accounts, and CIBA for step-up consent.

## 1. Create an Auth0 Account

1. Go to [auth0.com/signup](https://auth0.com/signup) and create a free account.
2. Choose a tenant name (e.g., `agentgate-dev`). This becomes your `AUTH0_DOMAIN` (e.g., `agentgate-dev.us.auth0.com`).

## 2. Create an Application

1. In the Auth0 Dashboard, go to **Applications > Applications > Create Application**.
2. Name it `AgentGate` and select **Regular Web Application**.
3. Click **Create**.
4. On the **Settings** tab, note the following values:
   - **Domain** -- your Auth0 domain
   - **Client ID**
   - **Client Secret**

### Configure URLs

Under **Application URIs**, set the following:

| Field | Value |
|-------|-------|
| Allowed Callback URLs | `http://localhost:8000/callback` |
| Allowed Logout URLs | `http://localhost:8000` |
| Allowed Web Origins | `http://localhost:8000` |

For production, replace `localhost:8000` with your deployment URL.

Click **Save Changes**.

## 3. Create an API (Audience)

1. Go to **Applications > APIs > Create API**.
2. Name: `AgentGate API`
3. Identifier: `https://agentgate.example.com/api` (this becomes your `AUTH0_AUDIENCE`)
4. Signing Algorithm: RS256
5. Click **Create**.

## 4. Enable Token Vault (Connected Accounts)

Token Vault allows AgentGate to exchange tokens for connected third-party services on behalf of users.

1. Go to **Authentication > Social** and enable the connections you want:
   - **GitHub**: Create a GitHub OAuth App, enter Client ID and Secret.
   - **Google**: Create a Google OAuth 2.0 Client, enter credentials.
   - **Slack**: Create a Slack App, enter credentials.
2. For each connection, make sure to enable it for the `AgentGate` application.

### Enable Token Vault for each connection

1. Go to **Authentication > Social > [Connection Name] > Token Vault**.
2. Toggle **Enable Token Vault** to ON.
3. Under **Scopes**, add the scopes your agents will need:
   - GitHub: `repo`, `read:user`, `read:org`, `gist`, `notifications`
   - Slack: `channels:read`, `chat:write`, `users:read`, `files:read`
   - Google: `gmail.readonly`, `calendar.readonly`, `drive.readonly`

### Configure Token Exchange

1. Go to **Applications > APIs > Auth0 Management API**.
2. Under **Machine to Machine Applications**, authorize the `AgentGate` application.
3. Grant the following scopes:
   - `read:users`
   - `read:user_idp_tokens`

## 5. Configure CIBA (Client-Initiated Backchannel Authentication)

CIBA enables step-up authentication where AgentGate pushes a consent request to the user for sensitive operations.

1. Go to **Applications > Applications > AgentGate > Settings > Advanced Settings**.
2. Under **Grant Types**, enable:
   - `authorization_code`
   - `client_credentials`
   - `urn:openid:params:grant-type:ciba`
3. Click **Save Changes**.

### CIBA Delivery Mode

1. Go to **Security > CIBA** (or configure via the Management API).
2. Set **Delivery Mode** to `poll` (AgentGate polls for user approval).
3. Configure the **Binding Message Template** if desired -- this is the message shown to users when an agent requests elevated access.

## 6. Enable My Account API

The My Account API lets users manage their connected accounts from the AgentGate dashboard.

1. Go to **Applications > APIs** and find **Auth0 My Account API**.
2. Enable it and authorize the `AgentGate` application.
3. Grant scopes:
   - `read:connected_accounts`
   - `delete:connected_accounts`
   - `read:sessions`

## 7. Environment Variables

Create a `.env` file in the project root (use `.env.example` as a template):

```env
# Auth0 Configuration
AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_CLIENT_ID=your_client_id_here
AUTH0_CLIENT_SECRET=your_client_secret_here
AUTH0_CALLBACK_URL=http://localhost:8000/callback
AUTH0_AUDIENCE=https://agentgate.example.com/api

# Application
APP_SECRET_KEY=generate-a-random-secret-here
DATABASE_URL=sqlite+aiosqlite:///./agentgate.db

# Token Vault
TOKEN_VAULT_ENABLED=true
```

### Generating a secret key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## 8. Verify the Setup

1. Start the application:
   ```bash
   uvicorn src.app:app --reload
   ```

2. Open `http://localhost:8000` and click **Sign in with Auth0**.

3. After logging in, you should see the dashboard with:
   - Your connected services (connect GitHub, Slack, etc.)
   - Agent policy management
   - API key management
   - Audit trail

4. Test the agent API:
   ```bash
   # Create a policy via the dashboard, then create an API key
   # Use the key to request a token:
   curl -X POST http://localhost:8000/api/v1/token \
     -H "Authorization: Bearer ag_your_key_here" \
     -H "Content-Type: application/json" \
     -d '{"agent_id": "my-bot", "service": "github", "scopes": ["repo"]}'
   ```

## Troubleshooting

| Issue | Solution |
|-------|---------|
| `invalid_client` error | Verify `AUTH0_CLIENT_ID` and `AUTH0_CLIENT_SECRET` match your application |
| Callback URL mismatch | Ensure `AUTH0_CALLBACK_URL` is in the Allowed Callback URLs list |
| Token exchange fails | Check that Token Vault is enabled for the social connection |
| CIBA not working | Verify the CIBA grant type is enabled in Advanced Settings |
| `401 Unauthorized` on API | Ensure `AUTH0_AUDIENCE` matches the API identifier you created |
