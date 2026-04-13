#!/bin/bash
# Seed Dkron with default jobs.
# Run once after first deployment, or after clearing Dkron data.
#
# Usage: docker exec emad-host-dkron sh /config/dkron/seed-jobs.sh
# Or:    curl -X POST from the host after dkron is running.

DKRON_URL="http://localhost:8080"

# Noon health notification — fires daily at 12:00 PM UTC
curl -s -X POST "${DKRON_URL}/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "health-notification",
    "schedule": "0 0 12 * * *",
    "timezone": "UTC",
    "executor": "http",
    "executor_config": {
      "method": "POST",
      "url": "http://emad-host-langgraph:8000/autoprompt",
      "headers": "{\"Content-Type\": \"application/json\"}",
      "body": "{\"job_name\": \"health-notification\", \"runbook_path\": \"health-notification.md\"}"
    },
    "retries": 2
  }'

# Nadella token keepalive — fires weekly Monday 10:00 AM Eastern
# Keeps the MSAL refresh token alive (expires after 90 days of inactivity)
curl -s -X POST "${DKRON_URL}/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "nadella-token-keepalive",
    "schedule": "0 0 10 * * 1",
    "timezone": "America/New_York",
    "executor": "http",
    "executor_config": {
      "method": "POST",
      "url": "http://emad-host-langgraph:8000/v1/chat/completions",
      "headers": "{\"Content-Type\": \"application/json\"}",
      "body": "{\"model\":\"nadella\",\"conversation_id\":\"new\",\"messages\":[{\"role\":\"user\",\"content\":\"Check for any unread emails from the last 24 hours and give me a brief summary.\"}]}",
      "timeout": "120000"
    },
    "retries": 2
  }'

echo ""
echo "Dkron jobs seeded."
