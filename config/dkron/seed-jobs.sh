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

echo ""
echo "Dkron jobs seeded."
