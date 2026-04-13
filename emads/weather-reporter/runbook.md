# Weather Reporter Runbook

## Task

Check the current weather conditions for New York, NY and send a brief summary notification.

## Steps

1. Use `web_search` to find the current weather conditions for New York, NY.
2. Extract the key information: temperature, conditions (sunny/cloudy/rain/etc.), and any weather alerts.
3. Format a brief summary (2-3 sentences).
4. Use `send_notification` to deliver the summary with type "weather-report".

## Notes

- Keep the summary concise — this is a daily check-in, not a detailed forecast.
- If the web search fails, send a notification indicating the weather check failed.
- If there are severe weather alerts, mention them prominently.
