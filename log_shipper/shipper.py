import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone

import aiodocker
import asyncpg

# Configure logging for the shipper itself


class _JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        entry = {
            "timestamp": _dt.now(_tz.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_JsonFormatter())
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(_handler)
logger = logging.getLogger("log_shipper")

# Configuration
POSTGRES_DSN = os.environ.get("POSTGRES_DSN", "")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "100"))
FLUSH_INTERVAL_SEC = float(os.environ.get("FLUSH_INTERVAL_SEC", "1.0"))


class LogShipper:
    def __init__(self):
        self.docker = None
        self.pg_pool = None
        self.network_id = None
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.active_tasks = {}  # container_id -> task
        self.running = False

    async def setup(self):
        """Initialize connections and discover network topology."""
        logger.info("Initializing Log Shipper...")

        # 1. Connect to Postgres
        try:
            self.pg_pool = await asyncpg.create_pool(
                POSTGRES_DSN, min_size=1, max_size=5
            )
            logger.info("Connected to PostgreSQL")
        except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as e:
            logger.error("Failed to connect to PostgreSQL: %s", e)
            sys.exit(1)

        # 2. Connect to Docker
        self.docker = aiodocker.Docker()

        # 3. Discover our own container and network
        # The hostname in a Docker container is its container ID by default
        my_container_id = os.environ.get("HOSTNAME")
        if not my_container_id:
            logger.warning(
                "Could not determine own container ID from HOSTNAME. Trying network discovery fallback."
            )
            # Fallback: Find the emad-host-net network by name
            networks = await self.docker.networks.list()
            for net in networks:
                if "emad-host-net" in net["Name"]:
                    self.network_id = net["Id"]
                    logger.info(
                        "Discovered network by name: %s (%s)", net['Name'], self.network_id[:12]
                    )
                    break
        else:
            try:
                my_container = await self.docker.containers.get(my_container_id)
                networks = my_container["NetworkSettings"]["Networks"]
                if not networks:
                    raise ValueError("Container is not attached to any networks")
                # Assume the first network is the primary internal one (emad-host-net)
                network_name = list(networks.keys())[0]
                self.network_id = networks[network_name]["NetworkID"]
                logger.info(
                    "Discovered network via self-inspection: %s (%s)", network_name, self.network_id[:12]
                )
            except (aiodocker.exceptions.DockerError, KeyError, ValueError) as e:
                logger.error("Failed to discover network topology: %s", e)
                sys.exit(1)

        if not self.network_id:
            logger.error("Could not determine the primary network ID. Exiting.")
            sys.exit(1)

    async def _get_last_timestamp(self, container_name: str) -> str:
        """Get the timestamp of the last log line written for this container."""
        # We query Postgres to find the high-water mark so we don't duplicate logs on restart.
        # This requires the system_logs table to have an index on (container_name, timestamp).
        # We pad the time slightly to ensure we don't miss anything that happened during the restart window.
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT log_timestamp FROM system_logs WHERE container_name = $1 ORDER BY log_timestamp DESC LIMIT 1",
                container_name,
            )

            if row and row["log_timestamp"]:
                # Docker API expects unix timestamp (integer) or formatted string
                # We return it as a string for the 'since' parameter
                # The datetime from asyncpg is already a datetime object
                ts = int(row["log_timestamp"].timestamp())
                return str(ts)

        # If no logs exist, return 0 to get everything
        return "0"

    async def tail_container(self, container_id: str):
        """Continuous task to tail a single container's logs and push to queue."""
        try:
            container = await self.docker.containers.get(container_id)
            # Remove the leading slash from the container name
            name = container["Name"].lstrip("/")

            # Don't tail ourselves to avoid an infinite loop of logging our own inserts
            if name == "emad-host-log-shipper":
                return

            logger.info("Starting tail for container: %s (%s)", name, container_id[:12])

            # Determine where to start tailing from
            since_ts = await self._get_last_timestamp(name)

            # Get the log stream
            # follow=True blocks and yields new lines as they arrive
            # timestamps=True prepends the Docker timestamp to each line
            stream = container.log(
                stdout=True, stderr=True, follow=True, timestamps=True, since=since_ts
            )

            async for line in stream:
                if not self.running:
                    break

                # line format with timestamps=True: "2024-03-24T12:00:00.000000000Z The actual log message..."
                try:
                    # Docker's multiplexed stream (stdout/stderr) is handled transparently by aiodocker in string mode,
                    # but we need to split the timestamp from the message.
                    parts = line.split(" ", 1)
                    if len(parts) < 2:
                        continue

                    timestamp_str = parts[0]
                    message = parts[1].strip()

                    if not message:
                        continue

                    # Try to parse the message as JSON if possible (for structured logs)
                    data = None
                    if message.startswith("{") and message.endswith("}"):
                        try:
                            data = message
                            # Also extract a simpler message if it's a JSON log
                            parsed = json.loads(message)
                            if "message" in parsed:
                                message = parsed["message"]
                            elif "msg" in parsed:
                                message = parsed["msg"]
                        except json.JSONDecodeError:
                            data = json.dumps({"raw": message})
                    else:
                        data = json.dumps({"raw": message})

                    # Try to parse the Docker timestamp to standard ISO format
                    try:
                        # Docker timestamps can have up to 9 decimal places (nanoseconds), python datetime only supports 6 (microseconds)
                        # Truncate to microseconds for parsing
                        ts_clean = timestamp_str
                        if "." in ts_clean:
                            base, frac = ts_clean.split(".", 1)
                            frac = frac.replace("Z", "")
                            frac = frac[:6].ljust(6, "0")
                            ts_clean = f"{base}.{frac}Z"

                        # Parse to datetime
                        dt = datetime.strptime(ts_clean, "%Y-%m-%dT%H:%M:%S.%fZ")
                        dt = dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        # Fallback to current time if parsing fails
                        dt = datetime.now(timezone.utc)

                    # Queue the payload
                    payload = {
                        "container_name": name,
                        "timestamp": dt,
                        "message": message,
                        "data": data,
                    }

                    try:
                        self.log_queue.put_nowait(payload)
                    except asyncio.QueueFull:
                        pass  # Drop log entry under backpressure

                except (ValueError, KeyError, UnicodeDecodeError) as e:
                    logger.debug("Error parsing log line from %s: %s", name, e)

        except aiodocker.exceptions.DockerError as e:
            if e.status == 404:
                logger.info(
                    "Container %s no longer exists. Stopping tail.", container_id[:12]
                )
            else:
                logger.error("Docker error tailing %s: %s", container_id[:12], e)
        except asyncio.CancelledError:
            logger.info("Tail task cancelled for %s", container_id[:12])
        except (OSError, RuntimeError) as e:
            logger.error("Unexpected error tailing %s: %s", container_id[:12], e)
        finally:
            if container_id in self.active_tasks:
                del self.active_tasks[container_id]

    async def _write_batch(self, batch):
        """Bulk insert a batch of logs into Postgres."""
        if not batch:
            return

        try:
            async with self.pg_pool.acquire() as conn:
                # Use executemany for bulk insert
                query = """
                    INSERT INTO system_logs (container_name, log_timestamp, message, data)
                    VALUES ($1, $2, $3, $4::jsonb)
                """

                records = [
                    (
                        item["container_name"],
                        item["timestamp"],
                        item["message"],
                        item["data"],
                    )
                    for item in batch
                ]

                await conn.executemany(query, records)
                logger.debug("Inserted batch of %d logs", len(batch))

        except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as e:
            logger.error("Failed to write batch to Postgres: %s", e)
            # If the DB fails, we could potentially requeue, but for logs it's usually
            # better to drop them than to exhaust memory if the DB is permanently down.
            # In a State 4 environment, simplicity > perfect reliability for diagnostic logs.

    async def postgres_writer_loop(self):
        """Continuous background loop to pull from queue and write batches."""
        logger.info("Starting Postgres writer loop")
        batch = []

        while self.running:
            try:
                # Try to get an item from the queue with a timeout
                # This ensures we periodically flush even if the batch isn't full
                try:
                    item = await asyncio.wait_for(
                        self.log_queue.get(), timeout=FLUSH_INTERVAL_SEC
                    )
                    batch.append(item)
                    self.log_queue.task_done()
                except asyncio.TimeoutError:
                    pass  # Timeout is expected, just flush what we have

                # Flush if we hit the batch size OR if we had a timeout and have *some* data
                if len(batch) >= BATCH_SIZE or (batch and self.log_queue.empty()):
                    await self._write_batch(batch)
                    batch = []

            except asyncio.CancelledError:
                break
            except (asyncpg.PostgresError, OSError, asyncio.TimeoutError) as e:
                logger.error("Error in postgres writer loop: %s", e)
                await asyncio.sleep(1)

        # Final flush on shutdown
        if batch:
            await self._write_batch(batch)

    async def scan_existing_containers(self):
        """Find all containers currently on our network and start tailing them."""
        logger.info("Scanning for existing containers on network...")
        try:
            containers = await self.docker.containers.list()
            count = 0

            for c in containers:
                # aiodocker returns DockerContainer objects — inspect to get network info
                c_info = await c.show()
                networks = c_info.get("NetworkSettings", {}).get("Networks", {})

                # Check if this container is on our network
                on_our_network = False
                for net_info in networks.values():
                    if net_info.get("NetworkID") == self.network_id:
                        on_our_network = True
                        break

                if on_our_network:
                    c_id = c_info["Id"]
                    if c_id not in self.active_tasks:
                        task = asyncio.create_task(self.tail_container(c_id))
                        self.active_tasks[c_id] = task
                        count += 1

            logger.info("Started tailing %d existing containers", count)
        except (aiodocker.exceptions.DockerError, OSError) as e:
            logger.error("Failed to scan existing containers: %s", e)

    async def event_watcher_loop(self):
        """Watch the Docker event stream for containers joining/leaving our network."""
        logger.info("Starting Docker event watcher")

        # We want to watch for network connect/disconnect events
        filters = {
            "type": ["network"],
            "event": ["connect", "disconnect"],
            "network": [self.network_id],
        }

        try:
            subscriber = self.docker.events.subscribe(filters=filters)

            while self.running:
                try:
                    event = await subscriber.get()
                    if not event:
                        continue

                    action = event.get("Action")
                    actor = event.get("Actor", {})
                    attributes = actor.get("Attributes", {})

                    container_id = attributes.get("container")
                    if not container_id:
                        continue

                    if action == "connect":
                        logger.info("Container joined network: %s", container_id[:12])
                        if container_id not in self.active_tasks:
                            task = asyncio.create_task(
                                self.tail_container(container_id)
                            )
                            self.active_tasks[container_id] = task

                    elif action == "disconnect":
                        logger.info("Container left network: %s", container_id[:12])
                        if container_id in self.active_tasks:
                            self.active_tasks[container_id].cancel()
                            # Intentional: no await here — the cancelled task will
                            # clean up in its own finally block (tail_container).
                            del self.active_tasks[container_id]

                except asyncio.CancelledError:
                    break
                except (aiodocker.exceptions.DockerError, KeyError, ValueError) as e:
                    logger.error("Error processing Docker event: %s", e)
                    await asyncio.sleep(1)

        except (aiodocker.exceptions.DockerError, OSError) as e:
            logger.error("Failed to subscribe to Docker events: %s", e)

    async def run(self):
        """Main execution flow."""
        self.running = True

        await self.setup()

        # Start the writer loop
        writer_task = asyncio.create_task(self.postgres_writer_loop())

        # Scan for existing containers
        await self.scan_existing_containers()

        # Start watching for new events
        event_task = asyncio.create_task(self.event_watcher_loop())

        # Wait for shutdown signal
        try:
            await asyncio.gather(writer_task, event_task)
        except asyncio.CancelledError:
            logger.info("Received shutdown signal")
        finally:
            self.running = False

            # Brief pause to let the writer flush remaining queued logs
            await asyncio.sleep(0.5)

            # Cancel all tail tasks
            for task in self.active_tasks.values():
                task.cancel()

            # Wait for writer to finish final flush
            if not writer_task.done():
                writer_task.cancel()

            if self.pg_pool:
                await self.pg_pool.close()

            if self.docker:
                await self.docker.close()

            logger.info("Log Shipper shut down cleanly")


def handle_sigterm(shipper, main_task):
    """Handle graceful shutdown."""
    logger.info("SIGTERM received, initiating shutdown...")
    shipper.running = False
    main_task.cancel()


if __name__ == "__main__":
    shipper = LogShipper()

    async def main():
        task = asyncio.ensure_future(shipper.run())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: handle_sigterm(shipper, task))
        await task

    try:
        asyncio.run(main())
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
