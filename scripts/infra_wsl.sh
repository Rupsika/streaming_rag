#!/usr/bin/env bash
# Docker-free alternative to docker-compose.yml: Apache Kafka (KRaft mode) + Prometheus
# as plain binaries inside WSL2 / Linux. Windows reaches them through WSL's localhost
# forwarding (Kafka at localhost:19092, Prometheus at localhost:9090).
#
#   bash scripts/infra_wsl.sh up      # download once, then start both
#   bash scripts/infra_wsl.sh down
#   bash scripts/infra_wsl.sh status
#
# On Windows call it through WSL:  wsl -d Ubuntu -- bash scripts/infra_wsl.sh up
set -euo pipefail

KAFKA_VER=3.9.1
PROM_VER=2.54.1
HOME_DIR="${XS_INFRA_DIR:-$HOME/xstream-infra}"
KAFKA_DIR="$HOME_DIR/kafka_2.13-$KAFKA_VER"
PROM_DIR="$HOME_DIR/prometheus-$PROM_VER.linux-amd64"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HOME_DIR"

fetch() {
  [ -d "$KAFKA_DIR" ] || curl -sL "https://archive.apache.org/dist/kafka/$KAFKA_VER/kafka_2.13-$KAFKA_VER.tgz" | tar -xz -C "$HOME_DIR"
  [ -d "$PROM_DIR" ] || curl -sL "https://github.com/prometheus/prometheus/releases/download/v$PROM_VER/prometheus-$PROM_VER.linux-amd64.tar.gz" | tar -xz -C "$HOME_DIR"
}

kafka_conf() {
  cat > "$HOME_DIR/kraft.properties" <<EOF
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:19093
listeners=PLAINTEXT://0.0.0.0:19092,CONTROLLER://localhost:19093
advertised.listeners=PLAINTEXT://localhost:19092
controller.listener.names=CONTROLLER
listener.security.protocol.map=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
log.dirs=$HOME_DIR/kafka-logs
num.partitions=1
offsets.topic.replication.factor=1
transaction.state.log.replication.factor=1
transaction.state.log.min.isr=1
message.max.bytes=4194304
replica.fetch.max.bytes=4194304
log.retention.hours=24
group.initial.rebalance.delay.ms=0
EOF
  if [ ! -f "$HOME_DIR/kafka-logs/meta.properties" ]; then
    "$KAFKA_DIR/bin/kafka-storage.sh" format -t "$("$KAFKA_DIR/bin/kafka-storage.sh" random-uuid)" \
      -c "$HOME_DIR/kraft.properties" >/dev/null
  fi
}

prom_conf() {
  # services run on Windows; from WSL2 (NAT mode) the Windows host is the default gateway
  local host
  host="$(ip route show default | awk '{print $3}')"
  sed -e "s/host.docker.internal/$host/g" -e "s/redpanda:9644/localhost:9644/" \
    "$REPO/prometheus/prometheus.yml" > "$HOME_DIR/prometheus.yml"
}

up() {
  fetch; kafka_conf; prom_conf
  if ! pgrep -f kafka.Kafka >/dev/null; then
    nohup "$KAFKA_DIR/bin/kafka-server-start.sh" "$HOME_DIR/kraft.properties" > "$HOME_DIR/kafka.log" 2>&1 &
  fi
  if ! pgrep -f "prometheus --config" >/dev/null; then
    nohup "$PROM_DIR/prometheus" --config.file="$HOME_DIR/prometheus.yml" \
      --storage.tsdb.path="$HOME_DIR/prom-data" --storage.tsdb.retention.time=2d \
      > "$HOME_DIR/prometheus.log" 2>&1 &
  fi
  for _ in $(seq 60); do
    if "$KAFKA_DIR/bin/kafka-broker-api-versions.sh" --bootstrap-server localhost:19092 >/dev/null 2>&1; then
      echo "kafka up on localhost:19092; prometheus on localhost:9090"; return 0
    fi
    sleep 1
  done
  echo "kafka did not come up; see $HOME_DIR/kafka.log" >&2; return 1
}

down() {
  pkill -f "prometheus --config" || true
  "$KAFKA_DIR/bin/kafka-server-stop.sh" >/dev/null 2>&1 || true
  echo "stopped"
}

status() {
  pgrep -af kafka.Kafka | cut -c1-80 || echo "kafka: not running"
  pgrep -af "prometheus --config" | cut -c1-80 || echo "prometheus: not running"
}

"${1:-up}"
