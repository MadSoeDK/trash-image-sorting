#!/usr/bin/env bash
# Script to run load tests for the trash sorting API
#
# Usage:
#   ./run_load_tests.sh [options]
#
# Options:
#   -h, --host URL        API endpoint (default: http://localhost:8000)
#   -u, --users N         Number of concurrent users (default: 50)
#   -r, --spawn-rate N    Users to spawn per second (default: 5)
#   -t, --time DURATION   Test duration, e.g., 1m, 30s (default: 2m)
#   -w, --web             Run with web UI (default: headless)
#   --help                Show this help message

set -e

# Default values
HOST="${MYENDPOINT:-http://localhost:8000}"
USERS=50
SPAWN_RATE=5
RUN_TIME="2m"
HEADLESS="--headless"
OUTPUT_DIR="load_test_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--host)
            HOST="$2"
            shift 2
            ;;
        -u|--users)
            USERS="$2"
            shift 2
            ;;
        -r|--spawn-rate)
            SPAWN_RATE="$2"
            shift 2
            ;;
        -t|--time)
            RUN_TIME="$2"
            shift 2
            ;;
        -w|--web)
            HEADLESS=""
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -h, --host URL        API endpoint (default: http://localhost:8000)"
            echo "  -u, --users N         Number of concurrent users (default: 50)"
            echo "  -r, --spawn-rate N    Users to spawn per second (default: 5)"
            echo "  -t, --time DURATION   Test duration, e.g., 1m, 30s (default: 2m)"
            echo "  -w, --web             Run with web UI (default: headless)"
            echo "  --help                Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MYENDPOINT           Default API endpoint if not specified with -h"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get timestamp for filenames
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "Running Load Tests for Trash Sorting API"
echo "============================================================"
echo "Host:        $HOST"
echo "Users:       $USERS"
echo "Spawn rate:  $SPAWN_RATE per second"
echo "Duration:    $RUN_TIME"
echo "Mode:        $([ -z "$HEADLESS" ] && echo "Web UI" || echo "Headless")"
echo "============================================================"
echo ""

# Check if API is accessible
echo "Checking API accessibility..."
if ! curl -sf "$HOST/api/health" > /dev/null; then
    echo "❌ Error: API is not accessible at $HOST"
    echo "Please make sure the API is running:"
    echo "  uvicorn trashsorting.api:app --host 0.0.0.0 --port 8000"
    exit 1
fi
echo "✅ API is accessible"
echo ""

# Run locust
if [ -z "$HEADLESS" ]; then
    # Web UI mode
    echo "Starting Locust web UI..."
    echo "Open http://localhost:8089 in your browser"
    uv run locust -f tests/performancetests/locustfile.py --host "$HOST"
else
    # Headless mode
    OUTPUT_PREFIX="$OUTPUT_DIR/locust_${TIMESTAMP}"

    uv run locust -f tests/performancetests/locustfile.py \
        $HEADLESS \
        --users "$USERS" \
        --spawn-rate "$SPAWN_RATE" \
        --run-time "$RUN_TIME" \
        --host "$HOST" \
        --csv="$OUTPUT_PREFIX" \
        --html="${OUTPUT_PREFIX}_report.html"

    echo ""
    echo "============================================================"
    echo "Load Test Results"
    echo "============================================================"

    # Display summary if stats file exists
    if [ -f "${OUTPUT_PREFIX}_stats.csv" ]; then
        echo ""
        echo "Results saved to:"
        echo "  - ${OUTPUT_PREFIX}_stats.csv"
        echo "  - ${OUTPUT_PREFIX}_stats_history.csv"
        echo "  - ${OUTPUT_PREFIX}_failures.csv"
        echo "  - ${OUTPUT_PREFIX}_report.html"
        echo ""

        # Extract and display key metrics
        echo "Key Metrics:"
        echo "============"
        awk -F',' 'NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i == "Name") name_idx=i
                if ($i == "Aggregate Request Count") req_idx=i
                if ($i == "Aggregate Failure Count") fail_idx=i
                if ($i == "Average Response Time") avg_idx=i
                if ($i == "95%") p95_idx=i
                if ($i == "99%") p99_idx=i
                if ($i == "Requests/s") rps_idx=i
            }
        }
        NR>1 && $name_idx=="Aggregated" {
            printf "  Total Requests:     %s\n", $req_idx
            printf "  Total Failures:     %s\n", $fail_idx
            if ($req_idx > 0) {
                failure_rate = ($fail_idx / $req_idx) * 100
                printf "  Failure Rate:       %.2f%%\n", failure_rate
            }
            printf "  Avg Response Time:  %s ms\n", $avg_idx
            printf "  95th Percentile:    %s ms\n", p95_idx
            printf "  99th Percentile:    %s ms\n", p99_idx
            printf "  Requests/Second:    %s\n", $rps_idx
        }' "${OUTPUT_PREFIX}_stats.csv"

        echo ""
        echo "Open ${OUTPUT_PREFIX}_report.html in a browser for detailed results."
    fi
fi

echo ""
echo "============================================================"
echo "Load test completed!"
echo "============================================================"

