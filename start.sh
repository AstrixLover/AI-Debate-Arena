#!/usr/bin/env bash
fuser -k 8080/tcp 2>/dev/null
sleep 1
exec /home/runner/workspace/.pythonlibs/bin/streamlit run main.py --server.port 8080 --server.address 0.0.0.0
