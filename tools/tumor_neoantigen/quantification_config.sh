#!/bin/bash
# Shared featureCounts/StringTie strand mode: 0=unstranded, 1=forward, 2=reverse.
STRAND_FLAG=${STRAND_FLAG:-0}

case "$STRAND_FLAG" in
    0|1|2) ;;
    *) echo "Error: STRAND_FLAG must be 0, 1, or 2." >&2; return 1 2>/dev/null || exit 1;;
esac
