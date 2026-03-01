#!/bin/bash
# PostToolUse hook: runs biome check --write on edited/written files
# - Safe fixes are auto-applied
# - Remaining issues are fed back to Claude via stderr + exit 2

FILE=$(cat - | jq -r '.tool_input.file_path')

# Skip non-existent files (e.g. deleted files)
if [ ! -f "$FILE" ]; then
  exit 0
fi

# Skip non-JS/TS files
case "$FILE" in
  *.ts|*.tsx|*.js|*.jsx|*.json) ;;
  *) exit 0 ;;
esac

# Find biome binary (check project, then hackathon-web-app)
BIOME="$CLAUDE_PROJECT_DIR/node_modules/.bin/biome"
[ ! -x "$BIOME" ] && BIOME="$HOME/coding/hackathon-web-app/node_modules/.bin/biome"
[ ! -x "$BIOME" ] && exit 0

# Run biome: apply safe fixes, capture remaining issues
OUTPUT=$("$BIOME" check --write "$FILE" 2>&1)
RC=$?

if [ $RC -ne 0 ] && [ -n "$OUTPUT" ]; then
  # Biome found issues it couldn't auto-fix — feed back to Claude
  echo "$OUTPUT" | tail -20 >&2
  exit 2
fi

exit 0
