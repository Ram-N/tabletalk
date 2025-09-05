# Paste Screenshot

Fetch the latest screenshot from the Windows Screenshots folder.

## Usage

```bash
/paste-ss
```

## Command

```bash
bash ~/paste_screenshot.sh
```

## Description

This command runs a shell script that finds and returns the path to the most recent screenshot from the Windows Screenshots directory (`/mnt/c/Users/ramna/Pictures/Screenshots`). The script:

1. Checks if the Screenshots directory exists
2. Finds the most recent PNG file (excluding desktop.ini)
3. Returns the full path to the latest screenshot

The returned path can be used with Claude Code to analyze or process the screenshot.