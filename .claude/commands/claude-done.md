# Claude Done

Play a notification tone when Claude wants your attention.

## Usage

```bash
/claude-done
```

## Command

```bash
bash ~/claude-done.sh
```

## Description

This command runs a shell script that plays an audio notification to get your attention when Claude has completed a task. The script:

1. Attempts to play a custom sound file (`~/claude_done.wav` or `~/claude_done.mp3`)
2. Uses `aplay` for WAV files or `mpg123` for MP3 files
3. Displays a completion message with musical note emoji
4. Waits for user confirmation (Enter key press)

This is useful for long-running tasks where you want to be notified when Claude has finished working.