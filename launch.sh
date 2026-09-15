#!/usr/bin/env bash
VENV="/home/king/Desktop/Thesis/myenv/bin/activate"
#CMD1="clear; source \"$VENV\"; python3.11 main.py; exec zsh"
CMD1="clear; source \"$VENV\"; python3.11 worker.py 192.168.1.211 alice; exec zsh"
CMD2="clear; source \"$VENV\"; python3.11 main.py; exec zsh"


if command -v konsole >/dev/null 2>&1; then
  konsole -e zsh -ic "$CMD1" &
  konsole -e zsh -ic "$CMD2" &
elif command -v xterm >/dev/null 2>&1; then
  xterm -hold -e "zsh -ic '$CMD1'" &
  xterm -hold -e "zsh -ic '$CMD2'" &
else
  echo "No GUI terminal found — using tmux fallback (requires tmux)."
  # create detached session with two panes and attach
  tmux new-session -d -s two_term "$CMD1"
  tmux split-window -h -t two_term "$CMD2"
  tmux select-layout -t two_term tiled
  tmux attach -t two_term
fi
