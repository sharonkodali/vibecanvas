#!/bin/bash
echo "🎵 VibeCanvas Setup"
echo "==================="

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

echo ""
echo "✅ Dependencies installed."
echo ""
echo "Next: Download the dataset"
echo "  → https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs"
echo "  → Place spotify_songs.csv in data/"
echo ""
echo "Then run the pipeline:"
echo "  python src/pipeline.py"
echo ""
echo "Then open the app:"
echo "  open vibecanvas.html"
