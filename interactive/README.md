# NPS Queue Simulation - Interactive Mode

This module provides a web interface for running queue prioritization simulation experiments interactively.

## Setup

1. Create a conda environment:
   ```
   conda create -n NPS_interactive python=3.8
   conda activate NPS_interactive
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask application:
   ```
   python app.py
   ```

4. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Features

- Configure and run simulation experiments through a web interface
- View experiment results immediately
- Browse previous experiment runs
- Dark GitHub-like theme for comfortable viewing

## Directory Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: CSS and JavaScript files
- `interactive_results/`: Directory where experiment results are stored

## Notes

- Results are stored in the `interactive_results` directory
- Each experiment gets its own subdirectory named with a timestamp
- The application runs in debug mode by default, which should not be used in production 