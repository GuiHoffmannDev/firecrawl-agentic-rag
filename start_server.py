"""Start the Streamlit app with the current interpreter: python start_server.py"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).with_name("app.py")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), *sys.argv[1:]]))
