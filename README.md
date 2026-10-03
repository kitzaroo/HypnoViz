# HypnoViz (Hypnosis)

A trippy music visualizer for Windows. Drop in a song and a spiral reacts to the bass, kicks and snares. Drop in videos, GIFs or images and they blend into the visuals and change scene on the beat.

- 6 visual styles: Classic, Prism, Neon, Kaleidoscope, Psychedelic, and Video (footage only)
- Media layer: stack videos, GIFs and images; scene changes on the beat, kick or snare; optional smooth video (in-between frames) and camera sway
- Scene manager: see every detected scene in a clip, hide or delete the ones you don't want
- Presets, Domination Mode (popup text) and Spotify Mode (listens to Spotify only)

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| Python | 3.11 (the version this project is developed and tested on). Needs to be on your PATH |
| Graphics | A GPU / driver with OpenGL 3.3 support (any recent NVIDIA, AMD or Intel GPU) |
| Internet | Only for the first install, to download the Python packages |

The Python packages are listed in `requirements.txt` and are installed for you automatically:
`pygame`, `moderngl`, `numpy`, `miniaudio`, `sounddevice`, `opencv-python`, `Pillow`, `av` and, on Windows only for Spotify Mode, `proc-tap`, `pycaw` (Spotify volume) and `winrt-*` (Spotify progress bar and seeking).

You do **not** need conda, ffmpeg or anything else. Everything is installed into a private virtual environment (`.venv`) inside the project folder, so nothing touches your system Python.

## First-time install

1. **Install Python 3.11** from [python.org](https://www.python.org/downloads/windows/). On the first installer screen, tick **"Add python.exe to PATH"**.
2. **Get the project**: click the green **Code** button on GitHub, then **Download ZIP**, and unzip it somewhere (for example `C:\HypnoViz`). Or clone it with git:
   ```
   git clone https://github.com/kitzaroo/HypnoViz.git
   ```
3. **Double-click `run.bat`.**

That's it. The first launch creates the `.venv` folder and downloads the packages, which takes a few minutes. A console window shows the progress. After that, `run.bat` starts the app right away.

If something goes wrong, the console window stays open and shows the error. The most common cause is Python not being on your PATH. Reinstall Python with the PATH box ticked, then run `run.bat` again.

### What the batch files do

| File | What it does |
|---|---|
| `setup_env.bat` | Creates the `.venv` virtual environment and installs `requirements.txt`. Only reinstalls when `requirements.txt` changes. You normally never run this yourself, because the two files below call it first. |
| `run.bat` | Runs `setup_env.bat`, then starts the app. This is the one you use day to day. |
| `build_exe.bat` | Runs `setup_env.bat`, then builds a standalone `.exe` with PyInstaller into `dist\Hypnosis\`. |

## Using it

1. Run `run.bat` and drag a music file (mp3, wav, flac, ogg, m4a, aac, opus, ...) onto the window.
2. Drag videos, GIFs or images onto the window as well to add them to the media stack.
3. Press **Esc** to open the menu (a player bar at the bottom has the song, transport, seek bar and volume; in Spotify Mode it shows the album art and controls Spotify, and the glass Spotify button switches the mode on and off): Queue, Visuals, Scenes, Beat and Settings (Visuals has the style cards, a live preview, motion and media-effect sliders, and the media clip and stack, each in a collapsible section).

| Key | Action |
|---|---|
| Space | Play / pause |
| Esc | Open / close the menu |
| F or F11 | Fullscreen |
| M or Tab | Next visual style |
| V | Next media scene |
| X | Hide the scene on screen (it never plays again) |
| Delete | Delete the scene on screen |
| Z | Undo the last hide / delete |
| Mouse wheel (Esc > Visuals) | Scroll the page; Ctrl + wheel over a slider nudges it |
| D | Domination Mode |
| S | Spotify Mode |
| N / P | Next / previous track |
| Left / Right | Seek 5 seconds |
| Up / Down | Volume |
| H | Show / hide the help text |
| Q | Quit |

Your settings, presets and hidden/deleted scenes are saved in `%APPDATA%\Hypnosis`.

## Building a standalone .exe

Double-click `build_exe.bat`. When it finishes, the app is in `dist\Hypnosis\Hypnosis.exe`.

Keep the whole `dist\Hypnosis` folder together: `Hypnosis.exe` needs the `_internal` folder next to it. To share it, zip that folder. The people you share it with need **no** Python and no setup, they just unzip it and run `Hypnosis.exe`.

## Updating

Replace the files with the new version (or `git pull`) and run `run.bat` again. If `requirements.txt` changed, the packages update automatically.

## Troubleshooting

- **"Could not create the virtual environment"**: Python isn't installed or isn't on your PATH. See step 1 above.
- **Videos don't play**: make sure the packages finished installing. Delete the `.venv` folder and run `run.bat` again to reinstall.
- **Spotify Mode hears nothing**: it listens to the Spotify desktop app only, so Spotify must be open and playing.
- **Slow with big videos**: turn off **Smooth video** in Visuals > Motion & effects (Media effects).
