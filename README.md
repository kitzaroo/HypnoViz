# HypnoViz (Hypnosis)

A trippy music visualizer for Windows. Drop in a song and a spiral reacts to the bass, kicks and snares. Drop in videos, GIFs or images and they blend into the visuals and change scene on the beat.

![The visual styles: Classic, Neon, Kaleidoscope and Psychedelic](screenshots/styles.jpg)

- 6 visual styles: Classic, Prism, Neon, Kaleidoscope, Psychedelic, and Video (footage only)
- Media layer: stack videos, GIFs and images; scene changes on the beat, kick or snare; optional smooth video (in-between frames) and camera sway
- Scene manager: preview every detected scene in a clip, trim it, hide or delete the ones you don't want
- A glass menu (Esc) with a live preview, a player bar, presets, Domination Mode (popup text) and Spotify Mode (listens to Spotify only, with album art, volume and seeking)

## Requirements

| | |
|---|---|
| OS | Windows 10 or 11 (64-bit) |
| Python | 3.11 (the version this project is developed and tested on). Needs to be on your PATH |
| Graphics | A GPU / driver with OpenGL 3.3 support (any recent NVIDIA, AMD or Intel GPU) |
| Internet | To install the Python packages the first time. Spotify Mode also uses it to look up album art |

The Python packages are listed in `requirements.txt` and are installed for you automatically:
`pygame`, `moderngl`, `numpy`, `miniaudio`, `sounddevice`, `opencv-python`, `Pillow`, `av` and, on Windows only (for Spotify Mode), `proc-tap` (hears Spotify), `pycaw` (Spotify volume) and `winrt-*` (Spotify progress bar and seeking).

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
3. Press **Esc** to open the glass menu. It fades in with a blur, and every page shares the player bar at the bottom.

### The menu

**Queue**: your playlist. Click a track to play it, reorder or remove tracks, shuffle or clear what's coming up, loop the queue.

![Queue](screenshots/queue.jpg)

**Visuals**: one scrolling page of collapsible sections.

- *Visual style*: the style cards. Hover a card and it plays a short looping demo of that style.
- *Live preview*: a mirror of what's on screen. Scroll down and it shrinks and docks in the top right corner, so you can keep watching it while you edit the sliders below.
- *Motion & effects*: spin, bass zoom, chromatic aberration, smoothing, colour drift and Domination text sliders. Under the **Media effects** heading are the media blend mode (spiral window, soft overlay, glow), beat-reactive opacity, kick/snare burst and flash, camera sway, scene change rate, transition style and **Smooth video**.
- *Media & stack*: the current clip with **Add files...**, **Scenes...** and the **Media layer** on/off switch, plus the stack of every clip. Click a clip to jump to it.

![Visuals](screenshots/visuals.jpg)
![Motion and effects](screenshots/motion.jpg)
![Media and stack](screenshots/media.jpg)

**Scenes**: every scene detected in a clip. Click one to loop it in the preview, drag the two handles under the preview to trim it (shorten or lengthen), and hide or delete scenes you don't want. Trims, hidden and deleted scenes are remembered.

![Scenes](screenshots/scenes.jpg)

**Beat**: a live spectrum with the kick and snare bands. Tune the ranges and sensitivities until KICK and SNARE flash only on the real drums.

![Beat](screenshots/beat.jpg)

**Settings**: the **Domination Mode** and **Spotify Mode** switches, display mode (windowed, borderless, exclusive fullscreen), help hints, remembering your media on startup, the Spotify sync delay, a shortcut cheat sheet, presets (save, update, delete) and **Reset all settings**.

![Settings](screenshots/settings.jpg)

### The player bar

At the bottom of every page: the current song, shuffle, previous, play/pause, next, repeat, a seek bar with times, a volume slider, and the glass Spotify button.

- **Shuffle** shuffles the upcoming tracks, **Repeat** loops the queue.
- Click the glass Spotify button to turn Spotify Mode on (it washes green in a colourful wave) or off (it fades back to grey).

### Spotify Mode

Spotify Mode makes the visuals react to the Spotify desktop app instead of your own files. Spotify must be installed, open and playing.

![Spotify Mode](screenshots/spotify.jpg)
*(The screenshot uses placeholder song info and album art.)*

In Spotify Mode the player bar shows Spotify's song, artist and album art, and:

- previous, play/pause and next control Spotify
- the **volume** slider sets Spotify's own volume in the Windows mixer (needs `pycaw`; Spotify must have played something so its audio session exists)
- the **progress bar** shows the track position and you can click or drag it to jump around (needs the `winrt` packages)
- the song name and artist stay on screen while the song is paused
- the album art is looked up online from the song name, so it can occasionally be a different release or missing (a green note is shown then)
- shuffle and repeat are greyed out, because Spotify can't be asked to change those

`setup_env.bat` installs those packages for you. If one is missing, the matching control tells you which to install.

`spotify_icon.png` (next to the program) is the picture used for the glass Spotify button. Without it a plain glass disc is drawn instead.

### Keys

| Key | Action |
|---|---|
| Space | Play / pause |
| Esc | Open / close the menu |
| F or F11 | Fullscreen |
| M or Tab | Next visual style |
| V | Next media scene |
| X | Hide the scene on screen (it never plays again; menu closed) |
| Delete | Delete the scene on screen (menu closed) |
| Z | Undo the last hide / delete (menu closed) |
| Mouse wheel (menu > Visuals) | Scroll the page; Ctrl + wheel over a slider nudges it |
| D | Domination Mode |
| S | Spotify Mode |
| N / P | Next / previous track |
| Left / Right | Seek 5 seconds |
| Up / Down or mouse wheel | Volume (menu closed) |
| H | Show / hide the help text |
| Q | Quit |

Your settings, presets, trims and hidden/deleted scenes are saved in `%APPDATA%\Hypnosis`.

## Building a standalone .exe

Double-click `build_exe.bat`. When it finishes, the app is in `dist\Hypnosis\Hypnosis.exe`.

Keep the whole `dist\Hypnosis` folder together: `Hypnosis.exe` needs the `_internal` folder next to it (and `spotify_icon.png`, which the build copies there). To share it, zip that folder. The people you share it with need **no** Python and no setup, they just unzip it and run `Hypnosis.exe`.

## Updating

Replace the files with the new version (or `git pull`) and run `run.bat` again. If `requirements.txt` changed, the packages update automatically.

## Troubleshooting

- **"Could not create the virtual environment"**: Python isn't installed or isn't on your PATH. See step 1 above.
- **Videos don't play**: make sure the packages finished installing. Delete the `.venv` folder and run `run.bat` again to reinstall.
- **Spotify Mode hears nothing**: it listens to the Spotify desktop app only, so Spotify must be open and playing.
- **Spotify volume or the progress bar does nothing**: the `pycaw` / `winrt` packages are missing. Run `setup_env.bat` (or delete `.venv\requirements.installed` and run `run.bat`). The volume slider also needs Spotify to have started playing at least once.
- **Album art is wrong or missing**: it is matched by song name from an online music search, so rare or renamed tracks may not match.
- **Slow with big videos**: turn off **Smooth video** in Visuals > Motion & effects (under Media effects).
