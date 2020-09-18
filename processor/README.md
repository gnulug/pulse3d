# Pulse 3D Processor

This is the component the renders mono audio to 3D, stereo audio.

## Setup

Currently this program requires `python3` running on Linux. You can set everything up with:

```bash
./setup.sh
```

## Usage

```bash
source .venv/bin/activate  # Brings "pulse3d-processor" into scope
pulse3d-processor -h
```

Examples:

```bash
# Renders 45 degrees to the right, 15 degrees up, and 20 units away (10 = default)
pulse3d-processor --azimuth 45 --elevation 15 --distance 20

# Render position spins in a circle (this is the same as the default mode)
pulse3d-processor --azimuth "10 * seconds"

# Renders with a custom HRTF (see below)
pulse3d-processor --hrir-zip data/some-hrir.zip
```

**Custom HRTF:**

Since we each have different ear and head shapes, it makes sense that we would need to
render audio differently for each person. To customize the HRTF to your head you can do the following:

 - Find [a recording here](http://recherche.ircam.fr/equipes/salles/listen/sounds.html) that
   sounds most realistic to you
 - Download [the corresponding zip file](http://recherche.ircam.fr/equipes/salles/listen/download.html)
   and pass it in with `--hrir-zip path/to/IRC_10XX.zip`
