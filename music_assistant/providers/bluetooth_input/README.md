# Bluetooth Input Provider

This provider allows Music Assistant to capture audio from locally connected Bluetooth receivers and stream it through the system using FFmpeg.

## Features

- Capture audio from any ALSA-compatible audio input device
- Support for Bluetooth receivers connected as audio input devices
- Configurable sample rate, channels, and buffer size
- Automatic restart of capture process if it fails
- Presents the input as a radio station in Music Assistant

## Configuration

The provider supports the following configuration options:

- **Audio Input Device**: The ALSA device to capture from (e.g., `default`, `hw:0`, `pulse`)
- **Sample Rate**: Audio sample rate in Hz (8000, 16000, 22050, 44100, 48000, 96000)
- **Channels**: Number of audio channels (1 for mono, 2 for stereo)
- **Buffer Size**: Audio buffer size in milliseconds (50-1000ms)
- **Auto Start**: Automatically start capturing when the provider loads

## Requirements

- FFmpeg with ALSA support
- A Bluetooth receiver or other audio input device connected to the system
- ALSA audio system (Linux)

## Usage

1. Connect your Bluetooth receiver to the system
2. Configure the provider with the appropriate audio device
3. The Bluetooth input will appear as a radio station in your Music Assistant library
4. Play the "Bluetooth Audio Input" radio station to hear the live audio feed

## Technical Details

The provider uses FFmpeg to capture audio from ALSA devices and streams it as PCM audio through Music Assistant's streaming system. The audio is captured in real-time and can be played on any Music Assistant-compatible player.

The provider implements the Music Provider interface and presents the audio input as a radio station, which allows it to be treated like any other audio source in the system.
