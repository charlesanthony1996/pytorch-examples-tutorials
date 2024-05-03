import itertools
import math
import wave

frames_per_second = 44100

def sound_wave(frequency, num_seconds):
    for frame in range(round(num_seconds * frames_per_second)):
        time = frame / frames_per_second
        amplitude = math.sin(2 * math.pi * frequency * time)
        yield round((amplitude + 1) / 2 * 255)

left_channel = sound_wave(440, 2.5)
right_channel = sound_wave(480, 2.5)
stereo_frames = itertools.chain(*zip(left_channel, right_channel))


with wave.open("/users/charles/downloads/bongo_sound.wav", mode="wb") as wav_file:
    wav_file.setnchannels(2)
    wav_file.setsampwidth(1)
    wav_file.setframerate(frames_per_second)
    wav_file.writeframes(bytes(stereo_frames))
    print(wav_file.setframerate(frames_per_second))