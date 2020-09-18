"""
A program that renders mono input as 3D, stereo output

:-a --azimuth evaluator 20*seconds
    Horizontal angle in degrees (0 = center). Can be expression like "90 * seconds"

:-e --elevation evaluator 0
    Vertical angle in degrees (0 = center). Can be expression like "30 * sin(0.1 * seconds))"')

:-d --distance evaluator 10
    Distance (10 = 100%% volume). Can be expression like "cos(seconds)"'

:-u --update-interval float 0.3
    Seconds between updating the position

:-c --chunk-size int 2048
    Samples per chunk (affects latency and cpu usage)

:-r --hrir-zip str {data_folder}/IRC_1013.zip
    ZIP file of HRIR data to use
"""
import inspect
import io
import math
import sys
from collections import namedtuple
from os.path import dirname, realpath, join
from time import sleep, monotonic
from traceback import format_exception_only
from typing import Callable
from zipfile import ZipFile

import numpy as np
import pyaudio
from prettyparse import Usage

HrirData = namedtuple('HrirData', 'left_filters right_filters azimuths elevations')


class Processor:
    def __init__(self,
                 hrir: HrirData,
                 calc_azimuth: Callable[[], float],
                 calc_elevation: Callable[[], float],
                 calc_distance: Callable[[], float],
                 update_interval: float = 0.3):
        filters = np.stack([
            hrir.left_filters,
            hrir.right_filters
        ]).transpose((1, 0, 2))  # [direction][side][timestep]
        mag = (filters ** 2).transpose((2, 0, 1))  # Place timesteps first
        chop_index = find_index_to_running_sum(mag, 0.05 * mag.sum())
        self.filters = filters[:, :, chop_index:2 * chop_index]
        self.buffer = np.zeros((self.filters.shape[-1],))
        self.angles = np.dstack([hrir.azimuths, hrir.elevations])
        self.calc_azimuth = calc_azimuth
        self.calc_elevation = calc_elevation
        self.calc_distance = calc_distance
        self.direction = self.calc_direction(self.calc_azimuth(), self.calc_elevation())
        self.distance = self.calc_distance()
        self.update_interval = update_interval
        self.last_update = monotonic()

    def calc_direction(self, azimuth, elevation) -> int:
        """Finds the filter index that has an angle that most closely corresponds to the given angle"""
        angle = np.array([azimuth, elevation])
        diffs = ((self.angles - angle) + 180) % 360 - 180
        dists = (diffs ** 2).sum(axis=-1)
        return dists.argmin()

    def update(self):
        self.direction = self.calc_direction(self.calc_azimuth(), self.calc_elevation())
        self.distance = self.calc_distance()
        self.last_update = monotonic()

    def process(self, audio: np.ndarray) -> np.ndarray:
        """Render positioned mono audio into 3D, stereo audio"""
        if monotonic() - self.last_update >= self.update_interval:
            self.update()
        full_audio = np.concatenate([self.buffer, audio]) * (10.0 / self.distance)
        self.buffer = audio[-len(self.buffer):]
        left = np.convolve(full_audio, self.filters[self.direction][0], mode='valid')
        right = np.convolve(full_audio, self.filters[self.direction][1], mode='valid')
        return left, right


def find_index_to_running_sum(arr: np.ndarray, amount: float) -> int:
    """Finds the index of arr that has a running sum at or above amount"""
    s = 0.0
    for i in range(len(arr)):
        s += arr[i].sum()
        if s >= amount:
            return i
    raise IndexError("No such index exists")


def hrir_data_from_mat(mat: dict):
    assert np.all(mat['l_hrir_S'][0][0]['azim_v'] == mat['r_hrir_S'][0][0]['azim_v'])
    assert np.all(mat['l_hrir_S'][0][0]['elev_v'] == mat['r_hrir_S'][0][0]['elev_v'])
    assert mat['l_hrir_S'][0][0]['sampling_hz'][0][0] == 44100
    assert mat['r_hrir_S'][0][0]['sampling_hz'][0][0] == 44100
    return HrirData(
        left_filters=mat['l_hrir_S'][0][0]['content_m'],
        right_filters=mat['r_hrir_S'][0][0]['content_m'],
        azimuths=mat['l_hrir_S'][0][0]['azim_v'][:, 0],
        elevations=mat['l_hrir_S'][0][0]['elev_v'][:, 0]
    )


def load_mat_from_zip(zip_filename: str) -> dict:
    from scipy.io import loadmat
    with ZipFile(zip_filename, 'r') as archive:
        mat_path = next(iter(
            [i for i in archive.namelist() if i.startswith('RAW/MAT/')] or
            [i for i in archive.namelist() if i.lower().endswith('.mat')]
        ))
        # noinspection PyTypeChecker
        return loadmat(io.BytesIO(archive.read(mat_path)))


def evaluator_type(expression: str) -> Callable[[], float]:
    def evaluator():
        return eval(expression, dict(inspect.getmembers(math)), dict(seconds=monotonic() - start_time))

    try:
        start_time = monotonic()
        evaluator()  # Verify no exceptions
    except Exception as e:
        msg = ''.join(format_exception_only(type(e), e)).strip()
        print(msg, file=sys.stderr)
        raise ValueError(msg)
    return evaluator


def main():
    Usage.types['evaluator'] = evaluator_type
    usage = Usage(__doc__)
    args = usage.parse()

    data_folder = join(dirname(realpath(__file__)), 'data')
    mat = load_mat_from_zip(args.hrir_zip.format(data_folder=data_folder))
    hrir = hrir_data_from_mat(mat)

    processor = Processor(
        hrir,
        calc_azimuth=args.azimuth,
        calc_elevation=args.elevation,
        calc_distance=args.distance,
        update_interval=args.update_interval
    )

    def callback(in_data, frame_count, time_info, status):
        audio = np.frombuffer(in_data, dtype=np.float32).reshape((-1, 2))[:, 0]
        left, right = processor.process(audio)
        out_data = np.vstack((left, right)).T.flatten().astype('float32').tobytes()
        return out_data, pyaudio.paContinue

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=2,
        rate=44100,
        input=True,
        output=True,
        frames_per_buffer=args.chunk_size,
        stream_callback=callback
    )
    stream.start_stream()

    try:
        while stream.is_active():
            sleep(0.2)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == '__main__':
    main()
