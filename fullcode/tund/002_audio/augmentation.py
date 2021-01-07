import io
import os
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def read_raw_audio(audio, sample_rate=16000, return_tensor=True):
    if isinstance(audio, str):
        wave, _ = librosa.load(os.path.expanduser(audio), sr=sample_rate, mono=True)
    elif isinstance(audio, bytes):
        wave, sr = sf.read(io.BytesIO(audio))
        if wave.ndim > 1: wave = np.mean(wave, axis=-1)
        wave = np.asfortranarray(wave)
        if sr != sample_rate: wave = librosa.resample(wave, sr, sample_rate)
    elif isinstance(audio, np.ndarray):
        if audio.ndim > 1: ValueError("input audio must be single channel")
        return audio
    else:
        raise ValueError("input audio must be either a path or bytes")
    return tf.convert_to_tensor(wave, tf.float32) if return_tensor else wave


class AudioTFRecordDataset():
    """ Dataset to load a pair of audio and noise signals from tfrecord files"""

    def __init__(self,
                 tfrecords_dir: str,
                 noise_tfrecords_dir: str = None,
                 shuffle: bool = True,
                 sample_rate: int = 16000,
                 num_parallel_calls: int = None,
                 prefetch: int = None):
        self.tfrecords_dir = tfrecords_dir
        self.noise_tfrecords_dir = noise_tfrecords_dir
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls else AUTOTUNE
        self.prefetch = prefetch if prefetch else AUTOTUNE

    def create(self):
        with tf.device("/CPU:0"):
            dataset = self.fusion()
            dataset = dataset.prefetch(AUTOTUNE)
            return dataset.batch(1)

    @tf.function
    def fusion(self):
        def get_dataset_from_tfrecords(tf_records_dir):
            pattern = os.path.join(tf_records_dir, f"*.tfrecord")
            files_ds = tf.data.Dataset.list_files(pattern)
            ignore_order = tf.data.Options()
            ignore_order.experimental_deterministic = False
            files_ds = files_ds.with_options(ignore_order)
            dataset = tf.data.TFRecordDataset(
                files_ds, compression_type='ZLIB', num_parallel_reads=self.num_parallel_calls)
            return dataset

        audio_dataset = get_dataset_from_tfrecords(self.tfrecords_dir)
        noise_dataset = get_dataset_from_tfrecords(self.noise_tfrecords_dir)

        audio_dataset = audio_dataset.map(self.parse_record, num_parallel_calls=self.num_parallel_calls)
        noise_dataset = noise_dataset.map(self.parse_record, num_parallel_calls=self.num_parallel_calls)
        if self.shuffle:
            audio_dataset = audio_dataset.shuffle(1000, reshuffle_each_iteration=True)
            noise_dataset = noise_dataset.shuffle(1000, reshuffle_each_iteration=True).repeat(-1)
        return tf.data.Dataset.zip((audio_dataset, noise_dataset))

    def parse_record(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        audio = tf.numpy_function(
            read_raw_audio,
            inp=[example["audio"], self.sample_rate],
            Tout=(tf.float32)
        )
        return audio


class AugmentTFRecordDataset(AudioTFRecordDataset):
    """ Dataset to load a pair of audio and noise signals from tfrecord files"""
    def __init__(self,
                 augmenter,
                 tfrecords_dir: str,
                 noise_tfrecords_dir: str = None,
                 shuffle: bool = True,
                 sample_rate: int = 16000,
                 num_parallel_calls: int = None,
                 prefetch: int = None):
        self.augmenter = augmenter
        self.tfrecords_dir = tfrecords_dir
        self.noise_tfrecords_dir = noise_tfrecords_dir
        self.shuffle = shuffle
        self.sample_rate = sample_rate
        self.num_parallel_calls = num_parallel_calls if num_parallel_calls else AUTOTUNE
        self.prefetch = prefetch if prefetch else AUTOTUNE

    def create(self):
        with tf.device("/CPU:0"):
            dataset = self.fusion()
            dataset = dataset.map(self.tf_function_mix, num_parallel_calls=self.num_parallel_calls)
            if self.prefetch != 0:
                dataset = dataset.prefetch(self.prefetch)
            return dataset.batch(1)

    def parse_record(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "transcript": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)
        return example["audio"]

    def tf_function_mix(self, audio, noise):
        return tf.numpy_function(
            self.mix,
            inp=[audio, noise],
            Tout=(tf.float32)
        )

    def mix(self, audio, noise):
        audio_signal = read_raw_audio(audio, self.sample_rate, return_tensor=False)
        noise_signal = read_raw_audio(noise, self.sample_rate, return_tensor=False)

        signal = self.augmenter.augment(audio_signal, noise_signal)
        return tf.convert_to_tensor(signal, tf.float32)
