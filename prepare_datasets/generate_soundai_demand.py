"""
Object-Oriented version of EARS-Reverb-DEMAND dataset generator

Usage:
    python ./generate_datasets/generate_soundai_demand.py --data_dir ~/colddisk/datasets/generate_soundai_datasets/
"""

import sys
import sofa
import mat73
import torch
import numpy as np
import pyloudnorm as pyln
import json
import os
from glob import glob
from os import listdir, makedirs
from os.path import join, isdir, exists, basename, dirname
from argparse import ArgumentParser
from soundfile import read, write
from tqdm import tqdm
from scipy.signal import convolve
from scipy import stats
from librosa import resample as resample_librosa
from torchaudio.functional import highpass_biquad
from torchaudio.transforms import Resample
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class DatasetSplit(Enum):
    """Dataset split enumeration"""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


@dataclass
class AudioMetadata:
    """Audio file metadata"""
    id: int
    speaker: str
    speech_file: str
    speech_start: int
    speech_end: int
    rir_file: str
    rir_channel: int
    gain_reverb: float
    rt60: float
    noise_file: str
    noise_channel: int
    noise_start: int
    noise_end: int
    loudness_speech: float
    loudness_reverb: float
    loudness_noise: float
    loudness_mixture: float
    snr_dB: float


class RIRProcessor:
    """RIR loading and processing"""

    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr

    def load(self, rir_file: str) -> Tuple[np.ndarray, int]:
        """Load RIR from various formats and resample to target_sr"""

        if "ARNI" in rir_file:
            rir, sr_rir = read(rir_file, always_2d=True)
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:, channel]
        elif rir_file.endswith(".wav"):
            rir, sr_rir = read(rir_file, always_2d=True)
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:, channel]
        elif rir_file.endswith(".sofa"):
            hrtf = sofa.Database.open(rir_file)
            rir = hrtf.Data.IR.get_values()
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[0, channel, :]
            sr_rir = hrtf.Data.SamplingRate.get_values().item()
        elif rir_file.endswith(".mat"):
            rir_data = mat73.loadmat(rir_file)
            sr_rir = rir_data["fs"].item()
            rir = rir_data["data"]
            channel = np.random.randint(0, rir.shape[1])
            rir = rir[:, channel]
        else:
            raise ValueError(f"Unknown RIR format: {rir_file}")

        # Resample if needed
        if sr_rir != self.target_sr:
            rir = resample_librosa(rir, orig_sr=sr_rir, target_sr=self.target_sr)

        # Process RIR
        rir = self._process_rir(rir)

        return rir, channel

    def _process_rir(self, rir: np.ndarray) -> np.ndarray:
        """Process RIR: cut to direct path and normalize"""
        # Cut to direct path
        max_index = np.argmax(np.abs(rir))
        rir = rir[max_index:]

        # Normalize to [0.1, 0.7] range
        max_val = np.max(np.abs(rir))
        if max_val < 0.1:
            rir = 0.1 * rir / max_val
        elif max_val > 0.7:
            rir = 0.7 * rir / max_val

        return rir

    @staticmethod
    def calc_rt60(rir: np.ndarray, sr: int, method: str = 't30') -> float:
        """Calculate RT60 using Schroeder's method"""
        method = method.lower()
        rt_params = {
            't30': (-5.0, -35.0, 2.0),
            't20': (-5.0, -25.0, 3.0),
            't10': (-5.0, -15.0, 6.0),
            'edt': (0.0, -10.0, 6.0)
        }

        init, end, factor = rt_params.get(method, rt_params['t30'])

        h_abs = np.abs(rir) / np.max(np.abs(rir))

        # Schroeder integration
        sch = np.cumsum(h_abs[::-1]**2)[::-1]
        sch_db = 10.0 * np.log10(sch / np.max(sch) + 1e-20)

        # Linear regression
        sch_init = sch_db[np.abs(sch_db - init).argmin()]
        sch_end = sch_db[np.abs(sch_db - end).argmin()]
        init_sample = np.where(sch_db == sch_init)[0][0]
        end_sample = np.where(sch_db == sch_end)[0][0]

        x = np.arange(init_sample, end_sample + 1) / sr
        y = sch_db[init_sample:end_sample + 1]
        slope, intercept = stats.linregress(x, y)[0:2]

        # Calculate RT60
        db_regress_init = (init - intercept) / slope
        db_regress_end = (end - intercept) / slope
        t60 = factor * (db_regress_end - db_regress_init)

        return t60


class AudioProcessor:
    """Audio signal processing utilities"""

    def __init__(self, target_sr: int = 16000, source_sr: int = 48000):
        self.target_sr = target_sr
        self.source_sr = source_sr
        self.meter = pyln.Meter(target_sr)

        # Resampler (48kHz → 16kHz)
        if source_sr != target_sr:
            self.resampler = Resample(source_sr, target_sr, dtype=torch.float64)
        else:
            self.resampler = None

    def preprocess_speech(self, speech: np.ndarray, cutoff_freq: float = 75.0) -> np.ndarray:
        """Highpass filter and downsample speech"""
        # Highpass filter at source sr
        speech = highpass_biquad(
            torch.from_numpy(speech),
            sample_rate=self.source_sr,
            cutoff_freq=cutoff_freq
        ).numpy()

        # Downsample to target sr
        if self.resampler is not None:
            speech = self.resampler(torch.from_numpy(speech)).numpy()

        return speech

    def apply_reverb(self, speech: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """Apply RIR convolution"""
        reverberant = convolve(speech, rir)[:len(speech)]
        return reverberant

    def normalize_loudness(self, signal: np.ndarray, target_loudness: float) -> Tuple[np.ndarray, float]:
        """Normalize signal to target loudness (LUFS)"""
        # Compute current loudness and guard against invalid values
        current_loudness = self.meter.integrated_loudness(signal)
        if not np.isfinite(current_loudness) or not np.isfinite(target_loudness):
            # If loudness measurement failed (e.g. silent or invalid), return original signal and unity gain
            return signal, 1.0

        delta = target_loudness - current_loudness
        gain = np.power(10.0, delta / 20.0)

        # Guard against NaN or infinite gain
        if not np.isfinite(gain) or gain == 0:
            return signal, 1.0

        normalized = gain * signal

        # Final safety clip to avoid values outside [-1,1]
        max_abs = np.max(np.abs(normalized)) if normalized.size > 0 else 0.0
        if max_abs > 1.0:
            normalized = normalized / max_abs

        return normalized, gain

    def add_noise_with_snr(self, reverberant: np.ndarray, noise: np.ndarray,
                          snr_dB: float) -> Tuple[np.ndarray, float, float]:
        """Add noise at specified SNR, preventing clipping"""
        loudness_reverb = self.meter.integrated_loudness(reverberant)
        loudness_noise = self.meter.integrated_loudness(noise)

        # Calculate initial noise gain
        target_loudness = loudness_reverb - snr_dB
        delta = target_loudness - loudness_noise
        gain_noise = np.power(10.0, delta / 20.0)

        # Mix and prevent clipping
        while True:
            noise_scaled = gain_noise * noise
            mixture = reverberant + noise_scaled

            if np.max(np.abs(mixture)) < 1.0:
                break

            # Increase SNR (reduce noise) to prevent clipping
            snr_dB += 1.0
            target_loudness = loudness_reverb - snr_dB
            delta = target_loudness - loudness_noise
            gain_noise = np.power(10.0, delta / 20.0)

        return mixture, snr_dB, gain_noise

    def apply_ramp(self, signal: np.ndarray, ramp_ms: int = 10) -> np.ndarray:
        """Apply fade-in/fade-out ramp"""
        ramp_samples = int(ramp_ms * self.target_sr / 1000)
        ramp = np.linspace(0, 1, ramp_samples)

        signal[:ramp_samples] *= ramp
        signal[-ramp_samples:] *= ramp[::-1]

        return signal


class DatasetConfig:
    """Dataset generation configuration"""

    def __init__(self, args):
        self.data_dir = args.data_dir
        self.target_sr = 16000  # Fixed to 16kHz for DEMAND compatibility
        self.source_sr = 48000  # EARS original sampling rate

        # Audio parameters
        self.min_snr = args.min_snr
        self.max_snr = args.max_snr
        self.min_length = args.min_length
        self.max_length = args.max_length
        self.cutoff_freq = args.cutoff_freq
        self.min_dB = args.min_dB
        self.max_rt60 = args.max_rt60
        self.ramp_time_ms = args.ramp_time_in_ms

        # Output options
        self.save_reverberant = args.save_reverberant

        # Paths
        self.speech_dir = join(self.data_dir, "dp_speech")
        self.noise_dir = join(self.data_dir, "DEMAND_16kHz")
        self.target_dir = join(self.data_dir, f"noisy_{self.target_sr//1000}k_soundai")

        audio_extensions = ['.flac']
        # 查找所有干净音频文件
        clean_flac_files = []
        for root, dirs, files in os.walk(self.speech_dir):
            for file in files:
                if any(file.endswith(ext) for ext in audio_extensions):
                    clean_flac_files.append(os.path.join(root, file))
                    
 
        # DEMAND noise splits
        self.noise_scenes = {
            "train": ["DKITCHEN", "DLIVING", "NFIELD", "NPARK", "OMEETING", "OOFFICE",
                     "PCAFETER", "PRESTO", "SPSQUARE", "STRAFFIC"],
            "valid": ["DWASHING", "NRIVER"],
            "test": ["OHALLWAY", "PSTATION", "TBUS", "TCAR"]
        }


class DatasetGenerator:
    """Main dataset generator"""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rir_processor = RIRProcessor(config.target_sr)
        self.audio_processor = AudioProcessor(config.target_sr, config.source_sr)

        # Initialize RIR and noise file lists
        self.rir_files: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}
        self.noise_files: Dict[str, List[str]] = {"train": [], "valid": [], "test": []}

        # Sample counter
        self.sample_ids: Dict[str, int] = {"train": 0, "valid": 0, "test": 0}

    def setup(self):
        """Setup dataset directories and file lists"""
        # Validate directories
        assert isdir(self.config.speech_dir), f"Directory not found: {self.config.speech_dir}"
        assert isdir(self.config.noise_dir), f"Directory not found: {self.config.noise_dir}"

        # Per-split directories are created on demand in _prepare_output_dirs

        # Load RIR files
        self._load_rir_files()

        # Load noise files
        self._load_noise_files()

        print(f"RIRs - Train: {len(self.rir_files['train'])}, "
              f"Valid: {len(self.rir_files['valid'])}, Test: {len(self.rir_files['test'])}")
        print(f"Noise - Train: {len(self.noise_files['train'])}, "
              f"Valid: {len(self.noise_files['valid'])}, Test: {len(self.noise_files['test'])}")

    def _load_rir_files(self):
        """Load RIR file paths"""
        data_dir = self.config.data_dir
        RIR_dir = join(data_dir, "RIR")
        # ACE-Challenge (test)
        ace_dir = join(RIR_dir, "ACE-Challenge")
        if isdir(ace_dir):
            names = ["Chromebook", "Crucif", "EM32", "Lin8Ch", "Mobile", "Single"]
            for name in names:
                self.rir_files["test"] += sorted(
                    glob(join(ace_dir, f"ACE_Corpus_RIRN_{name}", "**", "*RIR.wav"), recursive=True)
                )

        # ARNI (train, sample 144)
        arni_dir = join(RIR_dir, "ARNI")
        if isdir(arni_dir):
            all_arni = sorted(glob(join(arni_dir, "**", "*.wav"), recursive=True))
            # Remove corrupted file
            all_arni = [f for f in all_arni if "numClosed_28_numComb_2743_mic_4_sweep_5.wav" not in f]
            if len(all_arni) > 0:
                self.rir_files["train"] += sorted(
                    list(np.random.choice(all_arni, size=min(144, len(all_arni)), replace=False))
                )

        # BRUDEX (train)
        brudex_dir = join(RIR_dir, "BRUDEX")
        if isdir(brudex_dir):
            self.rir_files["train"] += sorted(glob(join(brudex_dir, "rir", "**", "*.mat"), recursive=True))

        # DetmoldSRIR (train)
        detmold_dir = join(RIR_dir, "DetmoldSRIR")
        if isdir(detmold_dir):
            self.rir_files["train"] += sorted(
                glob(join(detmold_dir, "SetA_SingleSources", "Data", "**", "*.wav"), recursive=True)
            )

        # Palimpsest (train)
        pal_dir = join(RIR_dir, "Palimpsest")
        if isdir(pal_dir):
            self.rir_files["train"] += sorted(glob(join(pal_dir, "**", "*.wav"), recursive=True))

        # davidscipka-MIT (train)
        david_dir = join(RIR_dir, "davidscipka-MIT_environmental_impulse_responses")
        if isdir(david_dir):
            david_rirs = sorted(glob(join(david_dir, "*.wav")))
            self.rir_files["train"] += david_rirs
            print(f"Added {len(david_rirs)} RIRs from davidscipka-MIT dataset")

    def _load_noise_files(self):
        """Load DEMAND noise file paths"""
        for split, scenes in self.config.noise_scenes.items():
            for scene in scenes:
                scene_dir = join(self.config.noise_dir, scene)
                if isdir(scene_dir):
                    self.noise_files[split] += sorted(glob(join(scene_dir, "*.wav")))

    def generate_train_valid(self, split: DatasetSplit):
        """Generate train or valid split"""
        subset = split.value
        print(f"\nGenerate {subset} split")

        # Create directories and CSV header
        self._prepare_output_dirs(subset)
        self._write_csv_header(subset)

        # Get speech files
        speech_files = self._get_speech_files(subset)

        # Process each speech file
        for speech_file in tqdm(speech_files, desc=f"Processing {subset}"):
            self._process_speech_file(speech_file, subset, is_test=False)

    def generate_test(self):
        """Generate test split with test_files.json"""
        print("\nGenerate test split")

        # Load test_files.json
        test_json = "test_files.json"
        if not exists(test_json):
            print(f"Warning: {test_json} not found, skipping test split")
            return

        with open(test_json, "r") as f:
            test_data = json.load(f)

        # Create directories and CSV header
        self._prepare_output_dirs("test")
        self._write_csv_header("test")

        # Prepare test files
        test_files = []
        for speaker in test_data.keys():
            for speech_file in test_data[speaker].keys():
                test_files.append({
                    "path": join(self.config.speech_dir, speaker, speech_file + ".wav"),
                    "speaker": speaker,
                    "cutting_times": test_data[speaker][speech_file]
                })

        # Shuffle for randomness
        np.random.seed(42)
        np.random.shuffle(test_files)

        # Process test files
        for test_info in tqdm(test_files, desc="Processing test"):
            self._process_test_file(test_info)

    def _prepare_output_dirs(self, subset: str):
        """Create output directories for a given subset (no per-speaker dirs)."""
        makedirs(join(self.config.target_dir, f"{subset}_clean"), exist_ok=True)
        makedirs(join(self.config.target_dir, f"{subset}_noisy"), exist_ok=True)
        if self.config.save_reverberant:
            makedirs(join(self.config.target_dir, f"{subset}_reverberant"), exist_ok=True)

    def _write_csv_header(self, subset: str):
        """Write CSV header"""
        csv_path = join(self.config.target_dir, f"{subset}.csv")
        with open(csv_path, "w") as f:
            # speaker removed; filenames are stored without per-speaker subfolders
            f.write("id,speech_file,speech_start,speech_end,"
                   "rir_file,rir_channel,gain_reverb,rt60,"
                   "noise_file,noise_channel,noise_start,noise_end,"
                   "speech_dB,reverb_dB,noise_dB,mixture_dB,snr_dB\n")

    def _get_speech_files(self, subset: str) -> List[str]:
        """Get filtered speech file list"""
        # No per-speaker structure: collect all wav/flac files under speech_dir
        exts = (".wav", ".flac")
        all_files = sorted(glob(join(self.config.speech_dir, "**", "*.*"), recursive=True))
        speech_files = [f for f in all_files if f.lower().endswith(exts)]
        return speech_files

    def _process_speech_file(self, speech_file: str, subset: str, is_test: bool = False):
        """Process single speech file"""
        # Read and preprocess speech
        try:
            speech, sr = read(speech_file)
        except Exception as e:
            print(f"\nError reading file: {speech_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            return

        assert sr == self.config.source_sr, f"Expected {self.config.source_sr}Hz, got {sr}Hz"

        speech = self.audio_processor.preprocess_speech(speech, self.config.cutoff_freq)

        # Check length
        min_samples = int(self.config.min_length * self.config.target_sr)
        if len(speech) < min_samples:
            return

        # Apply reverb
        reverberant, rir_info = self._apply_reverb_with_retry(speech, subset)
        if reverberant is None:
            return

        # Add noise
        mixture, snr_dB, noise_info = self._add_noise(reverberant, subset)

        # Limit length (train/valid only)
        if not is_test:
            max_samples = int(self.config.max_length * self.config.target_sr)
            if len(mixture) > max_samples:
                mixture = mixture[:max_samples]
                speech = speech[:max_samples]
                reverberant = reverberant[:max_samples]

        # Measure loudness
        loudness = self._measure_loudness(speech, reverberant, mixture)

        # Save if meets quality threshold
        if "whisper" in speech_file or loudness["speech"] > self.config.min_dB:
            self._save_sample(
                subset, speech_file, speech, reverberant, mixture,
                rir_info, noise_info, snr_dB, loudness,
                speech_start=0, speech_end=len(speech)
            )

    def _process_test_file(self, test_info: Dict):
        """Process test file with cutting times"""
        speech_file = test_info["path"]
        speaker = test_info["speaker"]
        cutting_times = test_info["cutting_times"]

        # Read and preprocess
        try:
            speech, sr = read(speech_file)
        except Exception as e:
            print(f"\nError reading test file: {speech_file}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            return

        assert sr == self.config.source_sr
        speech = self.audio_processor.preprocess_speech(speech, self.config.cutoff_freq)

        # Process each cutting time
        for start, end in cutting_times:
            speech_cut = speech[start:end]

            # Apply reverb
            reverberant, rir_info = self._apply_reverb_with_retry(speech_cut, "test")
            if reverberant is None:
                continue

            # Add noise
            mixture, snr_dB, noise_info = self._add_noise(reverberant, "test")

            # Apply ramp
            mixture = self.audio_processor.apply_ramp(mixture, self.config.ramp_time_ms)
            speech_cut = self.audio_processor.apply_ramp(speech_cut, self.config.ramp_time_ms)
            reverberant = self.audio_processor.apply_ramp(reverberant, self.config.ramp_time_ms)

            # Measure loudness
            loudness = self._measure_loudness(speech_cut, reverberant, mixture)

            # Save
            self._save_sample(
                "test", speech_file, speech_cut, reverberant, mixture,
                rir_info, noise_info, snr_dB, loudness,
                speech_start=start, speech_end=end
            )

    def _apply_reverb_with_retry(self, speech: np.ndarray, subset: str,
                                 max_attempts: int = 100) -> Tuple[Optional[np.ndarray], Dict]:
        """Apply reverb with RT60 constraint"""
        for _ in range(max_attempts):
            # Sample RIR
            if len(self.rir_files[subset]) == 0:
                print(f"Warning: No RIRs for {subset}, using dry signal")
                return speech.copy(), {"file": "no_rir", "channel": 0, "gain": 1.0, "rt60": 0.0}

            rir_file = np.random.choice(self.rir_files[subset])
            rir, channel = self.rir_processor.load(rir_file)
            rt60 = self.rir_processor.calc_rt60(rir, self.config.target_sr)

            if rt60 > self.config.max_rt60:
                continue

            # Apply reverb
            reverberant = self.audio_processor.apply_reverb(speech, rir)

            # Normalize loudness
            loudness_speech = self.audio_processor.meter.integrated_loudness(speech)
            reverberant_norm, gain = self.audio_processor.normalize_loudness(reverberant, loudness_speech)

            if reverberant_norm is None:
                continue

            # Clip protection
            if np.max(np.abs(reverberant_norm)) > 1.0:
                reverberant_norm = reverberant_norm / np.max(np.abs(reverberant_norm))

            return reverberant_norm, {
                "file": rir_file,
                "channel": channel,
                "gain": gain,
                "rt60": rt60
            }

        return None, {}

    def _add_noise(self, reverberant: np.ndarray, subset: str) -> Tuple[np.ndarray, float, Dict]:
        """Add noise at random SNR"""
        # Load noise
        noise_file = None
        while True:
            noise_file = np.random.choice(self.noise_files[subset])
            noise, sr_noise = read(noise_file, always_2d=True)

            assert sr_noise == self.config.target_sr, \
                f"Noise sr {sr_noise} != target sr {self.config.target_sr}"

            if noise.shape[0] >= len(reverberant):
                break

        # Select channel and cut
        channel = np.random.randint(0, noise.shape[1])
        noise = noise[:, channel]
        start = np.random.randint(len(noise) - len(reverberant) + 1)
        noise_cut = noise[start:start + len(reverberant)]

        # Add at random SNR
        snr_dB = np.round(np.random.uniform(self.config.min_snr, self.config.max_snr), 1)
        mixture, actual_snr, gain = self.audio_processor.add_noise_with_snr(
            reverberant, noise_cut, snr_dB
        )

        return mixture, actual_snr, {
            "file": noise_file,
            "channel": channel,
            "start": start,
            "end": start + len(reverberant)
        }

    def _measure_loudness(self, speech: np.ndarray, reverberant: np.ndarray,
                         mixture: np.ndarray) -> Dict[str, float]:
        """Measure loudness of all signals"""
        noise = mixture - reverberant
        meter = self.audio_processor.meter

        return {
            "speech": meter.integrated_loudness(speech),
            "reverberant": meter.integrated_loudness(reverberant),
            "noise": meter.integrated_loudness(noise),
            "mixture": meter.integrated_loudness(mixture)
        }

    def _save_sample(self, subset: str, speech_file: str,
                     speech: np.ndarray, reverberant: np.ndarray, mixture: np.ndarray,
                     rir_info: Dict, noise_info: Dict, snr_dB: float, loudness: Dict,
                     speech_start: int = 0, speech_end: int = -1):
         """Save processed audio sample"""

         sample_id = self.sample_ids[subset]
         self.sample_ids[subset] += 1

         # Save audio files
         rt60 = rir_info["rt60"]
         # Use subset-level folders without per-speaker subdirectories.
         clean_dir = join(self.config.target_dir, f"{subset}_clean")
         noisy_dir = join(self.config.target_dir, f"{subset}_noisy")
         if self.config.save_reverberant:
             reverb_dir = join(self.config.target_dir, f"{subset}_reverberant")

         # Filenames prefixed with clean / noisy. Reverberant files also use 'clean' prefix.
         noisy_fname = f"noisy_{sample_id:05d}_{rt60:.2f}_{snr_dB:.1f}dB.wav"
         clean_fname = f"clean_{sample_id:05d}.wav"
         reverb_fname = f"clean_{sample_id:05d}_{rt60:.2f}.wav"

         write(join(noisy_dir, noisy_fname), mixture, self.config.target_sr, subtype="FLOAT")
         write(join(clean_dir, clean_fname), speech, self.config.target_sr, subtype="FLOAT")

         if self.config.save_reverberant:
             write(join(reverb_dir, reverb_fname), reverberant, self.config.target_sr, subtype="FLOAT")

         # Write CSV metadata
         csv_path = join(self.config.target_dir, f"{subset}.csv")
         with open(csv_path, "a") as f:
             f.write(
                f"{sample_id:05d},{basename(speech_file)[:-4]},{speech_start},"
                f"{speech_end if speech_end != -1 else len(speech)},"
                f"{rir_info['file'].replace(self.config.data_dir, '')},{rir_info['channel']},"
                f"{rir_info['gain']},{rt60:.2f},"
                f"{noise_info['file'].replace(self.config.data_dir, '')},{noise_info['channel']},"
                f"{noise_info['start']},{noise_info['end']},"
                f"{loudness['speech']:.1f},{loudness['reverberant']:.1f},"
                f"{loudness['noise']:.1f},{loudness['mixture']:.1f},{snr_dB:.1f}\n"
             )

    def run(self):
        """Run full dataset generation"""
        # Set seed for reproducibility
        np.random.seed(42)

        # Setup
        self.setup()

        # Generate splits
        self.generate_train_valid(DatasetSplit.TRAIN)
        self.generate_train_valid(DatasetSplit.VALID)
        self.generate_test()

        print(f"\nDataset generation complete!")
        print(f"Output directory: {self.config.target_dir}")


def main():
    parser = ArgumentParser(description="EARS-Reverb-DEMAND Dataset Generator (OOP version)")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Data directory containing EARS, DEMAND_16kHz, and RIR datasets")
    parser.add_argument("--min_snr", type=float, default=-8, help="Minimum SNR (dB)")
    parser.add_argument("--max_snr", type=float, default=10, help="Maximum SNR (dB)")
    parser.add_argument("--min_length", type=float, default=3, help="Minimum audio length (s)")
    parser.add_argument("--max_length", type=float, default=6,
                       help="Maximum audio length for train/valid (s)")
    parser.add_argument("--cutoff_freq", type=float, default=75.0, help="Highpass cutoff (Hz)")
    parser.add_argument("--min_dB", type=float, default=-55.0,
                       help="Minimum loudness threshold (LUFS)")
    parser.add_argument("--max_rt60", type=float, default=3.0, help="Maximum RT60 (s)")
    parser.add_argument("--ramp_time_in_ms", type=int, default=10, help="Ramp time for test (ms)")
    parser.add_argument("--save_reverberant", action="store_true",
                       help="Save reverberant audio (no noise)")

    args = parser.parse_args()

    # Create config and generator
    config = DatasetConfig(args)
    generator = DatasetGenerator(config)

    # Run generation
    generator.run()


if __name__ == "__main__":
    main()
