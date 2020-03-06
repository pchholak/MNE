import mne
from mne.datasets import sample

fname = 'zoya_part1_tsss.fif'

raw = mne.io.read_raw_fif(fname)

# Set up pick list: MEG + STI 014 - bad channels
want_meg = True
want_eeg = False
want_stim = False

picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
                        exclude='bads')

some_picks = picks[:5]  # take 5 first
start, stop = raw.time_as_index([0, 15])    # read the first 15s of data
data, time = raw[some_picks, start:(stop + 1)]
