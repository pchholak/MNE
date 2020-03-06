import numpy as np
import matplotlib.pyplot as plt
import mne
from mne import io
from mne.time_frequency import tfr_multitaper


raw_fname = 'zoya_part1_tsss_raw.fif'
event_fname = 'zoya_part1_6.7-eve.fif'
event_id, tmin, tmax = 1, 0, 9.999
Fs = 1000

raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)
print(events)

# set up pick list: MEG - bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin - 2, tmax, picks=picks,
                    baseline=(tmin, tmax))

# Compute ERDS maps
freqs = np.arange(5, 8, .1)
n_cycles = freqs    # use constant t/f resolution

power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                    return_itc=False)

#ch = range(2, 306, 3)
ch = [68]
power.plot(ch, baseline=(-1., 0.), mode='mean')
