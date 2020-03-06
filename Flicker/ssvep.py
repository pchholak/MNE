import matplotlib.pyplot as plt
import mne
from mne import io

raw_fname = 'zoya_part1_tsss_raw.fif'
event_fname = 'zoya_part1_6.7-eve.fif'
event_id, tmin, tmax = 1, 5, 5.199

raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)
print(events)

# set up pick list: MEG - bad channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=False,
                        exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(tmin, tmax))

evoked = epochs.average()

evoked.plot(time_unit='s')
