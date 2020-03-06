import matplotlib.pyplot as plt
import mne
from mne.datasets import sample

print(__doc__)

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

events_1 = mne.read_events(fname, include=1)
events_1_2 = mne.read_events(fname, include=[1, 2])
events_not_4_32 = mne.read_events(fname, exclude=[4, 32])

print(events_1[:5], '\n\n---\n\n', events_1_2[:5], '\n\n')

for ind, before, after in events_1[:5]:
    print("At sample %d stim channel went from %d to %d"
            % (ind, before, after))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

mne.viz.plot_events(events_1, axes=axs[0], show=False)
axs[0].set(title="restricted to event 1")

mne.viz.plot_events(events_1_2, axes=axs[1], show=False)
axs[1].set(title="restricted to event 1 or 2")

mne.viz.plot_events(events_not_4_32, axes=axs[2], show=False)
axs[2].set(title="keep all but 4 and 32")
plt.setp([ax.get_xticklabels() for ax in axs], rotation=45)
plt.tight_layout()
plt.show()

mne.write_events('example-eve.fif', events_1)
