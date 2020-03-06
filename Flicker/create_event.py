import numpy as np
import mne

ev_list = [193.235, 483.938, 514.032, 684.375, 699.422, 714.469]
nev = len(ev_list)
ev = np.array(ev_list)
#print(ev)

t_start = 27.
t_stop = 944.999
sfreq = 1000
#times = np.arange(t_start, t_stop, 1 / sfreq)

#print(len(times))
#print(f"Starts from {times[0]}")
#print(f"Ends at {times[-1]}")
#print(f"{times[166235]}")

#ev_i = (ev - t_start) * sfreq

ev_i = ev * sfreq
ev_sample = ev_i.astype(int)
print(ev_sample)

prev = np.zeros((nev,), dtype=int)
ev_id = np.ones((nev,), dtype=int)
print(prev)
print(ev_id)

events = np.array([ev_sample, prev, ev_id])
events = events.transpose()
print(events)

#mne.write_events('zoya_part1_6.7-eve.fif', events)
mne.write_events('zoya_part1_6.7-eve.fif', events)
