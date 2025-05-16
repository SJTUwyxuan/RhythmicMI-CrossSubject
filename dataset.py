import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
import torch
from torch.utils.data import Dataset

# -----------------------------------------------
# Dataset:
# 1) Constructing Dataset
# 2) Conduct channel selection and preprocessing
# -----------------------------------------------

class EEGDataset(Dataset):
    def __init__(self,
                 data_root,
                 subject_ids,
                 session=1,
                 channel_indices=None,
                 highpass=None,
                 lowpass=None,
                 filter_order=2):
        
        """
        data_root
        subject_ids: list of int
        session: 1 or 2 or 3
        channel_indices: channel idx to be kept, None means all channels are used
        highpass: in Hz, None means no highpass
        lowpass: in Hz, None means no lowpass
        filter_order

        """
        all_data = []
        all_labels = []

        for subj in subject_ids:
            base = os.path.join(data_root, f"Session{session}")
            eeg_filepath   = os.path.join(base, f"Subj_{subj}_preprocessed_MIperiod.mat")
            label_filepath = os.path.join(base, f"Subj_{subj}_label.mat")

            eeg = loadmat(eeg_filepath)['eegdata'].astype(np.float32)    # (chan,sample,trial)
            labels = loadmat(label_filepath)['true_label'].squeeze().astype(np.int64)  # (120,)

            # raw label might be count from 1
            if labels.min() == 1:
                labels -= 1

            # Transform to (trial, chan, sample) for Neural Network
            eeg = np.transpose(eeg, (2,0,1)) 

            # bandpass filter
            fs = 128
            nyq = fs/2.0

            if highpass or lowpass:
                if highpass and lowpass:
                    wn = [highpass/nyq, lowpass/nyq]
                    btype = 'bandpass'
                elif highpass:
                    wn = highpass/nyq
                    btype = 'highpass'
                else:
                    wn = lowpass/nyq
                    btype = 'lowpass'
                b, a = butter(filter_order, wn, btype=btype)
                # filtfilt alongside the sample dimension
                eeg = filtfilt(b, a, eeg, axis=-1)

            # channel selection
            if channel_indices is not None:
                ch_idx = np.array(channel_indices) - 1
                eeg = eeg[:, ch_idx, :]

            all_data.append(eeg)
            all_labels.append(labels)
        
        self.data = np.concatenate(all_data, axis=0)    # (all_trials, chan, sample)
        self.labels = np.concatenate(all_labels, axis=0)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]    # numpy (chan, sample)
        y = self.labels[idx]
        return torch.from_numpy(x).float(), int(y)
    

if __name__ == "__main__":

    data_root = "D:/14-AORIonline/DataSet"
    subject_ids = list(range(1,11))
    session = 1
    channel_indices = [9,10,11,12,13,14,17,18,19,20,21,22,
                         44,45,46,49,50,51,54,55,56,57,58,59] + [25,26,30,63,62,27,29,64]
    highpass = 0.5
    lowpass = 5.0
    filter_order = 4
    Testdataset = EEGDataset(data_root, subject_ids, session, channel_indices, highpass, lowpass,  filter_order)