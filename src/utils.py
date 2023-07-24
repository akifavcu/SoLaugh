# Importantions
import os
from src.params import ACTIVE_RUN, PASSIVE_RUN, PREPROC_PATH

def get_bids_file(BIDS_PATH, stage, subj='all', run='all', task = "LaughterActive", measure=None, condition=None) :
    
    """
    conditions = list()
    TODO : check task param
    If task is set manually, and the run doesn't fit,
    task will be automatically change
    If run = 'all', by default task is LaughterActive
    """

    # Find active and passive runs
    if run in ACTIVE_RUN :
        task = "LaughterActive"
    elif run in PASSIVE_RUN :
        task = "LaughterPassive"

    # Take raw data 
    if  stage == "raw":
        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_meg.ds".format(subj, task, run)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)

    # Take preprocessed data 
    elif (stage == "ave"  
    or stage == "epo" 
    or stage == "ica"
    or stage == "cov"  
    or stage == "ica_epo"
    or stage == "proc-clean_epo") : 

        extension = ".fif"

        laughter_bidsname = "sub-{}_ses-recording_task-{}_{}{}".format(subj, task, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), 
        "ses-recording", "meg", laughter_bidsname)
    
    elif stage == "proc-filt_raw" :

        extension = ".fif"

        laughter_bidsname = "sub-{}_ses-recording_task-{}_run-{}_{}{}".format(subj, task, run, stage, extension)
        laughter_bidspath = os.path.join(BIDS_PATH, "sub-{}".format(subj), "ses-recording", "meg", laughter_bidsname)

    # Epochs, ERPs and PSD files
    elif ("psd" in stage 
    or "epo" in stage
    or "erp" in stage) :

        folder = 'sub-' + subj
        if measure == 'log' or measure == 'log_fooof':
            extension = '.pkl'
        else :
            extension = '.fif'

        if condition != None :
            if measure == None :
                laughter_bidsname = "sub-{}_task-{}_run-{}_cond-{}_{}{}".format(subj, task, run, 
                                                                                condition, stage, extension)

                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", folder, laughter_bidsname)
            else :
                laughter_bidsname = "sub-{}_task-{}_run-{}_cond-{}_meas-{}_{}{}".format(subj, task, 
                                                                                        run, condition,
                                                                                        measure, stage, extension)
                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", folder, laughter_bidsname)
        else :
            if measure == None :
                laughter_bidsname = "sub-{}_task-{}_run-{}_{}{}".format(subj, task, run, 
                                                                        stage, extension)

                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", folder, laughter_bidsname)
            else :
                laughter_bidsname = "sub-{}_task-{}_run-{}_meas-{}_{}{}".format(subj, task, 
                                                                                run,
                                                                                measure, stage, extension)
                laughter_bidspath = os.path.join(BIDS_PATH, "meg", "reports", folder, laughter_bidsname)


    return laughter_bidsname, laughter_bidspath


def compute_ch_adjacency(info, ch_type):
    """
    Code from : https://github.com/mne-tools/mne-python/blob/96a4bc2e928043a16ab23682fc818cf0a3e78aef/mne/channels/channels.py#L1524
    
    Compute channel adjacency matrix using Delaunay triangulations.
    Parameters
    ----------
    %(info_not_none)s
    ch_type : str
        The channel type for computing the adjacency matrix. Currently
        supports 'mag', 'grad' and 'eeg'.
    Returns
    -------
    ch_adjacency : scipy.sparse.csr_matrix, shape (n_channels, n_channels)
        The adjacency matrix.
    ch_names : list
        The list of channel names present in adjacency matrix.
    """
    import numpy as np
    from scipy import sparse
    from scipy.spatial import Delaunay
    from mne import spatial_tris_adjacency
    from mne.defaults import HEAD_SIZE_DEFAULT, _handle_default
    from mne.channels.layout import _find_topomap_coords, _pair_grad_sensors
    from mne.io.pick import (channel_type, pick_info, pick_types, _picks_by_type,
                       _check_excludes_includes, _contains_ch_type,
                       channel_indices_by_type, pick_channels, _picks_to_idx,
                       get_channel_type_constants,
                       _pick_data_channels)

    combine_grads = (ch_type == 'grad'
                     and any([coil_type in [ch['coil_type']
                                            for ch in info['chs']]
                              for coil_type in
                              [FIFF.FIFFV_COIL_VV_PLANAR_T1,
                               FIFF.FIFFV_COIL_NM_122]]))

    picks = dict(_picks_by_type(info, exclude=[]))[ch_type]
    ch_names = [info['ch_names'][pick] for pick in picks]
    if combine_grads:
        pairs = _pair_grad_sensors(info, topomap_coords=False, exclude=[])
        if len(pairs) != len(picks):
            raise RuntimeError('Cannot find a pair for some of the '
                               'gradiometers. Cannot compute adjacency '
                               'matrix.')
        # only for one of the pair
        xy = _find_topomap_coords(info, picks[::2], sphere=HEAD_SIZE_DEFAULT)
    else:
        xy = _find_topomap_coords(info, picks, sphere=HEAD_SIZE_DEFAULT)
    tri = Delaunay(xy)
    neighbors = spatial_tris_adjacency(tri.simplices)

    if combine_grads:
        ch_adjacency = np.eye(len(picks), dtype=bool)
        for idx, neigbs in zip(neighbors.row, neighbors.col):
            for ii in range(2):  # make sure each pair is included
                for jj in range(2):
                    ch_adjacency[idx * 2 + ii, neigbs * 2 + jj] = True
                    ch_adjacency[idx * 2 + ii, idx * 2 + jj] = True  # pair
        ch_adjacency = sparse.csr_matrix(ch_adjacency)
    else:
        ch_adjacency = sparse.lil_matrix(neighbors)
        ch_adjacency.setdiag(np.repeat(1, ch_adjacency.shape[0]))
        ch_adjacency = ch_adjacency.tocsr()

    return ch_adjacency, ch_names