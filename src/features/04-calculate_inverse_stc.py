import os
import glob
import pickle
import numpy as np
import logging

import mne

from sklearn.model_selection import KFold

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import study_path, subjects_dir, spacing, meg_dir, mindist
from arg_parser import source_rescontruction_parser

from params import RESULT_PATH, MRI_SCAN, BIDS_PATH
from utils import get_bids_file


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_filenames(subject: str, spacing: str, task: str, meg_picks: str = True) -> dict:
    """
    Set up filenames for various files related to source reconstruction.

    Parameters:
    -----------
    subject : str
        Subject identifier.
    spacing : str
        The spacing to be used for source space.
    meg_picks : str or bool

    Returns:
    --------
    filenames : dict
        Dictionary containing file paths for various related files.
    """
    epo_filename, _ = get_bids_file(RESULT_PATH, task=task, subj=subject[-2:], stage="AR_epo")

    subject_mri = 'sub-all'

    if meg_picks==True:
        meg_picks="mag"
    filenames = {
        'trans': os.path.join(RESULT_PATH, 'mri', f'{subject}/{subject}-trans.fif'),
        'src': os.path.join(subjects_dir, subject_mri, 'bem', f'{subject_mri}-{spacing}-src.fif'),
        'bem': os.path.join(subjects_dir, subject_mri, 'bem', f'{subject_mri}-5120-bem-sol.fif'),
        'epo': os.path.join(RESULT_PATH, 'meg/reports', subject, epo_filename),
        'fwd': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-{meg_picks}-eeg-{spacing}-fwd.fif'),
        'cov': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-cov.fif'),
        'inv': os.path.join(RESULT_PATH, 'mri', subject,'src', f'{subject}-{meg_picks}-{spacing}-inv.fif'),
        'stc': os.path.join(RESULT_PATH, 'mri', subject, 'src'),
        'mrph': os.path.join(RESULT_PATH, 'mri', subject, 'morph'),
        'avg_tcs': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-{meg_picks}-ROI-avg-time-courses.pkl'),
        'raw_tcs': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-{meg_picks}-ROI-raw-time-courses.pkl'),
        'trans_avg_tcs': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-{meg_picks}-ROI-trans-avg-time-courses.pkl'),
        'trans_raw_tcs': os.path.join(RESULT_PATH, 'mri', subject, 'src', f'{subject}-{meg_picks}-ROI-trans-raw-time-courses.pkl')
    }

    for folder in [filenames['stc'], filenames['mrph']]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    return filenames

def main():
    """
    Main function to perform source reconstruction pipeline.
    Reads command line arguments, processes subject data, and performs source reconstruction steps.
    """
    print(subjects_dir)
    parser = source_rescontruction_parser()
    args = parser.parse_args()
    subject = f"sub-{args.subject:02d}"
    task = args.task
    filenames = setup_filenames(subject, spacing, task, args.meg_picks)

    if not os.path.isfile(filenames['trans']) or args.overwrite:
        coreg = compute_coregistration(filenames['trans'], subject, task, args.overwrite)

    epochs = mne.read_epochs(filenames['epo'], preload=True)
    epochs = epochs.pick_types(meg=args.meg_picks, eeg=False, eog=False, ecg=False, stim=False)

    if not os.path.isfile(filenames['cov']) or args.overwrite:
        cov = compute_cov(filenames['cov'], epochs, args.overwrite)
    else:
        logger.info("Loading cov matrix")
        cov = mne.read_cov(filenames['cov'])

    if not os.path.isfile(filenames['fwd']) or args.overwrite:
        fwd = compute_forward_solution(filenames['fwd'], filenames['src'], filenames['epo'],
                                       filenames['trans'], filenames['bem'], mindist, 
                                       args.overwrite, args.meg_picks)
    else:
        logger.info("Loading forward solution")
        fwd = mne.read_forward_solution(filenames['fwd'])

    if not os.path.isfile(filenames['inv']): # or args.overwrite:
        inverse_operator = compute_inverse_problem(filenames['inv'], epochs, fwd, cov, args.overwrite)
    else:
        logger.info("Loading inverse operator")
        inverse_operator = mne.minimum_norm.read_inverse_operator(filenames['inv'])

    if not os.path.isdir(filenames['stc']) or args.overwrite:
        compute_source_estimates(filenames['stc'], epochs, inverse_operator, args.method, args.overwrite)
        del inverse_operator

    del epochs            

    if not os.path.isdir(filenames['mrph']) or args.overwrite:
        logger.info("Loading source estimates")
        stc_files = sorted(glob.glob(os.path.join(filenames['stc'], f'*{args.method}*.h5')))
        stcs = [mne.read_source_estimate(file_path) for file_path in stc_files]
        morph_source_estimates(filenames['mrph'], stcs, subject, args.method, args.overwrite)
        del stcs

    if not os.path.isfile(filenames['avg_tcs']) or args.overwrite:
        logger.info("Loading morphed source estimates")
        morph_files = sorted(glob.glob(os.path.join(filenames['mrph'], f'*{args.method}*.h5')))
        morphed = [mne.read_source_estimate(file_path) for file_path in morph_files]
        morphed = extract_source_estimates_by_ROIs(filenames['avg_tcs'], morphed, "mean")

    if not os.path.isfile(filenames['trans_avg_tcs']) or args.overwrite:
        with open(filenames['avg_tcs'], 'rb') as file:
            time_courses_dict = pickle.load(file)
        transform_data(filenames['trans_avg_tcs'], time_courses_dict, "mean")

    #if not os.path.isfile(filenames['raw_tcs']) or args.overwrite:
    #    logger.info("Loading morphed source estimates")
    #    morph_files = sorted(glob.glob(os.path.join(filenames['mrph'], f'*{args.method}*.h5')))
    #    morphed = [mne.read_source_estimate(file_path) for file_path in morph_files]
    #    morphed = extract_source_estimates_by_ROIs(filenames['raw_tcs'], morphed, 'auto')
    #    del morphed
        
    #if not os.path.isfile(filenames['trans_raw_tcs']) or args.overwrite:
    #    with open(filenames['raw_tcs'], 'rb') as file:
    #        time_courses_dict = pickle.load(file)
    #    transform_data(filenames['trans_raw_tcs'], time_courses_dict, 'auto')


def compute_coregistration(fname_trans: str, subject: str, task: str, overwrite: bool = False) -> mne.coreg.Coregistration:
    """
    Compute the coregistration matrix for aligning MEG and MRI coordinate spaces.

    This function computes the coregistration matrix by fitting an initial
    registration using iterative closest points (ICP) with fiducial points.
    It then omits head shape points that are far from the scalp surface
    and performs another ICP fitting. The resulting transformation matrix
    is saved to a specified file.

    Args:
        fname_trans (str): The path to save the computed transformation matrix.
        subject (str): The subject identifier.
        overwrite (bool, optional): Whether to overwrite existing transformation file.
            Defaults to False.

    Returns:
        mne.coreg.Coregistration: The coregistration object containing the transformation matrix.

    Note:
    - The function assumes that MEG data is available in the specified directory structure.
    - Ensure that the `subjects_dir` variable is set to the path containing the subject's MRI data.
    - The nasion_weight and distance parameters control the coregistration quality.
    - The computed transformation matrix is stored as a .trans file for further use.

    """
    print(" ========> Computing registration matrix")
    
    # Define file paths
    _, fname_raw = get_bids_file(BIDS_PATH, task=task, subj=subject[-2:], stage="raw", run='07')
    raw = mne.io.read_raw_ctf(fname_raw)
    
    # Read MEG info
    info = raw.info
    
    # Create a Coregistration object and perform initial ICP fitting
    coreg = mne.coreg.Coregistration(info, 'sub-all', subjects_dir, fiducials="estimated")
    coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
    
    # Omit head shape points far from the scalp surface and fit ICP again
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    
    # Write the computed transformation matrix to a file
    mne.write_trans(fname_trans, coreg.trans, overwrite=overwrite)
    
    return coreg


def compute_cov(fname_cov: str, epochs: mne.Epochs, overwrite: bool = False,
                tmax: float = 0, kfolds: int = 3) -> mne.Covariance:
    """
    Compute the covariance matrix for MEG data using cross-validation.

    This function computes the covariance matrix using cross-validation (KFold).
    It uses the shrunk covariance estimator for improved stability.

    Args:
        fname_cov (str): The path to save the computed covariance matrix.
        epochs (mne.Epochs): The epochs of MEG data for covariance computation.
        overwrite (bool, optional): Whether to overwrite existing covariance file.
            Defaults to False.
        tmax (float, optional): The end time for epoch selection. Defaults to 0.
        kfolds (int, optional): The number of folds for cross-validation. Defaults to 3.

    Returns:
        mne.Covariance: The computed covariance matrix.

    Note:
    - The function assumes that MEG data epochs are provided.
    - The covariance matrix is computed using the shrunk covariance estimator.
    - The computed covariance matrix is saved to the specified file.
    - Cross-validation is performed to improve stability and generalization of the covariance estimate.

    """
    logger.info("Computing cov matrix")
    
    # Define KFold cross-validation
    cv = KFold(n_splits=kfolds)
    
    # Compute the covariance matrix using shrunk covariance estimator
    cov = mne.compute_covariance(epochs, tmax=tmax, method='shrunk', cv=cv)
    
    # Save the computed covariance matrix to a file
    cov.save(fname_cov, overwrite=overwrite)
    
    return cov


def compute_forward_solution(fname_fwd: str, fname_src: str, fname_epo: str,
                             fname_trans: str, fname_bem: str, mindist: float, 
                             overwrite: bool, meg_picks: str = True) -> mne.Forward:
    """
    Compute the forward solution for MEG source estimation.

    This function computes the forward solution using the specified source space,
    transformation matrix, boundary element model (BEM), and other parameters.

    Args:
        fname_fwd (str): The path to save the computed forward solution.
        fname_src (str): The path to the source space file.
        fname_epo (str): The path to the epochs data file.
        fname_trans (str): The path to the transformation matrix file.
        fname_bem (str): The path to the BEM file.
        mindist (float): The minimum distance between sources and sensors.
        overwrite (bool): Whether to overwrite existing forward solution file.
        meg_picks (str or bool): "mag", "grad", "plana1", "planar2" or True to include all. Defaults to True.

    Returns:
        mne.Forward: The computed forward solution.

    Note:
    - The function uses the specified source space, transformation matrix, and BEM model.
    - The computed forward solution is saved to the specified file.
    - The forward solution contains information about source space and leadfield.
    """
    logger.info("Computing forward solution")

    # Read source space and epochs info
    src = mne.read_source_spaces(fname_src)
    info = mne.io.read_info(fname_epo)
    
    # Compute the forward solution using the specified parameters
    fwd = mne.make_forward_solution(info, fname_trans, fname_src, fname_bem,
                                    meg=meg_picks, eeg=False, mindist=mindist)
    
    # Save the computed forward solution to a file
    mne.write_forward_solution(fname_fwd, fwd, overwrite=overwrite)
    
    # Display information about source space and leadfield
    print(f"Before: {src}")
    print(f'After:  {fwd["src"]}')
    leadfield = fwd["sol"]["data"]
    print(f"Leadfield size : {leadfield.shape[0]} sensors x {leadfield.shape[1]} dipoles")
    
    return fwd


def compute_inverse_problem(fname_inv: str, epochs: mne.Epochs,
                            fwd: mne.Forward, cov: mne.Covariance,
                            overwrite: bool) -> mne.minimum_norm.InverseOperator:
    """
    Compute the inverse operator for source reconstruction.

    This function computes the inverse operator using the specified forward solution,
    covariance matrix, epochs data, and other parameters.

    Args:
        fname_inv (str): The path to save the computed inverse operator.
        epochs (mne.Epochs): The epochs data.
        fwd (mne.Forward): The forward solution.
        cov (mne.Covariance): The covariance matrix.
        overwrite (bool): Whether to overwrite existing inverse operator file.

    Returns:
        mne.minimum_norm.InverseOperator: The computed inverse operator.

    Note:
    - The function uses the specified forward solution, covariance matrix, and epochs data.
    - The computed inverse operator is saved to the specified file.
    - The inverse operator is used for source reconstruction using minimum norm estimation.
    """
    logger.info("Computing inverse operator")

    # Compute the inverse operator using the specified parameters
    inverse_operator = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
    
    # Save the computed inverse operator to a file
    mne.minimum_norm.write_inverse_operator(fname_inv, inverse_operator, overwrite=overwrite)
    
    return inverse_operator


def compute_source_estimates(fname_stc: str, epochs: mne.Epochs,
                              inverse_operator: mne.minimum_norm.InverseOperator,
                              method: str, overwrite: bool = False) -> list:
    """
    Compute source estimates using the specified inverse operator.

    This function computes source estimates using the provided inverse operator,
    epochs data, and other parameters.

    Args:
        fname_stc (str): The directory path to save the computed source estimates.
        epochs (mne.Epochs): The epochs data.
        inverse_operator (mne.minimum_norm.InverseOperator): The inverse operator.
        method (str): The method for computing source estimates (e.g., 'dSPM', 'sLORETA').
        overwrite (bool, optional): Whether to overwrite existing source estimates.

    Returns:
        list of mne.SourceEstimate: The computed source estimates.

    Note:
    - The function applies the inverse operator to epochs data to obtain source estimates.
    - Source estimates are saved in the specified directory with appropriate filenames.
    - The 'method' parameter determines the method used for source estimation.
    """
    logger.info("Computing source estimates")
    
    # Parameters for source estimation
    snr = 3.0
    lambda2 = 1.0 / snr**2
    
    # Apply inverse operator to epochs data
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method=method,
        # pick_ori="normal",
        verbose=False,
    )
    
    # Create the directory if it doesn't exist
    os.makedirs(fname_stc, exist_ok=True)
    print(len(stcs))
    # Save each source estimate with a suitable filename
    for idx, stc in enumerate(stcs):
        filename = os.path.join(fname_stc, f'{idx:03d}_{method}_src')
        stc.save(filename, ftype='h5', overwrite=overwrite)
    

def morph_source_estimates(fname_mrp: str, stcs: list, subject: str, method: str, overwrite: True) -> list:
    """
    Morph source estimates to a common brain space.

    This function morphs source estimates from the individual subject's brain space
    to a common brain space (e.g., fsaverage) using the provided morphing parameters.

    Args:
        fname_mrp (str): The directory path to save the morphed source estimates.
        stcs (list of mne.SourceEstimate): List of source estimates to be morphed.
        subject (str): The subject's ID.
        method (str): The method used for morphing (e.g., 'dSPM', 'sLORETA').
        overwrite (bool): Whether to overwrite existing morphed source estimates.

    Returns:
        list of mne.SourceEstimate: The morphed source estimates.

    Note:
    - The function morphs source estimates from the individual subject's brain space
      to a common brain space (e.g., fsaverage).
    - The 'method' parameter determines the method used for morphing.
    """
    logger.info("Morphing source estimates")
    
    # Path to fsaverage source space
    fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
    src_to = mne.read_source_spaces(fname_fsaverage_src)
    
    # Create the directory if it doesn't exist
    os.makedirs(fname_mrp, exist_ok=True)
        
    # Morph each source estimate and save
    for idx, stc in enumerate(stcs):
        morph = mne.compute_source_morph(
            stc,
            subject_from='sub-all',
            subject_to="fsaverage",
            src_to=src_to,
            subjects_dir=subjects_dir,
        ).apply(stc)
        
        filename = os.path.join(fname_mrp, f'{idx:03d}_{method}_morph')
        print('FILENAME :', filename)
        morph.save(filename, ftype='h5', overwrite=overwrite)


def extract_source_estimates_by_ROIs(fname_avg_time_courses: str, morphed: list, mode: str) -> dict:
    """
    Extract source estimates within predefined ROIs using specified mode.

    This function extracts time courses within predefined regions of interest (ROIs)
    for a list of source estimates.

    Args:
        fname_time_courses (str): The file path to save the averaged time courses in pickled format.
        morphed (list of mne.SourceEstimate): List of morphed source estimates.
        mode (str): The mode for extracting time courses (e.g., 'mean', 'raw').

    Returns:
        dict: A dictionary containing averaged time courses for each ROI and each source estimate.

    Note:
    - The function calculates the average time course within predefined ROIs for each source estimate.
    """
    logger.info("Averaging source estimates by ROIs")
    
    # Read label list from fsaverage parcellation
    label_list = mne.read_labels_from_annot("fsaverage", parc="aparc.a2009s", subjects_dir=subjects_dir)
    
    # Read fsaverage source space
    fname_fsaverage_src = os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-5-src.fif")
    src = mne.read_source_spaces(fname_fsaverage_src)
    time_courses_dict = {}

    for idx, source_estimate in enumerate(morphed):
        epoch_time_courses = {}
        # Loop through each label and extract the time course for the source estimate
        for label in label_list:
            label_name = label.name
            time_course = mne.extract_label_time_course(source_estimate, labels=label, src=src, mode=mode)
            epoch_time_courses[label_name] = time_course
        time_courses_dict[idx] = epoch_time_courses
    
    # Save the time courses in pickled format
    with open(fname_avg_time_courses, 'wb') as file:
        pickle.dump(time_courses_dict, file)
    
    return time_courses_dict


def transform_data(fname_transformed_time_courses: str, original_data: dict, mode: str) -> None:
    """
    Transform and save data based on the specified mode.

    Args:
        fname_transformed_time_courses (str): File path to save the transformed data.
        original_data (dict): Dictionary containing the original data.
        mode (str): Transformation mode ('raw' or 'mean').

    Returns:
        None
    """
    logger.info("Transforming data")
    transformed_data = {}

    if mode == "raw":
        # Transform data to 'raw' mode
        pre_transformed_data = {}
        for _, region_data in original_data.items():
            for region, data in region_data.items():
                if region not in pre_transformed_data:
                    pre_transformed_data[region] = []
                pre_transformed_data[region].append(data[0])
        for key, value in pre_transformed_data.items():
            transformed_data[key] = np.array(value)
                               
    elif mode == "mean":
        # Transform data to 'mean' mode
        regions = original_data[next(iter(original_data.keys()))].keys()

        for region in regions:
            region_matrix = []
            for _, epoch_dict in original_data.items():
                epoch_data = epoch_dict[region]
                region_matrix.append(epoch_data)
            region_matrix = np.vstack(region_matrix)
            transformed_data[region] = region_matrix

    else:
        raise ValueError("Invalid mode. Supported modes: 'raw' or 'mean'.")
    
    logger.info("Data transformation complete")

    # Save the transformed data to a pickle file
    with open(fname_transformed_time_courses, 'wb') as file:
        pickle.dump(transformed_data, file)

    logger.info("Data saved to file")

if __name__ == '__main__':
    main()