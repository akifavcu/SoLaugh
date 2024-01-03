import argparse

def get_task_condition_parser():
    """
    Basic argparse parser 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-task",
        "--task",
        default="LaughterActive",
        type=str,
        help="Task to process",
    )

    parser.add_argument(
        "-cond1",
        "--condition1",
        default="LaughReal",
        type=str,
        help="First condition",
    )

    parser.add_argument(
        "-cond2",
        "--condition2",
        default="LaughPosed",
        type=str,
        help="Second condition",
    )
    return parser

def source_rescontruction_parser():
    """
    Create an argparse parser for configuring source reconstrcution.
    """
    parser = argparse.ArgumentParser(description="Source Reconstruction.")

    source = parser.add_argument_group("Source Data")
    source.add_argument(
        "--subject", 
        type=int, 
        default=1, 
        help="Subject id."
    )
    source.add_argument(
        "--method", 
        type=str, 
        default="MNE", 
        help="Source Estimation method."
    )
    source.add_argument(
        "--overwrite",
        dest="overwrite", 
        action="store_true",
        help="If we want to overwrite existing files.",
    )
    source.add_argument(
        "--no-overwrite", 
        dest="overwrite", 
        action="store_false",
        help="Without overwrite."
    )
    source.add_argument(
        '--stimuli_file_name',
        type=str, 
        default="Fam",
        help='File name for stimuli images'
    )
    source.add_argument(
        '--meg_picks',
        type=str, 
        default="mag",
        help='Type of MEG sensors to use'
    )
    source.add_argument(
        '--task',
        type=str, 
        default="LaughterActive",
        help='Task to compute'
    )

    return parser