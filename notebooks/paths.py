import os, sys

# current  notebook directory
curr_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_script_dir)
sys.path.append(parent_dir)

parent_dir = os.path.dirname(curr_script_dir)
results_dir = os.path.join(parent_dir, 'results')
report_dir = os.path.join(parent_dir, 'report')
src_dir = os.path.join(parent_dir, 'src')
data_dir = os.path.join(parent_dir, 'data')