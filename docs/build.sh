# Use:
#   sh build.sh <path_to_pyl4c_repo> <path_to_output_dir>

pdoc --html -c latex_math=True --force -o $2 $1
