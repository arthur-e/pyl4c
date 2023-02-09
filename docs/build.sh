# Use:
#   sh build.sh <path_to_pyl4c_repo> <path_to_output_dir>

pdoc --html -c latex_math=True --template-dir ${2}/templates --force -o $2 $1
mv $2/pyl4c/* $2
rmdir $2/pyl4c
