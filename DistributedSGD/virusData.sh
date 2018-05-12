if test "$#" -ne 4; then
    echo "Illegal number of parameters"
else
    mpiexec -n $1 python3 SGDVirusDataset.py $2 $3 $4
fi
