if test "$#" -ne 5; then
    echo "Illegal number of parameters"
else
    mpiexec -n $1 python3 kmeans_sparse.py $2 $3 $4 $5
fi
