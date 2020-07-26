# Print env vars, debug purposes.
echo NEST_MODULE_PATH=$NEST_MODULE_PATH
echo NEST_DATA_DIR=$NEST_DATA_DIR
echo PYTHONPATH=$PYTHONPATH
echo NEST_PYTHON_PREFIX=$NEST_PYTHON_PREFIX
echo LTDL_LIBRARY_PATH=$LTDL_LIBRARY_PATH

#unset MPIHOME
#unset MPICH_PROCESS_GROUP
#unset OMPI_MCA_btl
#unset PYTHONPATH

echo MPIHOME=$MPIHOME
echo MPICH_PROCESS_GROUP=$MPICH_PROCESS_GROUP
echo OMPI_MCA_btl=$OMPI_MCA_btl

#real	3m21.671s
#mpirun -np 1 -hostfile hostfile --prefix $CONDA_PREFIX -x NEST_MODULE_PATH=$NEST_MODULE_PATH -x NEST_DATA_DIR=$NEST_DATA_DIR --mca btl vader,self python main.py

#real	0m50.965s 
#mpirun -np 1 -hostfile hostfile --prefix $CONDA_PREFIX -x NEST_MODULE_PATH=$NEST_MODULE_PATH -x NEST_DATA_DIR=$NEST_DATA_DIR -bind-to none -map-by slot --mca btl vader,self python main.py

#real	0m52.808s
#mpirun -np 1 -hostfile hostfile --prefix $CONDA_PREFIX -x NEST_MODULE_PATH=$NEST_MODULE_PATH -x NEST_DATA_DIR=$NEST_DATA_DIR -bind-to none -map-by slot --mca btl vader,self,tcp  python main.py


mpirun -np 68 -hostfile hostfile --prefix $CONDA_PREFIX -x NEST_MODULE_PATH=$NEST_MODULE_PATH -x NEST_DATA_DIR=$NEST_DATA_DIR -bind-to none -map-by slot --mca pml ob1 --mca btl tcp,self --mca btl_tcp_if_include enp6s0f0 python main.py

