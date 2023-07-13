for f in 1 2 3 4 5 6 7 8 9 10 11 12
do
	python3 fcn_era5.py ++custom.tstep=$f
done
t = "data/pdsi_Missouri_h5/train"
d = "data/pdsi_Missouri_h5/test"
for f in 1 2 3 4 5 6 7 8 9 10 11 12
do
	python3 fcn_era5.py ++custom.tstep=$f ++custom.train_dataset.data_path=$t ++custom.test_dataset.data_path=$d
done
t = "data/pdsi_CentralKZ_h5/train"
d = "data/pdsi_CentralKZ_h5/test"
for f in 1 2 3 4 5 6 7 8 9 10 11 12
do
	python3 fcn_era5.py ++custom.tstep=$f ++custom.train_dataset.data_path=$t ++custom.test_dataset.data_path=$d
done
