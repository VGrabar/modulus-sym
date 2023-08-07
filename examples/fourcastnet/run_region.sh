for f in 10 11 
do
	python3 fcn_era5.py ++custom.tstep=$f ++arch.afno.depth=8 ++arch.afno.num_blocks=32
done
for f in 10 11 
do
	python3 fcn_era5.py ++custom.tstep=$f ++arch.afno.depth=4 ++arch.afno.num_blocks=16
done
for f in 10 11 
do
	python3 fcn_era5.py ++custom.tstep=$f ++arch.afno.depth=2 ++arch.afno.num_blocks=4
done
for f in 10 11 
do
	python3 fcn_era5.py ++custom.tstep=$f ++arch.afno.depth=4 ++arch.afno.num_blocks=128
done
