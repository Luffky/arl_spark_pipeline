spark-submit --name dask_spark --properties-file spark-arl.conf --files sc512,LOWBD2.csv,SKA1_LOW_beam.fits --conf spark.executorEnv.PYTHONHASHSEED=353 --total-executor-cores 36 --py-files arl.zip pipeline-partitioning_auto_spark.py --nfreqwin 512 --ntimes 7 --context 2d --nfacets 1 --parallelism 64 
