from pyspark import SparkContext, SparkConf
import socket
import time
import numpy as np
from collections import defaultdict

def MapPartitioner(partitions):
    def _inter(key):
        partition = partitions
        return partition[key]
    return _inter

def git_hash():

    """ Get the hash for this git repository.

    :return: string or "unknown"
    """
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", 'HEAD'])
    except Exception as excp:
        print(excp)
        return "unknown"

def create_zero_array(scale):
    return np.zeros(max(1, int(scale)), dtype=np.int32)

def create_simulate_vis_graph_empty(sc, nfreqwin = 16, scale_data = 0.1):
    def telescope_management_kernel(ixs):
        ix = next(ixs)[0]
        result = create_zero_array(scale_data * 0)
        label = "Telescope Management"
        print(label + str(result))
        return iter([(ix, result)])

    def telescope_management_handle(sc):
        partitions = defaultdict(int)
        partition = 0
        initset = []
        partitions[()] = partition
        partition += 1
        initset.append(((), ()))
        partitioner = MapPartitioner(partitions)
        return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(
            telescope_management_kernel, True)

    def telescope_data_flatmap(ixs, dep):
        ret = []
        for key in dep:
            data = create_zero_array(scale_data * 0)
            ret.append((key, data))
        return iter(ret)

    def telescope_data_kernel(ixs):
        ret = []
        for data in ixs:
            value = create_zero_array(scale_data * 1823123744)
            value[0] = 0
            ret.append((data[0], value))
            label = "create_simulate_vis"
            print(label + str(data[0]))
        return iter(ret)

    def telescope_data_handle(telescope_management):
        # partitions = defaultdict(int)
        partition = 0
        dep_telescope_management = defaultdict(dict)
        beam = 0
        baseline = 0
        major_loop = 0
        facet = 0
        polarisation = 0
        time = 0
        for frequency in range(0, nfreqwin):
            partition += 1
            dep_telescope_management[(beam, major_loop, frequency, time, facet, polarisation)] = create_zero_array(scale_data * 0)
        input_telescope_management = telescope_management.flatMap(lambda record: telescope_data_flatmap(record, dep_telescope_management))
        return input_telescope_management.partitionBy(partition).mapPartitions(telescope_data_kernel, True)

    return telescope_data_handle(telescope_management_handle(sc))


def create_low_test_image_from_gleam_spark_empty(sc, nfreqwin, scale_data):
    def extract_lsm_kernel(ixs):
        index = None
        sc = None
        for data in ixs:
            sc = create_zero_array(scale_data * 0)
            index = data[0]
        label = "Extract_LSM"
        print(label + str((index, sc)))
        return iter([(index, sc)])

    def extract_lsm_handle(sc):
        partitions = defaultdict(int)
        initset = []
        beam = 0
        major_loop = 0
        partition = 0
        for i in range(nfreqwin):
            initset.append(((beam, major_loop, i), create_zero_array(scale_data * 0)))
            partitions[(beam, major_loop, i)] = partition
            partition += 1
        partitioner = MapPartitioner(partitions)

        return sc.parallelize(initset).partitionBy(partition, partitioner).mapPartitions(extract_lsm_kernel, True)

    def reppre_ifft_kernel(ix, data_extract_lsm, dep_image):
        comps = None
        for i in data_extract_lsm.value:
            if i[0][2] == ix[2]:
                comps = i[1]
        model = create_zero_array(scale_data * 3954302955)
        label = "create_gleam_image"
        print(label + str(ix))
        model[0] = 0
        return model

    def reppre_ifft_handle(sc, broadcast_lsm):
        initset = []
        dep_image = defaultdict(dict)
        beam = 0
        major_loop = 0
        for i in range(nfreqwin):
            time = 0
            facet = 0
            polarisation = 0
            npol = 1
            initset.append(((beam, major_loop, i, time, facet, polarisation), (beam, major_loop, i, time, facet, polarisation)))
            dep_image[(beam, major_loop, i, time, facet, polarisation)] = create_zero_array(scale_data * 0)
        dep_image_broadcast = sc.broadcast(dep_image)
        return sc.parallelize(initset).repartition(nfreqwin).mapValues(lambda ix: reppre_ifft_kernel(ix, broadcast_lsm, dep_image_broadcast))

    return reppre_ifft_handle(sc, sc.broadcast(extract_lsm_handle(sc).collect()))

def create_predict_graph_empty(reppre_ifft, telescope_data, context, nfacets, nslices):
    def degrid_kernel(ixs, context):
        iix, (data_image, data_visibility) = ixs
        if (context != "2d" and context != "facets"):
            iix = (iix[0], iix[1], iix[2], iix[3], data_image.facet_id, iix[5])
        label = "predict"
        print(label + str(iix))
        return (iix, data_visibility)

    def scatter_image_kernel(ixs, facets):
        ix, data = ixs
        step = int(float(data.size) / float(facets))
        ret = []
        for i in range(0, facets):
            slice = data[i*step: (i+1)*step]
            slice[0] = i
            ret.append((ix, slice))
        return iter(ret)

    def scatter_vis_kernel(ixs, slices):
        ix, data = ixs
        step = int(float(data.size) / float(slices))
        ret = []
        for i in range(0, slices):
            slice = data[i*step: (i+1)*step]
            slice[0] = i
            ret.append((ix, slice))
        return iter(ret)

    def sum_predict_vis_reduce_kernel(v1, v2):
        return v1

    def change_key(ix):
        ret_key = (ix[0][0], ix[0][1], ix[0][2], ix[0][3], 0, ix[0][5])
        return (ret_key, ix[1])

    def degrid_handle(reppre_ifft, telescope_data, context):
        if(context == "2d"):
            return reppre_ifft.join(telescope_data).map(lambda ix: degrid_kernel(ix, context))

        elif(context == "facets"):
            return reppre_ifft.flatMap(lambda ixs: scatter_image_kernel(ixs, nfacets)).join(telescope_data)\
            .map(lambda ix: degrid_kernel(ix, context)).reduceByKey(sum_predict_vis_reduce_kernel)

        elif(context == "facet_slice"):
            telescope_data_scatter = telescope_data.flatMap(lambda ix: scatter_vis_kernel(ix, nslices))
            return reppre_ifft.flatMap(lambda ixs: scatter_image_kernel(ixs, nfacets)).join(telescope_data_scatter)\
            .map(lambda ix: degrid_kernel(ix, context)).reduceByKeye(sum_predict_vis_reduce_kernel)\
            .map(change_key).map(lambda ix: (ix[0], create_zero_array(ix[1].size * nslices)))\
            .reduceByKey(sum_predict_vis_reduce_kernel)

        else:
            telescope_data_scatter = telescope_data.flatMap(lambda ix: scatter_vis_kernel(ix, nslices))
            return reppre_ifft.join(telescope_data_scatter).map(lambda ix: degrid_kernel(ix, context))\
            .reduceByKeye(sum_predict_vis_reduce_kernel).map(lambda ix: (ix[0], create_zero_array(ix[1].size * nslices)))

    return degrid_handle(reppre_ifft, telescope_data, context)

def create_corrupt_vis_graph_empty(vis, scale_data):
    def corrupt_kernel(viss):
        ret = []
        for v in viss:
            ret.append((v[0], create_zero_array(scale_data * 511780275)))
            label = "corrupt"
            print(label + str(v[0]))
        return iter(ret)
    return vis.mapPartitions(corrupt_kernel)

def create_empty_image_empty(vis, scale_data):
    def create_empty_image_kernel(viss):
        ret = []
        for v in viss:
            ret.append((v[0], create_zero_array(scale_data * 3954302955)))
            label = "create_empty_image"
            print(label + str(v[0]))
        return iter(ret)
    return vis.mapPartitions(create_empty_image_kernel)

def create_invert_graph_empty(vis, im, context, nfacets, nslices):
    def invert_kernel(ixs, context):
        ix, (im, vis) = ixs
        if context == "facet_slice":
            ix = (ix[0], ix[1], ix[2], ix[3], im[0], ix[5])
        label = "invert"
        print(label + str(ix))
        return (ix, im)

    def scatter_image_kernel(ixs, facets):
        ix, data = ixs
        step = int(float(data.size) / float(facets))
        ret = []
        for i in range(0, facets):
            slice = data[i*step: (i+1)*step]
            slice[0] = i
            ret.append((ix, slice))
        return iter(ret)

    def scatter_vis_kernel(ixs, slices):
        ix, data = ixs
        step = int(float(data.size) / float(slices))
        ret = []
        for i in range(0, slices):
            slice = data[i*step: (i+1)*step]
            slice[0] = i
            ret.append((ix, slice))
        return iter(ret)

    def sum_predict_vis_reduce_kernel(v1, v2):
        return v1

    def change_key(ix):
        ret_key = (ix[0][0], ix[0][1], ix[0][2], ix[0][3], 0, ix[0][5])
        return (ret_key, ix[1])

    def invert_handle(vis, im, context):

        image_metadata = im.mapValues(lambda im: create_zero_array(1))
        if context == "2d":
            return im.join(vis).map(lambda ixs: invert_kernel(ixs, context))

        elif context == "facets":
            return im.flatMap(lambda ixs: scatter_image_kernel(ixs, facets=nfacets)).join(vis).map(lambda ixs: invert_kernel(ixs, context))\
            .groupByKey().join(image_metadata).map(lambda ixs: (ixs[0], create_zero_array(list(ixs[1][0])[0].size * nfacets)))

        elif context == "facet_slice":
            vis_scatter = vis.flatMap(lambda vis: scatter_vis_kernel(vis, nslices))
            return im.flatMap(lambda ixs: scatter_image_kernel(ixs, facets=nfacets)).join(vis_scatter).map(lambda ixs: invert_kernel(ixs, context))\
            .reduceByKey(sum_predict_vis_reduce_kernel).map(change_key).groupByKey().join(image_metadata)\
            .map(lambda ixs: (ixs[0], create_zero_array(list(ixs[1][0])[0].size * nfacets)))

        else:
            vis_scatter = vis.flatMap(lambda vis: scatter_vis_kernel(vis, nslices))
            return im.join(vis_scatter).map(lambda ix: invert_kernel(ix, context)).reduceByKey(sum_predict_vis_reduce_kernel)

    return invert_handle(vis, im, context)


def trial_case(results, seed=180555, context='wstack', nworkers=8, threads_per_worker=1,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1, slices=1, scale_data=0.1):
    npol = 1
    np.random.seed(seed)
    results['seed'] = seed

    start_all = time.time()

    results['context'] = context
    results['hostname'] = socket.gethostname()
    results['git_hash'] = git_hash()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")

    zerow = False
    print("Context is %s" % context)

    results['nworkers'] = nworkers
    results['threads_per_worker'] = threads_per_worker
    results['processes'] = processes
    results['order'] = order
    results['nfreqwin'] = nfreqwin
    results['ntimes'] = ntimes
    results['rmax'] = rmax
    results['facets'] = facets
    results['wprojection_planes'] = wprojection_planes

    print("At start, configuration is {0!r}".format(results))

    conf = SparkConf()
    sc = SparkContext(conf=conf)

    vis_graph_list = create_simulate_vis_graph_empty(sc, nfreqwin=nfreqwin, scale_data=scale_data)

    print("****** Visibility creation ******")


    wprojection_planes = 1



    gleam_model_graph = create_low_test_image_from_gleam_spark_empty(sc=sc, nfreqwin=nfreqwin, scale_data=scale_data)

    start = time.time()
    print("****** Starting GLEAM model creation ******")
    # gleam_model_graph.cache()
    # gleam_model_graph.collect()

    print("****** Finishing GLEAM model creation *****")
    end = time.time()
    results['time create gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))


    vis_graph_list = create_predict_graph_empty(gleam_model_graph, vis_graph_list, context=context, nfacets=facets, nslices=slices)
    start = time.time()
    print("****** Starting GLEAM model visibility prediction ******")
    # vis_graph_list.cache()
    # vis_graph_list.collect()
    end = time.time()
    results['time predict'] = end - start
    print("GLEAM model Visibility prediction took %.2f seconds" % (end - start))

    # Corrupt the visibility for the GLEAM model
    print("****** Visibility corruption ******")
    vis_graph_list = create_corrupt_vis_graph_empty(vis=vis_graph_list, scale_data=scale_data)
    start = time.time()
    vis_graph_list.cache()
    # vis_graph_list.collect()
    end = time.time()
    results['time corrupt'] = end - start
    print("Visibility corruption took %.2f seconds" % (end - start))

    # Create an empty model image
    model_graph = create_empty_image_empty(vis_graph_list, scale_data=scale_data)

    model_graph.cache()
    # model_graph.collect()
    psf_graph = create_invert_graph_empty(vis_graph_list, model_graph, context, nslices=slices, nfacets=facets)

    start = time.time()
    print("****** Starting PSF calculation ******")
    psfs = psf_graph.collect()
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))



    dirty_graph = create_invert_graph_empty(vis_graph_list, model_graph, context, nslices=slices, nfacets=facets)

    start = time.time()
    print("****** Starting dirty image calculation ******")
    dirtys  = dirty_graph.collect()
    print(psfs[0][1].shape)
    print(dirtys[0][1].shape)

    end = time.time()
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))









    end_all = time.time()
    results['time overall'] = end_all - start_all

    print("At end, results are {0!r}".format(results))
    sc.stop()

    return results







def write_results(filename, fieldnames, results):
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writerow(results)
        csvfile.close()


def write_header(filename, fieldnames):
    with open(filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='|',
                                quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        csvfile.close()

def main(args):
    results = {}

    nworkers = args.nworkers
    results['nworkers'] = nworkers

    context = args.context
    results['context'] = context

    nnodes = args.nnodes
    results['nnodes'] = nnodes

    threads_per_worker = args.nthreads

    print("Using %s workers" % nworkers)
    print("Using %s threads per worker" % threads_per_worker)

    nfreqwin = args.nfreqwin
    results['nfreqwin'] = nfreqwin

    rmax = args.rmax
    results['rmax'] = rmax

    context = args.context
    results['context'] = context

    ntimes = args.ntimes
    results['ntimes'] = ntimes

    nfacets = args.nfacets
    nslices = args.nslices
    scale = args.scale

    results['hostname'] = socket.gethostname()
    results['epoch'] = time.strftime("%Y-%m-%d %H:%M:%S")
    results['driver'] = 'pipelines-timings-delayed'

    threads_per_worker = args.nthreads

    print("Trying %s workers" % nworkers)
    print("Using %s threads per worker" % threads_per_worker)
    print("Defining %d frequency windows" % nfreqwin)

    fieldnames = ['driver', 'nnodes', 'nworkers', 'time ICAL', 'time ICAL graph', 'time create gleam',
                  'time predict', 'time corrupt', 'time invert', 'time psf invert', 'time write', 'time overall',
                  'threads_per_worker', 'processes', 'order',
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'slices', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context']

    filename = seqfile.findNextFile(folder="./csv_spark", prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)

    write_header(filename, fieldnames)

    results = trial_case(results, nworkers=nworkers, rmax=rmax, context=context,
                         threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes, facets=nfacets, slices=nslices, scale_data=scale)
    write_results(filename, fieldnames, results)

    print('Exiting %s' % results['driver'])

if __name__ == '__main__':
    import csv
    import seqfile

    import argparse

    parser = argparse.ArgumentParser(description='Benchmark pipelines in numpy and spark')
    parser.add_argument('--nnodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--nthreads', type=int, default=1, help='Number of threads')
    parser.add_argument('--nworkers', type=int, default=1, help='Number of workers')
    parser.add_argument('--nfacets', type=int, default=1, help='Number of facets')
    parser.add_argument('--nslices', type=int, default=1, help='Number of slices')

    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='2d',
                        help='Imaging context: 2d|timeslice|timeslice|wstack|facets_slice|facets|facets_timeslice|facets_wstack')
    parser.add_argument('--rmax', type=float, default=200.0, help='Maximum baseline (m)')
    parser.add_argument('--scale', type=float, default = 0.1)

    main(parser.parse_args())

    exit()