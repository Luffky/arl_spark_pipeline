import socket
import time
import sys
from arl.imaging import advise_wide_field
from arl.image.operations import qa_image, export_images_to_fits
from arl.spark_transformation import *
from pyspark import SparkContext, SparkConf

log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler(sys.stdout))

#os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3"
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

def trial_case(results, seed=180555, context='wstack', nworkers=8, threads_per_worker=1,
               processes=True, order='frequency', nfreqwin=7, ntimes=3, rmax=750.0,
               facets=1, wprojection_planes=1):
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
    sc.addFile("./LOWBD2.csv")
    sc.addFile("./sc16")
    sc.addFile("./SKA1_LOW_beam.fits")
    # sc.addFile("./GLEAM_EGC.fits")

    frequency = np.linspace(0.8e8, 1.2e8, nfreqwin)
    if nfreqwin > 1:
        channel_bandwidth = np.array(nfreqwin * [frequency[1] - frequency[0]])
    else:
        channel_bandwidth = np.array([1e6])
    times = np.linspace(-np.pi / 3.0, np.pi / 3.0, ntimes)

    phasecentre = SkyCoord(ra=+30.0 * u.deg, dec=-60.0 * u.deg, frame='icrs', equinox='J2000')
    vis_graph_list = create_simulate_vis_graph(sc, 'LOWBD2', frequency=frequency, channel_bandwidth=channel_bandwidth,
                                               times=times, phasecentre=phasecentre, order=order, format='blockvis',
                                               rmax=rmax)

    print("****** Visibility creation ******")


    wprojection_planes = 1
    vis = None
    for v in vis_graph_list.collect():
        if v[0][2] == 0:
            vis = v[1]
            break

    advice = advise_wide_field(convert_blockvisibility_to_visibility(vis), guard_band_image=6.0,
                               delA=0.02, facets=facets, wprojection_planes=wprojection_planes,
                               oversampling_synthesised_beam=4.0)


    kernel = advice['kernel']

    npixel = advice['npixels2']
    cellsize = advice['cellsize']
    print(cellsize)
    print(npixel)

    if context == 'timeslice' or context == 'facets_timeslice':
        vis_slices = ntimes
    elif context == '2d' or context == 'facets':
        vis_slices = 1
        kernel = '2d'
    else:
        vis_slices = advice['vis_slices']

    # vis_slices = 4
    results['vis_slices'] = vis_slices
    results['cellsize'] = cellsize
    results['npixel'] = npixel
    print(vis_slices)

    gleam_model_graph = create_low_test_image_from_gleam_spark(sc=sc, npixel=npixel, frequency=frequency,
                                                               channel_bandwidth=channel_bandwidth, cellsize=cellsize,
                                                               phasecentre=phasecentre,
                                                               polarisation_frame=PolarisationFrame("stokesI"),
                                                               flux_limit=0.1, applybeam=False)

    start = time.time()
    print("****** Starting GLEAM model creation ******")
    # gleam_model_graph.cache()
    gleam_model_graph.collect()

    print("****** Finishing GLEAM model creation *****")
    end = time.time()
    results['time create gleam'] = end - start
    print("Creating GLEAM model took %.2f seconds" % (end - start))


    vis_graph_list = create_predict_graph(vis_graph_list, gleam_model_graph, vis_slices=vis_slices, facets=facets, context=context
                                          , kernel=kernel, nfrequency=nfreqwin)
    start = time.time()
    print("****** Starting GLEAM model visibility prediction ******")
    # vis_graph_list.cache()
    vis_graph_list.collect()
    end = time.time()
    results['time predict'] = end - start
    print("GLEAM model Visibility prediction took %.2f seconds" % (end - start))

    # Corrupt the visibility for the GLEAM model
    print("****** Visibility corruption ******")
    vis_graph_list = create_corrupt_vis_graph(vis_graph_list, phase_error=1.0)
    start = time.time()
    vis_graph_list.cache()
    vis_graph_list.collect()
    end = time.time()
    results['time corrupt'] = end - start
    print("Visibility corruption took %.2f seconds" % (end - start))

    # Create an empty model image
    model_graph = create_empty_image(vis_graph_list, npixel=npixel, cellsize=cellsize, frequency=frequency,
                                     channel_bandwidth=channel_bandwidth, polarisation_frame=PolarisationFrame("stokesI"))

    model_graph.cache()
    model_graph.collect()

    psf_graph = create_invert_graph(vis_graph_list, model_graph, vis_slices=vis_slices, context=context, facets=facets,
                                    dopsf=True, kernel=kernel)

    start = time.time()
    print("****** Starting PSF calculation ******")
    psfs = psf_graph.collect()
    psf = None
    for i in psfs:
        if i[0][2] == 0:
            psf = i[1][0]
    end = time.time()
    results['time psf invert'] = end - start
    print("PSF invert took %.2f seconds" % (end - start))

    results['psf_max'] = qa_image(psf).data['max']
    results['psf_min'] = qa_image(psf).data['min']

    print(results['psf_max'])
    print(results['psf_min'])


    dirty_graph = create_invert_graph(vis_graph_list, model_graph, vis_slices=vis_slices, context=context, facets=facets,
                                    kernel=kernel)

    start = time.time()
    print("****** Starting dirty image calculation ******")
    dirtys  = dirty_graph.collect()
    dirty, sumwt = (None, None)
    for i in dirtys:
        if i[0][2] == 0:
            dirty, sumwt = i[1]

    print(psf.shape)
    print(dirty.shape)
    end = time.time()
    results['time invert'] = end - start
    print("Dirty image invert took %.2f seconds" % (end - start))
    print("Maximum in dirty image is ", numpy.max(numpy.abs(dirty.data)), ", sumwt is ", sumwt)
    qa = qa_image(dirty)
    results['dirty_max'] = qa.data['max']
    results['dirty_min'] = qa.data['min']

    start = time.time()
    print("***** write data to file *****")
    export_images_to_fits(psfs, nfreqwin, "psf.fits")
    export_images_to_fits(dirtys, nfreqwin, "dirty.fits")
    end = time.time()
    results['time write'] = end - start

    print("****** Starting ICAL ******")
    start = time.time()
    residual_graph, deconvolve_graph, restore_graph = create_ical_graph(sc, vis_graph_list, model_graph, nchan=nfreqwin, context=context, vis_slices=vis_slices,
                                   facets=facets, first_selfcal=1, algorithm='msclean', nmoments=3, niter=1000,
                                   fractional_threshold=0.1, scales=[0, 3, 10], threshold=0.1, nmajor=2, gain=0.7,
                                   timeslice='auto', global_solution=True, window_shape='quarter')

    deconvolveds = deconvolve_graph.collect()
    residuals = residual_graph.collect()
    restores = restore_graph.collect()

    end = time.time()
    results['time ICAL'] = end - start
    print("ICAL graph execution took %.2f seconds" % (end - start))

    residual = None
    for i in residuals:
        if i[0][2] == 0:
            residual = i[1][0]
    print(residual)
    qa = qa_image(residual)
    results['residual_max'] = qa.data['max']
    results['residual_min'] = qa.data['min']
    export_images_to_fits(residuals, nfreqwin, "pipelines-timings-delayed-ical_residual.fits")

    deconvolve = None
    for i in deconvolveds:
        if i[0][2] == 0:
            deconvolve = i[1]
    print(deconvolve)
    qa = qa_image(deconvolve)
    results['deconvolved_max'] = qa.data['max']
    results['deconvolved_min'] = qa.data['min']
    export_images_to_fits(deconvolveds, nfreqwin, "pipelines-timings-delayed-deconvolved.fits", has_sumwt=False)

    restore = None
    for i in restores:
        if i[0][2] == 0:
            restore = i[1]
    print(restore)
    qa = qa_image(restore)
    results['restored_max'] = qa.data['max']
    results['restored_min'] = qa.data['min']
    export_images_to_fits(restores, nfreqwin, "pipelines-timings-delayed-restored.fits", has_sumwt=False)









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
                  'nfreqwin', 'ntimes', 'rmax', 'facets', 'wprojection_planes', 'vis_slices', 'npixel',
                  'cellsize', 'seed', 'dirty_max', 'dirty_min', 'psf_max', 'psf_min', 'deconvolved_max',
                  'deconvolved_min', 'restored_min', 'restored_max', 'residual_max', 'residual_min',
                  'hostname', 'git_hash', 'epoch', 'context']

    filename = seqfile.findNextFile(folder="./develop_csv", prefix='%s_%s_' % (results['driver'], results['hostname']), suffix='.csv')
    print('Saving results to %s' % filename)

    write_header(filename, fieldnames)

    results = trial_case(results, nworkers=nworkers, rmax=rmax, context=context,
                         threads_per_worker=threads_per_worker, nfreqwin=nfreqwin, ntimes=ntimes, facets=nfacets)
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

    parser.add_argument('--ntimes', type=int, default=7, help='Number of hour angles')
    parser.add_argument('--nfreqwin', type=int, default=16, help='Number of frequency windows')
    parser.add_argument('--context', type=str, default='2d',
                        help='Imaging context: 2d|timeslice|timeslice|wstack|facets_slice|facets|facets_timeslice|facets_wstack')
    parser.add_argument('--rmax', type=float, default=200.0, help='Maximum baseline (m)')

    main(parser.parse_args())

    exit()
