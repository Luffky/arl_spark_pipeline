from collections import defaultdict
from pyspark import SparkContext, SparkFiles
from astropy.wcs import WCS
from arl.image.operations import create_image_from_array, create_empty_image_like
from arl.skycomponent.operations import insert_skycomponent
from arl.util.testing_support import create_low_test_beam, create_low_test_skycomponents_from_gleam, simulate_gaintable
from arl.imaging.imaging_context import imaging_context
from arl.imaging.base import create_image_from_visibility, normalize_sumwt
from arl.visibility.coalesce import *
from arl.visibility.base import *
from astropy import units as u
from arl.calibration.solvers import create_gaintable_from_blockvisibility
from arl.calibration.calibration import apply_gaintable
import numpy as np
from arl.graphs.delayed import sum_invert_results
from arl.image.operations import copy_image
from arl.visibility.iterators import *

from arl.visibility.base import create_blockvisibility, create_visibility
from arl.util.testing_support import create_named_configuration, simulate_gaintable, create_low_test_image_from_gleam

log = logging.getLogger(__name__)

def MapPartitioner(partitions):
    def _inter(key):
        partition = partitions
        return partition[key]
    return _inter

def SDPPartitioner_by_frequency(key):
    '''
		Partitioner_function
	'''
    return int(str(key).split(',')[2])

def iterate_print_rdd(rdd, only_index=True):
    for item in rdd.collect():
        if only_index == True:
            print(item[0])
        else:
            print(item)

def SDPPartitioner_by_frequency_and_facetid(key):
    return int()

def create_simulate_vis_graph(sc: SparkContext, config='LOWBD2-CORE',
                              phasecentre=SkyCoord(ra=+15.0 * u.deg, dec=-60.0 * u.deg,
                                                   frame='icrs', equinox='J2000'),
                              frequency=None, channel_bandwidth=None, times=None,
                              polarisation_frame=PolarisationFrame("stokesI"), order='frequency',
                              format='blockvis',
                              rmax=1000.0):
    if format == 'vis':
        create_vis = create_visibility
    else:
        create_vis = create_blockvisibility

    if times is None:
        times = [0.0]
    if channel_bandwidth is None:
        channel_bandwidth = [1e6]
    if frequency is None:
        frequency = [1e8]

    telescope_management = telescope_management_handle(sc, config, rmax)
    vis_graph_list = telescope_data_handle(telescope_management, times=times, frequencys=frequency,
                                               channel_bandwidth=channel_bandwidth, weight=1.0, phasecentre=phasecentre,
                                               polarisation_frame=polarisation_frame, order=order, create_vis=create_vis)

    return vis_graph_list

def create_low_test_image_from_gleam_spark(sc: SparkContext, npixel=512, polarisation_frame=PolarisationFrame("stokesI"), cellsize=0.000015,
                                     frequency=np.array([1e8]), channel_bandwidth=np.array([1e6]),
                                     phasecentre=None, kind='cubic', applybeam=False, flux_limit=0.1,
                                     radius=None, insert_method='Nearest'):
    if phasecentre is None:
        phasecentre = SkyCoord(ra=+15.0 * u.deg, dec=-35.0 * u.deg, frame='icrs', equinox='J2000')

    if radius is None:
        radius = npixel * cellsize / np.sqrt(2.0)

    sc_filename = "sc" + str(len(frequency))
    comps = extract_lsm_handle(sc, flux_limit=flux_limit, polarisation_frame=polarisation_frame,
                               frequencys=frequency, phasecentre=phasecentre, kind=kind, radius=radius, filename=sc_filename)
    broadcast_lsm = sc.broadcast(comps.collect())

    if polarisation_frame is None:
        polarisation_frame = PolarisationFrame("stokesI")

    model = reppre_ifft_handle(sc, broadcast_lsm, polarisation_frame=polarisation_frame, frequencys=frequency,
                               npixel=npixel, cellsize=cellsize, phasecentre=phasecentre, channel_bandwidth=channel_bandwidth,
                               insert_method=insert_method, applybeam=applybeam)

    return model

def create_predict_graph(vis_graph_list, gleam_model_graph, vis_slices=1, facets=1, context='2d', nfrequency=16, **kwargs):
    results_vis_graph_list = degrid_handle(gleam_model_graph, vis_graph_list, context=context, vis_slices=vis_slices,
                                           facets=facets, nfrequency=nfrequency, **kwargs)

    return results_vis_graph_list

def create_corrupt_vis_graph(vis_graph_list, gt_graph=None, **kwargs):
    result = corrupt_handle(vis_graph_list, gt_graph, **kwargs)
    return result

def create_empty_image(vis_graph_list, npixel, cellsize, frequency, channel_bandwidth, polarisation_frame):
    result = create_empty_handle(vis_graph_list, npixel, cellsize, frequency, channel_bandwidth, polarisation_frame)
    return result

def create_invert_graph(sc, vis_graph_list, template_model_graph, dopsf=False, normalize=True, facets=1, vis_slices=None, context="2d", **kwargs):
    c = imaging_context(context)
    results_vis_graph_list = invert_handle(sc, template_model_graph, vis_graph_list, context=context, dopsf=dopsf,
                                           normalize=normalize, facets=facets, vis_slices=vis_slices, **kwargs)
    return results_vis_graph_list


def extract_lsm_handle(sc: SparkContext, flux_limit, polarisation_frame, frequencys, phasecentre,
                       kind, radius, filename):
    partitions = defaultdict(int)
    partition = 0
    initset = []
    beam = 0
    major_loop = 0
    for i, frequency in enumerate(frequencys):
        initset.append(((beam, major_loop, i), {"flux_limit": flux_limit, "polarisation_frame": polarisation_frame,
                                          "frequency": [frequency], "phasecentre": phasecentre, "kind": kind,
                                          "radius": radius}))
        partitions[(beam, major_loop, i)] = partition
        partition += 1
    partitioner = MapPartitioner(partitions)
    return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(lambda record: extract_lsm_kernel(record, frequencys.shape[0]), True)

def extract_lsm_kernel(ixs, nchan):
    '''
        生成skycomponent(s)
    :param ixs: key
    :return: iter[(key, skycoponent)]
    '''
    index = None
    sc = None
    for i in ixs:
        ix, para = i
        sc = create_low_test_skycomponents_from_gleam(flux_limit=para["flux_limit"], polarisation_frame=para["polarisation_frame"],
                                                  frequency=para["frequency"], phasecentre=para["phasecentre"], kind=para["kind"],
                                                  radius=para["radius"], nchan=nchan)
        index = ix

    label = "Ectract_LSM (0.0M MB, 0.00 Tflop) "
    print(label + str((index, sc)))
    return iter([(index, sc)])

def telescope_management_handle(sc: SparkContext, config, rmax):
	partitions = defaultdict(int)
	partition = 0
	initset = []
	partitions[()] = partition
	partition += 1
	initset.append(((), ()))
	partitioner = MapPartitioner(partitions)
	return sc.parallelize(initset).partitionBy(len(partitions), partitioner).mapPartitions(
        lambda a: telescope_management_kernel(a, config, rmax), True)

def telescope_management_kernel(ixs, config, rmax):
    '''
        生成总的conf类，留待telescope_data_kernel进一步划分
    :param ixs:
    :return: iter[(key, conf)]
    '''
    ix = next(ixs)[0]
    conf = create_named_configuration(config, rmax=rmax)
    result = (ix, conf)
    label = "Telescope Management (0.0 MB, 0.00 Tflop) "
    print(label + str(result))
    return iter([result])

def telescope_data_handle(telescope_management, times, frequencys, channel_bandwidth,
                          weight, phasecentre, polarisation_frame, order, create_vis):
    # partitions = defaultdict(int)
    partition = 0
    dep_telescope_management = defaultdict(dict)
    beam = 0
    baseline = 0
    major_loop = 0
    facet = 0
    polarisation = 0
    if order == 'time':
        j = 0
        for i, time in enumerate(times):
            # partitions[(beam, major_loop, j, i, facet, polarisation)] = partition
            partition += 1
            dep_telescope_management[(beam, major_loop, j, i, facet, polarisation)] = \
                {"times": [time], "frequencys": frequencys, "channel_bandwidth": channel_bandwidth}

    elif order == 'frequency':
        i = 0
        for j, (frequency, bandwidth) in enumerate(zip(frequencys, channel_bandwidth)):
            # partitions[(beam, major_loop, j, i, facet, polarisation)] = partition
            partition += 1
            dep_telescope_management[(beam, major_loop, j, i, facet, polarisation)] = \
                {"times": times, "frequencys": [frequency], "channel_bandwidth": [bandwidth]}

    elif order == "both":
        for j, (frequency, bandwidth) in enumerate(zip(frequencys, channel_bandwidth)):
            for i, time in enumerate(times):
                # partitions[(beam, major_loop, j, i, facet, polarisation)] = partition
                partition += 1
                dep_telescope_management[(beam, major_loop, j, i, facet, polarisation)] = \
                    {"times": [time], "frequencys": [frequency], "channel_bandwidth": [bandwidth]}


    input_telescope_management = telescope_management.flatMap(
        lambda management: telescope_data_flatmap(management, dep_telescope_management,phasecentre,polarisation_frame,weight=1.0))
    # partitioner = MapPartitioner(partitions)
    return input_telescope_management.partitionBy(partition).mapPartitions(lambda record: telescope_data_kernel(record, create_vis=create_vis), True)

def telescope_data_flatmap(ixs, dep_telescope_management: defaultdict(dict), phasecentre, polarisation, weight):
    ret = []
    _, conf = ixs
    for key in dep_telescope_management:
        dep_telescope_management[key]["conf"] = conf
        dep_telescope_management[key]["phasecentre"] = phasecentre
        dep_telescope_management[key]["polarisation_frame"] = polarisation
        dep_telescope_management[key]["weight"] = weight
        ret.append((key, dep_telescope_management[key]))
    return iter(ret)

def telescope_data_kernel(ixs, create_vis):
    '''
		分割visibility类为visibility_para
	:param ixs:
    :return: iter[(key, visibility_para)]
    '''
    result = []
    for data in ixs:
        ix, value = data
        times = value["times"]
        frequencys = value["frequencys"]
        channel_bandwidth = value["channel_bandwidth"]
        phasecentre = value["phasecentre"]
        polarisation_frame = value["polarisation_frame"]
        weight = value["weight"]
        conf = value["conf"]

        result.append((ix, create_vis(conf, times=times, frequency=frequencys,
                                      channel_bandwidth=channel_bandwidth,
                                      weight=weight, phasecentre=phasecentre,
                                      polarisation_frame=polarisation_frame)))

        label = "create_simulate_vis, key: " + str(ix)
        print(label)

    return iter(result)

def reppre_ifft_handle(sc: SparkContext, broadcast_lsm, polarisation_frame, frequencys, npixel, cellsize,
                       phasecentre, channel_bandwidth, insert_method, applybeam):
    initset = []
    dep_image = defaultdict(dict)
    beam = 0
    major_loop = 0
    nfrequency = frequencys.shape[0]
    for i, frequency in enumerate(frequencys):
        time = 0
        facet = 0
        polarisation = 0
        npol = polarisation_frame.npol
        frequency = [frequency]
        nchan = len(frequency)
        shape = [nchan, npol, npixel, npixel]

        w = WCS(naxis=4)
        # The negation in the longitude is needed by definition of RA, DEC
        w.wcs.cdelt = [-cellsize * 180.0 / np.pi, cellsize * 180.0 / np.pi, 1.0, channel_bandwidth[0]]
        w.wcs.crpix = [npixel // 2 + 1, npixel // 2 + 1, 1.0, 1.0]
        w.wcs.ctype = ["RA---SIN", "DEC--SIN", 'STOKES', 'FREQ']
        w.wcs.crval = [phasecentre.ra.deg, phasecentre.dec.deg, 1.0, frequency[0]]
        w.naxis = 4
        w.wcs.radesys = 'ICRS'
        w.wcs.equinox = 2000.0
        initset.append(((beam, major_loop, i, time, facet, polarisation), (beam, major_loop, i, time, facet, polarisation)))
        dep_image[(beam, major_loop, i, time, facet, polarisation)] = {
            "frequency": frequency, "shape": shape, "wcs": w, "polarisation_frame": polarisation_frame
        }
    dep_image = sc.broadcast(dep_image)
    # in this place, I use mapValues to preserve partitioner information.
    # TODO 注意点
    return sc.parallelize(initset).partitionBy(nfrequency).mapValues(lambda ix: reppre_ifft_kernel(ix, broadcast_lsm, dep_image.value[ix],
                                                                     insert_method=insert_method, applybeam=applybeam))

def reppre_ifft_kernel(ix, data_extract_lsm, dep_image, insert_method, applybeam):
    frequency = dep_image["frequency"]
    comps = []
    for i in data_extract_lsm.value:
        if i[0][2] == ix[2]:
            comps = i[1]

    wcs = dep_image["wcs"]
    shape = dep_image["shape"]
    polarisation_frame = dep_image["polarisation_frame"]
    model = create_image_from_array(np.zeros(shape), wcs, polarisation_frame=polarisation_frame)

    model = insert_skycomponent(model, comps, insert_method=insert_method)

    if applybeam:
        beam = create_low_test_beam(model)
        model.data = model.data * beam

    result = model
    label = "Reprojection Predict + IFFT (14645.6 MB, 2.56 Tflop) "
    # print(label + str(result))
    return result

def degrid_handle(reppre_ifft, telescope_data, context, vis_slices, facets, nfrequency, **kwargs) -> BlockVisibility :
    c = imaging_context(context)
    if context == "2d":
        telescope_data = telescope_data.mapValues(lambda vis: coalesce_visibility(vis, **kwargs))
        return reppre_ifft.join(telescope_data).map(lambda record: degrid_kernel(record, c["predict"], context=context, **kwargs))

    elif context == "facets":
        telescope_data = telescope_data.mapValues(lambda vis: coalesce_visibility(vis, **kwargs))
        print("scatter_image")
        return reppre_ifft.flatMap(lambda im: scatter_image_flatmap(im, facets=facets, image_iter=c["image_iterator"], **kwargs), True)\
            .join(telescope_data).map(lambda record: degrid_kernel(record, c["predict"], context=context, **kwargs ), True)\
            .reduceByKey(sum_predict_vis_reduce_kernel)

    elif context == "facets_slice" or context == "facets_timeslice" or context == "facets_wstack":
        telescope_data_origin = telescope_data.map(lambda vis: (vis[0], coalesce_visibility(vis[1], **kwargs)))
        telescope_data = telescope_data_origin.flatMap(lambda vis: scatter_vis_flatmap(vis, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs), True)
        # TODO 此处可优化
        return reppre_ifft.flatMap(lambda im: scatter_image_flatmap(im, facets=facets, image_iter=c["image_iterator"], **kwargs), True) \
            .join(telescope_data).map(lambda record: degrid_kernel(record, c["predict"], context=context, **kwargs)) \
            .combineByKey(gather_vis_createCombiner_kernel, gather_vis_mergeValue_kernel, gather_vis_mergeCombiner_kernel)\
            .map(change_key).join(telescope_data_origin).mapValues(lambda data: gather_vis_kernel(data, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs))\
            .reduceByKey(sum_predict_vis_reduce_kernel)


    elif context == "slice" or context == "timeslice" or context == "wstack":
        telescope_data_origin = telescope_data.map(lambda vis: (vis[0], coalesce_visibility(vis[1], **kwargs)))
        telescope_data = telescope_data_origin.flatMap(
            lambda vis: scatter_vis_flatmap(vis, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs), True)
        return reppre_ifft.join(telescope_data).map(lambda record: degrid_kernel(record, c["predict"], context=context, **kwargs)) \
        .combineByKey(gather_vis_createCombiner_kernel, gather_vis_mergeValue_kernel, gather_vis_mergeCombiner_kernel)\
        .join(telescope_data_origin).mapValues(lambda data: gather_vis_kernel(data, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs))

def scatter_image_flatmap(ixs, facets, image_iter, **kwargs):
    iix, im = ixs
    id = 0
    ret = []
    for subim in image_iter(im, facets=facets, **kwargs):
        subim.facet_id = id
        id += 1
        ret.append((iix, subim))
    return iter(ret)

def scatter_vis_flatmap(ixs, vis_slices, vis_iter, **kwargs):
    iix, vis = ixs
    ret = []
    id = 0
    for rows in vis_iter(vis, vis_slices=vis_slices, **kwargs):
        v = create_visibility_from_rows(vis, rows)
        v.iter_id = id
        id += 1
        ret.append((iix, v))
    return iter(ret)

def sum_predict_vis_reduce_kernel(v1, v2):
    assert v1.data['vis'].shape == v2.data['vis'].shape
    v1.data['vis'] += v2.data['vis']
    return v1

def gather_vis_createCombiner_kernel(v1: Visibility):
    return {v1.iter_id: v1.data['vis']}

def gather_vis_mergeValue_kernel(v1: dict, v2: Visibility):
    v1[v2.iter_id] = v2.data['vis']
    return v1

def gather_vis_mergeCombiner_kernel(d1: dict, d2: dict):
    for key, value in d2.items():
        d1[key] = value
    return d1

def change_key(kv):
    return ((kv[0][0], kv[0][1], kv[0][2], kv[0][3], 0, kv[0][5]), kv[1])

def gather_vis_kernel(data, vis_slices, vis_iter, **kwargs) -> BlockVisibility:
    data_dict, data_vis = data
    for i, rows in enumerate(vis_iter(data_vis, vis_slices=vis_slices, **kwargs)):
        assert i < len(data_dict), "Insufficient results for the gather"
        if rows is not None and data_dict[i] is not None:
            data_vis.data['vis'][rows] = data_dict[i]
    data_vis = decoalesce_visibility(data_vis, **kwargs)
    return data_vis

def degrid_kernel(ixs, function, context, **kwargs):
    iix, (data_image, data_visibility) = ixs
    iter_id = data_visibility.iter_id
    result = function(copy_visibility(data_visibility), data_image, context=context, **kwargs)
    if context != "2d" and context != "facets":
        iix = (iix[0], iix[1], iix[2], iix[3], data_image.facet_id, iix[5])
        result.iter_id = iter_id
    else:
        result = decoalesce_visibility(result, **kwargs)
    result = (iix, result)
    label = "Degridding Kernel Update + Degrid (674.8 MB, 0.59 Tflop) "
    print(label + str(result))
    return result

def corrupt_handle(vis_graph_list, gt_graph, **kwargs):
    return vis_graph_list.mapValues(lambda vis: corrupt_kernel(vis, gt_graph, **kwargs))

def corrupt_kernel(vis, gt, **kwargs):
    if gt is None:
        gt = create_gaintable_from_blockvisibility(vis, **kwargs)
        gt = simulate_gaintable(gt, **kwargs)

    result = apply_gaintable(vis, gt)
    label = "corrupt Kernel"
    print(label + str(result))

    return result

def create_empty_handle(vis_graph_list, npixel, cellsize, frequency, channel_bandwidth, polarisation_frame):
    return vis_graph_list.map(lambda vis: (vis[0], create_image_from_visibility(vis[1], npixel=npixel, cellsize=cellsize,
                                                                       frequency=[frequency[vis[0][2]]],
                                                                       channel_bandwidth=[channel_bandwidth[vis[0][2]]],
                                                                       polarisation_frame=PolarisationFrame("stokesI")
                                                                       )))

def invert_handle(sc, template_model_graph, vis_graph_list, context, dopsf, normalize, facets, vis_slices, **kwargs):
    c = imaging_context(context)
    image_metadata = template_model_graph.mapValues(lambda im: (im.wcs, im.polarisation_frame, im.shape))
    if context == "2d":
        visibility = vis_graph_list.mapValues(lambda vis: coalesce_visibility(vis, **kwargs))
        return template_model_graph.join(visibility).map(lambda record: invert_kernel(record, c["invert"], dopsf, normalize, context, **kwargs))
    elif context == "facets":
        visibility = vis_graph_list.mapValues(lambda vis: coalesce_visibility(vis, **kwargs))
        return template_model_graph.flatMap(lambda im: scatter_image_flatmap(im, facets=facets, image_iter=c["image_iterator"], **kwargs), True)\
        .join(visibility).map(lambda record: invert_kernel(record, c["invert"], dopsf, normalize, context, **kwargs), True)\
        .combineByKey(gather_img_createConbiner_kernel, gather_img_margeValue_kernel, gather_img_mergeCombiner)\
        .join(image_metadata).mapValues(lambda data: gather_image_kernel(data, facets=facets, image_iter=c["image_iterator"], **kwargs))

    elif context == "facets_slice" or context == "facets_wstack":
        visibility = vis_graph_list.mapValues(lambda vis: coalesce_visibility(vis, **kwargs)).\
            flatMap(lambda vis: scatter_vis_flatmap(vis, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs), True)
        return template_model_graph.flatMap(lambda im: scatter_image_flatmap(im, facets=facets, image_iter=c["image_iterator"], **kwargs), True)\
        .join(visibility).map(lambda record: invert_kernel(record, c["invert"], dopsf, normalize, context, **kwargs))\
        .reduceByKey(sum_inver_image_reduce_kernel).map(change_key).mapValues(lambda im_sumwt: (normalize_sumwt(im_sumwt[0], im_sumwt[1]), im_sumwt[1]))\
        .combineByKey(gather_img_createConbiner_kernel, gather_img_margeValue_kernel, gather_img_mergeCombiner)\
        .join(image_metadata).mapValues(lambda data: gather_image_kernel(data, facets=facets, image_iter=c["image_iterator"], **kwargs))

    elif context == "facets_timeslice":
        visibility = vis_graph_list.mapValues(lambda vis: coalesce_visibility(vis, **kwargs)). \
            flatMap(lambda vis: scatter_vis_flatmap(vis, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs), True)
        return template_model_graph.flatMap(lambda im: scatter_image_flatmap(im, facets=facets, image_iter=c["image_iterator"], **kwargs), True) \
        .join(visibility).map(lambda record: invert_kernel(record, c["invert"], dopsf, normalize, context, **kwargs)) \
        .combineByKey(gather_img_createConbiner_kernel, gather_img_margeValue_kernel, gather_img_mergeCombiner) \
        .join(image_metadata).mapValues(lambda data: gather_image_kernel(data, facets=facets, image_iter=c["image_iterator"], **kwargs))\
        .map(change_key).reduceByKey(sum_inver_image_reduce_kernel).mapValues(lambda im_sumwt: (normalize_sumwt(im_sumwt[0], im_sumwt[1]), im_sumwt[1]))

    elif context == "slice" or context == "timeslice" or context == "wstack":
        visibility = vis_graph_list.mapValues(lambda vis: coalesce_visibility(vis, **kwargs)). \
            flatMap(lambda vis: scatter_vis_flatmap(vis, vis_slices=vis_slices, vis_iter=c["vis_iterator"], **kwargs), True)
        return template_model_graph.join(visibility).map(lambda record: invert_kernel(record, c["invert"], dopsf, normalize, context, **kwargs)) \
        .reduceByKey(sum_inver_image_reduce_kernel).mapValues(lambda im_sumwt: (normalize_sumwt(im_sumwt[0], im_sumwt[1]), im_sumwt[1]))

def invert_kernel(ixs, function, dopsf, normalize, context, **kwargs):
    iix, (data_image, data_visibility) = ixs
    facet_id = data_image.facet_id
    ix = None
    if context == "facets_slice" or context == "facets_wstack":
        ix = (iix[0], iix[1], iix[2], iix[3], facet_id, iix[5])
    elif context in ["facets_timeslice"]:
        ix = (iix[0], iix[1], iix[2], iix[3], data_visibility.iter_id, iix[5])
    else:
        ix = iix
    if data_visibility is not None:
        result = function(data_visibility, data_image, dopsf=dopsf, normalize=normalize, **kwargs)
        if context == "2d" or context == "facets":
            result = (ix, sum_invert_results([result]))
        else:
            result = (ix, result)
        result[1][0].facet_id = facet_id
    else:
        result = (create_empty_image_like(data_image), 0.0)
        if context == "2d" or context == "facets":
            result = (ix, sum_invert_results([result]))
        else:
            result = (ix, result)
        result[1][0].facet_id = facet_id
    label = "Invert Kernel "
    print(label + str(result))
    return result

def gather_img_createConbiner_kernel(im_sumwt):
    return {im_sumwt[0].facet_id: (im_sumwt[0].data, im_sumwt[1])}

def gather_img_margeValue_kernel(im1: dict, im2):
    im1[im2[0].facet_id] = (im2[0].data, im2[1])
    return im1

def gather_img_mergeCombiner(d1: dict, d2: dict):
    for key, value in d2.items():
        d1[key] = value
    return d1

def gather_image_kernel(data, facets, image_iter, **kwargs):
    data_dict, image_metadata = data
    wcs, polarisation_frame, shape = image_metadata
    result = create_image_from_array(np.empty(shape), wcs=wcs, polarisation_frame=polarisation_frame)
    i = 0
    sumwt = np.zeros([shape[0], shape[1]])
    for dpatch in image_iter(result, facets=facets, **kwargs):
        assert i < len(data_dict), "Too few results in gather_image_iteration_results"
        if data_dict[i] is not None:
            assert len(data_dict[i]) == 2, data_dict[i]
            dpatch.data[...] = data_dict[i][0]
            sumwt += data_dict[i][1]
            i += 1
    return result, sumwt

def sum_inver_image_reduce_kernel(im_sumwt1, im_sumwt2):
    first = True
    total_sumwt = 0.0
    ret_im = None
    facet_id = im_sumwt1[0].facet_id
    for im, sumwt in [im_sumwt1, im_sumwt2]:
        if isinstance(sumwt, numpy.ndarray):
            scale = sumwt[..., numpy.newaxis, numpy.newaxis]
        else:
            scale = sumwt
        if first:
            ret_im = copy_image(im)
            ret_im.facet_id = facet_id
            ret_im.data *= scale
            total_sumwt = sumwt
            first = False
        else:
            ret_im.data += scale * im.data
            total_sumwt += sumwt
    return (ret_im, total_sumwt)


