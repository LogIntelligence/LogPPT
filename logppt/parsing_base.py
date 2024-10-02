import time

from logppt.parsing_cache import ParsingCache
from tqdm import tqdm
import string
from multiprocessing import set_start_method
import multiprocessing
from joblib import Parallel, delayed, Memory
import logging

logger = logging.getLogger("LogPPT")
# try:
#     set_start_method('spawn')
# except RuntimeError:
#     pass


def verify_template(template):
    template = template.replace("<*>", "")
    template = template.replace(" ", "")
    return any(char not in string.punctuation for char in template)

def get_template_line(device, model, log_line, vtoken, cache):
    model.to(device)
    model.eval()
    log = " ".join(log_line.strip().split())
    results = cache.match_event(log)
    if results[0] != "NoMatch":
        return results[0]
    else:
        template = model.parse(log, device=device, vtoken=vtoken)
        if verify_template(template):
            cache.add_templates(template, True, results[2])
        return template

def template_extraction_joblib(model, devices, log_lines, vtoken="virtual-param"):
    cache_dir = "./cache"
    memory = Memory(cache_dir, verbose=0)
    get_template_cached = memory.cache(get_template)

    logger.info("Starting template extraction")
    with Parallel(n_jobs=len(devices), verbose=1) as parallel:
        t0 = time.time()
        results = parallel(delayed(get_template_cached)(devices[i], model, log_lines, vtoken) for i in range(len(devices)))
        templates = []
        model_time = 0
        no_of_invocations = 0
        for result in results:
            templates.extend(result[0])
            model_time += result[1]
            no_of_invocations += result[2]
    logger.info(f"Total time taken: {time.time() - t0}")
    logger.info(f"Total time taken by model: {model_time}")
    logger.info(f"No of model invocations: {no_of_invocations}")
    return templates, model_time


def template_extraction(model, devices, log_lines, vtoken="virtual-param"):

    logger.info("Starting template extraction")

    #multi processing with multiple GPU devices
    if len(devices) > 1:
        t0 = time.time()
        # devide log into len(devices) chunks
        chunk_size = len(log_lines) // len(devices)
        chunks = [log_lines[i:i + chunk_size] for i in range(0, len(log_lines), chunk_size)]
        # create a pool of workers
        pool = multiprocessing.Pool(processes=len(devices))
        results = []
        for i in range(len(devices)):
            results.append(pool.apply_async(get_template, args=(devices[i], model, chunks[i], vtoken)))
        pool.close()
        pool.join()
        templates = []
        model_time = 0
        no_of_invocations = 0
        for result in results:
            templates.extend(result.get()[0])
            model_time += result.get()[1]
            no_of_invocations += result.get()[2]
        logger.info(f"Total time taken: {time.time() - t0}")
        logger.info(f"Total time taken by model: {model_time}")
        logger.info(f"No of model invocations: {no_of_invocations}")
        return templates, model_time
    
    
    model.eval()
    cache = ParsingCache()
    t0 = time.time()
    templates = []
    cache_for_all_invocations = {}
    model_time = 0
    pbar = tqdm(total=len(log_lines), desc='Parsing')
    templates = []
    for log in log_lines:
        log = " ".join(log.strip().split())
        try:
            template = cache_for_all_invocations[log]
        except KeyError:
            template = None
        if template is not None:
            templates.append(template)
            pbar.update(1)
            continue
        results = cache.match_event(log)
        if results[0] != "NoMatch":
            templates.append(results[0])
            pbar.update(1)
            continue
        else:
            t0 = time.time()
            template = model.parse(log, device=devices[0], vtoken=vtoken)
            model_time += time.time() - t0
            if verify_template(template):
                cache.add_templates(template, True, results[2])
            cache_for_all_invocations[log] = template
            templates.append(template)
            pbar.update(1)
    logger.info(f"Total time taken: {time.time() - t0}")
    logger.info(f"No of model invocations: {len(cache_for_all_invocations.keys())}")
    logger.info(f"Total time taken by model: {model_time}")
    return templates, model_time