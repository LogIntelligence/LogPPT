import time

from logppt.parsing_cache import ParsingCache
from tqdm import tqdm
import string
from multiprocessing import set_start_method
import multiprocessing
from joblib import Parallel, delayed, Memory
import logging

logger = logging.getLogger("LogPPT")


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

def template_extraction(model, device, log_lines, vtoken="virtual-param"):

    logger.info("Starting template extraction")
    model.eval()
    model.to(device)
    cache = ParsingCache()
    start_time = time.time()
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
            template = model.parse(log, device=device, vtoken=vtoken)
            model_time += time.time() - t0
            if verify_template(template):
                cache.add_templates(template, True, results[2])
            cache_for_all_invocations[log] = template
            templates.append(template)
            pbar.update(1)
    print(cache.template_tree)
    print(cache.threshold)
    logger.info(f"Total time taken: {time.time() - start_time}")
    logger.info(f"No of model invocations: {len(cache_for_all_invocations.keys())}")
    logger.info(f"Total time taken by model: {model_time}")
    return templates, model_time #, cache.get_template_list()