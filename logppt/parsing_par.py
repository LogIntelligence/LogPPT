import time

from logppt.parsing_cache import ParsingCache
import string

# import multiprocessing
import logging
from multiprocessing.managers import BaseManager
from tqdm import tqdm
import torch.multiprocessing as mp

try:
    mp.set_start_method('fork')
except RuntimeError:
    pass


# ctx_in_main = multiprocessing.get_context('forkserver')
# ctx_in_main.set_forkserver_preload(['logppt.parsing_cache', 'logppt.parsing_base', 'logppt.models.roberta', 'logppt.data.data_loader', 'logppt.arguments', 'logppt.trainer'])

cache_lock = mp.Lock()

logger = logging.getLogger("LogPPT")

def verify_template(template):
    template = template.replace("<*>", "")
    template = template.replace(" ", "")
    return any(char not in string.punctuation for char in template)

def get_template_line(log_line, device, model, vtoken, cache):
    # model.to(device)
    model.eval()
    log = " ".join(log_line.strip().split())
    results = cache.match_event(log)
    model_invocation = 0
    model_time = 0
    if results[0] != "NoMatch":
        template = (results[0], results[1]) # template, id
    else:
        with cache_lock:
            t0 = time.time()
            template = model.parse(log, device=device, vtoken=vtoken)
            model_time = time.time() - t0
            if verify_template(template):
                template_id = cache.add_templates(template, True, results[2])
                # print(f"Added template: {template} with id: {template_id}")
                # print(cache.template_list)
            else:
                template_id = None
            template = (template, template_id)
        model_invocation = 1
    return template, model_invocation, model_time

def template_extraction(model, device, log_lines, vtoken="virtual-param", n_workers=1):

    model.share_memory()
    logger.info("Starting template extraction")
    templates = []
    start_time = time.time()
    BaseManager.register('ParsingCache', ParsingCache)
    manager = BaseManager()
    manager.start()
    cache = manager.ParsingCache()
    with mp.Pool(n_workers) as executor:
        templates = list(executor.starmap(get_template_line, tqdm([(log_line, device, model, vtoken, cache) for log_line in log_lines], desc='Parsing')))
    
    manager.shutdown()
    no_of_invocations = sum([template[1] for template in templates])
    model_time = sum([template[2] for template in templates])
    templates = [template[0] for template in templates]

    logger.info(f"Total time taken: {time.time() - start_time}")
    logger.info(f"Total time taken by model: {model_time}")
    logger.info(f"No of model invocations: {no_of_invocations}")
    return templates, model_time, cache.get_template_list()
