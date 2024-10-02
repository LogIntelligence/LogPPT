import time

from logppt.parsing_cache import ParsingCache
from tqdm import tqdm
import string

from concurrent.futures import ThreadPoolExecutor
import threading
import logging

logger = logging.getLogger("LogPPT")

def template_extraction_concurrent(model, device, log_lines, vtoken="virtual-param", n_workers=1):
    cache_lock = threading.Lock()
    cache = ParsingCache()

    def verify_template(template):
        template = template.replace("<*>", "")
        template = template.replace(" ", "")
        return any(char not in string.punctuation for char in template)

    def get_template_line(device, model, log_line, vtoken, cache):
        model.to(device)
        model.eval()
        log = " ".join(log_line.strip().split())
        results = cache.match_event(log)
        model_invocation = 0
        model_time = 0
        if results[0] != "NoMatch":
            template = results[0]
        else:
            with cache_lock:
                t0 = time.time()
                template = model.parse(log, device=device, vtoken=vtoken)
                model_time = time.time() - t0
                if verify_template(template):
                    cache.add_templates(template, True, results[2])
            model_invocation = 1
        return template, model_invocation, model_time

    logger.info("Starting template extraction")

    templates = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for log_line in log_lines:
            futures.append(executor.submit(get_template_line, device, model, log_line, vtoken, cache))
        
        for future in tqdm(futures, desc='Parsing', total=len(futures)):
            templates.append(future.result())

    no_of_invocations = sum([template[1] for template in templates])
    model_time = sum([template[2] for template in templates])
    templates = [template[0] for template in templates]

    logger.info(f"Total time taken: {time.time() - start_time}")
    logger.info(f"Total time taken by model: {model_time}")
    logger.info(f"No of model invocations: {no_of_invocations}")
    return templates, model_time
